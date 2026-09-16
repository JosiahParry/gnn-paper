# CA -> WV census-tract health transfer. Same inductive protocol as
# R/kc-to-ames-core.R (train region gets its own KNN graph, target region
# gets its own disjoint KNN graph, no edges cross the boundary), same
# arm roster, same winning recipe (node-mode LayerNorm + dropout). No
# prediction-level ensembling anywhere in this file -- every arm is scored
# as a single deployable model; repeated seeds characterize typical
# performance and spread only, never combined into one predictor.
#
# Target: DIABETES (diagnosed diabetes prevalence, % of adults).
# Predictors: ten other CDC PLACES measures -- risk behaviors, access to
# care, social needs, self-rated health -- deliberately not other Health
# Outcomes measures, so they're drivers, not restatements, of the target.

library(dplyr)
library(mirai)
library(sf)
library(sfdep)
library(spdep)
library(torch)
library(torchgnn)
library(parsnip)
library(recipes)
library(rsample)
library(workflows)
library(yardstick)

k_nb <- 15L
val_prop <- 0.2
n_epochs <- 500L
lr <- 0.01
patience <- 20L
sage_hidden <- c(56, 32, 16)

feature_cols <- c(
  "BINGE", "CSMOKING", "LPA", "SLEEP", "ACCESS2", "CHECKUP",
  "FOODINSECU", "HOUSINSECU", "LACKTRPT", "GHLTH"
)

layer_layer_norm_node <- function(dim) layer_layer_norm(dim, mode = "node")

# Data ---------------------------------------------------------------------

raw <- readRDS("data/cdc-places-raw.rds")

wide <- raw |>
  mutate(data_value = as.numeric(data_value)) |>
  select(locationname, stateabbr, measureid, data_value, geolocation) |>
  tidyr::pivot_wider(names_from = measureid, values_from = data_value)

coords <- do.call(rbind, lapply(wide$geolocation$coordinates, function(c) c(lon = c[1], lat = c[2])))
wide <- wide |>
  mutate(lon = coords[, "lon"], lat = coords[, "lat"]) |>
  select(-geolocation) |>
  filter(complete.cases(across(all_of(c("DIABETES", feature_cols)))))

tracts <- st_as_sf(wide, coords = c("lon", "lat"), crs = 4326) |> st_transform(3857)

ca <- tracts |> filter(stateabbr == "CA")
wv <- tracts |> filter(stateabbr == "WV")

cat(sprintf("CA (source) n = %d\n", nrow(ca)))
cat(sprintf("WV (target) n = %d\n", nrow(wv)))

X_of <- function(region) {
  d <- if (region == "ca") ca else wv
  as.matrix(st_drop_geometry(d)[, feature_cols])
}
y_of <- function(region) if (region == "ca") ca$DIABETES else wv$DIABETES
geom_of <- function(region) if (region == "ca") st_geometry(ca) else st_geometry(wv)

# Graphs ---------------------------------------------------------------------

# Same kernel machinery as R/airbnb-core-brow-sd.R (sfdep::st_kernel_weights,
# threshold_mult scales the median k-NN neighbour distance). kernel="uniform"
# (the default) reproduces the original unweighted behaviour exactly.
make_sub_graph <- function(geom, kernel = "uniform", threshold_mult = 1) {
  sub_nb <- sfdep::st_knn(geom, k = min(k_nb, length(geom) - 1L))
  from <- rep.int(seq_along(sub_nb), lengths(sub_nb))
  to <- unlist(sub_nb)

  if (kernel == "uniform") {
    weight <- NULL
    kw_floored <- NULL
  } else {
    coords <- sf::st_coordinates(geom)
    dists <- unlist(spdep::nbdists(sub_nb, coords))
    thr <- stats::median(dists) * threshold_mult
    kw <- sfdep::st_kernel_weights(sub_nb, geom, kernel = kernel, threshold = thr, adaptive = FALSE)
    weight <- unlist(kw)
    kw_floored <- lapply(kw, function(w) pmax(w, 1e-8))
  }

  adj <- adj_from_edgelist(from, to, weight = weight) |> add_graph_self_loops()
  lw <- if (is.null(kw_floored)) nb2listw(sub_nb) else nb2listw(sub_nb, glist = kw_floored, style = "W")
  list(adj = adj, lw = lw)
}

make_scaler <- function(ids) {
  mu <- colMeans(X_of("ca")[ids, , drop = FALSE])
  sdv <- apply(X_of("ca")[ids, , drop = FALSE], 2, sd)
  sdv[sdv == 0] <- 1
  function(ids2, region) scale(X_of(region)[ids2, , drop = FALSE], mu, sdv)
}

# Model arms -------------------------------------------------------------

frame_of <- function(ids, region) {
  d <- as.data.frame(X_of(region)[ids, , drop = FALSE])
  d$y <- y_of(region)[ids]
  d
}

fit_ols <- function(fit_ids) {
  m <- lm(y ~ ., data = frame_of(fit_ids, "ca"))
  function(ids, region) as.numeric(predict(m, newdata = frame_of(ids, region)))
}

build_df <- function(ids, region, lw, with_lags) {
  X <- X_of(region)[ids, , drop = FALSE]
  out <- as.data.frame(X)
  if (with_lags) {
    lags <- as.data.frame(lag.listw(lw, X))
    colnames(lags) <- paste0("lag_", feature_cols)
    out <- cbind(out, lags)
  }
  out$y <- y_of(region)[ids]
  out
}

xgb_spec <- boost_tree(trees = 500, stop_iter = patience) |>
  set_engine("xgboost", objective = "reg:squarederror", validation = val_prop) |>
  set_mode("regression")

xgb_workflow <- function(template) {
  workflow() |>
    add_recipe(recipe(y ~ ., data = template) |> step_normalize(all_numeric_predictors())) |>
    add_model(xgb_spec)
}

default_kspec <- list(kernel = "uniform", threshold_mult = 1)

fit_xgb <- function(fit_ids, with_lags, seed, kspec = default_kspec) {
  g <- make_sub_graph(geom_of("ca")[fit_ids], kspec$kernel, kspec$threshold_mult)
  train_df <- build_df(fit_ids, "ca", g$lw, with_lags)
  set.seed(seed)
  fitted <- suppressWarnings(fit(xgb_workflow(train_df), data = train_df))
  function(ids, region) {
    g2 <- make_sub_graph(geom_of(region)[ids], kspec$kernel, kspec$threshold_mult)
    predict(fitted, new_data = build_df(ids, region, g2$lw, with_lags))$.pred
  }
}

fit_sage <- function(train_id, val_id, norm, dropout = 0, wd = 0, kspec = default_kspec) {
  ids <- c(train_id, val_id)
  scaler <- make_scaler(ids)
  g <- make_sub_graph(geom_of("ca")[ids], kspec$kernel, kspec$threshold_mult)

  x_t <- torch_tensor(scaler(ids, "ca"), dtype = torch_float32())
  y_t <- torch_tensor(y_of("ca")[ids], dtype = torch_float32())$view(c(-1, 1))

  train_idx <- seq_along(train_id)
  val_idx <- seq_along(val_id) + length(train_id)

  model <- model_sage(
    in_features = length(feature_cols), hidden_dims = sage_hidden, out_features = 1,
    norm = norm, dropout = dropout
  )
  optimizer <- optim_adam(model$parameters, lr = lr, weight_decay = wd)

  best_val <- Inf; best_state <- NULL; no_improve <- 0L
  for (epoch in seq_len(n_epochs)) {
    model$train()
    optimizer$zero_grad()
    out <- model(x_t, g$adj)
    loss <- nnf_l1_loss(out[train_idx, ], y_t[train_idx, ])
    loss$backward()
    optimizer$step()
    with_no_grad({
      model$eval()
      vl <- nnf_l1_loss(model(x_t, g$adj)[val_idx, ], y_t[val_idx, ])$item()
    })
    if (vl < best_val) {
      best_val <- vl; best_state <- lapply(model$state_dict(), \(t) t$clone()); no_improve <- 0L
    } else {
      no_improve <- no_improve + 1L
    }
    if (no_improve >= patience) break
  }
  model$load_state_dict(best_state)
  model$eval()

  function(ids2, region) {
    g2 <- make_sub_graph(geom_of(region)[ids2], kspec$kernel, kspec$threshold_mult)
    x2 <- torch_tensor(scaler(ids2, region), dtype = torch_float32())
    with_no_grad({ as.numeric(model(x2, g2$adj)$squeeze()) })
  }
}

arms <- list(
  OLS = function(train_id, val_id, seed, kspec = default_kspec) fit_ols(c(train_id, val_id)),
  XGBoost = function(train_id, val_id, seed, kspec = default_kspec) fit_xgb(c(train_id, val_id), with_lags = FALSE, seed = seed),
  `XGBoost + lags` = function(train_id, val_id, seed, kspec = default_kspec) fit_xgb(c(train_id, val_id), with_lags = TRUE, seed = seed, kspec = kspec),
  GraphSAGE = function(train_id, val_id, seed, kspec = default_kspec) {
    torch_manual_seed(seed)
    fit_sage(train_id, val_id, norm = NULL, kspec = kspec)
  },
  `GraphSAGE + LayerNorm` = function(train_id, val_id, seed, kspec = default_kspec) {
    torch_manual_seed(seed)
    fit_sage(train_id, val_id, norm = layer_layer_norm_node, dropout = 0.1, wd = 1e-4, kspec = kspec)
  }
)

stochastic <- c("XGBoost", "XGBoost + lags", "GraphSAGE", "GraphSAGE + LayerNorm")

# Scoring --------------------------------------------------------------------

reg_metrics <- metric_set(mae, rmse, rsq, rsq_trad)
score <- function(truth, estimate) {
  d <- data.frame(truth = truth, estimate = estimate)
  m <- reg_metrics(d, truth = truth, estimate = estimate)
  data.frame(
    mae = m$.estimate[m$.metric == "mae"],
    rmse = m$.estimate[m$.metric == "rmse"],
    rsq = m$.estimate[m$.metric == "rsq"],
    rsq_trad = m$.estimate[m$.metric == "rsq_trad"],
    bias = mean(estimate - truth),
    cal_slope = unname(coef(lm(truth ~ estimate))[2])
  )
}

# Folds ------------------------------------------------------------------

set.seed(0)
final_val <- sample(nrow(ca), size = floor(val_prop * nrow(ca)))
final_train <- setdiff(seq_len(nrow(ca)), final_val)

# Worker pool ------------------------------------------------------------

start_daemons <- function(n_daemons = max(1L, parallel::detectCores() - 2L)) {
  core <- normalizePath("R/cdc-places-core.R", mustWork = TRUE)
  daemons(n_daemons)
  everywhere(
    { source(core_path, local = FALSE); torch_set_num_threads(1L) },
    .args = list(core_path = core), .min = n_daemons
  )
  cat(sprintf("%d daemons up, torch pinned to 1 thread each\n", n_daemons))
  invisible(n_daemons)
}

wv_task <- function(spec) {
  kspec <- if (is.null(spec$kspec)) default_kspec else spec$kspec
  predictor <- arms[[spec$arm]](final_train, final_val, spec$seed, kspec)
  cbind(arm = spec$arm, seed = spec$seed,
        kernel = kspec$kernel, threshold_mult = kspec$threshold_mult,
        score(y_of("wv"), predictor(seq_len(nrow(wv)), "wv")))
}

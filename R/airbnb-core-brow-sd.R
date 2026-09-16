# Broward County FL -> San Diego CA Airbnb listing-price transfer.
# Picked via R/airbnb-screen-pairs.R: a cheap cross-region OLS screen across
# 13 cities x 12 pairs each, before spending any GraphSAGE compute, to find
# a pair where the underlying covariate -> relative-price relationship is
# actually similar (unlike LA -> NYC or Chicago -> Nashville, both of which
# showed genuinely different fitted coefficients between regions). Broward
# -> San Diego topped the large-city list: rsq_trad=0.648, bias=-0.085,
# cal_slope=1.095 from OLS alone.
#
# Same protocol as R/airbnb-core.R: inductive KNN graphs, Gaussian-kernel
# edge weights, node-mode LayerNorm + dropout=0.1 + weight_decay=1e-4, no
# prediction ensembling -- single deployable models only.

library(dplyr)
library(mirai)
library(sf)
library(sfdep)
library(spdep)
library(torch)
library(torchgnn)
library(parsnip)
library(recipes)
library(workflows)
library(yardstick)

k_nb <- 30L
val_prop <- 0.2
n_epochs <- 500L
lr <- 0.01
patience <- 20L
sage_hidden <- c(56, 32, 16)

feature_cols <- c(
  "accommodates", "bathrooms", "bedrooms", "beds", "minimum_nights",
  "availability_365", "number_of_reviews", "is_entire_home"
)

layer_layer_norm_node <- function(dim) layer_layer_norm(dim, mode = "node")

# Data -------------------------------------------------------------------

cleaned <- readRDS("data/airbnb-all-cities-clean.rds")
brow <- cleaned$broward |> st_as_sf(coords = c("lon", "lat"), crs = 4326) |> st_transform(3857)
sd_ <- cleaned$sandiego |> st_as_sf(coords = c("lon", "lat"), crs = 4326) |> st_transform(3857)

cat(sprintf("Broward (source)   n = %d\n", nrow(brow)))
cat(sprintf("San Diego (target) n = %d\n", nrow(sd_)))

X_of <- function(region) as.matrix(st_drop_geometry(if (region == "brow") brow else sd_)[, feature_cols])
y_of <- function(region) if (region == "brow") brow$y else sd_$y
geom_of <- function(region) st_geometry(if (region == "brow") brow else sd_)

# Graphs ---------------------------------------------------------------------

# Kernel weighting via sfdep::st_kernel_weights() (the package's own,
# idiomatic mechanism) instead of a hand-rolled Gaussian formula.
# threshold_mult scales sfdep::critical_threshold(geom) -- the minimum
# distance that still gives every point >=1 neighbour, i.e. the tightest
# threshold that doesn't strand anyone. adaptive=TRUE uses each node's own
# max-neighbour-distance as its threshold instead of one global value
# (per sfdep docs). Because these kernels have compact support (K(z)=0 for
# |z|>=1), a small threshold_mult also *prunes* the nominal k_nb neighbour
# set down to genuinely close points -- a KNN-candidate-pool + distance-band
# hybrid in one mechanism, without a slow true distance-band query.
#
# Diagnostic that motivated pushing this tight (Moran's I of OLS residuals
# vs. covariates on the San Diego graph, k=30): the *ratio* of covariate
# spatial signal to residual spatial signal only exceeds 1 at a much
# steeper bandwidth than the naive default -- a tighter kernel lets the
# covariates' own spatial pattern actually cover what the residual needs.
# NOTE: sfdep::critical_threshold() turned out to be a bad scale reference
# for this data -- Airbnb listings have many exact-duplicate coordinates
# (privacy jittering / building-level rounding), producing sub-graphs that
# inflate critical_threshold to ~4.7km, far looser than the ~20-50m that
# actually worked in the earlier ad-hoc sweep. threshold_mult here instead
# scales the median k-NN neighbour distance directly (consistent with that
# earlier, working scale) unless adaptive=TRUE, which ignores threshold_mult
# entirely and uses sfdep's own per-node adaptive threshold.
make_sub_graph <- function(geom, kernel = "gaussian", threshold_mult = 1, adaptive = FALSE) {
  sub_nb <- sfdep::st_knn(geom, k = min(k_nb, length(geom) - 1L))
  from <- rep.int(seq_along(sub_nb), lengths(sub_nb))
  to <- unlist(sub_nb)
  coords <- sf::st_coordinates(geom)
  dists <- unlist(spdep::nbdists(sub_nb, coords))
  thr <- stats::median(dists) * threshold_mult
  kw <- sfdep::st_kernel_weights(sub_nb, geom, kernel = kernel, threshold = thr, adaptive = adaptive)
  weight <- unlist(kw)
  adj <- adj_from_edgelist(from, to, weight = weight) |> add_graph_self_loops()
  # lw carries the SAME kernel weights (row-standardized), not nb2listw's
  # silent binary default -- XGBoost+lags' lag.listw() was computing an
  # unweighted neighbour mean even when kernel="gaussian" was requested, so
  # it never actually saw the tuned kernel. Fixed here. A tight threshold
  # can zero out every neighbour for some nodes (compact kernel support);
  # adj_row_normalize() handles that gracefully (deg_inv Inf -> 0) but
  # nb2listw() errors on a zero-sum row, so floor every weight at a tiny
  # epsilon -- negligible for populated rows, keeps sparse ones finite.
  kw_floored <- lapply(kw, function(w) pmax(w, 1e-8))
  lw <- nb2listw(sub_nb, glist = kw_floored, style = "W")
  list(adj = adj, lw = lw)
}

make_scaler <- function(ids) {
  mu <- colMeans(X_of("brow")[ids, , drop = FALSE])
  sdv <- apply(X_of("brow")[ids, , drop = FALSE], 2, sd)
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
  m <- lm(y ~ ., data = frame_of(fit_ids, "brow"))
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

default_kspec <- list(kernel = "gaussian", threshold_mult = 1, adaptive = FALSE)

fit_xgb <- function(fit_ids, with_lags, seed, kspec = default_kspec) {
  g <- make_sub_graph(geom_of("brow")[fit_ids], kspec$kernel, kspec$threshold_mult, kspec$adaptive)
  train_df <- build_df(fit_ids, "brow", g$lw, with_lags)
  set.seed(seed)
  fitted <- suppressWarnings(fit(xgb_workflow(train_df), data = train_df))
  function(ids, region) {
    g2 <- make_sub_graph(geom_of(region)[ids], kspec$kernel, kspec$threshold_mult, kspec$adaptive)
    predict(fitted, new_data = build_df(ids, region, g2$lw, with_lags))$.pred
  }
}

fit_sage <- function(train_id, val_id, norm, dropout = 0, wd = 0, kspec = default_kspec) {
  ids <- c(train_id, val_id)
  scaler <- make_scaler(ids)
  g <- make_sub_graph(geom_of("brow")[ids], kspec$kernel, kspec$threshold_mult, kspec$adaptive)

  x_t <- torch_tensor(scaler(ids, "brow"), dtype = torch_float32())
  y_t <- torch_tensor(y_of("brow")[ids], dtype = torch_float32())$view(c(-1, 1))

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
    g2 <- make_sub_graph(geom_of(region)[ids2], kspec$kernel, kspec$threshold_mult, kspec$adaptive)
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
final_val <- sample(nrow(brow), size = floor(val_prop * nrow(brow)))
final_train <- setdiff(seq_len(nrow(brow)), final_val)

# Worker pool ------------------------------------------------------------

start_daemons <- function(n_daemons = max(1L, parallel::detectCores() - 2L)) {
  core <- normalizePath("R/airbnb-core-brow-sd.R", mustWork = TRUE)
  daemons(n_daemons)
  everywhere(
    { source(core_path, local = FALSE); torch_set_num_threads(1L) },
    .args = list(core_path = core), .min = n_daemons
  )
  cat(sprintf("%d daemons up, torch pinned to 1 thread each\n", n_daemons))
  invisible(n_daemons)
}

sd_task <- function(spec) {
  kspec <- if (is.null(spec$kspec)) default_kspec else spec$kspec
  predictor <- arms[[spec$arm]](final_train, final_val, spec$seed, kspec)
  cbind(arm = spec$arm, seed = spec$seed,
        kernel = kspec$kernel, threshold_mult = kspec$threshold_mult, adaptive = kspec$adaptive,
        score(y_of("sd"), predictor(seq_len(nrow(sd_)), "sd")))
}

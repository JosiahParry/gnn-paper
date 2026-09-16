# Raster transfer core: gridded target, one region -> another.
#
# Differs from R/pair-core.R in one way that matters: a raster has a natural
# neighbour definition (the cells that touch you), so this core can build the
# graph three ways and compare them. Every other transfer in this project
# used KNN-30 because the points were irregular and there was no alternative;
# on a grid there is, and it has never been tested.
#
#   queen   - the 8 touching cells. What an ArcGIS user means by "neighbours".
#   queen2  - the 24 cells within two steps. A wider natural neighbourhood.
#   knn30   - 30 nearest cell centres, matching every other test here.
#
# Coordinates are METRES in an equal-area projection (see R/heat-fetch.R), not
# degrees. sf objects are built with crs = NA so every distance is plain
# Euclidean on those metres; labelling them 4326 would silently treat metres
# as degrees.
#
# Configure before sourcing:
#   options(raster_src = "cornbelt", raster_tgt = "ohiovalley",
#           raster_graph = "queen")

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
CELL <- getOption("raster_cell", 5000)   # grid spacing in metres (v1 5 km, v2 1 km)

layer_layer_norm_node <- function(dim) layer_layer_norm(dim, mode = "node")

SRC <- getOption("raster_src", "cornbelt")
TGT <- getOption("raster_tgt", "ohiovalley")
DATA <- getOption("raster_data", "data/heat-clean.rds")

cleaned <- readRDS(DATA)
stopifnot(all(c(SRC, TGT) %in% names(cleaned)))

as_grid <- function(d) st_as_sf(d, coords = c("lon", "lat"), crs = NA)
src_sf <- as_grid(cleaned[[SRC]])
tgt_sf <- as_grid(cleaned[[TGT]])

feature_cols <- setdiff(names(st_drop_geometry(src_sf)), "y")

cat(sprintf("%s (source) n = %d\n%s (target) n = %d\nfeatures (%d): %s\n",
            SRC, nrow(src_sf), TGT, nrow(tgt_sf), length(feature_cols),
            paste(feature_cols, collapse = ", ")))

reg_sf <- function(region) if (region == "src") src_sf else tgt_sf
X_of <- function(region) as.matrix(st_drop_geometry(reg_sf(region))[, feature_cols])
y_of <- function(region) reg_sf(region)$y
geom_of <- function(region) st_geometry(reg_sf(region))

# Graphs -----------------------------------------------------------------

# Queen adjacency by distance band: orthogonal neighbours sit at CELL,
# diagonals at CELL*sqrt(2) = 7071 m, and the next ring starts at 2*CELL.
# A 1.45-cell radius therefore captures exactly the 8 touching cells.
GRAPH_RADIUS <- c(queen = 1.45, queen2 = 2.9)

make_sub_graph <- function(geom, graph = "queen", kernel = "uniform", threshold_mult = 1) {
  coords <- sf::st_coordinates(geom)
  if (graph == "knn30") {
    sub_nb <- sfdep::st_knn(geom, k = min(k_nb, length(geom) - 1L))
  } else {
    sub_nb <- spdep::dnearneigh(coords, 0, CELL * GRAPH_RADIUS[[graph]])
    # Isolated cells (holes in the raster) would give zero-weight rows, which
    # spdep rejects; give them a self-neighbour so they are still predicted.
    empty <- which(vapply(sub_nb, function(z) identical(z, 0L), logical(1)))
    for (i in empty) sub_nb[[i]] <- i
    class(sub_nb) <- "nb"
  }

  from <- rep.int(seq_along(sub_nb), lengths(sub_nb))
  to <- unlist(sub_nb)

  if (kernel == "uniform") {
    weight <- NULL; kw_floored <- NULL
  } else {
    dists <- unlist(spdep::nbdists(sub_nb, coords))
    thr <- stats::median(dists) * threshold_mult
    kw <- sfdep::st_kernel_weights(sub_nb, geom, kernel = kernel,
                                   threshold = thr, adaptive = FALSE)
    weight <- unlist(kw)
    kw_floored <- lapply(kw, function(w) pmax(w, 1e-8))
  }

  adj <- adj_from_edgelist(from, to, weight = weight) |> add_graph_self_loops()
  lw <- if (is.null(kw_floored)) nb2listw(sub_nb, zero.policy = TRUE)
        else nb2listw(sub_nb, glist = kw_floored, style = "W", zero.policy = TRUE)
  list(adj = adj, lw = lw)
}

make_scaler <- function(ids) {
  mu <- colMeans(X_of("src")[ids, , drop = FALSE])
  sdv <- apply(X_of("src")[ids, , drop = FALSE], 2, sd)
  sdv[sdv == 0] <- 1
  function(ids2, region) scale(X_of(region)[ids2, , drop = FALSE], mu, sdv)
}

# Arms -------------------------------------------------------------------

frame_of <- function(ids, region) {
  d <- as.data.frame(X_of(region)[ids, , drop = FALSE])
  d$y <- y_of(region)[ids]
  d
}

fit_ols <- function(fit_ids) {
  m <- lm(y ~ ., data = frame_of(fit_ids, "src"))
  function(ids, region) as.numeric(predict(m, newdata = frame_of(ids, region)))
}

build_df <- function(ids, region, lw, with_lags) {
  X <- X_of(region)[ids, , drop = FALSE]
  out <- as.data.frame(X)
  if (with_lags) {
    lags <- as.data.frame(lag.listw(lw, X, zero.policy = TRUE))
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

default_g <- list(graph = getOption("raster_graph", "queen"),
                  kernel = "uniform", threshold_mult = 1)

sub_graph <- function(geom, g) make_sub_graph(geom, g$graph, g$kernel, g$threshold_mult)

fit_xgb <- function(fit_ids, with_lags, seed, g = default_g) {
  gr <- sub_graph(geom_of("src")[fit_ids], g)
  train_df <- build_df(fit_ids, "src", gr$lw, with_lags)
  set.seed(seed)
  fitted <- suppressWarnings(fit(xgb_workflow(train_df), data = train_df))
  function(ids, region) {
    g2 <- sub_graph(geom_of(region)[ids], g)
    predict(fitted, new_data = build_df(ids, region, g2$lw, with_lags))$.pred
  }
}

fit_sage <- function(train_id, val_id, norm, dropout = 0, wd = 0, g = default_g) {
  ids <- c(train_id, val_id)
  scaler <- make_scaler(ids)
  gr <- sub_graph(geom_of("src")[ids], g)

  x_t <- torch_tensor(scaler(ids, "src"), dtype = torch_float32())
  y_t <- torch_tensor(y_of("src")[ids], dtype = torch_float32())$view(c(-1, 1))
  train_idx <- seq_along(train_id)
  val_idx <- seq_along(val_id) + length(train_id)

  model <- model_sage(in_features = length(feature_cols), hidden_dims = sage_hidden,
                      out_features = 1, norm = norm, dropout = dropout)
  optimizer <- optim_adam(model$parameters, lr = lr, weight_decay = wd)

  best_val <- Inf; best_state <- NULL; no_improve <- 0L
  for (epoch in seq_len(n_epochs)) {
    model$train(); optimizer$zero_grad()
    out <- model(x_t, gr$adj)
    loss <- nnf_l1_loss(out[train_idx, ], y_t[train_idx, ])
    loss$backward(); optimizer$step()
    with_no_grad({
      model$eval()
      vl <- nnf_l1_loss(model(x_t, gr$adj)[val_idx, ], y_t[val_idx, ])$item()
    })
    if (vl < best_val) {
      best_val <- vl; best_state <- lapply(model$state_dict(), \(t) t$clone()); no_improve <- 0L
    } else no_improve <- no_improve + 1L
    if (no_improve >= patience) break
  }
  model$load_state_dict(best_state); model$eval()

  function(ids2, region) {
    g2 <- sub_graph(geom_of(region)[ids2], g)
    x2 <- torch_tensor(scaler(ids2, region), dtype = torch_float32())
    with_no_grad({ as.numeric(model(x2, g2$adj)$squeeze()) })
  }
}

arms <- list(
  OLS = function(tr, va, seed, g = default_g) fit_ols(c(tr, va)),
  XGBoost = function(tr, va, seed, g = default_g) fit_xgb(c(tr, va), FALSE, seed),
  `XGBoost + lags` = function(tr, va, seed, g = default_g) fit_xgb(c(tr, va), TRUE, seed, g),
  GraphSAGE = function(tr, va, seed, g = default_g) {
    torch_manual_seed(seed); fit_sage(tr, va, norm = NULL, g = g)
  },
  `GraphSAGE + LayerNorm` = function(tr, va, seed, g = default_g) {
    torch_manual_seed(seed)
    fit_sage(tr, va, norm = layer_layer_norm_node, dropout = 0.1, wd = 1e-4, g = g)
  }
)

stochastic <- c("XGBoost", "XGBoost + lags", "GraphSAGE", "GraphSAGE + LayerNorm")

reg_metrics <- metric_set(mae, rmse, rsq, rsq_trad)
score <- function(truth, estimate) {
  d <- data.frame(truth = truth, estimate = estimate)
  m <- reg_metrics(d, truth = truth, estimate = estimate)
  data.frame(mae = m$.estimate[m$.metric == "mae"],
             rmse = m$.estimate[m$.metric == "rmse"],
             rsq = m$.estimate[m$.metric == "rsq"],
             rsq_trad = m$.estimate[m$.metric == "rsq_trad"],
             bias = mean(estimate - truth),
             cal_slope = unname(coef(lm(truth ~ estimate))[2]))
}

set.seed(0)
final_val <- sample(nrow(src_sf), size = floor(val_prop * nrow(src_sf)))
final_train <- setdiff(seq_len(nrow(src_sf)), final_val)

start_daemons <- function(n_daemons = max(1L, parallel::detectCores() - 2L)) {
  core <- normalizePath("R/raster-core.R", mustWork = TRUE)
  daemons(n_daemons)
  everywhere({
    options(raster_src = ps, raster_tgt = pt, raster_data = pd, raster_cell = pc)
    source(core_path, local = FALSE)
    torch_set_num_threads(1L)
  }, .args = list(core_path = core, ps = SRC, pt = TGT, pd = DATA, pc = CELL),
     .min = n_daemons)
  cat(sprintf("%d daemons up (%s -> %s)\n", n_daemons, SRC, TGT))
  invisible(n_daemons)
}

target_task <- function(spec) {
  g <- if (is.null(spec$g)) default_g else spec$g
  predictor <- arms[[spec$arm]](final_train, final_val, spec$seed, g)
  cbind(src = SRC, tgt = TGT, arm = spec$arm, seed = spec$seed, graph = g$graph,
        kernel = g$kernel, threshold_mult = g$threshold_mult,
        score(y_of("tgt"), predictor(seq_len(nrow(tgt_sf)), "tgt")))
}

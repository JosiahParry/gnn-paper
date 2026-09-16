# Controlled domain-shift transfer experiment.
#
# KC -> Ames tells us GraphSAGE + LayerNorm recovers most of a catastrophic
# transfer failure, but it is one noisy real-world data point: we don't know
# the true function, and we can't dial the amount of distribution shift up or
# down. This experiment builds two disjoint synthetic regions -- "source"
# (fit) and "target" (predict, never seen during training, own KNN graph, no
# edges back to source) -- under the SAME coefficient function but with the
# target's covariates shifted by a controllable number of source standard
# deviations. That isolates exactly the mechanism the KC -> Ames calibration
# diagnostics pointed at: does an arm's *prediction scale* survive a covariate
# shift it was never fit on.
#
# Every design choice mirrors R/kc-to-ames-core.R: L1 loss, Adam, early
# stopping on a validation slice of the source region, node-mode LayerNorm,
# the same five covariate/beta shape as R/sim-core.R's Scenario A/C so this
# reads as the same DGP family under genuine (not within-lattice) shift.

library(dplyr)
library(mirai)
library(sf)
library(sfdep)
library(spdep)
library(torch)
library(torchgnn)
library(yardstick)
library(parsnip)
library(recipes)
library(workflows)

k_nb <- 15L
val_prop <- 0.2
n_epochs <- 500L
lr <- 0.01
patience <- 20L
sage_hidden <- c(56, 32, 16)

beta <- c(1, rep(0.3, 5))
sigma <- 1
mu0 <- c(0, 1, -1, 2.5, 0.5)
sd0 <- sqrt(c(1, 0.5, 1.5, 2, 1))

xb <- function(x, b = beta) as.vector(cbind(1, as.matrix(x)) %*% b)

# Same nonlinear form as R/sim-core.R's Scenario B, so the nonlinear
# domain-shift variant reads as the same DGP family. Linear OLS is
# misspecified for this one -- unlike the pure-linear DGP, it no longer gets
# a by-construction advantage under a location shift.
m_nonlinear <- function(x) {
  2 * sin(x[, 1]) + x[, 2]^2 - 1.5 * x[, 3] * x[, 4]
}

layer_layer_norm_node <- function(dim) layer_layer_norm(dim, mode = "node")

# Region builders ----------------------------------------------------------
#
# Each region is its own lattice with its own KNN-k graph; source and target
# never share a node or an edge, matching the inductive protocol used
# everywhere else in this project. dgp = "linear" matches the original
# experiment (xb only); "nonlinear" adds m_nonlinear(x) so OLS is
# misspecified too, not just disadvantaged relative to a shift it handles
# perfectly by construction.
build_region <- function(cols, rows, shift, seed, dgp = "linear") {
  set.seed(seed)
  grid <- make_square_grid(cols, rows) |> st_set_crs(3857)
  centroids <- st_centroid(grid)
  n <- nrow(centroids)

  nb <- sfdep::st_knn(centroids, k = min(k_nb, n - 1L))
  lw <- nb2listw(nb)
  edges <- list(
    from = rep.int(seq_along(nb), lengths(nb)),
    to = unlist(nb)
  )
  adj <- adj_from_edgelist(edges$from, edges$to) |> add_graph_self_loops()

  mu <- mu0 + shift * sd0
  x <- vapply(seq_len(5), \(j) rnorm(n, mu[j], sd0[j]), numeric(n))
  colnames(x) <- paste0("x", 1:5)
  mean_fn <- if (dgp == "nonlinear") xb(x) + m_nonlinear(x) else xb(x)
  y <- mean_fn + rnorm(n, sd = sigma)

  list(n = n, x = x, y = y, adj = adj, lw = lw, centroids = centroids)
}

make_square_grid <- function(cols, rows) {
  pts <- expand.grid(x = seq_len(cols), y = seq_len(rows))
  st_as_sf(pts, coords = c("x", "y"))
}

# Model arms -----------------------------------------------------------------
#
# All fit on source (train/val split inside it), predict on the whole target
# region's own graph. XGBoost's internal validation split is unseeded by the
# engine, so it (like GraphSAGE) needs a fresh seed per replicate; OLS is the
# only deterministic arm and is fit once per shift level.

fit_predict_ols <- function(src, tgt) {
  train <- as.data.frame(cbind(y = src$y, src$x))
  test <- as.data.frame(tgt$x)
  fit <- lm(y ~ ., data = train)
  data.frame(truth = tgt$y, estimate = as.numeric(predict(fit, newdata = test)))
}

xgb_spec <- boost_tree(trees = 500, stop_iter = patience) |>
  set_engine("xgboost", objective = "reg:absoluteerror", validation = val_prop) |>
  set_mode("regression")

build_lagged <- function(x, y, lw, with_lags) {
  out <- as.data.frame(x)
  if (with_lags) {
    wx <- lag.listw(lw, x)
    colnames(wx) <- paste0("lag_", colnames(x))
    out <- cbind(out, wx)
  }
  out$y <- y
  out
}

fit_predict_xgb <- function(src, tgt, with_lags, seed) {
  train_df <- build_lagged(src$x, src$y, src$lw, with_lags)
  wf <- workflow() |>
    add_recipe(recipe(y ~ ., data = train_df) |> step_normalize(all_numeric_predictors())) |>
    add_model(xgb_spec)
  set.seed(seed)
  fitted <- suppressWarnings(fit(wf, data = train_df))
  test_df <- build_lagged(tgt$x, tgt$y, tgt$lw, with_lags)
  data.frame(truth = tgt$y, estimate = predict(fitted, new_data = test_df)$.pred)
}

fit_predict_sage <- function(src, tgt, norm, seed, dropout = 0) {
  torch_manual_seed(seed)
  n <- src$n
  val_id <- sample(n, size = floor(val_prop * n))
  train_id <- setdiff(seq_len(n), val_id)

  mu <- colMeans(src$x[train_id, , drop = FALSE])
  sdv <- apply(src$x[train_id, , drop = FALSE], 2, sd)

  x_t <- torch_tensor(scale(src$x, mu, sdv), dtype = torch_float32())
  y_t <- torch_tensor(src$y, dtype = torch_float32())$view(c(-1, 1))

  model <- model_sage(
    in_features = ncol(src$x), hidden_dims = sage_hidden, out_features = 1,
    norm = norm, dropout = dropout
  )
  optimizer <- optim_adam(model$parameters, lr = lr)

  best_val <- Inf
  best_state <- NULL
  no_improve <- 0L
  for (epoch in seq_len(n_epochs)) {
    model$train()
    optimizer$zero_grad()
    out <- model(x_t, src$adj)
    loss <- nnf_l1_loss(out[train_id, ], y_t[train_id, ])
    loss$backward()
    optimizer$step()
    with_no_grad({
      model$eval()
      vl <- nnf_l1_loss(model(x_t, src$adj)[val_id, ], y_t[val_id, ])$item()
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

  x_tgt <- torch_tensor(scale(tgt$x, mu, sdv), dtype = torch_float32())
  with_no_grad({
    est <- as.numeric(model(x_tgt, tgt$adj)$squeeze())
  })
  data.frame(truth = tgt$y, estimate = est)
}

# Scoring --------------------------------------------------------------------

reg_metrics <- metric_set(mae, rmse, rsq, rsq_trad)
score <- function(d) {
  m <- reg_metrics(d, truth = truth, estimate = estimate)
  data.frame(
    mae = m$.estimate[m$.metric == "mae"],
    rmse = m$.estimate[m$.metric == "rmse"],
    rsq = m$.estimate[m$.metric == "rsq"],
    rsq_trad = m$.estimate[m$.metric == "rsq_trad"],
    bias = mean(d$estimate - d$truth),
    cal_slope = unname(coef(lm(truth ~ estimate, data = d))[2])
  )
}

# Sweep grid -------------------------------------------------------------

shift_levels <- c(0, 1, 2, 3)
n_seeds <- 8L
src_cols <- 60L; src_rows <- 50L   # n_source = 3000
tgt_cols <- 30L; tgt_rows <- 30L   # n_target =  900

region_cache <- new.env()

get_regions <- function(shift, dgp = "linear") {
  key <- paste(dgp, shift, sep = "_")
  if (!is.null(region_cache[[key]])) return(region_cache[[key]])
  src <- build_region(src_cols, src_rows, shift = 0, seed = 1000L, dgp = dgp)
  tgt <- build_region(tgt_cols, tgt_rows, shift = shift, seed = 2000L + round(shift * 100), dgp = dgp)
  out <- list(src = src, tgt = tgt)
  region_cache[[key]] <- out
  out
}

domain_shift_task <- function(spec) {
  r <- get_regions(spec$shift, dgp = if (is.null(spec$dgp)) "linear" else spec$dgp)
  seed <- 3000L + spec$seed
  d <- switch(spec$arm,
    OLS = fit_predict_ols(r$src, r$tgt),
    XGBoost = fit_predict_xgb(r$src, r$tgt, with_lags = FALSE, seed = seed),
    `XGBoost + lags` = fit_predict_xgb(r$src, r$tgt, with_lags = TRUE, seed = seed),
    GraphSAGE = fit_predict_sage(r$src, r$tgt, norm = NULL, seed = seed),
    `GraphSAGE + LayerNorm` = fit_predict_sage(
      r$src, r$tgt, norm = layer_layer_norm_node, seed = seed, dropout = 0.1
    )
  )
  cbind(shift = spec$shift, arm = spec$arm, seed = spec$seed, score(d))
}

# Prediction-capturing variant, for testing whether the same prediction-level
# ensembling that rescued GraphSAGE at KC -> Ames (rsq_trad 0.38 -> 0.583)
# also helps here.
domain_shift_task_preds <- function(spec) {
  r <- get_regions(spec$shift)
  seed <- 3000L + spec$seed
  d <- switch(spec$arm,
    XGBoost = fit_predict_xgb(r$src, r$tgt, with_lags = FALSE, seed = seed),
    `XGBoost + lags` = fit_predict_xgb(r$src, r$tgt, with_lags = TRUE, seed = seed),
    GraphSAGE = fit_predict_sage(r$src, r$tgt, norm = NULL, seed = seed),
    `GraphSAGE + LayerNorm` = fit_predict_sage(
      r$src, r$tgt, norm = layer_layer_norm_node, seed = seed, dropout = 0.1
    )
  )
  list(shift = spec$shift, arm = spec$arm, seed = spec$seed,
       metrics = cbind(shift = spec$shift, arm = spec$arm, seed = spec$seed, score(d)),
       truth = d$truth, preds = d$estimate)
}

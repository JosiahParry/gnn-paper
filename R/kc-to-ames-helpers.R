# Wave 2 of the KC -> Ames architecture ablation. All variants build on top of
# node-mode LayerNorm (wave 1's confirmed win: rsq_trad 0.199 -> 0.247), and
# test one additional change each against that new baseline:
#
#   no_self_loop    -- pure-neighbour aggregation (concat still carries the
#                       focal feature, so this only removes the *redundant*
#                       copy of it inside the neighbour mean)
#   gaussian_kernel  -- KNN edges weighted by exp(-d^2 / 2*bandwidth^2)
#                        instead of uniform weight 1, bandwidth = median
#                        neighbour distance
#   narrow_net       -- hidden_dims c(32, 16) instead of c(56, 32, 16)
#   dropout          -- dropout = 0.2
#   weight_decay     -- Adam weight_decay = 1e-4
#   lag_input        -- model sees [X, lag(X)] at the input layer, doubling
#                        in_features, on top of its own graph aggregation

# kernel="gaussian" now goes through sfdep::st_kernel_weights() (the
# package's own mechanism) rather than a hand-rolled formula, matching
# R/airbnb-core-brow-sd.R. threshold_mult scales the median k-NN neighbour
# distance. Found at Broward -> San Diego: a MUCH tighter kernel than the
# naive default (0.02x median, not 1x) is what actually helps -- screened
# across 4 kernel shapes x 6 thresholds there, gaussian was the only stable
# one (triangular/epanechnikov blew up at some thresholds) and kept
# improving down to 0.02x before turning over.
make_sub_graph_variant <- function(geom, self_loop = TRUE, kernel = "uniform", threshold_mult = 1) {
  sub_nb <- sfdep::st_knn(geom, k = min(k_nb, length(geom) - 1L))
  from <- rep.int(seq_along(sub_nb), lengths(sub_nb))
  to <- unlist(sub_nb)

  if (kernel == "gaussian") {
    coords <- sf::st_coordinates(geom)
    dists <- unlist(spdep::nbdists(sub_nb, coords))
    thr <- stats::median(dists) * threshold_mult
    weight <- unlist(sfdep::st_kernel_weights(sub_nb, geom, kernel = "gaussian", threshold = thr, adaptive = FALSE))
  } else {
    weight <- NULL
  }

  adj <- adj_from_edgelist(from, to, weight = weight)
  if (self_loop) adj <- add_graph_self_loops(adj)

  list(adj = adj, lw = nb2listw(sub_nb))
}

variants2 <- list(
  no_self_loop = list(self_loop = FALSE, kernel = "uniform", hidden = sage_hidden, dropout = 0, wd = 0, lag_input = FALSE),
  gaussian_kernel = list(self_loop = TRUE, kernel = "gaussian", hidden = sage_hidden, dropout = 0, wd = 0, lag_input = FALSE),
  narrow_net = list(self_loop = TRUE, kernel = "uniform", hidden = c(32, 16), dropout = 0, wd = 0, lag_input = FALSE),
  dropout = list(self_loop = TRUE, kernel = "uniform", hidden = sage_hidden, dropout = 0.2, wd = 0, lag_input = FALSE),
  weight_decay = list(self_loop = TRUE, kernel = "uniform", hidden = sage_hidden, dropout = 0, wd = 1e-4, lag_input = FALSE),
  lag_input = list(self_loop = TRUE, kernel = "uniform", hidden = sage_hidden, dropout = 0, wd = 0, lag_input = TRUE)
)

fit_sage_variant2 <- function(train_id, val_id, v) {
  ids <- c(train_id, val_id)
  scaler <- make_scaler(ids)

  build_x <- function(ids2, region, lw = NULL) {
    x <- scaler(ids2, region)
    if (v$lag_input) {
      stopifnot(!is.null(lw))
      wx <- lag.listw(lw, x)
      colnames(wx) <- paste0("lag_", colnames(x))
      x <- cbind(x, wx)
    }
    x
  }

  g <- make_sub_graph_variant(geom_of("kc")[ids], self_loop = v$self_loop, kernel = v$kernel, threshold_mult = if (is.null(v$threshold_mult)) 1 else v$threshold_mult)

  x_t <- torch_tensor(build_x(ids, "kc", g$lw), dtype = torch_float32())
  y_t <- torch_tensor(kc$y[ids], dtype = torch_float32())$view(c(-1, 1))

  train_idx <- seq_along(train_id)
  val_idx <- seq_along(val_id) + length(train_id)

  in_features <- length(feature_cols) * if (v$lag_input) 2L else 1L

  model <- model_sage(
    in_features = in_features,
    hidden_dims = v$hidden,
    out_features = 1,
    norm = layer_layer_norm_node,
    dropout = v$dropout
  )
  optimizer <- optim_adam(model$parameters, lr = lr, weight_decay = v$wd)

  best_val <- Inf
  best_state <- NULL
  no_improve <- 0L

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
      best_val <- vl
      best_state <- lapply(model$state_dict(), \(t) t$clone())
      no_improve <- 0L
    } else {
      no_improve <- no_improve + 1L
    }
    if (no_improve >= patience) break
  }

  model$load_state_dict(best_state)
  model$eval()

  function(ids2, region) {
    g2 <- make_sub_graph_variant(geom_of(region)[ids2], self_loop = v$self_loop, kernel = v$kernel, threshold_mult = if (is.null(v$threshold_mult)) 1 else v$threshold_mult)
    x2 <- torch_tensor(build_x(ids2, region, g2$lw), dtype = torch_float32())
    with_no_grad({
      as.numeric(model(x2, g2$adj)$squeeze())
    })
  }
}

sweep2_task <- function(spec) {
  v <- variants2[[spec$variant]]
  torch_manual_seed(spec$seed)
  predictor <- fit_sage_variant2(final_train, final_val, v)
  cbind(
    variant = spec$variant,
    seed = spec$seed,
    score(ames$y, predictor(seq_len(nrow(ames)), "ames"))
  )
}

# Wave 3: combinations of the wave-2 winners (dropout, lag_input,
# gaussian_kernel), built on the same fit_sage_variant2().
variants3 <- list(
  dropout_01           = list(self_loop = TRUE, kernel = "uniform", hidden = sage_hidden, dropout = 0.1, wd = 0, lag_input = FALSE),
  dropout_03           = list(self_loop = TRUE, kernel = "uniform", hidden = sage_hidden, dropout = 0.3, wd = 0, lag_input = FALSE),
  dropout_04           = list(self_loop = TRUE, kernel = "uniform", hidden = sage_hidden, dropout = 0.4, wd = 0, lag_input = FALSE),
  dropout_lag          = list(self_loop = TRUE, kernel = "uniform", hidden = sage_hidden, dropout = 0.2, wd = 0, lag_input = TRUE),
  dropout_gauss        = list(self_loop = TRUE, kernel = "gaussian", hidden = sage_hidden, dropout = 0.2, wd = 0, lag_input = FALSE),
  dropout_lag_gauss    = list(self_loop = TRUE, kernel = "gaussian", hidden = sage_hidden, dropout = 0.2, wd = 0, lag_input = TRUE)
)

sweep3_task <- function(spec) {
  v <- variants3[[spec$variant]]
  torch_manual_seed(spec$seed)
  predictor <- fit_sage_variant2(final_train, final_val, v)
  cbind(
    variant = spec$variant,
    seed = spec$seed,
    score(ames$y, predictor(seq_len(nrow(ames)), "ames"))
  )
}

# Final-validation task: like sweep2/3 but also returns raw per-node
# predictions, so an ensemble-of-predictions (not ensemble-of-metrics) can be
# built afterward. variant_list lets the driver pick which named list
# (variants2 or variants3, or a bespoke one) to pull `spec$variant` from.
final_task <- function(spec, variant_list) {
  v <- variant_list[[spec$variant]]
  torch_manual_seed(spec$seed)
  predictor <- fit_sage_variant2(final_train, final_val, v)
  preds <- predictor(seq_len(nrow(ames)), "ames")
  m <- cbind(variant = spec$variant, seed = spec$seed, score(ames$y, preds))
  list(metrics = m, preds = preds)
}

# Wave 4: refine around the dropout=0.1 optimum found in wave 3.
variants4 <- list(
  dropout_005       = list(self_loop = TRUE, kernel = "uniform", hidden = sage_hidden, dropout = 0.05, wd = 0, lag_input = FALSE),
  dropout_015       = list(self_loop = TRUE, kernel = "uniform", hidden = sage_hidden, dropout = 0.15, wd = 0, lag_input = FALSE),
  dropout01_gauss   = list(self_loop = TRUE, kernel = "gaussian", hidden = sage_hidden, dropout = 0.1, wd = 0, lag_input = FALSE)
)

sweep4_task <- function(spec) {
  v <- variants4[[spec$variant]]
  torch_manual_seed(spec$seed)
  predictor <- fit_sage_variant2(final_train, final_val, v)
  cbind(
    variant = spec$variant,
    seed = spec$seed,
    score(ames$y, predictor(seq_len(nrow(ames)), "ames"))
  )
}

# Wave 5: attack the mechanism instead of regularizing around it. make_scaler
# z-scores every region against King County's OWN mean/sd -- exactly the
# covariate-shift-at-the-input problem, not just a symptom of it. This
# rank-transforms each feature to its percentile *within its own region*
# instead: a node's covariates become "where does this house sit in ITS
# market" rather than "how many KC standard deviations from KC's mean." Both
# regions' raw covariates are available under the inductive protocol (no
# labels needed), so this uses no information a real deployment wouldn't
# have.
make_scaler_percentile <- function(ids) {
  function(ids2, region) {
    x <- X_of(region)[ids2, , drop = FALSE]
    apply(x, 2, function(col) (rank(col, ties.method = "average") - 0.5) / length(col))
  }
}

variants5 <- list(
  percentile              = list(self_loop = TRUE, kernel = "uniform", hidden = sage_hidden, dropout = 0.1, wd = 0, lag_input = FALSE, scaler = "percentile"),
  percentile_dropout05    = list(self_loop = TRUE, kernel = "uniform", hidden = sage_hidden, dropout = 0.05, wd = 0, lag_input = FALSE, scaler = "percentile"),
  percentile_dropout0     = list(self_loop = TRUE, kernel = "uniform", hidden = sage_hidden, dropout = 0, wd = 0, lag_input = FALSE, scaler = "percentile"),
  dropout01_wd            = list(self_loop = TRUE, kernel = "uniform", hidden = sage_hidden, dropout = 0.1, wd = 1e-4, lag_input = FALSE, scaler = "zscore")
)

fit_sage_variant5 <- function(train_id, val_id, v) {
  ids <- c(train_id, val_id)
  scaler <- if (identical(v$scaler, "percentile")) make_scaler_percentile(ids) else make_scaler(ids)

  build_x <- function(ids2, region, lw = NULL) {
    x <- scaler(ids2, region)
    if (v$lag_input) {
      stopifnot(!is.null(lw))
      wx <- lag.listw(lw, x)
      colnames(wx) <- paste0("lag_", colnames(x))
      x <- cbind(x, wx)
    }
    x
  }

  g <- make_sub_graph_variant(geom_of("kc")[ids], self_loop = v$self_loop, kernel = v$kernel, threshold_mult = if (is.null(v$threshold_mult)) 1 else v$threshold_mult)

  x_t <- torch_tensor(build_x(ids, "kc", g$lw), dtype = torch_float32())
  y_t <- torch_tensor(kc$y[ids], dtype = torch_float32())$view(c(-1, 1))

  train_idx <- seq_along(train_id)
  val_idx <- seq_along(val_id) + length(train_id)

  in_features <- length(feature_cols) * if (v$lag_input) 2L else 1L

  model <- model_sage(
    in_features = in_features, hidden_dims = v$hidden, out_features = 1,
    norm = layer_layer_norm_node, dropout = v$dropout
  )
  optimizer <- optim_adam(model$parameters, lr = lr, weight_decay = v$wd)

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
    g2 <- make_sub_graph_variant(geom_of(region)[ids2], self_loop = v$self_loop, kernel = v$kernel, threshold_mult = if (is.null(v$threshold_mult)) 1 else v$threshold_mult)
    x2 <- torch_tensor(build_x(ids2, region, g2$lw), dtype = torch_float32())
    with_no_grad({ as.numeric(model(x2, g2$adj)$squeeze()) })
  }
}

sweep5_task <- function(spec) {
  v <- variants5[[spec$variant]]
  torch_manual_seed(spec$seed)
  predictor <- fit_sage_variant5(final_train, final_val, v)
  cbind(variant = spec$variant, seed = spec$seed, score(ames$y, predictor(seq_len(nrow(ames)), "ames")))
}

# Wave 6: refine around the dropout=0.1 + weight_decay=1e-4 finding (wave 5's
# best: rsq_trad=0.467, beating dropout=0.1 alone at 0.402).
variants6 <- list(
  wd_3e5  = list(self_loop = TRUE, kernel = "uniform", hidden = sage_hidden, dropout = 0.1, wd = 3e-5, lag_input = FALSE, scaler = "zscore"),
  wd_3e4  = list(self_loop = TRUE, kernel = "uniform", hidden = sage_hidden, dropout = 0.1, wd = 3e-4, lag_input = FALSE, scaler = "zscore"),
  wd_1e3  = list(self_loop = TRUE, kernel = "uniform", hidden = sage_hidden, dropout = 0.1, wd = 1e-3, lag_input = FALSE, scaler = "zscore"),
  wd_1e4_dropout15 = list(self_loop = TRUE, kernel = "uniform", hidden = sage_hidden, dropout = 0.15, wd = 1e-4, lag_input = FALSE, scaler = "zscore"),
  wd_1e4_dropout05 = list(self_loop = TRUE, kernel = "uniform", hidden = sage_hidden, dropout = 0.05, wd = 1e-4, lag_input = FALSE, scaler = "zscore")
)

sweep6_task <- function(spec) {
  v <- variants6[[spec$variant]]
  torch_manual_seed(spec$seed)
  predictor <- fit_sage_variant5(final_train, final_val, v)
  cbind(variant = spec$variant, seed = spec$seed, score(ames$y, predictor(seq_len(nrow(ames)), "ames")))
}

# Wave 7: does the tight Gaussian kernel found at Broward -> San Diego
# (0.02x median, sfdep::st_kernel_weights) also help KC -> Ames, stacked on
# the confirmed dropout=0.1 + wd=1e-4 winner (wave 5/6, rsq_trad=0.467)?
variants7 <- list(
  kernel_002 = list(self_loop = TRUE, kernel = "gaussian", threshold_mult = 0.02, hidden = sage_hidden, dropout = 0.1, wd = 1e-4, lag_input = FALSE, scaler = "zscore"),
  kernel_005 = list(self_loop = TRUE, kernel = "gaussian", threshold_mult = 0.05, hidden = sage_hidden, dropout = 0.1, wd = 1e-4, lag_input = FALSE, scaler = "zscore"),
  kernel_01  = list(self_loop = TRUE, kernel = "gaussian", threshold_mult = 0.1,  hidden = sage_hidden, dropout = 0.1, wd = 1e-4, lag_input = FALSE, scaler = "zscore"),
  kernel_025 = list(self_loop = TRUE, kernel = "gaussian", threshold_mult = 0.25, hidden = sage_hidden, dropout = 0.1, wd = 1e-4, lag_input = FALSE, scaler = "zscore"),
  kernel_05  = list(self_loop = TRUE, kernel = "gaussian", threshold_mult = 0.5,  hidden = sage_hidden, dropout = 0.1, wd = 1e-4, lag_input = FALSE, scaler = "zscore")
)

sweep7_task <- function(spec) {
  v <- variants7[[spec$variant]]
  torch_manual_seed(spec$seed)
  predictor <- fit_sage_variant5(final_train, final_val, v)
  cbind(
    variant = spec$variant,
    seed = spec$seed,
    score(ames$y, predictor(seq_len(nrow(ames)), "ames"))
  )
}

# Wave 8: refine kernel bandwidth for KC -> Ames in the OTHER direction from
# Broward -> San Diego -- wave 7 found looser beats tighter here (0.5x:
# 0.488 > 0.02x: 0.365), opposite of the Airbnb pair. Push toward/past 1x.
variants8 <- list(
  kernel_05  = list(self_loop = TRUE, kernel = "gaussian", threshold_mult = 0.5,  hidden = sage_hidden, dropout = 0.1, wd = 1e-4, lag_input = FALSE, scaler = "zscore"),
  kernel_075 = list(self_loop = TRUE, kernel = "gaussian", threshold_mult = 0.75, hidden = sage_hidden, dropout = 0.1, wd = 1e-4, lag_input = FALSE, scaler = "zscore"),
  kernel_1   = list(self_loop = TRUE, kernel = "gaussian", threshold_mult = 1.0,  hidden = sage_hidden, dropout = 0.1, wd = 1e-4, lag_input = FALSE, scaler = "zscore"),
  kernel_15  = list(self_loop = TRUE, kernel = "gaussian", threshold_mult = 1.5,  hidden = sage_hidden, dropout = 0.1, wd = 1e-4, lag_input = FALSE, scaler = "zscore"),
  kernel_2   = list(self_loop = TRUE, kernel = "gaussian", threshold_mult = 2.0,  hidden = sage_hidden, dropout = 0.1, wd = 1e-4, lag_input = FALSE, scaler = "zscore")
)

sweep8_task <- function(spec) {
  v <- variants8[[spec$variant]]
  torch_manual_seed(spec$seed)
  predictor <- fit_sage_variant5(final_train, final_val, v)
  cbind(
    variant = spec$variant,
    seed = spec$seed,
    score(ames$y, predictor(seq_len(nrow(ames)), "ames"))
  )
}

# Final-validation config: dropout=0.1 + wd=1e-4 + gaussian kernel @ 0.75x
# median distance (wave 7/8's found optimum for KC -> Ames, rsq_trad=0.490
# at 15 seeds vs 0.467 without the kernel).
final2_best_v <- list(self_loop = TRUE, kernel = "gaussian", threshold_mult = 0.75,
                       hidden = sage_hidden, dropout = 0.1, wd = 1e-4, lag_input = FALSE, scaler = "zscore")

final2_task <- function(spec) {
  torch_manual_seed(spec$seed)
  predictor <- fit_sage_variant5(final_train, final_val, final2_best_v)
  cbind(seed = spec$seed, score(ames$y, predictor(seq_len(nrow(ames)), "ames")))
}

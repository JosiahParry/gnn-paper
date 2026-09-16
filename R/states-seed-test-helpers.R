# Variance test for the states holdout: is plain GraphSAGE genuinely better
# than GraphSAGE+LayerNorm there, or is the 0.645 vs 0.617 gap seed noise?
#
# The production states.R runs ONE fit per state per arm at a fixed seed
# (123), so it carries no uncertainty estimate -- unlike every other dataset
# in this project, which uses 15-30 seeds. It also runs the LayerNorm arm
# WITHOUT dropout/weight-decay, so it isn't the same model as the arm called
# "GraphSAGE + LayerNorm" elsewhere. Both are tested here.
#
# Parallelised across states/seeds for the variance estimate. Cross-process
# torch nondeterminism is fine for estimating a distribution; it is not fine
# for a single quoted per-state number, which is why states.R stays serial.

source("R/states-core.R")
library(mirai)  # states-core.R is serial by design and doesn't load it

sage_configs <- list(
  plain            = list(norm = "none",  dropout = 0,   wd = 0),
  layernorm        = list(norm = "node",  dropout = 0,   wd = 0),
  layernorm_full   = list(norm = "node",  dropout = 0.1, wd = 1e-4)
)

# states-core.R's fit_predict_sage has no dropout/wd arguments; this is the
# same fit with those two knobs exposed, everything else identical.
fit_predict_sage_cfg <- function(split, cfg) {
  train <- model_frame(split$train_nodes)
  test <- model_frame(split$test_nodes)

  x_all <- as.matrix(train[, feature_cols])
  y_all <- train$y
  x_test_all <- as.matrix(test[, feature_cols])

  x_mu <- colMeans(x_all[split$train_idx, , drop = FALSE])
  x_sd <- apply(x_all[split$train_idx, , drop = FALSE], 2, sd)
  x_sd[x_sd == 0] <- 1
  y_mu <- mean(y_all[split$train_idx])
  y_sd <- stats::sd(y_all[split$train_idx])

  adj_train <- adj_of(split$edges_train)
  adj_test <- adj_of(split$edges_test)

  x_t <- nodes_to_tensor(as.data.frame(scale(x_all, x_mu, x_sd)), adj_train)
  y_t <- torch_tensor((y_all - y_mu) / y_sd, dtype = torch_float32())$view(c(-1, 1))

  norm_fn <- if (cfg$norm == "node") layer_layer_norm_node else NULL

  model <- model_sage(
    in_features = length(feature_cols), hidden_dims = sage_hidden,
    out_features = 1, norm = norm_fn, dropout = cfg$dropout
  )
  optimizer <- optim_adam(model$parameters, lr = lr, weight_decay = cfg$wd)

  best_val <- Inf; best_state <- NULL; no_improve <- 0L
  for (epoch in seq_len(n_epochs)) {
    model$train()
    optimizer$zero_grad()
    out <- model(x_t, adj_train)
    loss <- nnf_l1_loss(out[split$train_idx, ], y_t[split$train_idx, ])
    loss$backward()
    optimizer$step()
    with_no_grad({
      model$eval()
      vl <- nnf_l1_loss(model(x_t, adj_train)[split$val_idx, ], y_t[split$val_idx, ])$item()
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

  x_ho <- nodes_to_tensor(as.data.frame(scale(x_test_all, x_mu, x_sd)), adj_test)
  with_no_grad({ preds <- model(x_ho, adj_test) })

  data.frame(truth = test$y, estimate = as.numeric(preds$detach()) * y_sd + y_mu)
}

state_seed_task <- function(spec) {
  split <- make_state_split(spec$fips)
  torch_manual_seed(spec$seed)
  d <- fit_predict_sage_cfg(split, sage_configs[[spec$config]])
  cbind(
    fips = spec$fips, stusps = split$stusps, config = spec$config, seed = spec$seed,
    score(d, split$lw_test)
  )
}

# ---------------------------------------------------------------------------
# Kernel-weighted queen contiguity.
#
# states-core.R builds a FIXED topology (counties sharing a border), unlike
# the KNN+bandwidth setups where kernel tuning was developed. So the kernel
# here weights the edges that already exist by inter-centroid distance --
# it does not prune the neighbour set. Pruning would disconnect contiguous
# but geographically distant counties (large Western states especially),
# which is why weights are floored rather than allowed to reach zero.
#
# Two threshold modes, because county sizes vary enormously:
#   global   -- one bandwidth = median contiguous-pair centroid distance
#   adaptive -- each county's own max neighbour distance as its bandwidth
# (adaptive is plausible here in a way it wasn't for Airbnb listings, which
# had duplicate coordinates producing zero-width bandwidths.)

make_state_graph_kernel <- function(geoids, geometry, kernel = "uniform",
                                    threshold_mult = 1, adaptive = FALSE) {
  base <- make_state_graph(geoids, geometry)
  if (kernel == "uniform") return(base)

  nb <- poly2nb(geometry)
  for (link in island_links) {
    from <- which(geoids == link[1])
    to <- which(geoids %in% link[-1])
    if (length(from) == 1L && length(to) >= 1L) nb <- addlinks1(nb, from, to)
  }
  isolated <- which(card(nb) == 0L)
  if (length(isolated) > 0L) {
    xy <- st_coordinates(suppressWarnings(st_centroid(geometry)))
    for (i in isolated) {
      d <- (xy[, 1] - xy[i, 1])^2 + (xy[, 2] - xy[i, 2])^2
      d[i] <- Inf
      nb <- addlinks1(nb, i, which.min(d))
    }
  }
  nb_self <- include.self(nb)

  pts <- suppressWarnings(st_centroid(geometry))
  dlist <- spdep::nbdists(nb_self, st_coordinates(pts))
  all_d <- unlist(dlist)
  all_d <- all_d[all_d > 0]
  global_bw <- stats::median(all_d) * threshold_mult

  wlist <- lapply(dlist, function(d) {
    bw <- if (adaptive) max(max(d), .Machine$double.eps) * threshold_mult else global_bw
    z <- d / bw
    w <- switch(kernel,
      gaussian = exp(-(z^2) / 2),
      triangular = pmax(1 - abs(z), 0),
      epanechnikov = pmax(0.75 * (1 - z^2), 0),
      stop("unsupported kernel")
    )
    pmax(w, 1e-8)  # never disconnect an existing contiguity edge
  })

  # The weights must line up with base$edges, which make_state_graph() built
  # from its own independently-constructed nb_self. Same deterministic code
  # path, but verify rather than assume -- a silent misalignment would
  # scramble every edge weight without erroring.
  stopifnot(
    length(unlist(wlist)) == length(base$edges$from),
    identical(rep.int(seq_along(nb_self), lengths(nb_self)), base$edges$from),
    identical(as.integer(unlist(nb_self, use.names = FALSE)),
              as.integer(base$edges$to))
  )

  # lw is left as the unweighted base version: it feeds XGBoost+lags and
  # residual Moran's I, neither of which is under test here. Only the
  # GraphSAGE arms (which use adj) see the kernel.
  list(edges = base$edges, edge_weight = unlist(wlist), lw = base$lw)
}

adj_of_weighted <- function(g) {
  if (is.null(g$edge_weight)) return(adj_of(g$edges))
  adj_from_edgelist(g$edges$from, g$edges$to, weight = g$edge_weight) |>
    add_graph_self_loops()
}

# Kernel-aware version of make_state_split(): identical splits (same seed 42
# validation draw, same train/test rows), but carries edge weights so the
# GraphSAGE adjacency can be kernel-weighted.
make_state_split_kernel <- function(holdout_fips, kernel = "uniform",
                                    threshold_mult = 1, adaptive = FALSE, seed = 42L) {
  is_holdout <- counties$STATEFP == holdout_fips
  train_rows <- which(!is_holdout)
  test_rows <- which(is_holdout)

  set.seed(seed)
  val_local <- sample(seq_along(train_rows), size = floor(val_prop * length(train_rows)))
  train_local <- setdiff(seq_along(train_rows), val_local)

  g_train <- make_state_graph_kernel(
    counties$GEOID[train_rows], st_geometry(counties)[train_rows],
    kernel, threshold_mult, adaptive
  )
  g_test <- make_state_graph_kernel(
    counties$GEOID[test_rows], st_geometry(counties)[test_rows],
    kernel, threshold_mult, adaptive
  )

  list(
    fips = holdout_fips,
    stusps = unique(counties$STUSPS[test_rows]),
    train_nodes = train_rows, test_nodes = test_rows,
    train_idx = train_local, val_idx = val_local,
    edges_train = g_train$edges, edges_test = g_test$edges,
    w_train = g_train$edge_weight, w_test = g_test$edge_weight,
    lw_train = g_train$lw, lw_test = g_test$lw
  )
}

# As fit_predict_sage_cfg(), but uses kernel-weighted adjacencies when the
# split carries edge weights.
fit_predict_sage_kernel <- function(split, cfg) {
  train <- model_frame(split$train_nodes)
  test <- model_frame(split$test_nodes)

  x_all <- as.matrix(train[, feature_cols])
  y_all <- train$y
  x_test_all <- as.matrix(test[, feature_cols])

  x_mu <- colMeans(x_all[split$train_idx, , drop = FALSE])
  x_sd <- apply(x_all[split$train_idx, , drop = FALSE], 2, sd)
  x_sd[x_sd == 0] <- 1
  y_mu <- mean(y_all[split$train_idx])
  y_sd <- stats::sd(y_all[split$train_idx])

  mk_adj <- function(edges, w) {
    if (is.null(w)) adj_from_edgelist(edges$from, edges$to) |> add_graph_self_loops()
    else adj_from_edgelist(edges$from, edges$to, weight = w) |> add_graph_self_loops()
  }
  adj_train <- mk_adj(split$edges_train, split$w_train)
  adj_test <- mk_adj(split$edges_test, split$w_test)

  x_t <- nodes_to_tensor(as.data.frame(scale(x_all, x_mu, x_sd)), adj_train)
  y_t <- torch_tensor((y_all - y_mu) / y_sd, dtype = torch_float32())$view(c(-1, 1))

  norm_fn <- if (cfg$norm == "node") layer_layer_norm_node else NULL
  model <- model_sage(
    in_features = length(feature_cols), hidden_dims = sage_hidden,
    out_features = 1, norm = norm_fn, dropout = cfg$dropout
  )
  optimizer <- optim_adam(model$parameters, lr = lr, weight_decay = cfg$wd)

  best_val <- Inf; best_state <- NULL; no_improve <- 0L
  for (epoch in seq_len(n_epochs)) {
    model$train()
    optimizer$zero_grad()
    out <- model(x_t, adj_train)
    loss <- nnf_l1_loss(out[split$train_idx, ], y_t[split$train_idx, ])
    loss$backward()
    optimizer$step()
    with_no_grad({
      model$eval()
      vl <- nnf_l1_loss(model(x_t, adj_train)[split$val_idx, ], y_t[split$val_idx, ])$item()
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

  x_ho <- nodes_to_tensor(as.data.frame(scale(x_test_all, x_mu, x_sd)), adj_test)
  with_no_grad({ preds <- model(x_ho, adj_test) })
  data.frame(truth = test$y, estimate = as.numeric(preds$detach()) * y_sd + y_mu)
}

state_kernel_task <- function(spec) {
  split <- make_state_split_kernel(spec$fips, spec$kernel, spec$threshold_mult, spec$adaptive)
  torch_manual_seed(spec$seed)
  d <- fit_predict_sage_kernel(split, sage_configs[[spec$config]])
  cbind(
    fips = spec$fips, stusps = split$stusps, config = spec$config,
    kernel = spec$kernel, threshold_mult = spec$threshold_mult, adaptive = spec$adaptive,
    seed = spec$seed, score(d, split$lw_test)
  )
}

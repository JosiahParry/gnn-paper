# Shared engine for both simulation drivers.
#
# R/simulate-block.R and R/simulate-random.R source this file and differ only
# in how they build folds. Everything else -- the lattice, the graphs, the
# data-generating processes, and all six model arms -- lives here, so the two
# runs cannot drift apart.
#
# Every design is inductive. Test nodes are pulled out and given their own
# subgraph with no edges back to the training nodes, so no arm can read a
# training node's features or response at prediction time.

library(spdgp)
library(torch)
library(torchgnn)
library(spdep)
library(sphet)
library(mirai)
library(sf)
library(yardstick)
library(parsnip)
library(recipes)
library(rsample)
library(tune)
library(workflows)
library(dplyr)

# Constants --------------------------------------------------------------

n <- 7000L
grid_cols <- 100L
grid_rows <- 70L
k_nb <- 15L
k_folds <- 6L
val_prop <- 0.1

n_epochs <- 500L
lr <- 0.01
patience <- 20L

sage_hidden <- c(56, 32, 16)

# layer_layer_norm's mode defaults to "graph" (the first level of
# c("graph","node")) whenever it is passed bare as `norm = layer_layer_norm`,
# because model_sage calls it as norm(hidden_dim) with no mode override. That
# collapses every node in the current graph to one shared scalar mean/var per
# layer, not the per-node normalization "LayerNorm" usually means. Verified by
# ablation (2026-09-14, KC -> Ames): switching to mode="node" moved rsq_trad
# from 0.199 to 0.247 and cal_slope from 0.774 to 0.854 on the same seeds.
layer_layer_norm_node <- function(dim) layer_layer_norm(dim, mode = "node")

# Worker pool ------------------------------------------------------------
#
# Daemons cannot be handed torch objects, so rather than shipping them state
# they rebuild it: each sources this file once at startup, which gives it the
# model arms, the scoring function and the lattice. After that a task only has
# to carry its fold's plain-R split.
#
# torch_set_num_threads(1) is the part that matters for wall time. torch
# defaults to one intra-op thread per core, so six daemons each claiming the
# whole machine oversubscribe it and can finish slower than a serial loop. The
# parallelism worth having here is across folds -- a single forward pass on a
# 6300-node sparse graph is far too small to repay threading it.
#
# Call before building splits; run daemons(0) when finished.
start_daemons <- function(n_daemons = 6L) {
  core <- normalizePath("R/sim-core.R", mustWork = TRUE)
  daemons(n_daemons)

  # local = FALSE is source()'s default and is load-bearing here: it evaluates
  # into the daemon's global environment, so the definitions persist for every
  # later mirai_map call rather than vanishing with this expression. core_path
  # goes through .args, which everywhere() keeps local to the evaluation -- it
  # is needed only for the duration of the source() call.
  #
  # .min holds until every daemon has connected, so none can pick up a fold
  # before it has been set up.
  everywhere(
    {
      source(core_path, local = FALSE)
      torch_set_num_threads(1L)
    },
    .args = list(core_path = core),
    .min = n_daemons
  )

  cat(sprintf("%d daemons up, torch pinned to 1 thread each\n", n_daemons))
  invisible(n_daemons)
}

# Lattice and full graph -------------------------------------------------

grid <- make_square_grid(grid_cols, grid_rows) |> st_set_crs(3857)
centroids <- st_centroid(grid)
coords <- st_coordinates(centroids)

nb_full <- sfdep::st_knn(centroids, k_nb)
lw_full <- nb2listw(nb_full)

# The subgraph a fold sees. Rebuilt from the held-out nodes' own coordinates,
# which is what makes the design inductive: the test graph is a KNN graph over
# test nodes only.
#
# Returns the edge list as plain integers rather than a built adjacency tensor.
# A torch tensor holds an external pointer and cannot be serialised to a mirai
# daemon -- shipping one produces a null pointer on the far side. Keeping splits
# free of torch objects is what lets the folds be fanned out at all; each daemon
# calls adj_of() to build its own tensors locally.
make_sub_graph <- function(node_ids) {
  sub_nb <- sfdep::st_knn(
    centroids[node_ids],
    k = min(k_nb, length(node_ids) - 1L)
  )
  list(
    edges = list(
      from = rep.int(seq_along(sub_nb), lengths(sub_nb)),
      to = unlist(sub_nb)
    ),
    lw = nb2listw(sub_nb),
    ids = node_ids
  )
}

adj_of <- function(edges) {
  adj_from_edgelist(edges$from, edges$to) |> add_graph_self_loops()
}

# Given the held-out nodes, build everything a fold needs. Both drivers call
# this, so block and random folds differ only in which nodes are held out.
# train_nodes is c(train_id, val_id) in that order, matching the row order of
# edges_train, which fit_predict_sage relies on.
#
# fold both seeds the validation draw and identifies the fold downstream, so a
# split is self-describing once it reaches a daemon.
make_split <- function(test_ids, fold = 1L) {
  fit_all <- setdiff(seq_len(n), test_ids)
  set.seed(fold)
  val_id <- sample(fit_all, size = floor(val_prop * length(fit_all)))
  train_id <- setdiff(fit_all, val_id)

  g_train <- make_sub_graph(c(train_id, val_id))
  g_test <- make_sub_graph(test_ids)

  list(
    fold = fold,
    train_id = train_id,
    val_id = val_id,
    test_id = test_ids,
    edges_train = g_train$edges,
    edges_test = g_test$edges,
    lw_train = g_train$lw,
    lw_test = g_test$lw,
    train_nodes = g_train$ids,
    test_nodes = g_test$ids
  )
}

# Data-generating processes ----------------------------------------------

moran_of <- function(x, lw = lw_full) {
  moran(x, lw, length(x), sum(unlist(lw$weights)))$I
}

# A single spatially autocorrelated field, standardised so that lambda controls
# spatial arrangement and nothing else. lambda = 0 gives an iid field.
make_field <- function(lambda, lw = lw_full) {
  e <- make_error(n, mu = 0, var = 1, method = "normal")
  z <- if (lambda == 0) e else sim_error(e, lw, lambda = lambda)
  as.vector(scale(z))
}

# Scenario B's covariates. Each of the five columns is its own field at the
# same lambda, rescaled to fixed moments so the mean function sees comparably
# scaled inputs at every Ix level. The mean function itself is held constant
# across the three levels, so they are one process at three levels of Ix.
make_x <- function(lambda, lw = lw_full) {
  mu <- c(0, 1, -1, 2.5, 0.5)
  sdv <- sqrt(c(1, 0.5, 1.5, 2, 1))
  x <- vapply(seq_len(5), \(j) make_field(lambda, lw) * sdv[j] + mu[j], numeric(n))
  colnames(x) <- paste0("x", 1:5)
  x
}

m_nonlinear <- function(x) {
  2 * sin(x[, 1]) + x[, 2]^2 - 1.5 * x[, 3] * x[, 4]
}

m_nonlinear_sp <- function(x, x_sp) {
  1.5 * x[, 1] + 2 * sin(x[, 2]) + x[, 3]^2 + 1.5 * x_sp^2
}

beta <- c(1, rep(0.3, 5))
sigma <- 1

xb <- function(x, b = beta) as.vector(cbind(1, as.matrix(x)) %*% b)

# Scenario E's spillover coefficient, set from a target share of Var(y) rather
# than fixed. Var(Wx1) depends on the covariate field's own autocorrelation --
# it is 1/k for an iid x1 under row-standardised KNN weights, and larger as the
# field smooths -- so a fixed theta would put the spillover at a different
# fraction of the response variance in each E row. Solving
#   share = theta^2 Var(Wx) / (Var(Xb) + theta^2 Var(Wx) + sigma^2)
# for theta gives the expression below. It ignores Cov(Xb, theta Wx), which is
# zero for an iid field but not for an autocorrelated one, so the realised
# share is reported alongside rather than assumed.
theta_for_share <- function(wx, x, share = 0.2, b = beta) {
  var_signal <- var(xb(x, b)) + sigma^2
  sqrt(share * var_signal / ((1 - share) * var(wx)))
}

# Model arms -------------------------------------------------------------
#
# OLS, SEM and both XGBoost variants are fit on train + val, since none of them
# uses a validation set. GraphSAGE takes gradient steps on train and selects its
# checkpoint on val. All four see the same rows; only GraphSAGE partitions them.

fit_predict_ols <- function(sim_df, split) {
  train <- as.data.frame(sim_df[split$train_nodes, , drop = FALSE])
  test <- as.data.frame(sim_df[split$test_nodes, , drop = FALSE])
  fit <- lm(y ~ ., data = train)
  data.frame(
    truth = test$y,
    estimate = as.numeric(predict(fit, newdata = test))
  )
}

# GMM spatial error model. Under the inductive protocol the held-out region's
# error field is unobservable -- there are no edges across the boundary to carry
# residual information -- so prediction is X_test %*% beta and nothing more.
fit_predict_sem <- function(sim_df, split) {
  train <- as.data.frame(sim_df[split$train_nodes, , drop = FALSE])
  test <- as.data.frame(sim_df[split$test_nodes, , drop = FALSE])

  fit <- spreg(y ~ ., data = train, listw = split$lw_train, model = "error")

  # spreg returns the spatial parameter alongside the betas, so match on the
  # design matrix's own column names rather than trusting position. The
  # coefficients come back as a one-column matrix, which carries its names in
  # rownames rather than in a names attribute -- indexing it by name without
  # this coercion yields NA and, downstream, NaN predictions.
  mm <- model.matrix(y ~ ., data = test)
  cf_m <- as.matrix(coef(fit))
  cf <- setNames(as.numeric(cf_m), rownames(cf_m))

  b <- cf[colnames(mm)]
  if (anyNA(b)) {
    stop(sprintf(
      "spreg coefficients do not cover the design matrix.\n  need: %s\n  have: %s",
      paste(colnames(mm), collapse = ", "),
      paste(names(cf), collapse = ", ")
    ))
  }

  data.frame(truth = test$y, estimate = as.numeric(mm %*% b))
}

# XGBoost runs across all folds at once through fit_resamples, so tune picks up
# whatever daemons() is set to and parallelises the folds itself.
#
# with_lags = TRUE pairs each covariate with its neighbourhood mean, built on
# the subgraph the rows belong to: training lags from the training subgraph,
# test lags from the held-out subgraph's own listw. Because each fold's rows
# carry lags computed on their
# own graph, the folds cannot share one data frame -- hence manual_rset over
# per-fold analysis/assessment pairs rather than an rsample split of one table.
xgb_spec <- boost_tree(trees = 500) |>
  set_engine("xgboost") |>
  set_mode("regression")

xgb_workflow <- function(template) {
  workflow() |>
    add_recipe(
      recipe(y ~ ., data = template) |> step_normalize(all_numeric_predictors())
    ) |>
    add_model(xgb_spec)
}

build_lagged <- function(sim_df, node_ids, lw, with_lags) {
  predictors <- setdiff(names(sim_df), "y")
  x <- as.matrix(sim_df[node_ids, predictors, drop = FALSE])
  out <- as.data.frame(x)
  if (with_lags) {
    wx <- lag.listw(lw, x)
    colnames(wx) <- paste0("lag_", predictors)
    out <- cbind(out, as.data.frame(wx))
  }
  out$y <- sim_df$y[node_ids]
  out
}

xgb_rset <- function(sim_df, splits, with_lags) {
  rsplits <- lapply(splits, function(s) {
    ana <- build_lagged(sim_df, s$train_nodes, s$lw_train, with_lags)
    ass <- build_lagged(sim_df, s$test_nodes, s$lw_test, with_lags)
    combined <- rbind(ana, ass)
    make_splits(
      list(
        analysis = seq_len(nrow(ana)),
        assessment = nrow(ana) + seq_len(nrow(ass))
      ),
      data = combined
    )
  })
  manual_rset(rsplits, paste0("Fold", seq_along(rsplits)))
}

xgb_cv_folds <- function(sim_df, splits, with_lags) {
  rset <- xgb_rset(sim_df, splits, with_lags)
  res <- suppressWarnings(fit_resamples(
    xgb_workflow(analysis(rset$splits[[1]])),
    rset,
    metrics = reg_metrics
  ))

  collect_metrics(res, summarize = FALSE) |>
    transmute(
      fold = as.integer(sub("Fold", "", id)),
      metric = .metric,
      value = .estimate
    ) |>
    tidyr::pivot_wider(names_from = metric, values_from = value) |>
    as.data.frame()
}

# Fit once with norm = NULL and once with norm = layer_layer_norm. The caller
# seeds both identically, so the norm argument is the only difference between
# the pair and the gap between them is attributable to it.
fit_predict_sage <- function(sim_df, split, norm = NULL) {
  x_all <- as.matrix(sim_df[, -1, drop = FALSE])
  y_all <- sim_df[, 1]

  # Built here rather than carried in the split: torch tensors do not survive
  # serialisation to a daemon.
  adj_train <- adj_of(split$edges_train)
  adj_test <- adj_of(split$edges_test)

  fit_ids <- c(split$train_id, split$val_id)

  # Scaling constants come from the training rows only, and are reused for the
  # held-out region. A test-set mean would leak.
  x_mu <- colMeans(x_all[split$train_id, , drop = FALSE])
  x_sd <- apply(x_all[split$train_id, , drop = FALSE], 2, sd)
  x_sd[x_sd == 0] <- 1
  y_mu <- mean(y_all[split$train_id])
  y_sd <- sd(y_all[split$train_id])

  x_train <- nodes_to_tensor(
    as.data.frame(scale(x_all[fit_ids, , drop = FALSE], x_mu, x_sd)),
    adj_train
  )
  y_train <- torch_tensor(
    (y_all[fit_ids] - y_mu) / y_sd,
    dtype = torch_float32()
  )$view(c(-1, 1))

  train_idx <- seq_along(split$train_id)
  val_idx <- seq_along(split$val_id) + length(split$train_id)

  model <- model_sage(
    in_features = ncol(x_all),
    hidden_dims = sage_hidden,
    out_features = 1,
    norm = norm
  )
  optimizer <- optim_adam(model$parameters, lr = lr)

  best_val <- Inf
  best_state <- NULL
  no_improve <- 0L

  for (epoch in seq_len(n_epochs)) {
    model$train()
    optimizer$zero_grad()
    out <- model(x_train, adj_train)
    loss <- nnf_mse_loss(out[train_idx, ], y_train[train_idx, ])
    loss$backward()
    optimizer$step()

    with_no_grad({
      model$eval()
      v <- nnf_mse_loss(
        model(x_train, adj_train)[val_idx, ],
        y_train[val_idx, ]
      )$item()
    })

    if (v < best_val) {
      best_val <- v
      best_state <- lapply(model$state_dict(), \(t) t$clone())
      no_improve <- 0L
    } else {
      no_improve <- no_improve + 1L
    }
    if (no_improve >= patience) break
  }

  model$load_state_dict(best_state)
  model$eval()

  x_test <- nodes_to_tensor(
    as.data.frame(scale(x_all[split$test_nodes, , drop = FALSE], x_mu, x_sd)),
    adj_test
  )
  with_no_grad({
    preds <- model(x_test, adj_test)
  })

  data.frame(
    truth = y_all[split$test_nodes],
    estimate = as.numeric(preds$detach()) * y_sd + y_mu
  )
}

# Scoring ----------------------------------------------------------------

reg_metrics <- metric_set(mae, rmse, rsq, rsq_trad)

score <- function(d) {
  m <- reg_metrics(d, truth = truth, estimate = estimate)
  data.frame(
    mae = m$.estimate[m$.metric == "mae"],
    rmse = m$.estimate[m$.metric == "rmse"],
    rsq = m$.estimate[m$.metric == "rsq"],
    rsq_trad = m$.estimate[m$.metric == "rsq_trad"]
  )
}

# Fold and scenario runners ----------------------------------------------

# The arms that are fit one fold at a time. XGBoost is not among them -- it goes
# through fit_resamples across all folds at once, in run_scenario.
#
# sem = TRUE only for scenarios C and D, the two with spatially autocorrelated
# errors. Elsewhere it has nothing to estimate that OLS does not.
run_fold <- function(sim_df, split, sem = FALSE) {
  seed <- 100L + split$fold
  arms <- list(
    OLS = \() fit_predict_ols(sim_df, split),
    GraphSAGE = \() {
      torch_manual_seed(seed)
      fit_predict_sage(sim_df, split, norm = NULL)
    },
    `GraphSAGE + LayerNorm` = \() {
      torch_manual_seed(seed)
      fit_predict_sage(sim_df, split, norm = layer_layer_norm_node)
    }
  )
  if (sem) arms <- append(arms, list(SEM = \() fit_predict_sem(sim_df, split)), 1L)

  do.call(
    rbind,
    lapply(names(arms), \(a) cbind(arm = a, score(arms[[a]]())))
  )
}

# The unit of parallel work: one fold, all of its per-fold arms. Top level so
# that mirai serialises it against the global environment -- see the note in
# run_scenario.
fold_task <- function(split, sim_df, sem) {
  run_fold(sim_df, split, sem = sem)
}

run_scenario <- function(label, sim_df, splits, sem = FALSE) {
  cat(sprintf("\n=== %s ===\n", label))
  sim_df <- as.data.frame(sim_df)

  # One task per fold, fanned out across the daemon pool. This is where the time
  # goes: two GraphSAGE fits per fold, twelve per scenario.
  #
  # A split carries its own fold index, node ids and integer edge lists, so a
  # task is self-contained and nothing torch-shaped has to be serialised.
  #
  # fold_task lives at the top level of this file rather than inline here on
  # purpose: a function carries its enclosing environment through
  # serialisation, and R follows the parent chain until it reaches the global
  # environment, which it sends by reference. A closure defined in this frame
  # would drag the frame -- splits included -- into every task.
  #
  # [.stop] cancels the remaining folds at the first failure and signals it,
  # rather than returning a miraiError object that an unchecked rbind would
  # silently drop.
  t0 <- Sys.time()
  fold_res <- mirai_map(
    splits,
    fold_task,
    .args = list(sim_df = sim_df, sem = sem)
  )[.stop]

  per_fold <- do.call(
    rbind,
    Map(\(i, r) cbind(fold = i, r), seq_along(fold_res), fold_res)
  )

  cat(sprintf(
    "  %d folds in %.1f min\n",
    length(splits),
    as.numeric(difftime(Sys.time(), t0, units = "mins"))
  ))

  cat("  xgboost (fit_resamples)\n")
  xgb <- rbind(
    cbind(arm = "XGBoost", xgb_cv_folds(sim_df, splits, with_lags = FALSE)),
    cbind(arm = "XGBoost + lags", xgb_cv_folds(sim_df, splits, with_lags = TRUE))
  )

  cols <- c("fold", "arm", "mae", "rmse", "rsq", "rsq_trad")
  folds <- rbind(per_fold[, cols], xgb[, cols])

  summary <- folds |>
    group_by(arm) |>
    summarise(
      across(c(mae, rmse, rsq, rsq_trad), list(mean = mean, sd = sd)),
      .groups = "drop"
    ) |>
    as.data.frame()

  print(
    summary[, c("arm", "rsq_mean", "rsq_sd", "rsq_trad_mean", "rsq_trad_sd")],
    row.names = FALSE, digits = 3
  )

  list(
    folds = cbind(label = label, folds),
    summary = cbind(label = label, summary)
  )
}

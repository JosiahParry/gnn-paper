# Simulation grid, scenarios A-E.
# Rscript R/sim.R <block|random|calibrate>

MODE <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(MODE)) {
  MODE <- "block"
}

MODE <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(MODE)) {
  MODE <- "block"
}

library(sf)
library(tune)
library(mirai)
library(dplyr)
library(sphet)
library(spdep)
library(spdgp)
library(torch)
library(recipes)
library(rsample)
library(parsnip)
library(torchgnn)
library(yardstick)
library(workflows)

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

# mode defaults to "graph": one scalar over the whole tensor, not per node.
norm_node <- function(dim) layer_layer_norm(dim, mode = "node")

# Worker pool ------------------------------------------------------------
start_daemons <- function(n_daemons = 6L) {
  core <- normalizePath("R/sim-core.R", mustWork = TRUE)
  daemons(n_daemons)

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

make_field <- function(lambda, lw = lw_full) {
  e <- make_error(n, mu = 0, var = 1, method = "normal")
  z <- if (lambda == 0) e else sim_error(e, lw, lambda = lambda)
  as.vector(scale(z))
}

make_x <- function(lambda, lw = lw_full) {
  mu <- c(0, 1, -1, 2.5, 0.5)
  sdv <- sqrt(c(1, 0.5, 1.5, 2, 1))
  x <- vapply(
    seq_len(5),
    \(j) make_field(lambda, lw) * sdv[j] + mu[j],
    numeric(n)
  )
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

theta_for_share <- function(wx, x, share = 0.2, b = beta) {
  var_signal <- var(xb(x, b)) + sigma^2
  sqrt(share * var_signal / ((1 - share) * var(wx)))
}

# Model arms -------------------------------------------------------------

fit_predict_ols <- function(sim_df, split) {
  train <- as.data.frame(sim_df[split$train_nodes, , drop = FALSE])
  test <- as.data.frame(sim_df[split$test_nodes, , drop = FALSE])
  fit <- lm(y ~ ., data = train)
  data.frame(
    truth = test$y,
    estimate = as.numeric(predict(fit, newdata = test))
  )
}

fit_predict_sem <- function(sim_df, split) {
  train <- as.data.frame(sim_df[split$train_nodes, , drop = FALSE])
  test <- as.data.frame(sim_df[split$test_nodes, , drop = FALSE])

  fit <- spreg(y ~ ., data = train, listw = split$lw_train, model = "error")

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

fit_predict_sage <- function(sim_df, split, norm = NULL) {
  x_all <- as.matrix(sim_df[, -1, drop = FALSE])
  y_all <- sim_df[, 1]

  adj_train <- adj_of(split$edges_train)
  adj_test <- adj_of(split$edges_test)

  fit_ids <- c(split$train_id, split$val_id)

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
  if (sem) {
    arms <- append(arms, list(SEM = \() fit_predict_sem(sim_df, split)), 1L)
  }

  do.call(
    rbind,
    lapply(names(arms), \(a) cbind(arm = a, score(arms[[a]]())))
  )
}

fold_task <- function(split, sim_df, sem) {
  run_fold(sim_df, split, sem = sem)
}

run_scenario <- function(label, sim_df, splits, sem = FALSE) {
  cat(sprintf("\n=== %s ===\n", label))
  sim_df <- as.data.frame(sim_df)

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
    cbind(
      arm = "XGBoost + lags",
      xgb_cv_folds(sim_df, splits, with_lags = TRUE)
    )
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
    row.names = FALSE,
    digits = 3
  )

  list(
    folds = cbind(label = label, folds),
    summary = cbind(label = label, summary)
  )
}

# --- scenarios ---

calib <- readRDS("data/calibration.rds")$calibrated

lam <- function(role, target) {
  calib$lambda[calib$role == role & calib$target_moran == target]
}

lam_ix_040 <- lam("covariate (Ix)", 0.4)
lam_ix_070 <- lam("covariate (Ix)", 0.7)
lam_iu_040 <- lam("error (Iu)", 0.4)
lam_iu_080 <- lam("error (Iu)", 0.8)

set.seed(0)

# Covariates -------------------------------------------------------------

x_iid <- make_x(0)
x_ix040 <- make_x(lam_ix_040)
x_ix070 <- make_x(lam_ix_070)

x_sp_040 <- make_field(lam_ix_040)
x_sp_070 <- make_field(lam_ix_070)

# Error fields -----------------------------------------------------------

u_iid <- rnorm(n, sd = sigma)
u_iu040 <- make_field(lam_iu_040)
u_iu080 <- make_field(lam_iu_080)

# Scenarios --------------------------------------------------------------

# A: linear mean, iid errors. Negative control -- the graph carries nothing.
y_A <- xb(x_iid) + rnorm(n, sd = sigma)

y_B_000 <- m_nonlinear(x_iid) + rnorm(n, sd = sigma)
y_B_040 <- m_nonlinear(x_ix040) + rnorm(n, sd = sigma)
y_B_070 <- m_nonlinear(x_ix070) + rnorm(n, sd = sigma)

# C: linear mean, spatially autocorrelated errors.
y_C_000 <- xb(x_iid) + u_iid
y_C_040 <- xb(x_iid) + u_iu040
y_C_080 <- xb(x_iid) + u_iu080

# D: nonlinear mean with a spatial covariate, plus spatial errors.
y_D_000_040 <- m_nonlinear_sp(x_iid, x_sp_040) + rnorm(n, sd = sigma)
y_D_040_040 <- m_nonlinear_sp(x_iid, x_sp_040) + u_iu040
y_D_080_070 <- m_nonlinear_sp(x_iid, x_sp_070) + u_iu080

wx1_iid <- lag.listw(lw_full, x_iid[, 1])
wx1_040 <- lag.listw(lw_full, x_ix040[, 1])

theta_iid <- theta_for_share(wx1_iid, x_iid)
theta_040 <- theta_for_share(wx1_040, x_ix040)

y_E_iid <- xb(x_iid) + theta_iid * wx1_iid + rnorm(n, sd = sigma)
y_E_040 <- xb(x_ix040) + theta_040 * wx1_040 + rnorm(n, sd = sigma)

# Assembly ---------------------------------------------------------------

scenarios <- list(
  list(label = "A", sem = FALSE, df = cbind(y = y_A, x_iid)),
  list(label = "B | Ix=0.0", sem = FALSE, df = cbind(y = y_B_000, x_iid)),
  list(label = "B | Ix=0.4", sem = FALSE, df = cbind(y = y_B_040, x_ix040)),
  list(label = "B | Ix=0.7", sem = FALSE, df = cbind(y = y_B_070, x_ix070)),
  list(label = "C | Iu=0.0", sem = TRUE, df = cbind(y = y_C_000, x_iid)),
  list(label = "C | Iu=0.4", sem = TRUE, df = cbind(y = y_C_040, x_iid)),
  list(label = "C | Iu=0.8", sem = TRUE, df = cbind(y = y_C_080, x_iid)),
  list(
    label = "D | Iu=0.0 Ix=0.4",
    sem = TRUE,
    df = cbind(y = y_D_000_040, x_iid, x_sp = x_sp_040)
  ),
  list(
    label = "D | Iu=0.4 Ix=0.4",
    sem = TRUE,
    df = cbind(y = y_D_040_040, x_iid, x_sp = x_sp_040)
  ),
  list(
    label = "D | Iu=0.8 Ix=0.7",
    sem = TRUE,
    df = cbind(y = y_D_080_070, x_iid, x_sp = x_sp_070)
  ),
  list(label = "E | Ix=0.0", sem = FALSE, df = cbind(y = y_E_iid, x_iid)),
  list(label = "E | Ix=0.4", sem = FALSE, df = cbind(y = y_E_040, x_ix040))
)

e_params <- data.frame(
  label = c("E | Ix=0.0", "E | Ix=0.4"),
  theta = c(theta_iid, theta_040),
  var_wx = c(var(wx1_iid), var(wx1_040)),
  share_wx = c(
    theta_iid^2 * var(wx1_iid) / var(y_E_iid),
    theta_040^2 * var(wx1_040) / var(y_E_040)
  ),
  cor_x_wx = c(
    cor(x_iid[, 1], wx1_iid),
    cor(x_ix040[, 1], wx1_040)
  )
)

cat("\nScenario E parameters:\n")
print(e_params, row.names = FALSE, digits = 3)

# Realised Moran's I of every field and response, for labelling the charts.
morans <- data.frame(
  variable = c(
    "x_ix040",
    "x_ix070",
    "x_sp_040",
    "x_sp_070",
    "u_iu040",
    "u_iu080",
    "y_A",
    "y_B_000",
    "y_B_040",
    "y_B_070",
    "y_C_000",
    "y_C_040",
    "y_C_080",
    "y_D_000_040",
    "y_D_040_040",
    "y_D_080_070",
    "y_E_iid",
    "y_E_040"
  ),
  moran_i = c(
    moran_of(x_ix040[, 1]),
    moran_of(x_ix070[, 1]),
    moran_of(x_sp_040),
    moran_of(x_sp_070),
    moran_of(u_iu040),
    moran_of(u_iu080),
    moran_of(y_A),
    moran_of(y_B_000),
    moran_of(y_B_040),
    moran_of(y_B_070),
    moran_of(y_C_000),
    moran_of(y_C_040),
    moran_of(y_C_080),
    moran_of(y_D_000_040),
    moran_of(y_D_040_040),
    moran_of(y_D_080_070),
    moran_of(y_E_iid),
    moran_of(y_E_040)
  )
)

if (MODE == "block") {
  library(blockCV)
  library(mirai)

  # Six vertical strips, each spanning the full height of the lattice.
  block_cv <- cv_spatial(
    grid,
    k = k_folds,
    hexagon = FALSE,
    rows_cols = c(1, k_folds),
    selection = "systematic",
    progress = FALSE,
    report = FALSE
  )

  start_daemons(6)

  splits_block <- lapply(seq_len(k_folds), function(i) {
    make_split(which(block_cv$folds_ids == i), fold = i)
  })

  cat("Block fold sizes (test nodes):\n")
  print(vapply(splits_block, \(s) length(s$test_id), integer(1)))

  results <- lapply(scenarios, function(s) {
    run_scenario(s$label, s$df, splits_block, sem = s$sem)
  })

  daemons(0)

  folds <- do.call(rbind, lapply(results, \(r) r$folds))
  summary <- do.call(rbind, lapply(results, \(r) r$summary))

  dir.create("data", showWarnings = FALSE)
  saveRDS(
    list(
      split = "block",
      folds = folds,
      summary = summary,
      morans = morans,
      e_params = e_params
    ),
    "data/scenario-results-block.rds"
  )

  cat(
    "\n\n=== Block CV, mean out-of-sample R2 across folds (squared correlation) ===\n"
  )
  print(
    summary |>
      select(label, arm, rsq_mean, rsq_sd) |>
      tidyr::pivot_wider(names_from = arm, values_from = c(rsq_mean, rsq_sd)) |>
      as.data.frame(),
    row.names = FALSE,
    digits = 3
  )

  cat(
    "\n=== Block CV, mean out-of-sample R2 across folds (traditional, 1 - SSE/SST) ===\n"
  )
  print(
    summary |>
      select(label, arm, rsq_trad_mean, rsq_trad_sd) |>
      tidyr::pivot_wider(
        names_from = arm,
        values_from = c(rsq_trad_mean, rsq_trad_sd)
      ) |>
      as.data.frame(),
    row.names = FALSE,
    digits = 3
  )

  cat("\nSaved to data/scenario-results-block.rds\n")
}

if (MODE == "random") {
  library(mirai)

  start_daemons(6)

  set.seed(1)
  fold_id <- sample(rep_len(seq_len(k_folds), n))

  splits_random <- lapply(seq_len(k_folds), function(i) {
    make_split(which(fold_id == i), fold = i)
  })

  cat("Random fold sizes (test nodes):\n")
  print(vapply(splits_random, \(s) length(s$test_id), integer(1)))

  results <- lapply(scenarios, function(s) {
    run_scenario(s$label, s$df, splits_random, sem = s$sem)
  })

  daemons(0)

  folds <- do.call(rbind, lapply(results, \(r) r$folds))
  summary <- do.call(rbind, lapply(results, \(r) r$summary))

  dir.create("data", showWarnings = FALSE)
  saveRDS(
    list(
      split = "random",
      folds = folds,
      summary = summary,
      morans = morans,
      e_params = e_params
    ),
    "data/scenario-results-random.rds"
  )

  cat(
    "\n\n=== Random CV, mean out-of-sample R2 across folds (squared correlation) ===\n"
  )
  print(
    summary |>
      select(label, arm, rsq_mean, rsq_sd) |>
      tidyr::pivot_wider(names_from = arm, values_from = c(rsq_mean, rsq_sd)) |>
      as.data.frame(),
    row.names = FALSE,
    digits = 3
  )

  cat(
    "\n=== Random CV, mean out-of-sample R2 across folds (traditional, 1 - SSE/SST) ===\n"
  )
  print(
    summary |>
      select(label, arm, rsq_trad_mean, rsq_trad_sd) |>
      tidyr::pivot_wider(
        names_from = arm,
        values_from = c(rsq_trad_mean, rsq_trad_sd)
      ) |>
      as.data.frame(),
    row.names = FALSE,
    digits = 3
  )

  cat("\nSaved to data/scenario-results-random.rds\n")
}

if (MODE == "calibrate") {
  n_draws <- 5L
  lambda_grid <- c(
    0,
    0.3,
    0.5,
    0.65,
    0.75,
    0.82,
    0.87,
    0.9,
    0.93,
    0.95,
    0.97,
    0.99
  )

  target_ix <- c(0.4, 0.7)
  target_iu <- c(0.4, 0.8)

  set.seed(0)

  # lambda -> Moran's I ----------------------------------------------------

  cat(sprintf(
    "Measuring Moran's I at %d values of lambda, %d draws each (n = %d, KNN-%d)\n",
    length(lambda_grid),
    n_draws,
    n,
    k_nb
  ))

  curve <- do.call(
    rbind,
    lapply(lambda_grid, function(lam) {
      i_vals <- vapply(
        seq_len(n_draws),
        \(d) moran_of(make_field(lam)),
        numeric(1)
      )
      cat(sprintf(
        "  lambda = %.2f  ->  I = %.3f (sd %.3f)\n",
        lam,
        mean(i_vals),
        sd(i_vals)
      ))
      data.frame(
        lambda = lam,
        moran_mean = mean(i_vals),
        moran_sd = sd(i_vals),
        moran_min = min(i_vals),
        moran_max = max(i_vals)
      )
    })
  )

  solve_lambda <- function(target) {
    approx(x = curve$moran_mean, y = curve$lambda, xout = target)$y
  }

  calibrated <- data.frame(
    role = c(
      rep("covariate (Ix)", length(target_ix)),
      rep("error (Iu)", length(target_iu))
    ),
    target_moran = c(target_ix, target_iu),
    lambda = c(solve_lambda(target_ix), solve_lambda(target_iu))
  )

  cat("\n=== Calibrated parameters ===\n")
  print(calibrated, row.names = FALSE, digits = 3)

  cat("\nVerifying at the solved lambdas:\n")

  verify <- do.call(
    rbind,
    lapply(seq_len(nrow(calibrated)), function(i) {
      lam <- calibrated$lambda[i]
      i_vals <- vapply(
        seq_len(n_draws),
        \(d) moran_of(make_field(lam)),
        numeric(1)
      )
      data.frame(
        role = calibrated$role[i],
        target = calibrated$target_moran[i],
        lambda = lam,
        realised = mean(i_vals),
        realised_sd = sd(i_vals)
      )
    })
  )

  print(verify, row.names = FALSE, digits = 3)

  # Parameter table --------------------------------------------------------

  n_test <- floor(n / k_folds)
  n_fit <- n - n_test
  n_val <- floor(val_prop * n_fit)
  n_train <- n_fit - n_val

  params <- rbind(
    data.frame(parameter = "n", value = as.character(n)),
    data.frame(
      parameter = "lattice",
      value = sprintf("%d x %d", grid_cols, grid_rows)
    ),
    data.frame(
      parameter = "graph",
      value = sprintf("KNN, k = %d, row-standardised", k_nb)
    ),
    data.frame(parameter = "folds", value = sprintf("%d, inductive", k_folds)),
    data.frame(
      parameter = "n train / val / test per fold",
      value = sprintf("%d / %d / %d", n_train, n_val, n_test)
    ),
    data.frame(parameter = "beta", value = paste(beta, collapse = ", ")),
    data.frame(
      parameter = "Wx1 target share of Var(y), scenario E",
      value = as.character(formals(theta_for_share)$share)
    ),
    data.frame(parameter = "sigma", value = as.character(sigma)),
    data.frame(
      parameter = "lambda for Ix = 0.4 / 0.7",
      value = sprintf("%.3f / %.3f", calibrated$lambda[1], calibrated$lambda[2])
    ),
    data.frame(
      parameter = "rho for Iu = 0.4 / 0.8",
      value = sprintf("%.3f / %.3f", calibrated$lambda[3], calibrated$lambda[4])
    ),
    data.frame(
      parameter = "GraphSAGE hidden dims",
      value = paste(sage_hidden, collapse = ", ")
    ),
    data.frame(
      parameter = "epochs / lr / patience",
      value = sprintf("%d / %s / %d", n_epochs, format(lr), patience)
    ),
    data.frame(parameter = "XGBoost trees", value = "500")
  )

  cat("\n=== Parameter table (for main.tex) ===\n")
  print(params, row.names = FALSE, right = FALSE)

  dir.create("data", showWarnings = FALSE)
  saveRDS(
    list(
      curve = curve,
      calibrated = calibrated,
      verify = verify,
      params = params
    ),
    "data/calibration.rds"
  )

  cat("\nSaved to data/calibration.rds\n")
}

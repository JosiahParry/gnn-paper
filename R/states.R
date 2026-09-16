# Held-out US states, county-level vote share.
# Rscript R/states.R <run|seeds|kernel>

MODE <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(MODE)) MODE <- "run"

MODE <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(MODE)) MODE <- "run"

library(torch)
library(torchgnn)
library(spdep)
library(sphet)
library(sf)
library(yardstick)
library(parsnip)
library(recipes)
library(workflows)
library(dplyr)

# Constants --------------------------------------------------------------

val_prop <- 0.1

n_epochs <- 500L
lr <- 0.01
patience <- 20L

# mode defaults to "graph": one scalar over the whole tensor, not per node.
layer_layer_norm_node <- function(dim) layer_layer_norm(dim, mode = "node")

sage_hidden <- c(56, 32, 16)

target <- "pct_dem_dr"

feature_cols <- c(
  "Sexratio",
  "Pct1829",
  "Pct65",
  "PctBlack",
  "PctHispanic",
  "MedIncome",
  "PctBach",
  "Gini",
  "PctManuf",
  "lnPopden",
  "Pct3party",
  "Turnout",
  "PctFB",
  "PctInsured"
)

# Data -------------------------------------------------------------------

counties <- read_sf("data/election-repro-data.fgb")

min_counties <- 20L

state_sizes <- table(counties$STATEFP)
all_fips <- sort(names(state_sizes)[state_sizes >= min_counties])

# Graphs -----------------------------------------------------------------

island_links <- list(
  c("25001", "25007", "25019"),
  c("25007", "25001", "25019"),
  c("25019", "25001", "25007"),
  c("53055", "53057"),
  c("53057", "53055")
)

make_state_graph <- function(geoids, geometry) {
  nb <- poly2nb(geometry)

  for (link in island_links) {
    from <- which(geoids == link[1])
    to <- which(geoids %in% link[-1])
    if (length(from) == 1L && length(to) >= 1L) {
      nb <- addlinks1(nb, from, to)
    }
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

  list(
    edges = list(
      from = rep.int(seq_along(nb_self), lengths(nb_self)),
      to = unlist(nb_self, use.names = FALSE)
    ),
    lw = nb2listw(nb)
  )
}

adj_of <- function(edges) {
  adj_from_edgelist(edges$from, edges$to) |> add_graph_self_loops()
}

make_state_split <- function(holdout_fips, seed = 42L) {
  is_holdout <- counties$STATEFP == holdout_fips

  train_rows <- which(!is_holdout)
  test_rows <- which(is_holdout)

  set.seed(seed)
  val_local <- sample(
    seq_along(train_rows),
    size = floor(val_prop * length(train_rows))
  )
  train_local <- setdiff(seq_along(train_rows), val_local)

  g_train <- make_state_graph(
    counties$GEOID[train_rows],
    st_geometry(counties)[train_rows]
  )
  g_test <- make_state_graph(
    counties$GEOID[test_rows],
    st_geometry(counties)[test_rows]
  )

  list(
    fips = holdout_fips,
    stusps = unique(counties$STUSPS[test_rows]),
    # Indices into counties, and into the training frame respectively.
    train_nodes = train_rows,
    test_nodes = test_rows,
    train_idx = train_local,
    val_idx = val_local,
    edges_train = g_train$edges,
    edges_test = g_test$edges,
    lw_train = g_train$lw,
    lw_test = g_test$lw
  )
}

# Model frames -----------------------------------------------------------

model_frame <- function(rows) {
  out <- counties |>
    st_drop_geometry() |>
    dplyr::slice(rows) |>
    dplyr::select(dplyr::all_of(c(target, feature_cols)))
  names(out)[1] <- "y"
  as.data.frame(out)
}

# Model arms -------------------------------------------------------------

fit_predict_ols <- function(split) {
  train <- model_frame(split$train_nodes)
  test <- model_frame(split$test_nodes)
  fit <- lm(y ~ ., data = train)
  data.frame(
    truth = test$y,
    estimate = as.numeric(predict(fit, newdata = test))
  )
}

fit_predict_sem <- function(split) {
  train <- model_frame(split$train_nodes)
  test <- model_frame(split$test_nodes)

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

build_lagged <- function(rows, lw, with_lags) {
  d <- model_frame(rows)
  x <- as.matrix(d[, feature_cols, drop = FALSE])
  out <- as.data.frame(x)
  if (with_lags) {
    wx <- lag.listw(lw, x)
    colnames(wx) <- paste0("lag_", feature_cols)
    out <- cbind(out, as.data.frame(wx))
  }
  out$y <- d$y
  out
}

xgb_spec <- boost_tree(trees = 500) |>
  set_engine("xgboost") |>
  set_mode("regression")

fit_predict_xgb <- function(split, with_lags) {
  train <- build_lagged(split$train_nodes, split$lw_train, with_lags)
  test <- build_lagged(split$test_nodes, split$lw_test, with_lags)

  wf <- workflow() |>
    add_recipe(
      recipe(y ~ ., data = train) |> step_normalize(all_numeric_predictors())
    ) |>
    add_model(xgb_spec)

  fit <- suppressWarnings(fit(wf, data = train))

  data.frame(
    truth = test$y,
    estimate = predict(fit, new_data = test)$.pred
  )
}

fit_predict_sage <- function(split, norm = NULL) {
  train <- model_frame(split$train_nodes)
  test <- model_frame(split$test_nodes)

  x_train_all <- as.matrix(train[, feature_cols, drop = FALSE])
  x_test_all <- as.matrix(test[, feature_cols, drop = FALSE])

  adj_train <- adj_of(split$edges_train)
  adj_test <- adj_of(split$edges_test)

  fit_rows <- split$train_idx
  x_mu <- colMeans(x_train_all[fit_rows, , drop = FALSE])
  x_sd <- apply(x_train_all[fit_rows, , drop = FALSE], 2, sd)
  x_sd[x_sd == 0] <- 1
  y_mu <- mean(train$y[fit_rows])
  y_sd <- sd(train$y[fit_rows])

  x_t <- nodes_to_tensor(
    as.data.frame(scale(x_train_all, x_mu, x_sd)),
    adj_train
  )
  y_t <- torch_tensor(
    (train$y - y_mu) / y_sd,
    dtype = torch_float32()
  )$view(c(-1, 1))

  model <- model_sage(
    in_features = length(feature_cols),
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
    out <- model(x_t, adj_train)
    loss <- nnf_mse_loss(out[split$train_idx, ], y_t[split$train_idx, ])
    loss$backward()
    optimizer$step()

    with_no_grad({
      model$eval()
      v <- nnf_mse_loss(
        model(x_t, adj_train)[split$val_idx, ],
        y_t[split$val_idx, ]
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

  x_ho <- nodes_to_tensor(
    as.data.frame(scale(x_test_all, x_mu, x_sd)),
    adj_test
  )
  with_no_grad({
    preds <- model(x_ho, adj_test)
  })

  data.frame(
    truth = test$y,
    estimate = as.numeric(preds$detach()) * y_sd + y_mu
  )
}

# Scoring ----------------------------------------------------------------

reg_metrics <- metric_set(mae, rmse, rsq, rsq_trad)

score <- function(d, lw) {
  m <- reg_metrics(d, truth = truth, estimate = estimate)
  resid <- d$truth - d$estimate
  data.frame(
    n = nrow(d),
    mae = m$.estimate[m$.metric == "mae"],
    rmse = m$.estimate[m$.metric == "rmse"],
    rsq = m$.estimate[m$.metric == "rsq"],
    rsq_trad = m$.estimate[m$.metric == "rsq_trad"],
    mean_bias = mean(resid),
    resid_moran = moran.test(resid, lw)$estimate[[1]]
  )
}

# State runner -----------------------------------------------------------

state_task <- function(holdout_fips) {
  split <- make_state_split(holdout_fips)

  arms <- list(
    OLS = \() fit_predict_ols(split),
    SEM = \() fit_predict_sem(split),
    XGBoost = \() fit_predict_xgb(split, with_lags = FALSE),
    `XGBoost + lags` = \() fit_predict_xgb(split, with_lags = TRUE),
    GraphSAGE = \() {
      torch_manual_seed(123L)
      fit_predict_sage(split, norm = NULL)
    },
    `GraphSAGE + LayerNorm` = \() {
      torch_manual_seed(123L)
      fit_predict_sage(split, norm = layer_layer_norm_node)
    }
  )

  do.call(
    rbind,
    lapply(names(arms), function(a) {
      cbind(
        STATEFP = holdout_fips,
        STUSPS = split$stusps,
        arm = a,
        score(arms[[a]](), split$lw_test)
      )
    })
  )
}

# --- experimental harness ---

library(mirai)  # states-core.R is serial by design and doesn't load it

sage_configs <- list(
  plain            = list(norm = "none",  dropout = 0,   wd = 0),
  layernorm        = list(norm = "node",  dropout = 0,   wd = 0),
  layernorm_full   = list(norm = "node",  dropout = 0.1, wd = 1e-4)
)

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

  stopifnot(
    length(unlist(wlist)) == length(base$edges$from),
    identical(rep.int(seq_along(nb_self), lengths(nb_self)), base$edges$from),
    identical(as.integer(unlist(nb_self, use.names = FALSE)),
              as.integer(base$edges$to))
  )

  list(edges = base$edges, edge_weight = unlist(wlist), lw = base$lw)
}

adj_of_weighted <- function(g) {
  if (is.null(g$edge_weight)) return(adj_of(g$edges))
  adj_from_edgelist(g$edges$from, g$edges$to, weight = g$edge_weight) |>
    add_graph_self_loops()
}

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

if (MODE == "run") {

  cat(sprintf("%d states\n", length(all_fips)))

  t0 <- Sys.time()

  state_res <- lapply(seq_along(all_fips), function(i) {
    cat(sprintf("[%2d/%d] %s\n", i, length(all_fips), all_fips[i]))
    state_task(all_fips[i])
  })

  results <- do.call(rbind, state_res)

  cat(sprintf(
    "\n%d states in %.1f min\n",
    length(all_fips),
    as.numeric(difftime(Sys.time(), t0, units = "mins"))
  ))

  dir.create("data", showWarnings = FALSE)
  saveRDS(results, "data/state-results.rds")

  cat(sprintf(
    "\n\n=== Held-out states (%d) ===\n",
    dplyr::n_distinct(results$STUSPS)
  ))

  print(
    results |>
      group_by(arm) |>
      summarise(
        mae = mean(mae),
        rmse = mean(rmse),
        rsq = mean(rsq),
        rsq_trad = mean(rsq_trad),
        mean_abs_bias = mean(abs(mean_bias)),
        median_abs_bias = median(abs(mean_bias)),
        max_abs_bias = max(abs(mean_bias)),
        resid_moran = mean(resid_moran, na.rm = TRUE),
        .groups = "drop"
      ) |>
      as.data.frame(),
    row.names = FALSE,
    digits = 3
  )

  cat("\n=== Per-state R2 (squared correlation) ===\n")
  print(
    results |>
      select(STUSPS, arm, rsq) |>
      tidyr::pivot_wider(names_from = arm, values_from = rsq) |>
      arrange(STUSPS) |>
      as.data.frame(),
    row.names = FALSE,
    digits = 3
  )

  cat("\n=== Per-state R2 (traditional, 1 - SSE/SST) ===\n")
  print(
    results |>
      select(STUSPS, arm, rsq_trad) |>
      tidyr::pivot_wider(names_from = arm, values_from = rsq_trad) |>
      arrange(STUSPS) |>
      as.data.frame(),
    row.names = FALSE,
    digits = 3
  )

  cat("\n=== GraphSAGE + LayerNorm vs GraphSAGE, states won (rsq_trad) ===\n")
  sage_wide <- results |>
    filter(arm %in% c("GraphSAGE", "GraphSAGE + LayerNorm")) |>
    select(STUSPS, arm, rsq_trad) |>
    tidyr::pivot_wider(names_from = arm, values_from = rsq_trad)
  n_layernorm_wins <- sum(sage_wide[["GraphSAGE + LayerNorm"]] > sage_wide[["GraphSAGE"]])
  cat(sprintf(
    "LayerNorm wins %d / %d states\n",
    n_layernorm_wins, nrow(sage_wide)
  ))

  cat("\nSaved to data/state-results.rds\n")
}

if (MODE == "seeds") {

  cli_args <- commandArgs(trailingOnly = TRUE)
  n_daemons <- if (length(cli_args) >= 1) as.integer(cli_args[1]) else 14L

  seeds <- 1001:1005

  daemons(n_daemons)
  core <- normalizePath("R/states-seed-test-helpers.R", mustWork = TRUE)
  everywhere(
    { source(core_path, local = FALSE); torch_set_num_threads(1L) },
    .args = list(core_path = core), .min = n_daemons
  )
  cat(sprintf("%d daemons up\n", n_daemons))

  jobs <- unlist(unlist(lapply(names(sage_configs), function(cfg) {
    lapply(all_fips, function(f) {
      lapply(seeds, function(s) list(fips = f, seed = s, config = cfg))
    })
  }), recursive = FALSE), recursive = FALSE)

  cat(sprintf("\nStates seed test: %d fits (%d states x %d configs x %d seeds)\n",
              length(jobs), length(all_fips), length(sage_configs), length(seeds)))
  t0 <- Sys.time()
  res <- do.call(rbind, mirai_map(jobs, state_seed_task)[.progress, .stop])
  daemons(0)
  cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

  per_seed <- res |>
    group_by(config, seed) |>
    summarise(rsq_trad = mean(rsq_trad), mae = mean(mae), .groups = "drop")

  cat("\n=== National mean rsq_trad, per seed ===\n")
  print(tidyr::pivot_wider(per_seed[, c("config","seed","rsq_trad")],
                            names_from = config, values_from = rsq_trad),
        row.names = FALSE, digits = 4)

  cat("\n=== Across-seed summary ===\n")
  summ <- per_seed |>
    group_by(config) |>
    summarise(rsq_trad_mean = mean(rsq_trad), rsq_trad_sd = sd(rsq_trad),
              rsq_trad_min = min(rsq_trad), rsq_trad_max = max(rsq_trad),
              mae_mean = mean(mae), .groups = "drop") |>
    as.data.frame()
  print(summ, row.names = FALSE, digits = 4)

  cat("\n=== Paired per-state comparison, plain vs layernorm (all seeds pooled) ===\n")
  w <- res |>
    filter(config %in% c("plain", "layernorm")) |>
    select(stusps, seed, config, rsq_trad) |>
    tidyr::pivot_wider(names_from = config, values_from = rsq_trad)
  cat(sprintf("plain better in %d of %d state-seed pairs (%.1f%%)\n",
              sum(w$plain > w$layernorm), nrow(w), 100 * mean(w$plain > w$layernorm)))
  cat(sprintf("mean paired difference (plain - layernorm): %.4f (sd %.4f)\n",
              mean(w$plain - w$layernorm), sd(w$plain - w$layernorm)))
  print(t.test(w$plain, w$layernorm, paired = TRUE))

  saveRDS(list(folds = res, per_seed = per_seed, summary = summ), "data/states-seed-test.rds")
  cat("\nSaved to data/states-seed-test.rds\n")
}

if (MODE == "kernel") {

  cli_args <- commandArgs(trailingOnly = TRUE)
  n_daemons <- if (length(cli_args) >= 1) as.integer(cli_args[1]) else 14L

  seeds <- 1001:1005

  settings <- list(
    list(kernel = "uniform",  threshold_mult = 1,    adaptive = FALSE, label = "uniform"),
    list(kernel = "gaussian", threshold_mult = 0.25, adaptive = FALSE, label = "gauss_0.25x"),
    list(kernel = "gaussian", threshold_mult = 0.5,  adaptive = FALSE, label = "gauss_0.5x"),
    list(kernel = "gaussian", threshold_mult = 1,    adaptive = FALSE, label = "gauss_1x"),
    list(kernel = "gaussian", threshold_mult = 2,    adaptive = FALSE, label = "gauss_2x"),
    list(kernel = "gaussian", threshold_mult = 4,    adaptive = FALSE, label = "gauss_4x"),
    list(kernel = "gaussian", threshold_mult = 1,    adaptive = TRUE,  label = "gauss_adaptive")
  )

  daemons(n_daemons)
  core <- normalizePath("R/states-seed-test-helpers.R", mustWork = TRUE)
  everywhere(
    { source(core_path, local = FALSE); torch_set_num_threads(1L) },
    .args = list(core_path = core), .min = n_daemons
  )
  cat(sprintf("%d daemons up\n", n_daemons))

  jobs <- unlist(unlist(lapply(settings, function(st) {
    lapply(all_fips, function(f) {
      lapply(seeds, function(s) {
        list(fips = f, seed = s, config = "layernorm_full",
             kernel = st$kernel, threshold_mult = st$threshold_mult,
             adaptive = st$adaptive, label = st$label)
      })
    })
  }), recursive = FALSE), recursive = FALSE)

  cat(sprintf("\nStates kernel test: %d fits (%d settings x %d states x %d seeds)\n",
              length(jobs), length(settings), length(all_fips), length(seeds)))

  kernel_task <- function(spec) {
    r <- state_kernel_task(spec)
    cbind(label = spec$label, r)
  }

  t0 <- Sys.time()
  res <- do.call(rbind, mirai_map(jobs, kernel_task)[.progress, .stop])
  daemons(0)
  cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

  per_seed <- res |>
    group_by(label, seed) |>
    summarise(rsq_trad = mean(rsq_trad), mae = mean(mae), .groups = "drop")

  summ <- per_seed |>
    group_by(label) |>
    summarise(rsq_trad_mean = mean(rsq_trad), rsq_trad_sd = sd(rsq_trad),
              mae_mean = mean(mae), .groups = "drop") |>
    as.data.frame()

  cat("\n=== National mean rsq_trad by kernel setting (5 seeds) ===\n")
  print(summ[order(-summ$rsq_trad_mean), ], row.names = FALSE, digits = 4)

  base_lab <- "uniform"
  cat(sprintf("\n=== Paired per-state tests vs %s ===\n", base_lab))
  for (lab in setdiff(summ$label, base_lab)) {
    w <- res |>
      filter(label %in% c(base_lab, lab)) |>
      select(stusps, seed, label, rsq_trad) |>
      tidyr::pivot_wider(names_from = label, values_from = rsq_trad)
    tt <- t.test(w[[lab]], w[[base_lab]], paired = TRUE)
    cat(sprintf("%-16s diff=%+.4f  p=%.3f  CI[%+.4f, %+.4f]  better in %d/%d\n",
                lab, unname(tt$estimate), tt$p.value, tt$conf.int[1], tt$conf.int[2],
                sum(w[[lab]] > w[[base_lab]]), nrow(w)))
  }

  saveRDS(list(folds = res, per_seed = per_seed, summary = summ), "data/states-kernel-test.rds")
  cat("\nSaved to data/states-kernel-test.rds\n")
}

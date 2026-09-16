# Shared engine for the US county analysis: every state held out in turn.
#
# County-level 2020 presidential vote share against fourteen demographic and
# economic covariates. Each state is held out as a region, given its own
# contiguity graph with no edges back to the training counties, and predicted
# by all six arms under one seed.
#
# The design is inductive, matching R/sim-core.R. A held-out state's counties
# are neighbours of one another and of nothing else, so no arm can read a
# training county's features or response at prediction time.
#
# States run one at a time: torch's CPU sparse operations are not deterministic
# across processes, so a daemon pool does not reproduce.

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

# See R/sim-core.R for why this wrapper exists: layer_layer_norm's mode
# defaults to "graph" (a single scalar over the whole tensor) unless
# overridden, not the per-node normalization "LayerNorm" usually means.
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

# A state needs enough counties for its held-out graph to have structure and
# for an R2 to mean anything. Below this a single bad prediction swings the
# score, and the graph is too small for a GNN to aggregate over.
min_counties <- 20L

state_sizes <- table(counties$STATEFP)
all_fips <- sort(names(state_sizes)[state_sizes >= min_counties])

# Graphs -----------------------------------------------------------------

# Queen contiguity leaves island counties with no neighbours, which nb2listw
# will not accept and which would give a GNN node nothing to aggregate. These
# five are joined to the mainland counties they are ferried to: Nantucket,
# Dukes and Barnstable in Massachusetts, San Juan and Island in Washington.
island_links <- list(
  c("25001", "25007", "25019"),
  c("25007", "25001", "25019"),
  c("25019", "25001", "25007"),
  c("53055", "53057"),
  c("53057", "53055")
)

# Two graphs come out of one neighbour list. The adjacency handed to GraphSAGE
# includes each county as its own neighbour, so a node keeps its own features
# through aggregation. The listw used for lag features and residual Moran's I
# excludes it, so that WX is the neighbourhood mean and not a blend of the
# county with its neighbours.
make_state_graph <- function(geoids, geometry) {
  nb <- poly2nb(geometry)

  for (link in island_links) {
    from <- which(geoids == link[1])
    to <- which(geoids %in% link[-1])
    if (length(from) == 1L && length(to) >= 1L) {
      nb <- addlinks1(nb, from, to)
    }
  }

  # Holding out a state can strand a county whose only shared boundary was
  # with it -- Newport County RI when Massachusetts goes, Richmond County NY
  # when New Jersey does. A region with no neighbours has no place in a
  # row-standardised weights matrix, and gives a GNN node nothing to aggregate
  # beyond itself, so it is joined to the nearest county by centroid distance.
  # This subsumes the island list above for every case that list does not name.
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
    # No zero.policy: nothing should be isolated by this point, and if the
    # nearest-neighbour repair above ever fails to hold, the run should stop
    # here rather than quietly propagate a lag of zero.
    lw = nb2listw(nb)
  )
}

adj_of <- function(edges) {
  adj_from_edgelist(edges$from, edges$to) |> add_graph_self_loops()
}

# Given a held-out state, build everything the six arms need. Row order is
# c(train_id, val_id) for the training frame, so the GraphSAGE index blocks
# line up with it.
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
#
# Every arm fits on the training states and predicts the held-out state. OLS,
# SEM and XGBoost use all training counties; GraphSAGE splits them into train
# and validation so it can early stop, so it sees strictly less data.

fit_predict_ols <- function(split) {
  train <- model_frame(split$train_nodes)
  test <- model_frame(split$test_nodes)
  fit <- lm(y ~ ., data = train)
  data.frame(
    truth = test$y,
    estimate = as.numeric(predict(fit, newdata = test))
  )
}

# GMM spatial error model. The held-out state's error field is unobservable --
# no edges cross the boundary to carry residual information -- so prediction is
# X_test %*% beta and nothing more.
fit_predict_sem <- function(split) {
  train <- model_frame(split$train_nodes)
  test <- model_frame(split$test_nodes)

  fit <- spreg(y ~ ., data = train, listw = split$lw_train, model = "error")

  # spreg returns the spatial parameter alongside the betas, and hands back a
  # one-column matrix whose names live in rownames rather than in a names
  # attribute. Indexing that by name without coercing first yields NA, and NaN
  # predictions downstream.
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

# with_lags = TRUE pairs each covariate with the mean of that covariate over the
# county's neighbours. The lag comes from the graph the rows belong to: training
# counties from the training states' graph, held-out counties from the held-out
# state's own listw. The listw excludes self, so lag_ is the neighbourhood mean
# and not a blend of the county with its neighbours.
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

# Fit once with norm = NULL and once with norm = layer_layer_norm. The caller
# seeds both identically, so the norm argument is the only difference between
# the pair and the gap between them is attributable to it.
fit_predict_sage <- function(split, norm = NULL) {
  train <- model_frame(split$train_nodes)
  test <- model_frame(split$test_nodes)

  x_train_all <- as.matrix(train[, feature_cols, drop = FALSE])
  x_test_all <- as.matrix(test[, feature_cols, drop = FALSE])

  adj_train <- adj_of(split$edges_train)
  adj_test <- adj_of(split$edges_test)

  # Scaling constants come from the training rows the model actually fits on,
  # not from the validation rows and not from the held-out state.
  fit_rows <- split$train_idx
  x_mu <- colMeans(x_train_all[fit_rows, , drop = FALSE])
  x_sd <- apply(x_train_all[fit_rows, , drop = FALSE], 2, sd)
  x_sd[x_sd == 0] <- 1
  y_mu <- mean(train$y[fit_rows])
  y_sd <- sd(train$y[fit_rows])

  # The whole training graph goes through the model every epoch; the loss is
  # taken on the train block and the checkpoint on the validation block.
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

# mean_bias is truth - estimate, so a negative value is a model predicting the
# state too high. resid_moran is Moran's I of the held-out residuals on the
# state's own graph: what spatial structure the arm left on the table.
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

# One held-out state, all six arms. The two GraphSAGE fits are seeded
# identically, so the norm argument is the only thing separating them.
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

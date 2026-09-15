# Shared engine for the King County to Ames analysis.
#
# R/kc-to-ames.R sources this and dispatches the work; every mirai daemon
# sources it too, which is what lets a task carry nothing but a fold index, an
# arm name and a seed. Loading it has no side effects beyond reading the two
# data files.
#
# y is log price centred within its own region, so the transfer is asked to
# reproduce relative variation rather than the level, which is not identified
# across markets.

library(dplyr)
library(mirai)
library(sf)
library(spdep)
library(torch)
library(torchgnn)
library(parsnip)
library(recipes)
library(rsample)
library(tune)
library(workflows)
library(yardstick)

k_nb <- 30L
k_folds <- 4L
val_prop <- 0.2
n_epochs <- 500L
lr <- 0.01
patience <- 20L
sage_hidden <- c(56, 32, 16)

feature_cols <- c(
  "lot_area",
  "living_area",
  "above_grade",
  "basement",
  "bedrooms",
  "bathrooms",
  "yr_built",
  "renovated"
)

# Data -------------------------------------------------------------------

kc <- readr::read_csv("data/kc_house_data.csv", show_col_types = FALSE) |>
  transmute(
    price,
    lot_area = sqft_lot,
    living_area = sqft_living,
    above_grade = sqft_above,
    basement = sqft_basement,
    bedrooms,
    bathrooms,
    yr_built,
    renovated = as.integer(yr_renovated > 0),
    long,
    lat
  ) |>
  st_as_sf(coords = c("long", "lat"), crs = 4326) |>
  st_transform(3857)

ames <- modeldata::ames |>
  transmute(
    price = Sale_Price,
    lot_area = Lot_Area,
    living_area = Gr_Liv_Area,
    above_grade = First_Flr_SF + Second_Flr_SF,
    basement = Total_Bsmt_SF,
    bedrooms = Bedroom_AbvGr,
    bathrooms = Full_Bath +
      Bsmt_Full_Bath +
      0.5 * (Half_Bath + Bsmt_Half_Bath),
    yr_built = Year_Built,
    renovated = as.integer(Year_Remod_Add > Year_Built),
    Longitude,
    Latitude
  ) |>
  st_as_sf(coords = c("Longitude", "Latitude"), crs = 4326) |>
  st_transform(3857)

kc <- kc |> mutate(y = log(price) - mean(log(price)))
ames <- ames |> mutate(y = log(price) - mean(log(price)))

cat(sprintf("King County  n = %d\n", nrow(kc)))
cat(sprintf("Ames         n = %d\n", nrow(ames)))

X_kc <- as.matrix(st_drop_geometry(kc)[, feature_cols])
X_ames <- as.matrix(st_drop_geometry(ames)[, feature_cols])

geom_of <- function(region) {
  if (region == "kc") st_geometry(kc) else st_geometry(ames)
}
X_of <- function(region) if (region == "kc") X_kc else X_ames
y_of <- function(region) if (region == "kc") kc$y else ames$y

# Graphs -----------------------------------------------------------------

# Held-out rows are always given their own KNN graph over themselves, with no
# edges back to the rows the model was fit on.
make_sub_graph <- function(geom) {
  sub_nb <- sfdep::st_knn(geom, k = min(k_nb, length(geom) - 1L))
  adj <- adj_from_edgelist(
    rep.int(seq_along(sub_nb), lengths(sub_nb)),
    unlist(sub_nb)
  ) |>
    add_graph_self_loops()
  list(adj = adj, lw = nb2listw(sub_nb))
}

make_scaler <- function(ids) {
  mu <- colMeans(X_kc[ids, , drop = FALSE])
  sdv <- apply(X_kc[ids, , drop = FALSE], 2, sd)
  sdv[sdv == 0] <- 1
  function(ids2, region) scale(X_of(region)[ids2, , drop = FALSE], mu, sdv)
}

# Model arms -------------------------------------------------------------
#
# Each returns a predictor: function(ids, region) -> numeric. Fitting always
# happens on King County; region says where the prediction is being asked for.
#
# y ~ X for every arm but one. XGBoost + lags is y ~ X + WX, where WX is the
# mean of each covariate over a node's neighbours in whichever graph its rows
# belong to.

frame_of <- function(ids, region) {
  d <- as.data.frame(X_of(region)[ids, , drop = FALSE])
  d$y <- y_of(region)[ids]
  d
}

fit_ols <- function(fit_ids) {
  m <- lm(y ~ ., data = frame_of(fit_ids, "kc"))
  function(ids, region) {
    as.numeric(predict(m, newdata = frame_of(ids, region)))
  }
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

# MAE objective with an internal validation split, so the tree count is chosen
# the same way GraphSAGE's epoch count is.
xgb_spec <- boost_tree(trees = 500, stop_iter = patience) |>
  set_engine("xgboost", objective = "reg:absoluteerror", validation = val_prop) |>
  set_mode("regression")

xgb_workflow <- function(template) {
  workflow() |>
    add_recipe(
      recipe(y ~ ., data = template) |>
        step_normalize(all_numeric_predictors())
    ) |>
    add_model(xgb_spec)
}

# The engine draws its own validation split from R's RNG, so this arm is not
# deterministic and has to be seeded like the others.
fit_xgb <- function(fit_ids, with_lags, seed) {
  g <- make_sub_graph(geom_of("kc")[fit_ids])
  train_df <- build_df(fit_ids, "kc", g$lw, with_lags)
  set.seed(seed)
  fitted <- suppressWarnings(fit(xgb_workflow(train_df), data = train_df))

  function(ids, region) {
    g2 <- make_sub_graph(geom_of(region)[ids])
    predict(fitted, new_data = build_df(ids, region, g2$lw, with_lags))$.pred
  }
}

# LayerNorm recenters each node's hidden vector using that node's own values,
# so a shift in the target region's feature distribution cannot propagate
# through the layers as a level error. The pair is seeded identically by the
# caller, so norm is the only difference between them.
fit_sage <- function(train_id, val_id, norm) {
  ids <- c(train_id, val_id)
  scaler <- make_scaler(ids)
  g <- make_sub_graph(geom_of("kc")[ids])

  x_t <- torch_tensor(scaler(ids, "kc"), dtype = torch_float32())
  y_t <- torch_tensor(kc$y[ids], dtype = torch_float32())$view(c(-1, 1))

  train_idx <- seq_along(train_id)
  val_idx <- seq_along(val_id) + length(train_id)

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
    out <- model(x_t, g$adj)
    loss <- nnf_l1_loss(out[train_idx, ], y_t[train_idx, ])
    loss$backward()
    optimizer$step()

    with_no_grad({
      model$eval()
      v <- nnf_l1_loss(
        model(x_t, g$adj)[val_idx, ],
        y_t[val_idx, ]
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

  function(ids2, region) {
    g2 <- make_sub_graph(geom_of(region)[ids2])
    x_te <- torch_tensor(scaler(ids2, region), dtype = torch_float32())
    with_no_grad({
      preds <- model(x_te, g2$adj)
    })
    as.numeric(preds$detach())
  }
}

# GraphSAGE is the only arm that holds rows back for early stopping, so it sees
# strictly less data than the others.
#
# The transfer to Ames is a single fit rather than an average over folds, so a
# single initialisation does not describe it: plain GraphSAGE's Ames R2 ranges
# from 0.04 to 0.60 across seeds on identical data. Both GraphSAGE arms are
# therefore refit under every seed and reported as a mean with its spread. OLS
# and XGBoost are deterministic at these settings and are fit once.
seeds <- 1001:1030

arms <- list(
  OLS = function(train_id, val_id, seed) fit_ols(c(train_id, val_id)),
  XGBoost = function(train_id, val_id, seed) {
    fit_xgb(c(train_id, val_id), with_lags = FALSE, seed = seed)
  },
  `XGBoost + lags` = function(train_id, val_id, seed) {
    fit_xgb(c(train_id, val_id), with_lags = TRUE, seed = seed)
  },
  GraphSAGE = function(train_id, val_id, seed) {
    torch_manual_seed(seed)
    fit_sage(train_id, val_id, norm = NULL)
  },
  `GraphSAGE + LayerNorm` = function(train_id, val_id, seed) {
    torch_manual_seed(seed)
    fit_sage(train_id, val_id, norm = layer_layer_norm)
  }
)

# Only OLS is deterministic. Both GraphSAGE arms vary with initialisation and
# both XGBoost arms vary with their internal validation split, so all four are
# refit under every seed.
stochastic <- c(
  "XGBoost", "XGBoost + lags", "GraphSAGE", "GraphSAGE + LayerNorm"
)

# Scoring ----------------------------------------------------------------

reg_metrics <- metric_set(mae, rmse, rsq)

# bias is estimate - truth, so a positive value is a model predicting too high.
# cal_slope is the calibration slope, the coefficient from regressing the
# observed value on the prediction. One is correct. Above one means the
# predictions are compressed toward their mean and need spreading out; below
# one means they are more dispersed than the truth.
score <- function(truth, estimate) {
  d <- data.frame(truth = truth, estimate = estimate)
  m <- reg_metrics(d, truth = truth, estimate = estimate)
  data.frame(
    mae = m$.estimate[m$.metric == "mae"],
    rmse = m$.estimate[m$.metric == "rmse"],
    rsq = m$.estimate[m$.metric == "rsq"],
    bias = mean(estimate - truth),
    cal_slope = unname(coef(lm(truth ~ estimate))[2])
  )
}


# Folds ------------------------------------------------------------------

set.seed(1)
kc_fold_id <- sample(rep_len(seq_len(k_folds), nrow(kc)))

kc_splits <- lapply(seq_len(k_folds), function(i) {
  test_id <- which(kc_fold_id == i)
  ana <- setdiff(seq_len(nrow(kc)), test_id)
  set.seed(100L + i)
  val_id <- sample(ana, size = floor(val_prop * length(ana)))
  list(train_id = setdiff(ana, val_id), val_id = val_id, test_id = test_id)
})

set.seed(0)
final_val <- sample(nrow(kc), size = floor(val_prop * nrow(kc)))
final_train <- setdiff(seq_len(nrow(kc)), final_val)

# Worker pool ------------------------------------------------------------
#
# Daemons cannot be handed torch objects, so each one sources this file at
# startup and rebuilds what it needs. A task then carries only scalars.
#
# torch_set_num_threads(1) is what makes the pool pay. torch defaults to one
# intra-op thread per core, so every daemon would otherwise try to claim the
# whole machine. A forward pass on a 21,613-node sparse graph is too small to
# repay threading anyway; the parallelism worth having is across fits.
start_daemons <- function(n_daemons = max(1L, parallel::detectCores() - 2L)) {
  core <- normalizePath("R/kc-to-ames-core.R", mustWork = TRUE)
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

# Tasks ------------------------------------------------------------------
#
# Both are top level so mirai serialises them against the global environment,
# which it sends by reference. A closure defined inside the driver would drag
# the driver's frame into every task.

kc_fold_task <- function(spec) {
  s <- kc_splits[[spec$fold]]
  predictor <- arms[[spec$arm]](s$train_id, s$val_id, spec$seed)
  cbind(
    fold = spec$fold,
    arm = spec$arm,
    score(kc$y[s$test_id], predictor(s$test_id, "kc"))
  )
}

ames_task <- function(spec) {
  predictor <- arms[[spec$arm]](final_train, final_val, spec$seed)
  cbind(
    arm = spec$arm,
    seed = spec$seed,
    score(ames$y, predictor(seq_len(nrow(ames)), "ames"))
  )
}

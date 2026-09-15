# Which design choices make the King County to Ames transfer work.
#
# The transfer currently centres y within each region but standardises X using
# King County's moments. That is inconsistent: centring y concedes the price
# level does not transfer, while scaling X by the source region asserts the
# covariate level does. A 2,000 square foot house is above the median in Ames
# and below it in King County, and the network is handed the same input for
# both.
#
# Two axes are varied here.
#
#   x_scale  source  features standardised with King County's mean and sd
#            self    each region standardised with its own moments
#
#   k_nb     30      the inherited neighbourhood size
#            15      matches the simulations
#
# Self-scaling uses only covariates of the target region, which are available
# at prediction time, so it introduces no outcome leakage. The counter-argument
# is that a fitted preprocessor is part of the model; both positions are set out
# in 2026-04-09-gnn-proposal/R/kc-to-ames-selfscale.R.
#
# XGBoost is the control. Tree splits are invariant to a monotone rescaling of
# any single feature, so its within-region numbers should not move at all. Its
# transfer numbers can move, because the two regions are then rescaled by
# different constants.
#
# Run from the repo root:
#   R -f R/kc-to-ames-designs.R

library(dplyr)
library(sf)
library(spdep)
library(torch)
library(torchgnn)
library(parsnip)
library(recipes)
library(tune)
library(workflows)
library(yardstick)

val_prop <- 0.2
n_epochs <- 500L
lr <- 0.01
patience <- 20L
sage_hidden <- c(56, 32, 16)

feature_cols <- c(
  "lot_area", "living_area", "above_grade", "basement",
  "bedrooms", "bathrooms", "yr_built", "renovated"
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
    long, lat
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
    bathrooms = Full_Bath + Bsmt_Full_Bath +
      0.5 * (Half_Bath + Bsmt_Half_Bath),
    yr_built = Year_Built,
    renovated = as.integer(Year_Remod_Add > Year_Built),
    Longitude, Latitude
  ) |>
  st_as_sf(coords = c("Longitude", "Latitude"), crs = 4326) |>
  st_transform(3857)

kc <- kc |> mutate(y = log(price) - mean(log(price)))
ames <- ames |> mutate(y = log(price) - mean(log(price)))

X_kc <- as.matrix(st_drop_geometry(kc)[, feature_cols])
X_ames <- as.matrix(st_drop_geometry(ames)[, feature_cols])

geom_of <- function(r) if (r == "kc") st_geometry(kc) else st_geometry(ames)
X_of <- function(r) if (r == "kc") X_kc else X_ames
y_of <- function(r) if (r == "kc") kc$y else ames$y

# How far apart the two regions actually are, which is what the scaling choice
# has to survive.
cat("\n=== Covariate moments by region ===\n")
print(
  data.frame(
    feature = feature_cols,
    kc_mean = colMeans(X_kc),
    ames_mean = colMeans(X_ames),
    kc_sd = apply(X_kc, 2, sd),
    ames_sd = apply(X_ames, 2, sd),
    # Where an average Ames house sits on King County's standardised scale.
    ames_mean_in_kc_units = (colMeans(X_ames) - colMeans(X_kc)) / apply(X_kc, 2, sd)
  ),
  row.names = FALSE, digits = 3
)

cat(sprintf(
  "\nsd(y): King County %.3f, Ames %.3f, ratio %.3f\n",
  sd(kc$y), sd(ames$y), sd(ames$y) / sd(kc$y)
))

# Graphs and scaling -----------------------------------------------------

make_sub_graph <- function(geom, k_nb) {
  nb <- sfdep::st_knn(geom, k = min(k_nb, length(geom) - 1L))
  list(
    adj = adj_from_edgelist(
      rep.int(seq_along(nb), lengths(nb)), unlist(nb)
    ) |> add_graph_self_loops(),
    lw = nb2listw(nb)
  )
}

# Returns a function(ids, region) giving the standardised design matrix.
make_scaler <- function(fit_ids, x_scale) {
  mu_src <- colMeans(X_kc[fit_ids, , drop = FALSE])
  sd_src <- apply(X_kc[fit_ids, , drop = FALSE], 2, sd)
  sd_src[sd_src == 0] <- 1

  mu_self <- list(kc = mu_src, ames = colMeans(X_ames))
  sd_self <- list(kc = sd_src, ames = apply(X_ames, 2, sd))
  sd_self$ames[sd_self$ames == 0] <- 1

  function(ids, region) {
    if (x_scale == "source") {
      scale(X_of(region)[ids, , drop = FALSE], mu_src, sd_src)
    } else {
      scale(X_of(region)[ids, , drop = FALSE], mu_self[[region]], sd_self[[region]])
    }
  }
}

# Arms -------------------------------------------------------------------

frame_of <- function(ids, region, scaler) {
  d <- as.data.frame(scaler(ids, region))
  d$y <- y_of(region)[ids]
  d
}

fit_ols <- function(fit_ids, scaler, k_nb) {
  m <- lm(y ~ ., data = frame_of(fit_ids, "kc", scaler))
  function(ids, region) as.numeric(predict(m, newdata = frame_of(ids, region, scaler)))
}

build_df <- function(ids, region, lw, with_lags, scaler) {
  X <- scaler(ids, region)
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
  set_engine("xgboost", objective = "reg:absoluteerror", validation = val_prop) |>
  set_mode("regression")

fit_xgb <- function(fit_ids, scaler, k_nb, with_lags) {
  g <- make_sub_graph(geom_of("kc")[fit_ids], k_nb)
  train_df <- build_df(fit_ids, "kc", g$lw, with_lags, scaler)
  wf <- workflow() |>
    add_recipe(recipe(y ~ ., data = train_df)) |>
    add_model(xgb_spec)
  fitted <- suppressWarnings(fit(wf, data = train_df))
  function(ids, region) {
    g2 <- make_sub_graph(geom_of(region)[ids], k_nb)
    predict(fitted, new_data = build_df(ids, region, g2$lw, with_lags, scaler))$.pred
  }
}

fit_sage <- function(train_id, val_id, scaler, k_nb, norm) {
  ids <- c(train_id, val_id)
  g <- make_sub_graph(geom_of("kc")[ids], k_nb)

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
    loss <- nnf_l1_loss(model(x_t, g$adj)[train_idx, ], y_t[train_idx, ])
    loss$backward()
    optimizer$step()

    with_no_grad({
      model$eval()
      v <- nnf_l1_loss(model(x_t, g$adj)[val_idx, ], y_t[val_idx, ])$item()
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
    g2 <- make_sub_graph(geom_of(region)[ids2], k_nb)
    x_te <- torch_tensor(scaler(ids2, region), dtype = torch_float32())
    with_no_grad({
      p <- model(x_te, g2$adj)
    })
    as.numeric(p$detach())
  }
}

# A design cell is one fit per seed for the two GraphSAGE arms. OLS is
# deterministic and XGBoost nearly so, so they are fit once and their numbers
# repeated, which keeps the table readable without pretending to a spread they
# do not have.
seeds <- c(123L, 224L, 325L, 426L, 527L)

arms <- list(
  OLS = function(tr, va, scaler, k, seed) fit_ols(c(tr, va), scaler, k),
  `XGBoost + lags` = function(tr, va, scaler, k, seed) {
    fit_xgb(c(tr, va), scaler, k, with_lags = TRUE)
  },
  GraphSAGE = function(tr, va, scaler, k, seed) {
    torch_manual_seed(seed)
    fit_sage(tr, va, scaler, k, norm = NULL)
  },
  `GraphSAGE + LayerNorm` = function(tr, va, scaler, k, seed) {
    torch_manual_seed(seed)
    fit_sage(tr, va, scaler, k, norm = layer_layer_norm)
  }
)

stochastic <- c("GraphSAGE", "GraphSAGE + LayerNorm")

reg_metrics <- metric_set(mae, rmse, rsq)

score <- function(truth, estimate) {
  d <- data.frame(truth = truth, estimate = estimate)
  m <- reg_metrics(d, truth = truth, estimate = estimate)
  data.frame(
    mae = m$.estimate[m$.metric == "mae"],
    rsq = m$.estimate[m$.metric == "rsq"],
    bias = mean(estimate - truth),
    cal_slope = unname(coef(lm(truth ~ estimate))[2])
  )
}

# Run --------------------------------------------------------------------

set.seed(0)
final_val <- sample(nrow(kc), size = floor(val_prop * nrow(kc)))
final_train <- setdiff(seq_len(nrow(kc)), final_val)

designs <- expand.grid(
  x_scale = c("source", "self"),
  k_nb = c(30L, 15L),
  stringsAsFactors = FALSE
)

ames_ids <- seq_len(nrow(ames))

results <- do.call(rbind, lapply(seq_len(nrow(designs)), function(d) {
  x_scale <- designs$x_scale[d]
  k_nb <- designs$k_nb[d]
  cat(sprintf("\n=== x_scale = %s, k = %d ===\n", x_scale, k_nb))
  scaler <- make_scaler(c(final_train, final_val), x_scale)

  do.call(rbind, lapply(names(arms), function(a) {
    use_seeds <- if (a %in% stochastic) seeds else seeds[1]
    per_seed <- do.call(rbind, lapply(use_seeds, function(s) {
      predictor <- arms[[a]](final_train, final_val, scaler, k_nb, s)
      kc_fit <- score(kc$y[final_val], predictor(final_val, "kc"))
      am <- score(ames$y, predictor(ames_ids, "ames"))
      data.frame(
        x_scale = x_scale, k_nb = k_nb, arm = a, seed = s,
        kc_val_rsq = kc_fit$rsq,
        ames_mae = am$mae, ames_rsq = am$rsq,
        ames_bias = am$bias, ames_cal_slope = am$cal_slope
      )
    }))
    cat(sprintf(
      "  %-22s  Ames rsq %.3f (sd %.3f, min %.3f, max %.3f)  slope %.3f\n",
      a, mean(per_seed$ames_rsq), sd(per_seed$ames_rsq),
      min(per_seed$ames_rsq), max(per_seed$ames_rsq),
      mean(per_seed$ames_cal_slope)
    ))
    per_seed
  }))
}))

summary_tbl <- results |>
  group_by(arm, x_scale, k_nb) |>
  summarise(
    n = dplyr::n(),
    kc_rsq = mean(kc_val_rsq),
    ames_rsq = mean(ames_rsq),
    ames_rsq_sd = sd(ames_rsq),
    ames_bias = mean(ames_bias),
    cal_slope = mean(ames_cal_slope),
    .groups = "drop"
  ) |>
  as.data.frame()

cat("\n\n=== Ames transfer across designs, averaged over seeds ===\n")
print(summary_tbl |> arrange(arm, x_scale, k_nb), row.names = FALSE, digits = 3)

cat("\n=== Seed spread versus design effect, GraphSAGE arms ===\n")
spread <- results |>
  filter(arm %in% stochastic) |>
  group_by(arm, x_scale, k_nb) |>
  summarise(sd_within = sd(ames_rsq), .groups = "drop") |>
  group_by(arm) |>
  summarise(
    mean_sd_within_cell = mean(sd_within),
    sd_across_cells = sd(
      summary_tbl$ames_rsq[summary_tbl$arm == dplyr::first(arm)]
    ),
    .groups = "drop"
  ) |>
  as.data.frame()
print(spread, row.names = FALSE, digits = 3)

dir.create("data", showWarnings = FALSE)
saveRDS(
  list(per_seed = results, summary = summary_tbl),
  "data/kc-to-ames-designs.rds"
)
cat("\nSaved to data/kc-to-ames-designs.rds\n")

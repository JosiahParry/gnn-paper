# Classification transfer core: binary land cover, one region -> another.
#
# The project's first classification task. Everything else here is regression,
# and the deliverable (an ArcGIS Pro GNN tool) needs to handle both.
#
# WHY A GRAPH SHOULD HELP HERE, CONCRETELY. Bare soil in a city and bare soil
# in a farm field have nearly identical spectra. Pixel by pixel they are not
# separable -- no tabular model can tell them apart, because the information
# is not in the pixel. What separates them is what SURROUNDS them: one is
# ringed by roofs and asphalt, the other by crops. Neighbour aggregation
# supplies exactly that context, and it supplies it from neighbours'
# COVARIATES, which is what GraphSAGE can actually see in a region where no
# labels exist.
#
# That makes this a cleaner test of the graph than any regression here: the
# benefit has a concrete, checkable mechanism rather than a vague appeal to
# "spatial structure".
#
# Metrics are classification metrics throughout. Accuracy alone is misleading
# when one class dominates, so balanced accuracy and AUC are the headline
# numbers and the class balance is always reported alongside.
#
# Configure before sourcing:
#   options(clf_data = "data/lulc-clean.rds", clf_src = "desmoines",
#           clf_tgt = "cedarrapids", clf_cell = 60)

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

val_prop <- 0.2
n_epochs <- 500L
lr <- 0.01
patience <- 20L
sage_hidden <- c(56, 32, 16)
k_nb <- 30L

CELL <- getOption("clf_cell", 60)
SRC  <- getOption("clf_src", "desmoines")
TGT  <- getOption("clf_tgt", "cedarrapids")
DATA <- getOption("clf_data", "data/lulc-clean.rds")

layer_layer_norm_node <- function(dim) layer_layer_norm(dim, mode = "node")

cleaned <- readRDS(DATA)
stopifnot(all(c(SRC, TGT) %in% names(cleaned)))

as_grid <- function(d) st_as_sf(d, coords = c("lon", "lat"), crs = NA)
src_sf <- as_grid(cleaned[[SRC]])
tgt_sf <- as_grid(cleaned[[TGT]])

feature_cols <- setdiff(names(st_drop_geometry(src_sf)), c("y", "lulc_raw"))

cat(sprintf("%s (source) n = %d, positive rate %.3f\n", SRC, nrow(src_sf), mean(src_sf$y)))
cat(sprintf("%s (target) n = %d, positive rate %.3f\n", TGT, nrow(tgt_sf), mean(tgt_sf$y)))
cat(sprintf("features (%d): %s\n", length(feature_cols), paste(feature_cols, collapse = ", ")))

reg_sf <- function(r) if (r == "src") src_sf else tgt_sf
X_of <- function(r) as.matrix(st_drop_geometry(reg_sf(r))[, feature_cols])
y_of <- function(r) reg_sf(r)$y
geom_of <- function(r) st_geometry(reg_sf(r))

# Graphs -----------------------------------------------------------------

GRAPH_RADIUS <- c(queen = 1.45, queen2 = 2.9)

make_sub_graph <- function(geom, graph = "queen") {
  coords <- sf::st_coordinates(geom)
  if (graph == "knn30") {
    sub_nb <- sfdep::st_knn(geom, k = min(k_nb, length(geom) - 1L))
  } else {
    sub_nb <- spdep::dnearneigh(coords, 0, CELL * GRAPH_RADIUS[[graph]])
    for (i in which(vapply(sub_nb, function(z) identical(z, 0L), logical(1)))) sub_nb[[i]] <- i
    class(sub_nb) <- "nb"
  }
  from <- rep.int(seq_along(sub_nb), lengths(sub_nb))
  adj <- adj_from_edgelist(from, unlist(sub_nb)) |> add_graph_self_loops()
  list(adj = adj, lw = nb2listw(sub_nb, zero.policy = TRUE))
}

make_scaler <- function(ids) {
  mu <- colMeans(X_of("src")[ids, , drop = FALSE])
  sdv <- apply(X_of("src")[ids, , drop = FALSE], 2, sd); sdv[sdv == 0] <- 1
  function(ids2, r) scale(X_of(r)[ids2, , drop = FALSE], mu, sdv)
}

# Arms -------------------------------------------------------------------
# Every arm returns PROBABILITIES, so all arms are scored identically.

frame_of <- function(ids, r) {
  d <- as.data.frame(X_of(r)[ids, , drop = FALSE]); d$y <- y_of(r)[ids]; d
}

fit_logit <- function(fit_ids) {
  m <- suppressWarnings(glm(y ~ ., data = frame_of(fit_ids, "src"), family = binomial()))
  function(ids, r) as.numeric(suppressWarnings(
    predict(m, newdata = frame_of(ids, r), type = "response")))
}

build_df <- function(ids, r, lw, with_lags) {
  X <- X_of(r)[ids, , drop = FALSE]
  out <- as.data.frame(X)
  if (with_lags) {
    lags <- as.data.frame(lag.listw(lw, X, zero.policy = TRUE))
    colnames(lags) <- paste0("lag_", feature_cols)
    out <- cbind(out, lags)
  }
  out$y <- factor(y_of(r)[ids], levels = c(0, 1))
  out
}

xgb_spec <- boost_tree(trees = 500, stop_iter = patience) |>
  set_engine("xgboost", validation = val_prop) |>
  set_mode("classification")

fit_xgb <- function(fit_ids, with_lags, seed, graph) {
  g <- make_sub_graph(geom_of("src")[fit_ids], graph)
  train_df <- build_df(fit_ids, "src", g$lw, with_lags)
  set.seed(seed)
  wf <- workflow() |>
    add_recipe(recipe(y ~ ., data = train_df) |> step_normalize(all_numeric_predictors())) |>
    add_model(xgb_spec)
  fitted <- suppressWarnings(fit(wf, data = train_df))
  function(ids, r) {
    g2 <- make_sub_graph(geom_of(r)[ids], graph)
    predict(fitted, new_data = build_df(ids, r, g2$lw, with_lags), type = "prob")$.pred_1
  }
}

fit_sage <- function(train_id, val_id, norm, dropout = 0, wd = 0, graph = "queen") {
  ids <- c(train_id, val_id)
  scaler <- make_scaler(ids)
  g <- make_sub_graph(geom_of("src")[ids], graph)

  x_t <- torch_tensor(scaler(ids, "src"), dtype = torch_float32())
  # torch cross-entropy wants 1-based class indices, so 0/1 becomes 1/2.
  y_t <- torch_tensor(y_of("src")[ids] + 1L, dtype = torch_long())

  train_idx <- seq_along(train_id)
  val_idx <- seq_along(val_id) + length(train_id)

  model <- model_sage(in_features = length(feature_cols), hidden_dims = sage_hidden,
                      out_features = 2L, norm = norm, dropout = dropout)
  optimizer <- optim_adam(model$parameters, lr = lr, weight_decay = wd)

  best_val <- Inf; best_state <- NULL; no_improve <- 0L
  for (epoch in seq_len(n_epochs)) {
    model$train(); optimizer$zero_grad()
    out <- model(x_t, g$adj)
    loss <- nnf_cross_entropy(out[train_idx, ], y_t[train_idx])
    loss$backward(); optimizer$step()
    with_no_grad({
      model$eval()
      vl <- nnf_cross_entropy(model(x_t, g$adj)[val_idx, ], y_t[val_idx])$item()
    })
    if (vl < best_val) {
      best_val <- vl; best_state <- lapply(model$state_dict(), \(t) t$clone()); no_improve <- 0L
    } else no_improve <- no_improve + 1L
    if (no_improve >= patience) break
  }
  model$load_state_dict(best_state); model$eval()

  function(ids2, r) {
    g2 <- make_sub_graph(geom_of(r)[ids2], graph)
    x2 <- torch_tensor(scaler(ids2, r), dtype = torch_float32())
    with_no_grad({
      p <- nnf_softmax(model(x2, g2$adj), dim = 2)
      as.numeric(p[, 2])
    })
  }
}

arms <- list(
  Logistic = function(tr, va, seed, graph) fit_logit(c(tr, va)),
  XGBoost = function(tr, va, seed, graph) fit_xgb(c(tr, va), FALSE, seed, graph),
  `XGBoost + lags` = function(tr, va, seed, graph) fit_xgb(c(tr, va), TRUE, seed, graph),
  GraphSAGE = function(tr, va, seed, graph) {
    torch_manual_seed(seed); fit_sage(tr, va, norm = NULL, graph = graph)
  },
  `GraphSAGE + LayerNorm` = function(tr, va, seed, graph) {
    torch_manual_seed(seed)
    fit_sage(tr, va, norm = layer_layer_norm_node, dropout = 0.1, wd = 1e-4, graph = graph)
  }
)
stochastic <- c("XGBoost", "XGBoost + lags", "GraphSAGE", "GraphSAGE + LayerNorm")

# Scoring ----------------------------------------------------------------
# Accuracy alone is misleading when one class dominates (built area is ~85%
# of these scenes), so balanced accuracy and AUC lead.

score <- function(truth, prob) {
  pred <- as.integer(prob > 0.5)
  tp <- sum(pred == 1 & truth == 1); tn <- sum(pred == 0 & truth == 0)
  fp <- sum(pred == 1 & truth == 0); fn <- sum(pred == 0 & truth == 1)
  sens <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
  spec <- if ((tn + fp) > 0) tn / (tn + fp) else NA_real_
  d <- data.frame(t = factor(truth, levels = c(0, 1)), p = prob)
  auc <- tryCatch(
    yardstick::roc_auc_vec(d$t, d$p, event_level = "second"), error = function(e) NA_real_)
  data.frame(
    accuracy = (tp + tn) / length(truth),
    bal_accuracy = mean(c(sens, spec), na.rm = TRUE),
    auc = auc, sensitivity = sens, specificity = spec,
    brier = mean((prob - truth)^2),
    pred_rate = mean(pred)
  )
}

set.seed(0)
final_val <- sample(nrow(src_sf), size = floor(val_prop * nrow(src_sf)))
final_train <- setdiff(seq_len(nrow(src_sf)), final_val)

start_daemons <- function(n_daemons = max(1L, parallel::detectCores() - 2L)) {
  core <- normalizePath("R/clf-core.R", mustWork = TRUE)
  daemons(n_daemons)
  everywhere({
    options(clf_data = pd, clf_src = ps, clf_tgt = pt, clf_cell = pc)
    source(core_path, local = FALSE)
    torch_set_num_threads(1L)
  }, .args = list(core_path = core, pd = DATA, ps = SRC, pt = TGT, pc = CELL),
     .min = n_daemons)
  cat(sprintf("%d daemons up (%s -> %s)\n", n_daemons, SRC, TGT))
  invisible(n_daemons)
}

target_task <- function(spec) {
  graph <- if (is.null(spec$graph)) "queen" else spec$graph
  predictor <- arms[[spec$arm]](final_train, final_val, spec$seed, graph)
  cbind(src = SRC, tgt = TGT, arm = spec$arm, seed = spec$seed, graph = graph,
        score(y_of("tgt"), predictor(seq_len(nrow(tgt_sf)), "tgt")))
}

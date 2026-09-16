# Transfer core. Configure with options(gnn_*) then source.
#
#   gnn_data   rds holding a named list of regions; each has lon, lat, y, features
#   gnn_src    source region name
#   gnn_tgt    target region name
#   gnn_crs    4326 for degrees, NA for projected metres
#   gnn_graph  knn30 | queen | queen2
#   gnn_cell   grid spacing in metres, for queen graphs
#   gnn_mode   reg | clf

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

opt <- function(n, d) getOption(paste0("gnn_", n), d)

DATA <- opt("data", NULL)
SRC <- opt("src", NULL)
TGT <- opt("tgt", NULL)
CRS <- opt("crs", 4326)
GRAPH <- opt("graph", "knn30")
CELL <- opt("cell", 60)
MODE <- opt("mode", "reg")

k_nb <- 30L
val_prop <- 0.2
n_epochs <- 500L
lr <- 0.01
patience <- 20L
sage_hidden <- c(56, 32, 16)

# layer_layer_norm defaults to mode "graph", a single scalar over the whole
# tensor. Per-node is what makes transfer work.
norm_node <- function(dim) layer_layer_norm(dim, mode = "node")

cleaned <- nanoparquet::read_parquet(DATA)
stopifnot("region" %in% names(cleaned), all(c(SRC, TGT) %in% cleaned$region))
region_rows <- function(r) {
  d <- cleaned[cleaned$region == r, , drop = FALSE]
  d[, setdiff(names(d), "region"), drop = FALSE]
}

as_sf <- function(d) {
  g <- st_as_sf(d, coords = c("lon", "lat"), crs = CRS)
  if (!is.na(CRS) && CRS == 4326) {
    g <- st_transform(g, 3857)
  }
  g
}
src_sf <- as_sf(region_rows(SRC))
tgt_sf <- as_sf(region_rows(TGT))

# Drop the target, non-numeric label columns, and any raw column the target
# was derived from -- price would let the model predict itself.
drop_cols <- c("y", "lulc_raw", "boro", "price", "soc", "lst", "ndvi_raw")
feature_cols <- setdiff(names(st_drop_geometry(src_sf)), drop_cols)
feature_cols <- feature_cols[vapply(
  st_drop_geometry(src_sf)[feature_cols],
  is.numeric,
  logical(1)
)]
stopifnot(length(feature_cols) > 0)

reg_sf <- function(r) if (r == "src") src_sf else tgt_sf
X_of <- function(r) as.matrix(st_drop_geometry(reg_sf(r))[, feature_cols])
y_of <- function(r) reg_sf(r)$y
geom_of <- function(r) st_geometry(reg_sf(r))

# Graph -------------------------------------------------------------------

RADIUS <- c(queen = 1.45, queen2 = 2.9)

make_graph <- function(geom, graph = GRAPH, kernel = "uniform", mult = 1) {
  coords <- st_coordinates(geom)
  if (graph == "knn30") {
    nb <- sfdep::st_knn(geom, k = min(k_nb, length(geom) - 1L))
  } else {
    nb <- spdep::dnearneigh(coords, 0, CELL * RADIUS[[graph]])
    for (i in which(vapply(nb, function(z) identical(z, 0L), logical(1)))) {
      nb[[i]] <- i
    }
    class(nb) <- "nb"
  }
  from <- rep.int(seq_along(nb), lengths(nb))

  if (kernel == "uniform") {
    w <- NULL
    glist <- NULL
  } else {
    thr <- stats::median(unlist(spdep::nbdists(nb, coords))) * mult
    kw <- sfdep::st_kernel_weights(
      nb,
      geom,
      kernel = kernel,
      threshold = thr,
      adaptive = FALSE
    )
    w <- unlist(kw)
    glist <- lapply(kw, function(v) pmax(v, 1e-8)) # spdep rejects all-zero rows
  }
  adj <- adj_from_edgelist(from, unlist(nb), weight = w) |>
    add_graph_self_loops()
  lw <- if (is.null(glist)) {
    nb2listw(nb, zero.policy = TRUE)
  } else {
    nb2listw(nb, glist = glist, style = "W", zero.policy = TRUE)
  }
  list(adj = adj, lw = lw)
}

default_k <- list(kernel = "uniform", mult = 1)
sub_graph <- function(geom, k) make_graph(geom, GRAPH, k$kernel, k$mult)

scaler_for <- function(ids) {
  mu <- colMeans(X_of("src")[ids, , drop = FALSE])
  sdv <- apply(X_of("src")[ids, , drop = FALSE], 2, sd)
  sdv[sdv == 0] <- 1
  function(ids2, r) scale(X_of(r)[ids2, , drop = FALSE], mu, sdv)
}

# Arms --------------------------------------------------------------------

frame_of <- function(ids, r) {
  d <- as.data.frame(X_of(r)[ids, , drop = FALSE])
  d$y <- y_of(r)[ids]
  d
}

fit_linear <- function(ids) {
  if (MODE == "clf") {
    m <- suppressWarnings(glm(
      y ~ .,
      data = frame_of(ids, "src"),
      family = binomial()
    ))
    function(i, r) {
      as.numeric(suppressWarnings(predict(
        m,
        frame_of(i, r),
        type = "response"
      )))
    }
  } else {
    m <- lm(y ~ ., data = frame_of(ids, "src"))
    function(i, r) as.numeric(predict(m, newdata = frame_of(i, r)))
  }
}

build_df <- function(ids, r, lw, lags) {
  X <- X_of(r)[ids, , drop = FALSE]
  out <- as.data.frame(X)
  if (lags) {
    l <- as.data.frame(lag.listw(lw, X, zero.policy = TRUE))
    names(l) <- paste0("lag_", feature_cols)
    out <- cbind(out, l)
  }
  out$y <- if (MODE == "clf") {
    factor(y_of(r)[ids], levels = c(0, 1))
  } else {
    y_of(r)[ids]
  }
  out
}

xgb_spec <- function() {
  s <- boost_tree(trees = 500, stop_iter = patience) |>
    set_engine("xgboost", validation = val_prop)
  if (MODE == "clf") {
    set_mode(s, "classification")
  } else {
    set_mode(s, "regression") |>
      set_engine(
        "xgboost",
        objective = "reg:squarederror",
        validation = val_prop
      )
  }
}

fit_xgb <- function(ids, lags, seed, k = default_k) {
  g <- sub_graph(geom_of("src")[ids], k)
  tr <- build_df(ids, "src", g$lw, lags)
  set.seed(seed)
  wf <- workflow() |>
    add_recipe(
      recipe(y ~ ., data = tr) |> step_normalize(all_numeric_predictors())
    ) |>
    add_model(xgb_spec())
  fitted <- suppressWarnings(fit(wf, data = tr))
  function(i, r) {
    g2 <- sub_graph(geom_of(r)[i], k)
    nd <- build_df(i, r, g2$lw, lags)
    if (MODE == "clf") {
      predict(fitted, nd, type = "prob")$.pred_1
    } else {
      predict(fitted, nd)$.pred
    }
  }
}

fit_sage <- function(
  train_id,
  val_id,
  norm,
  dropout = 0,
  wd = 0,
  k = default_k
) {
  ids <- c(train_id, val_id)
  scaler <- scaler_for(ids)
  g <- sub_graph(geom_of("src")[ids], k)
  x_t <- torch_tensor(scaler(ids, "src"), dtype = torch_float32())

  clf <- MODE == "clf"
  y_t <- if (clf) {
    torch_tensor(y_of("src")[ids] + 1L, dtype = torch_long())
  } else {
    torch_tensor(y_of("src")[ids], dtype = torch_float32())$view(c(-1, 1))
  }
  loss_fn <- if (clf) {
    function(o, y) nnf_cross_entropy(o, y)
  } else {
    function(o, y) nnf_l1_loss(o, y)
  }

  ti <- seq_along(train_id)
  vi <- seq_along(val_id) + length(train_id)
  model <- model_sage(
    in_features = length(feature_cols),
    hidden_dims = sage_hidden,
    out_features = if (clf) 2L else 1L,
    norm = norm,
    dropout = dropout
  )
  opt_ <- optim_adam(model$parameters, lr = lr, weight_decay = wd)

  best <- Inf
  state <- NULL
  stall <- 0L
  for (e in seq_len(n_epochs)) {
    model$train()
    opt_$zero_grad()
    o <- model(x_t, g$adj)
    l <- if (clf) loss_fn(o[ti, ], y_t[ti]) else loss_fn(o[ti, ], y_t[ti, ])
    l$backward()
    opt_$step()
    with_no_grad({
      model$eval()
      o2 <- model(x_t, g$adj)
      vl <- if (clf) {
        loss_fn(o2[vi, ], y_t[vi])$item()
      } else {
        loss_fn(o2[vi, ], y_t[vi, ])$item()
      }
    })
    if (vl < best) {
      best <- vl
      state <- lapply(model$state_dict(), \(t) t$clone())
      stall <- 0L
    } else {
      stall <- stall + 1L
    }
    if (stall >= patience) break
  }
  model$load_state_dict(state)
  model$eval()

  function(i, r) {
    g2 <- sub_graph(geom_of(r)[i], k)
    x2 <- torch_tensor(scaler(i, r), dtype = torch_float32())
    with_no_grad({
      o <- model(x2, g2$adj)
      if (clf) {
        as.numeric(nnf_softmax(o, dim = 2)[, 2])
      } else {
        as.numeric(o$squeeze())
      }
    })
  }
}

arms <- list(
  Linear = function(tr, va, seed, k) fit_linear(c(tr, va)),
  XGBoost = function(tr, va, seed, k) fit_xgb(c(tr, va), FALSE, seed),
  `XGBoost + lags` = function(tr, va, seed, k) {
    fit_xgb(c(tr, va), TRUE, seed, k)
  },
  GraphSAGE = function(tr, va, seed, k) {
    torch_manual_seed(seed)
    fit_sage(tr, va, NULL, k = k)
  },
  `GraphSAGE + LayerNorm` = function(tr, va, seed, k) {
    torch_manual_seed(seed)
    fit_sage(tr, va, norm_node, 0.1, 1e-4, k)
  }
)
stochastic <- c(
  "XGBoost",
  "XGBoost + lags",
  "GraphSAGE",
  "GraphSAGE + LayerNorm"
)

# Scoring -----------------------------------------------------------------

reg_metrics <- metric_set(mae, rmse, rsq, rsq_trad)

score <- function(truth, est) {
  if (MODE == "clf") {
    pred <- as.integer(est > 0.5)
    tp <- sum(pred == 1 & truth == 1)
    tn <- sum(pred == 0 & truth == 0)
    fp <- sum(pred == 1 & truth == 0)
    fn <- sum(pred == 0 & truth == 1)
    sens <- tp / max(tp + fn, 1)
    spec <- tn / max(tn + fp, 1)
    data.frame(
      accuracy = (tp + tn) / length(truth),
      bal_accuracy = mean(c(sens, spec)),
      auc = tryCatch(
        roc_auc_vec(
          factor(truth, levels = c(0, 1)),
          est,
          event_level = "second"
        ),
        error = function(e) NA_real_
      ),
      sensitivity = sens,
      specificity = spec,
      brier = mean((est - truth)^2)
    )
  } else {
    m <- reg_metrics(
      data.frame(truth = truth, estimate = est),
      truth = truth,
      estimate = estimate
    )
    g <- function(n) m$.estimate[m$.metric == n]
    data.frame(
      mae = g("mae"),
      rmse = g("rmse"),
      rsq = g("rsq"),
      rsq_trad = g("rsq_trad"),
      bias = mean(est - truth),
      cal_slope = unname(coef(lm(truth ~ est))[2])
    )
  }
}

set.seed(0)
final_val <- sample(nrow(src_sf), size = floor(val_prop * nrow(src_sf)))
final_train <- setdiff(seq_len(nrow(src_sf)), final_val)

# Bandwidth chosen on a third source split, never on the target.
BANDWIDTHS <- list(
  list(kernel = "uniform", mult = 1),
  list(kernel = "gaussian", mult = 2),
  list(kernel = "gaussian", mult = 1),
  list(kernel = "gaussian", mult = 0.5),
  list(kernel = "gaussian", mult = 0.25),
  list(kernel = "gaussian", mult = 0.1),
  list(kernel = "gaussian", mult = 0.02)
)
k_label <- function(k) {
  if (k$kernel == "uniform") "uniform" else sprintf("gauss_%g", k$mult)
}

set.seed(0)
.p <- sample(nrow(src_sf))
.n <- nrow(src_sf)
tune_train <- .p[seq_len(floor(0.6 * .n))]
tune_stop <- .p[(floor(0.6 * .n) + 1):floor(0.8 * .n)]
tune_select <- .p[(floor(0.8 * .n) + 1):.n]

fit_tuned <- function(seed, arm) {
  ys <- y_of("src")[tune_select]
  sel <- function(p) {
    e <- p(tune_select, "src")
    if (MODE == "clf") {
      mean((e > 0.5) == ys)
    } else {
      1 - sum((ys - e)^2) / sum((ys - mean(ys))^2)
    }
  }
  best <- NULL
  bs <- -Inf
  for (k in BANDWIDTHS) {
    p <- if (grepl("GraphSAGE", arm)) {
      torch_manual_seed(seed)
      fit_sage(tune_train, tune_stop, norm_node, 0.1, 1e-4, k)
    } else {
      fit_xgb(tune_train, TRUE, seed, k)
    }
    s <- sel(p)
    if (is.finite(s) && s > bs) {
      bs <- s
      best <- k
    }
  }
  p <- if (grepl("GraphSAGE", arm)) {
    torch_manual_seed(seed)
    fit_sage(c(tune_train, tune_stop), tune_select, norm_node, 0.1, 1e-4, best)
  } else {
    fit_xgb(c(tune_train, tune_stop), TRUE, seed, best)
  }
  list(predictor = p, chosen = k_label(best))
}

task <- function(spec) {
  k <- if (is.null(spec$k)) default_k else spec$k
  if (isTRUE(spec$tuned)) {
    r <- fit_tuned(spec$seed, spec$arm)
    p <- r$predictor
    chosen <- r$chosen
  } else {
    p <- arms[[spec$arm]](final_train, final_val, spec$seed, k)
    chosen <- k_label(k)
  }
  cbind(
    src = SRC,
    tgt = TGT,
    arm = spec$arm,
    seed = spec$seed,
    graph = GRAPH,
    chosen = chosen,
    score(y_of("tgt"), p(seq_len(nrow(tgt_sf)), "tgt"))
  )
}

start_daemons <- function(n = max(1L, parallel::detectCores() - 2L)) {
  core <- normalizePath("R/core.R", mustWork = TRUE)
  daemons(n)
  everywhere(
    {
      options(
        gnn_data = a,
        gnn_src = b,
        gnn_tgt = cc,
        gnn_crs = d,
        gnn_graph = e,
        gnn_cell = f,
        gnn_mode = g
      )
      source(path, local = FALSE)
      torch_set_num_threads(1L)
    },
    .args = list(
      path = core,
      a = DATA,
      b = SRC,
      cc = TGT,
      d = CRS,
      e = GRAPH,
      f = CELL,
      g = MODE
    ),
    .min = n
  )
  invisible(n)
}

# Neighbourhood size as a trained hyperparameter.
#
# TWO PROBLEMS THIS FIXES.
#
# 1. THE COMPARISON WAS UNFAIR TO THE GNN. XGBoost tunes its own complexity
#    by early stopping on a validation split. GraphSAGE was handed a fixed
#    neighbourhood (KNN-30, uniform weights) chosen by hand and never tuned.
#    Every head-to-head in this project compared a tuned tabular model against
#    an untuned graph model.
#
# 2. THE KERNEL SCREENS WERE ORACLE-TUNED. They swept bandwidths and reported
#    the best score on the TARGET region. That uses the labels you are trying
#    to predict, so it is not a number anyone could obtain in deployment. It
#    was reported as a sensitivity check but reads as a result.
#
# THE FIX. Treat the neighbourhood bandwidth exactly like any other
# hyperparameter: choose it inside the source region, then apply the single
# chosen value to the target, once.
#
# The source region is split three ways rather than two:
#   train  (60%) - gradient steps
#   stop   (20%) - early stopping, as before
#   select (20%) - bandwidth choice, held out from both of the above
#
# Using the early-stopping split to also choose the bandwidth would be mildly
# optimistic, since that split already influenced when training halted. A
# third split costs a little data and removes the objection.
#
# Nothing here touches target labels at any point.

source("R/pair-core.R")

BANDWIDTHS <- list(
  list(kernel = "uniform",  threshold_mult = 1),
  list(kernel = "gaussian", threshold_mult = 2),
  list(kernel = "gaussian", threshold_mult = 1),
  list(kernel = "gaussian", threshold_mult = 0.5),
  list(kernel = "gaussian", threshold_mult = 0.25),
  list(kernel = "gaussian", threshold_mult = 0.1),
  list(kernel = "gaussian", threshold_mult = 0.02)
)

bw_label <- function(k) if (k$kernel == "uniform") "uniform" else
  sprintf("gauss_%g", k$threshold_mult)

# Three-way split of the source region, fixed across arms and seeds so every
# arm sees identical data.
set.seed(0)
.perm <- sample(nrow(src_sf))
.n <- nrow(src_sf)
tune_train  <- .perm[seq_len(floor(0.60 * .n))]
tune_stop   <- .perm[(floor(0.60 * .n) + 1):floor(0.80 * .n)]
tune_select <- .perm[(floor(0.80 * .n) + 1):.n]

rsq_trad_vec <- function(truth, est) 1 - sum((truth - est)^2) / sum((truth - mean(truth))^2)

# Fit at every candidate bandwidth, score each on the held-out SELECT split of
# the source region, keep the best, and return its predictor unchanged.
fit_sage_tuned <- function(seed, norm, dropout = 0, wd = 0) {
  y_sel <- y_of("src")[tune_select]
  best <- NULL; best_score <- -Inf; trace <- list()

  for (kspec in BANDWIDTHS) {
    torch_manual_seed(seed)
    pred <- fit_sage(tune_train, tune_stop, norm = norm, dropout = dropout,
                     wd = wd, kspec = kspec)
    s <- rsq_trad_vec(y_sel, pred(tune_select, "src"))
    trace[[bw_label(kspec)]] <- s
    if (is.finite(s) && s > best_score) { best_score <- s; best <- kspec }
  }
  list(predictor = fit_sage(c(tune_train, tune_stop), tune_select, norm = norm,
                            dropout = dropout, wd = wd, kspec = best),
       chosen = bw_label(best), select_score = best_score, trace = trace)
}

# XGBoost + lags gets the same treatment, so the tabular graph-using arm is
# tuned on equal terms rather than left at the default.
fit_xgb_tuned <- function(seed, with_lags = TRUE) {
  y_sel <- y_of("src")[tune_select]
  best <- NULL; best_score <- -Inf
  for (kspec in BANDWIDTHS) {
    pred <- fit_xgb(tune_train, with_lags = with_lags, seed = seed, kspec = kspec)
    s <- rsq_trad_vec(y_sel, pred(tune_select, "src"))
    if (is.finite(s) && s > best_score) { best_score <- s; best <- kspec }
  }
  list(predictor = fit_xgb(c(tune_train, tune_stop), with_lags = with_lags,
                           seed = seed, kspec = best),
       chosen = bw_label(best), select_score = best_score)
}

tuned_task <- function(spec) {
  if (spec$arm == "GraphSAGE + LayerNorm (tuned)") {
    r <- fit_sage_tuned(spec$seed, norm = layer_layer_norm_node, dropout = 0.1, wd = 1e-4)
  } else if (spec$arm == "GraphSAGE (tuned)") {
    r <- fit_sage_tuned(spec$seed, norm = NULL)
  } else if (spec$arm == "XGBoost + lags (tuned)") {
    r <- fit_xgb_tuned(spec$seed)
  } else stop("unknown arm: ", spec$arm)

  cbind(src = SRC, tgt = TGT, arm = spec$arm, seed = spec$seed,
        chosen = r$chosen, select_score = r$select_score,
        score(y_of("tgt"), r$predictor(seq_len(nrow(tgt_sf)), "tgt")))
}

start_daemons_tuned <- function(n_daemons = max(1L, parallel::detectCores() - 2L)) {
  core <- normalizePath("R/tuned-core.R", mustWork = TRUE)
  daemons(n_daemons)
  everywhere(
    {
      options(pair_data = pd, pair_src = ps, pair_tgt = pt)
      source(core_path, local = FALSE)
      torch_set_num_threads(1L)
    },
    .args = list(core_path = core, pd = PAIR_DATA, ps = SRC, pt = TGT),
    .min = n_daemons
  )
  cat(sprintf("%d daemons up (%s -> %s, tuned neighbourhood)\n", n_daemons, SRC, TGT))
  invisible(n_daemons)
}

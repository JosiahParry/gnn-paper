# Train in King County, predict Ames.
#
# Two evaluations of the same five arms. King County cross-validation holds out
# a quarter of the county at a time and gives it its own subgraph, so it
# measures prediction into unseen parts of a region the model was fit on. The
# Ames transfer fits on all of King County and predicts a different housing
# market entirely -- different price level, different covariate distribution,
# a graph that shares no edges with the training one.
#
# Every fit is independent of every other, so they are fanned out across a
# daemon pool. Set the pool size with the first argument:
#   R -f R/kc-to-ames.R
#   Rscript R/kc-to-ames.R 10
#
# Results shift slightly against a serial run. torch's CPU sparse operations
# are not deterministic across processes, which matters for a quoted per-state
# number but not for a distribution over initialisations, which is what the
# transfer reports.

source("R/kc-to-ames-core.R")

cli_args <- commandArgs(trailingOnly = TRUE)
n_daemons <- if (length(cli_args) >= 1) {
  as.integer(cli_args[1])
} else {
  max(1L, parallel::detectCores() - 2L)
}

t0 <- Sys.time()
start_daemons(n_daemons)

# King County cross-validation -------------------------------------------
#
# Averaging over four folds already damps initialisation noise here, so one
# seed per fold is enough. The transfer below has no folds to average over,
# which is why it needs the seed sweep.

kc_jobs <- unlist(
  lapply(seq_len(k_folds), function(i) {
    lapply(names(arms), function(a) list(fold = i, arm = a, seed = seeds[i]))
  }),
  recursive = FALSE
)

cat(sprintf("\nKing County cross-validation: %d fits\n", length(kc_jobs)))
kc_folds <- do.call(rbind, mirai_map(kc_jobs, kc_fold_task)[.progress, .stop])

kc_summary <- kc_folds |>
  group_by(arm) |>
  summarise(
    across(c(mae, rmse, rsq, rsq_trad, bias, cal_slope), mean),
    .groups = "drop"
  ) |>
  as.data.frame()

cat(sprintf("\n=== King County, mean across %d folds ===\n", k_folds))
print(kc_summary, row.names = FALSE, digits = 3)

# Transfer to Ames -------------------------------------------------------
#
# All of Ames is scored at once for each fit. The spread reported here is
# across initialisations of the same model on the same data, not across
# partitions of the target region.

ames_jobs <- unlist(
  lapply(names(arms), function(a) {
    use <- if (a %in% stochastic) seeds else seeds[1]
    lapply(use, function(s) list(arm = a, seed = s))
  }),
  recursive = FALSE
)

cat(sprintf("\nAmes transfer: %d fits\n", length(ames_jobs)))
ames_seeds <- do.call(rbind, mirai_map(ames_jobs, ames_task)[.progress, .stop])

daemons(0)

ames_summary <- ames_seeds |>
  group_by(arm) |>
  summarise(
    n_fits = dplyr::n(),
    mae = mean(mae),
    rmse = mean(rmse),
    rsq_sd = stats::sd(rsq),
    rsq_min = min(rsq),
    rsq_max = max(rsq),
    rsq = mean(rsq),
    rsq_trad_sd = stats::sd(rsq_trad),
    rsq_trad_min = min(rsq_trad),
    rsq_trad_max = max(rsq_trad),
    rsq_trad = mean(rsq_trad),
    bias = mean(bias),
    cal_slope = mean(cal_slope),
    .groups = "drop"
  ) |>
  as.data.frame()

cat("\n\n=== Ames transfer, across initialisations ===\n")
cat("rsq is squared Pearson correlation (invariant to affine rescaling).\n")
cat("rsq_trad is 1 - SSE/SST (penalizes miscalibration; see cal_slope).\n")
print(
  ames_summary[order(-ames_summary$rsq_trad), ],
  row.names = FALSE, digits = 3
)

cat("\n=== King County vs Ames ===\n")
print(
  merge(
    kc_summary[, c("arm", "mae", "rsq", "rsq_trad")],
    ames_summary[, c("arm", "mae", "rsq", "rsq_trad", "rsq_trad_sd", "bias", "cal_slope")],
    by = "arm",
    suffixes = c("_kc", "_ames")
  ),
  row.names = FALSE,
  digits = 3
)

cat(sprintf(
  "\n%d fits in %.1f min on %d daemons\n",
  length(kc_jobs) + length(ames_jobs),
  as.numeric(difftime(Sys.time(), t0, units = "mins")),
  n_daemons
))

dir.create("data", showWarnings = FALSE)
saveRDS(
  list(
    kc_folds = kc_folds,
    kc_summary = kc_summary,
    ames_seeds = ames_seeds,
    ames_summary = ames_summary
  ),
  "data/kc-to-ames-results.rds"
)

cat("\nSaved to data/kc-to-ames-results.rds\n")

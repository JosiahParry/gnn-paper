# Controlled domain-shift transfer sweep. See R/domain-shift-sim-core.R for
# the design: two disjoint synthetic regions, same coefficient function,
# target's covariates shifted by `shift` source-sds, source-trained models
# evaluated on the whole (never-seen) target region.
#
# Run from the repo root:
#   Rscript R/domain-shift-sim.R [n_daemons]

source("R/domain-shift-sim-core.R")

cli_args <- commandArgs(trailingOnly = TRUE)
n_daemons <- if (length(cli_args) >= 1) as.integer(cli_args[1]) else 10L

arms <- c("OLS", "XGBoost", "XGBoost + lags", "GraphSAGE", "GraphSAGE + LayerNorm")
stochastic <- c("XGBoost", "XGBoost + lags", "GraphSAGE", "GraphSAGE + LayerNorm")

# Three levels deep (shift -> arm -> seed -> spec), so unlist(recursive=FALSE)
# has to run twice to fully flatten to one spec per element.
jobs <- unlist(unlist(lapply(shift_levels, function(s) {
  lapply(arms, function(a) {
    seeds <- if (a %in% stochastic) seq_len(n_seeds) else 1L
    lapply(seeds, function(sd) list(shift = s, arm = a, seed = sd))
  })
}), recursive = FALSE), recursive = FALSE)

cat(sprintf(
  "\nDomain-shift sweep: %d fits (%d shift levels x arms, %d seeds for stochastic arms)\n",
  length(jobs), length(shift_levels), n_seeds
))
cat(sprintf("source n=%d (%dx%d), target n=%d (%dx%d)\n",
  src_cols * src_rows, src_cols, src_rows, tgt_cols * tgt_rows, tgt_cols, tgt_rows))

daemons(n_daemons)
core_path <- normalizePath("R/domain-shift-sim-core.R", mustWork = TRUE)
everywhere({ source(core_path, local = FALSE) }, .args = list(core_path = core_path), .min = n_daemons)
cat(sprintf("%d daemons up\n", n_daemons))

t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, domain_shift_task)[.progress, .stop])
daemons(0)

cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

# var(target truth) is constant within a shift level (deterministic DGP draw),
# so recover it exactly from any one row's rmse/rsq_trad pair rather than
# averaging the already-nonlinear per-fit rsq_trad.
var_by_shift <- res |>
  group_by(shift) |>
  summarise(var_truth = (rmse[1]^2) / (1 - rsq_trad[1]), .groups = "drop")

summary <- res |>
  left_join(var_by_shift, by = "shift") |>
  group_by(shift, arm) |>
  summarise(
    mae = mean(mae), rmse = mean(rmse),
    rsq = mean(rsq),
    rsq_trad = 1 - mean(rmse)^2 / mean(var_truth),
    bias = mean(bias), cal_slope = mean(cal_slope),
    n_fits = dplyr::n(),
    .groups = "drop"
  ) |>
  as.data.frame()

cat("\n=== Domain-shift transfer, by shift level (source sds) and arm ===\n")
print(summary[order(summary$shift, -summary$rsq_trad), ], row.names = FALSE, digits = 3)

dir.create("data", showWarnings = FALSE)
saveRDS(list(folds = res, summary = summary), "data/domain-shift-sim-results.rds")
cat("\nSaved to data/domain-shift-sim-results.rds\n")

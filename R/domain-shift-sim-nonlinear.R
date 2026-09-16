# Nonlinear-DGP version of the domain-shift sweep. The original (linear)
# experiment handed OLS a by-construction advantage under a pure location
# shift -- it's exactly OLS's home turf. This adds R/sim-core.R's Scenario B
# nonlinear term (2*sin(x1) + x2^2 - 1.5*x3*x4) on top of the same linear
# part, so OLS is misspecified here too and the GraphSAGE-vs-XGBoost
# comparison isn't confounded by which arm happens to match the DGP's form.
#
# Run from the repo root:
#   Rscript R/domain-shift-sim-nonlinear.R [n_daemons]

source("R/domain-shift-sim-core.R")

cli_args <- commandArgs(trailingOnly = TRUE)
n_daemons <- if (length(cli_args) >= 1) as.integer(cli_args[1]) else 10L

arms <- c("OLS", "XGBoost", "XGBoost + lags", "GraphSAGE", "GraphSAGE + LayerNorm")
stochastic <- c("XGBoost", "XGBoost + lags", "GraphSAGE", "GraphSAGE + LayerNorm")

jobs <- unlist(unlist(lapply(shift_levels, function(s) {
  lapply(arms, function(a) {
    seeds <- if (a %in% stochastic) seq_len(n_seeds) else 1L
    lapply(seeds, function(sd) list(shift = s, arm = a, seed = sd, dgp = "nonlinear"))
  })
}), recursive = FALSE), recursive = FALSE)

cat(sprintf(
  "\nNonlinear domain-shift sweep: %d fits (%d shift levels x arms, %d seeds for stochastic arms)\n",
  length(jobs), length(shift_levels), n_seeds
))

daemons(n_daemons)
core_path <- normalizePath("R/domain-shift-sim-core.R", mustWork = TRUE)
everywhere({ source(core_path, local = FALSE) }, .args = list(core_path = core_path), .min = n_daemons)
cat(sprintf("%d daemons up\n", n_daemons))

t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, domain_shift_task)[.progress, .stop])
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

var_by_shift <- res |>
  group_by(shift) |>
  summarise(var_truth = (rmse[1]^2) / (1 - rsq_trad[1]), .groups = "drop")

summary <- res |>
  left_join(var_by_shift, by = "shift") |>
  group_by(shift, arm) |>
  summarise(
    mae = mean(mae), rmse = mean(rmse), rsq = mean(rsq),
    rsq_trad = 1 - mean(rmse)^2 / mean(var_truth),
    bias = mean(bias), cal_slope = mean(cal_slope), n_fits = dplyr::n(),
    .groups = "drop"
  ) |>
  as.data.frame()

cat("\n=== Nonlinear domain-shift transfer, by shift level and arm ===\n")
print(summary[order(summary$shift, -summary$rsq_trad), ], row.names = FALSE, digits = 3)

saveRDS(list(folds = res, summary = summary), "data/domain-shift-sim-nonlinear-results.rds")
cat("\nSaved to data/domain-shift-sim-nonlinear-results.rds\n")

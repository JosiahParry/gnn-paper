# Screen kernel shape x threshold_mult for Broward -> San Diego, using
# sfdep::st_kernel_weights() properly (see R/airbnb-core-brow-sd.R). Fast
# screen: 5 seeds, GraphSAGE + LayerNorm only. Confirm the winner at full
# seed count afterward.

source("R/airbnb-core-brow-sd.R")

cli_args <- commandArgs(trailingOnly = TRUE)
n_daemons <- if (length(cli_args) >= 1) as.integer(cli_args[1]) else 14L

seeds <- 1001:1005
kernels <- c("gaussian", "triangular", "epanechnikov", "quartic")
thresholds <- c(0.02, 0.05, 0.1, 0.25, 0.5, 1)

start_daemons(n_daemons)

jobs <- unlist(unlist(lapply(kernels, function(k) {
  lapply(thresholds, function(t) {
    lapply(seeds, function(s) {
      list(arm = "GraphSAGE + LayerNorm", seed = s,
           kspec = list(kernel = k, threshold_mult = t, adaptive = FALSE))
    })
  })
}), recursive = FALSE), recursive = FALSE)

cat(sprintf("\nKernel screen: %d fits (%d kernels x %d thresholds x %d seeds)\n",
            length(jobs), length(kernels), length(thresholds), length(seeds)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, sd_task)[.progress, .stop])
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

summary <- res |>
  group_by(kernel, threshold_mult) |>
  summarise(
    mae = mean(mae), rmse = mean(rmse),
    rsq_trad = mean(rsq_trad), bias = mean(bias), cal_slope = mean(cal_slope),
    .groups = "drop"
  ) |>
  as.data.frame()

cat("\n=== Kernel x threshold screen (5 seeds), GraphSAGE + LayerNorm ===\n")
cat("reference: XGBoost=0.650, OLS=0.648, old best (manual gaussian 0.1x median)=0.672\n")
print(summary[order(-summary$rsq_trad), ], row.names = FALSE, digits = 3)

saveRDS(list(folds = res, summary = summary), "data/airbnb-brow-sd-kernel-screen.rds")
cat("\nSaved to data/airbnb-brow-sd-kernel-screen.rds\n")

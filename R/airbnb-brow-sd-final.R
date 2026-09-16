# Final confirmation: gaussian kernel, threshold_mult=0.02 (the found
# optimum), full 15 seeds, both GraphSAGE arms plus reference OLS/XGBoost.
source("R/airbnb-core-brow-sd.R")

cli_args <- commandArgs(trailingOnly = TRUE)
n_daemons <- if (length(cli_args) >= 1) as.integer(cli_args[1]) else 14L

seeds <- 1001:1015
best_kspec <- list(kernel = "gaussian", threshold_mult = 0.02, adaptive = FALSE)

start_daemons(n_daemons)

jobs <- c(
  lapply(seeds, function(s) list(arm = "GraphSAGE", seed = s, kspec = best_kspec)),
  lapply(seeds, function(s) list(arm = "GraphSAGE + LayerNorm", seed = s, kspec = best_kspec)),
  lapply(seeds, function(s) list(arm = "XGBoost + lags", seed = s, kspec = best_kspec)),
  list(list(arm = "OLS", seed = seeds[1], kspec = default_kspec)),
  list(list(arm = "XGBoost", seed = seeds[1], kspec = default_kspec))
)

cat(sprintf("\nFinal confirmation: %d fits\n", length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, sd_task)[.progress, .stop])
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

summary <- res |>
  group_by(arm, threshold_mult) |>
  summarise(
    n_fits = dplyr::n(),
    mae = mean(mae), rmse = mean(rmse), rsq = mean(rsq),
    rsq_trad_sd = sd(rsq_trad), rsq_trad = mean(rsq_trad),
    bias = mean(bias), cal_slope = mean(cal_slope), .groups = "drop"
  ) |>
  as.data.frame()

cat("\n=== Final: Broward -> San Diego, gaussian kernel @ 0.02x median ===\n")
print(summary[order(-summary$rsq_trad), ], row.names = FALSE, digits = 3)

saveRDS(list(folds = res, summary = summary), "data/airbnb-brow-sd-final.rds")
cat("\nSaved to data/airbnb-brow-sd-final.rds\n")

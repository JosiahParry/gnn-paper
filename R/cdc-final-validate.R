# Full 15-seed confirmation of CDC PLACES CA -> WV best: gaussian kernel @
# 4x median distance (0.798 at 5 seeds, vs uniform-edge baseline 0.741).
source("R/cdc-places-core.R")

cli_args <- commandArgs(trailingOnly = TRUE)
n_daemons <- if (length(cli_args) >= 1) as.integer(cli_args[1]) else 14L

seeds <- 1001:1015
best_kspec <- list(kernel = "gaussian", threshold_mult = 4)

start_daemons(n_daemons)

jobs <- c(
  lapply(seeds, function(s) list(arm = "GraphSAGE", seed = s, kspec = best_kspec)),
  lapply(seeds, function(s) list(arm = "GraphSAGE + LayerNorm", seed = s, kspec = best_kspec)),
  lapply(seeds, function(s) list(arm = "XGBoost + lags", seed = s, kspec = best_kspec)),
  list(list(arm = "OLS", seed = seeds[1], kspec = list(kernel = "uniform", threshold_mult = 1))),
  list(list(arm = "XGBoost", seed = seeds[1], kspec = list(kernel = "uniform", threshold_mult = 1)))
)

cat(sprintf("\nCDC final confirmation: %d fits\n", length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, wv_task)[.progress, .stop])
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

summary <- res |>
  group_by(arm, threshold_mult) |>
  summarise(n_fits = dplyr::n(), mae = mean(mae), rmse = mean(rmse),
            rsq_trad_sd = sd(rsq_trad), rsq_trad = mean(rsq_trad),
            bias = mean(bias), cal_slope = mean(cal_slope), .groups = "drop") |>
  as.data.frame()

cat("\n=== CDC PLACES final: gaussian kernel @ 4x median, 15 seeds ===\n")
cat("reference: uniform-edge GraphSAGE+LayerNorm rsq_trad=0.741; XGBoost+lags(uniform)=0.801\n")
print(summary[order(-summary$rsq_trad), ], row.names = FALSE, digits = 3)

saveRDS(list(folds = res, summary = summary), "data/cdc-places-final.rds")
cat("\nSaved to data/cdc-places-final.rds\n")

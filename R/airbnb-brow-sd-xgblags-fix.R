# Retest XGBoost + lags at the winning kernel (gaussian, 0.02x median) now
# that lag.listw() actually sees the tuned kernel weights instead of
# nb2listw's silent unweighted default. Full 15 seeds.
source("R/airbnb-core-brow-sd.R")

cli_args <- commandArgs(trailingOnly = TRUE)
n_daemons <- if (length(cli_args) >= 1) as.integer(cli_args[1]) else 4L

seeds <- 1001:1015
best_kspec <- list(kernel = "gaussian", threshold_mult = 0.02, adaptive = FALSE)

start_daemons(n_daemons)

jobs <- lapply(seeds, function(s) list(arm = "XGBoost + lags", seed = s, kspec = best_kspec))

cat(sprintf("\nXGBoost+lags weighting fix retest: %d fits\n", length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, sd_task)[.progress, .stop])
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

cat("\n=== XGBoost + lags, gaussian kernel @ 0.02x, weighting bug fixed ===\n")
cat("old (buggy, unweighted lags): rsq_trad=0.627\n")
cat(sprintf(
  "new: mae=%.3f rmse=%.3f rsq_trad=%.3f bias=%.3f cal_slope=%.3f\n",
  mean(res$mae), mean(res$rmse), mean(res$rsq_trad), mean(res$bias), mean(res$cal_slope)
))

saveRDS(res, "data/airbnb-brow-sd-xgblags-fix.rds")
cat("\nSaved to data/airbnb-brow-sd-xgblags-fix.rds\n")

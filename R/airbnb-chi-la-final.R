source("R/airbnb-core-chi-la.R")
cli_args <- commandArgs(trailingOnly = TRUE)
n_daemons <- if (length(cli_args) >= 1) as.integer(cli_args[1]) else 14L
seeds <- 1001:1015
best_kspec <- list(kernel = "gaussian", threshold_mult = 2.0, adaptive = FALSE)
start_daemons(n_daemons)
jobs <- lapply(seeds, function(s) list(arm = "GraphSAGE + LayerNorm", seed = s, kspec = best_kspec))
cat(sprintf("\nFinal confirmation: %d fits\n", length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, la_task)[.progress, .stop])
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))
cat(sprintf(
  "\n=== GraphSAGE+LayerNorm, gaussian @ 2.0x, 15 seeds ===\nmae=%.3f rmse=%.3f rsq_trad_sd=%.3f rsq_trad=%.3f bias=%.3f cal_slope=%.3f\n",
  mean(res$mae), mean(res$rmse), sd(res$rsq_trad), mean(res$rsq_trad), mean(res$bias), mean(res$cal_slope)
))
cat("reference: OLS=0.632, XGBoost=0.626, uniform-edge GraphSAGE+LayerNorm=0.621\n")
saveRDS(res, "data/airbnb-chi-la-final.rds")
cat("\nSaved to data/airbnb-chi-la-final.rds\n")

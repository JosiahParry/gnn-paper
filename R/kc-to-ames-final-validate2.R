# Full 30-seed validation of the new KC -> Ames best: dropout=0.1 +
# wd=1e-4 + gaussian kernel @ 0.75x median distance (rsq_trad=0.490 at 15
# seeds, up from 0.467 without the kernel). Single deployable models only.
# final2_best_v / final2_task live in the helpers file so daemons see them.

source("R/kc-to-ames-core.R")
source("R/kc-to-ames-arch-sweep2-helpers.R")

final_seeds <- 1001:1030

start_daemons(14)
everywhere(source("R/kc-to-ames-arch-sweep2-helpers.R", local = FALSE))

jobs <- lapply(final_seeds, function(s) list(seed = s))

cat(sprintf("\nFinal validation (kernel @ 0.75x): %d fits\n", length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, final2_task)[.progress, .stop])
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

var_ames <- var(ames$y)
res$rsq_trad <- 1 - res$rmse^2 / var_ames

cat("\n=== KC -> Ames, dropout=0.1 + wd=1e-4 + gaussian kernel @ 0.75x, 30 seeds ===\n")
cat(sprintf(
  "mean-of-metrics: mae=%.3f rmse=%.3f rsq_trad_sd=%.3f rsq_trad=%.3f bias=%.3f cal_slope=%.3f\n",
  mean(res$mae), mean(res$rmse), sd(res$rsq_trad), mean(res$rsq_trad), mean(res$bias), mean(res$cal_slope)
))
cat(sprintf(
  "aggregate (1 - mean(rmse)^2/var): rsq_trad=%.3f\n",
  1 - mean(res$rmse)^2 / var_ames
))
cat("\nreference: dropout=0.1 alone (no kernel), 30 seeds: rsq_trad=0.380\n")
cat("reference: XGBoost + lags, 30 seeds: rsq_trad=0.283\n")

saveRDS(res, "data/kc-to-ames-final-validation2.rds")
cat("\nSaved to data/kc-to-ames-final-validation2.rds\n")

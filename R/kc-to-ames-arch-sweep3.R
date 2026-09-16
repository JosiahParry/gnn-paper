# Wave 3: combine the wave-2 winners (dropout=0.2, lag_input, gaussian
# kernel) and probe a couple more dropout rates.
#
# Reference points:
#   node_norm (wave 1):                     rsq_trad = 0.247
#   dropout=0.2 (wave 2):                    rsq_trad = 0.354  cal_slope=1.048
#   lag_input (wave 2):                      rsq_trad = 0.291  cal_slope=1.010
#   gaussian_kernel (wave 2):                rsq_trad = 0.282  cal_slope=0.839
#   XGBoost + lags (the number to beat):     rsq_trad = 0.283

source("R/kc-to-ames-core.R")
source("R/kc-to-ames-arch-sweep2-helpers.R")

sweep_seeds <- 1001:1015

start_daemons(10)
everywhere(source("R/kc-to-ames-arch-sweep2-helpers.R", local = FALSE))

jobs <- unlist(
  lapply(names(variants3), function(v) lapply(sweep_seeds, function(s) list(variant = v, seed = s))),
  recursive = FALSE
)

cat(sprintf("\nArchitecture sweep 3: %d fits (%d variants x %d seeds)\n", length(jobs), length(variants3), length(sweep_seeds)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, sweep3_task)[.progress, .stop])
daemons(0)

var_ames <- var(ames$y)
res$rsq_trad <- 1 - res$rmse^2 / var_ames

summary <- res |>
  group_by(variant) |>
  summarise(
    mae = mean(mae), rmse = mean(rmse),
    rsq = mean(rsq),
    rsq_trad = 1 - mean(rmse)^2 / var_ames,
    bias = mean(bias), cal_slope = mean(cal_slope),
    .groups = "drop"
  ) |>
  as.data.frame()

cat("\n=== Architecture sweep 3, Ames transfer, mean across 15 seeds ===\n")
cat("dropout=0.2 alone (wave 2): rsq_trad=0.354 cal_slope=1.048\n")
cat("XGBoost + lags (target to beat): rsq_trad=0.283\n")
print(summary[order(-summary$rsq_trad), ], row.names = FALSE, digits = 3)

cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

saveRDS(res, "data/kc-to-ames-arch-sweep3.rds")
cat("\nSaved to data/kc-to-ames-arch-sweep3.rds\n")

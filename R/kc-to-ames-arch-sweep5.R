# Wave 5: attack the covariate-shift mechanism directly (region-relative
# percentile scaling instead of z-scoring against King County alone), plus
# one more regularization combo (dropout=0.1 + weight_decay).
#
# Reference: current best, z-scored + dropout=0.1: rsq_trad = 0.402 (15 seeds)

source("R/kc-to-ames-core.R")
source("R/kc-to-ames-arch-sweep2-helpers.R")

sweep_seeds <- 1001:1015
start_daemons(10)
everywhere(source("R/kc-to-ames-arch-sweep2-helpers.R", local = FALSE))

jobs <- unlist(
  lapply(names(variants5), function(v) lapply(sweep_seeds, function(s) list(variant = v, seed = s))),
  recursive = FALSE
)
cat(sprintf("\nArchitecture sweep 5: %d fits\n", length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, sweep5_task)[.progress, .stop])
daemons(0)

var_ames <- var(ames$y)
res$rsq_trad <- 1 - res$rmse^2 / var_ames
summary <- res |>
  group_by(variant) |>
  summarise(
    mae = mean(mae), rmse = mean(rmse), rsq = mean(rsq),
    rsq_trad = 1 - mean(rmse)^2 / var_ames,
    bias = mean(bias), cal_slope = mean(cal_slope), .groups = "drop"
  ) |> as.data.frame()

cat("\n=== Architecture sweep 5 (percentile scaling) ===\n")
cat("reference: z-scored + dropout=0.1 (wave 3): rsq_trad=0.402 cal_slope=0.994\n")
print(summary[order(-summary$rsq_trad), ], row.names = FALSE, digits = 3)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))
saveRDS(res, "data/kc-to-ames-arch-sweep5.rds")
cat("\nSaved to data/kc-to-ames-arch-sweep5.rds\n")

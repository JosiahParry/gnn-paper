# Wave 6: refine around dropout=0.1 + weight_decay=1e-4 (wave 5's best:
# rsq_trad=0.467).
source("R/kc-to-ames-core.R")
source("R/kc-to-ames-arch-sweep2-helpers.R")

sweep_seeds <- 1001:1015
start_daemons(10)
everywhere(source("R/kc-to-ames-arch-sweep2-helpers.R", local = FALSE))

jobs <- unlist(
  lapply(names(variants6), function(v) lapply(sweep_seeds, function(s) list(variant = v, seed = s))),
  recursive = FALSE
)
cat(sprintf("\nArchitecture sweep 6: %d fits\n", length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, sweep6_task)[.progress, .stop])
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

cat("\n=== Architecture sweep 6 ===\n")
cat("wave5 best: dropout=0.1 + wd=1e-4  rsq_trad=0.467 cal_slope=0.970\n")
print(summary[order(-summary$rsq_trad), ], row.names = FALSE, digits = 3)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))
saveRDS(res, "data/kc-to-ames-arch-sweep6.rds")
cat("\nSaved to data/kc-to-ames-arch-sweep6.rds\n")

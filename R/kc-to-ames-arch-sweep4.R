# Wave 4: refine around dropout=0.1 (wave 3's best: rsq_trad=0.402,
# cal_slope=0.994) and dropout=0.2+gaussian (0.396, cal_slope=1.163).

source("R/kc-to-ames-core.R")
source("R/kc-to-ames-arch-sweep2-helpers.R")

sweep_seeds <- 1001:1015
start_daemons(10)
everywhere(source("R/kc-to-ames-arch-sweep2-helpers.R", local = FALSE))

jobs <- unlist(
  lapply(names(variants4), function(v) lapply(sweep_seeds, function(s) list(variant = v, seed = s))),
  recursive = FALSE
)
cat(sprintf("\nArchitecture sweep 4: %d fits\n", length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, sweep4_task)[.progress, .stop])
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

cat("\n=== Architecture sweep 4 ===\n")
cat("wave3 best: dropout_01 rsq_trad=0.402 cal_slope=0.994\n")
print(summary[order(-summary$rsq_trad), ], row.names = FALSE, digits = 3)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))
saveRDS(res, "data/kc-to-ames-arch-sweep4.rds")
cat("\nSaved to data/kc-to-ames-arch-sweep4.rds\n")

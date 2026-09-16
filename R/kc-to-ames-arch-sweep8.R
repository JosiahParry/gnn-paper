# Wave 8: refine kernel bandwidth for KC -> Ames, looser range (0.5-2.0x
# median), since wave 7 found looser beats tighter here (opposite of the
# Broward -> San Diego Airbnb pair).

source("R/kc-to-ames-core.R")
source("R/kc-to-ames-arch-sweep2-helpers.R")

sweep_seeds <- 1001:1015

start_daemons(14)
everywhere(source("R/kc-to-ames-arch-sweep2-helpers.R", local = FALSE))

jobs <- unlist(
  lapply(names(variants8), function(v) lapply(sweep_seeds, function(s) list(variant = v, seed = s))),
  recursive = FALSE
)
cat(sprintf("\nArchitecture sweep 8 (looser kernel on KC -> Ames): %d fits\n", length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, sweep8_task)[.progress, .stop])
daemons(0)

var_ames <- var(ames$y)
res$rsq_trad <- 1 - res$rmse^2 / var_ames
summary <- res |>
  group_by(variant) |>
  summarise(
    mae = mean(mae), rmse = mean(rmse), rsq = mean(rsq),
    rsq_trad_sd = sd(rsq_trad),
    rsq_trad = 1 - mean(rmse)^2 / var_ames,
    bias = mean(bias), cal_slope = mean(cal_slope), .groups = "drop"
  ) |> as.data.frame()

cat("\n=== Architecture sweep 8: looser kernel range ===\n")
cat("wave 7 best: kernel_05 (0.5x) rsq_trad=0.488\n")
print(summary[order(-summary$rsq_trad), ], row.names = FALSE, digits = 3)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))
saveRDS(res, "data/kc-to-ames-arch-sweep8.rds")
cat("\nSaved to data/kc-to-ames-arch-sweep8.rds\n")

# Wave 2 of the KC -> Ames architecture ablation. See
# R/kc-to-ames-arch-sweep2-helpers.R for what each variant changes.
#
# Reference points already measured (wave 1, seeds 1001:1015):
#   baseline (mode=graph, concat=TRUE):        rsq_trad = 0.199
#   node_norm (mode=node, concat=TRUE):         rsq_trad = 0.247  <- new floor
#   no_concat / node_no_concat:                 much worse, ruled out

source("R/kc-to-ames-core.R")
source("R/kc-to-ames-arch-sweep2-helpers.R")

sweep_seeds <- 1001:1015

start_daemons(10)
everywhere(source("R/kc-to-ames-arch-sweep2-helpers.R", local = FALSE))

jobs <- unlist(
  lapply(names(variants2), function(v) lapply(sweep_seeds, function(s) list(variant = v, seed = s))),
  recursive = FALSE
)

cat(sprintf("\nArchitecture sweep 2: %d fits (%d variants x %d seeds)\n", length(jobs), length(variants2), length(sweep_seeds)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, sweep2_task)[.progress, .stop])
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

cat("\n=== Architecture sweep 2, Ames transfer, mean across 15 seeds ===\n")
cat("reference: node_norm (wave 1 baseline) rsq_trad=0.247 cal_slope=0.854\n")
print(summary[order(-summary$rsq_trad), ], row.names = FALSE, digits = 3)

cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

saveRDS(res, "data/kc-to-ames-arch-sweep2.rds")
cat("\nSaved to data/kc-to-ames-arch-sweep2.rds\n")

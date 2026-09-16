# Wave 7: does the tight Gaussian kernel found at Broward -> San Diego
# (0.02x median distance, via sfdep::st_kernel_weights) also help the
# flagship KC -> Ames result, stacked on the confirmed dropout=0.1 +
# wd=1e-4 winner (rsq_trad=0.467, wave 5/6, uniform edges)?

source("R/kc-to-ames-core.R")
source("R/kc-to-ames-arch-sweep2-helpers.R")

sweep_seeds <- 1001:1015

start_daemons(10)
everywhere(source("R/kc-to-ames-arch-sweep2-helpers.R", local = FALSE))

jobs <- unlist(
  lapply(names(variants7), function(v) lapply(sweep_seeds, function(s) list(variant = v, seed = s))),
  recursive = FALSE
)
cat(sprintf("\nArchitecture sweep 7 (kernel on KC -> Ames): %d fits\n", length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, sweep7_task)[.progress, .stop])
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

cat("\n=== Architecture sweep 7: kernel on top of dropout=0.1+wd=1e-4 ===\n")
cat("reference (uniform edges, no kernel): rsq_trad=0.467 (wave 5/6)\n")
print(summary[order(-summary$rsq_trad), ], row.names = FALSE, digits = 3)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))
saveRDS(res, "data/kc-to-ames-arch-sweep7.rds")
cat("\nSaved to data/kc-to-ames-arch-sweep7.rds\n")

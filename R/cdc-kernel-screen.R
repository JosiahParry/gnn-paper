# CDC PLACES CA -> WV kernel bandwidth screen. Moran's I diagnostic
# suggested this probably won't help much (6/10 covariates already carry
# MORE spatial signal than the residual needs -- the opposite of Broward ->
# San Diego's gap) -- checking empirically rather than trusting that alone.
source("R/cdc-places-core.R")

cli_args <- commandArgs(trailingOnly = TRUE)
n_daemons <- if (length(cli_args) >= 1) as.integer(cli_args[1]) else 14L

seeds <- 1001:1005
thresholds <- c(0.1, 0.25, 0.5, 1, 2, 4)

start_daemons(n_daemons)

jobs <- unlist(lapply(thresholds, function(t) {
  lapply(seeds, function(s) {
    list(arm = "GraphSAGE + LayerNorm", seed = s, kspec = list(kernel = "gaussian", threshold_mult = t))
  })
}), recursive = FALSE)

cat(sprintf("\nCDC kernel screen: %d fits\n", length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, wv_task)[.progress, .stop])
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

summary <- res |>
  group_by(threshold_mult) |>
  summarise(mae = mean(mae), rmse = mean(rmse), rsq_trad = mean(rsq_trad),
            bias = mean(bias), cal_slope = mean(cal_slope), .groups = "drop") |>
  as.data.frame()

cat("\n=== CDC PLACES kernel screen (5 seeds), GraphSAGE + LayerNorm ===\n")
cat("reference: uniform edges (original) rsq_trad=0.741\n")
print(summary[order(-summary$rsq_trad), ], row.names = FALSE, digits = 3)

saveRDS(list(folds = res, summary = summary), "data/cdc-kernel-screen.rds")
cat("\nSaved to data/cdc-kernel-screen.rds\n")

# Broward -> San Diego, Gaussian bandwidth sweep. Tests whether a steeper
# kernel (smaller bandwidth relative to median neighbour distance) closes
# the gap with XGBoost, per the Moran's I diagnostic: covariate spatial
# signal only exceeds residual spatial signal at bandwidth ~0.25x median.
#
# Run: Rscript R/airbnb-brow-sd-bandwidth.R [n_daemons]

source("R/airbnb-core-brow-sd.R")

cli_args <- commandArgs(trailingOnly = TRUE)
n_daemons <- if (length(cli_args) >= 1) as.integer(cli_args[1]) else 14L

seeds <- 1001:1015
bandwidths <- c(1, 0.5, 0.25, 0.1)
graph_arms <- c("GraphSAGE", "GraphSAGE + LayerNorm", "XGBoost + lags")

start_daemons(n_daemons)

jobs <- unlist(unlist(lapply(bandwidths, function(bw) {
  lapply(graph_arms, function(a) lapply(seeds, function(s) list(arm = a, seed = s, bandwidth_mult = bw)))
}), recursive = FALSE), recursive = FALSE)

# Reference points that don't depend on bandwidth, one fit each.
jobs <- c(jobs, list(list(arm = "OLS", seed = seeds[1], bandwidth_mult = 1)))
jobs <- c(jobs, list(list(arm = "XGBoost", seed = seeds[1], bandwidth_mult = 1)))

cat(sprintf("\nBandwidth sweep: %d fits\n", length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, sd_task)[.progress, .stop])
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

summary <- res |>
  group_by(arm, bandwidth_mult) |>
  summarise(
    n_fits = dplyr::n(),
    mae = mean(mae), rmse = mean(rmse), rsq = mean(rsq),
    rsq_trad_sd = sd(rsq_trad), rsq_trad = mean(rsq_trad),
    bias = mean(bias), cal_slope = mean(cal_slope), .groups = "drop"
  ) |>
  as.data.frame()

cat("\n=== Bandwidth sweep, Broward -> San Diego ===\n")
cat("reference: XGBoost (no lags) rsq_trad=0.650, OLS rsq_trad=0.648\n")
print(summary[order(-summary$rsq_trad), ], row.names = FALSE, digits = 3)

saveRDS(list(folds = res, summary = summary), "data/airbnb-brow-sd-bandwidth.rds")
cat("\nSaved to data/airbnb-brow-sd-bandwidth.rds\n")

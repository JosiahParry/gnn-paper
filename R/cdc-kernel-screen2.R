# CDC PLACES CA -> WV kernel screen, wider range -- rsq_trad was still
# climbing at 4x median (0.798, vs uniform-edge reference 0.741), not yet
# plateaued.
source("R/cdc-places-core.R")

cli_args <- commandArgs(trailingOnly = TRUE)
n_daemons <- if (length(cli_args) >= 1) as.integer(cli_args[1]) else 14L

seeds <- 1001:1005
thresholds <- c(3, 6, 8, 12, 16, 24)

start_daemons(n_daemons)

jobs <- unlist(lapply(thresholds, function(t) {
  lapply(seeds, function(s) {
    list(arm = "GraphSAGE + LayerNorm", seed = s, kspec = list(kernel = "gaussian", threshold_mult = t))
  })
}), recursive = FALSE)

cat(sprintf("\nCDC kernel screen 2 (wider): %d fits\n", length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, wv_task)[.progress, .stop])
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

summary <- res |>
  group_by(threshold_mult) |>
  summarise(mae = mean(mae), rmse = mean(rmse), rsq_trad = mean(rsq_trad),
            bias = mean(bias), cal_slope = mean(cal_slope), .groups = "drop") |>
  as.data.frame()

cat("\n=== CDC PLACES kernel screen 2 (5 seeds), GraphSAGE + LayerNorm ===\n")
cat("reference: uniform edges (original) rsq_trad=0.741; 4x median from screen1: 0.798\n")
print(summary[order(-summary$rsq_trad), ], row.names = FALSE, digits = 3)

saveRDS(list(folds = res, summary = summary), "data/cdc-kernel-screen2.rds")
cat("\nSaved to data/cdc-kernel-screen2.rds\n")

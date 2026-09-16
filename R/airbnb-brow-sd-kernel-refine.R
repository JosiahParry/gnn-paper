# Refine: gaussian kernel was the clear winner and still improving at the
# tightest threshold tested (0.02x median = 0.678, vs 1.0x = 0.633). Push
# tighter to find where it plateaus or turns over.

source("R/airbnb-core-brow-sd.R")

cli_args <- commandArgs(trailingOnly = TRUE)
n_daemons <- if (length(cli_args) >= 1) as.integer(cli_args[1]) else 14L

seeds <- 1001:1005
thresholds <- c(0.001, 0.005, 0.01, 0.02, 0.03)

start_daemons(n_daemons)

# Only two levels of nesting here (thresholds x seeds), so a single
# unlist(recursive=FALSE) flattens correctly -- a second pass would tear
# each job's own {arm, seed, kspec} list apart too (as it briefly did).
jobs <- unlist(lapply(thresholds, function(t) {
  lapply(seeds, function(s) {
    list(arm = "GraphSAGE + LayerNorm", seed = s,
         kspec = list(kernel = "gaussian", threshold_mult = t, adaptive = FALSE))
  })
}), recursive = FALSE)

cat(sprintf("\nGaussian refine: %d fits\n", length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, sd_task)[.progress, .stop])
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

summary <- res |>
  group_by(threshold_mult) |>
  summarise(
    mae = mean(mae), rmse = mean(rmse),
    rsq_trad = mean(rsq_trad), bias = mean(bias), cal_slope = mean(cal_slope),
    .groups = "drop"
  ) |>
  as.data.frame()

cat("\n=== Gaussian refine, GraphSAGE + LayerNorm (5 seeds) ===\n")
cat("reference at 0.02: rsq_trad=0.6777\n")
print(summary[order(-summary$rsq_trad), ], row.names = FALSE, digits = 4)

saveRDS(list(folds = res, summary = summary), "data/airbnb-brow-sd-kernel-refine.rds")
cat("\nSaved to data/airbnb-brow-sd-kernel-refine.rds\n")

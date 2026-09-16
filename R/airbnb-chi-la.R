# Chicago -> LA Airbnb transfer: baseline (uniform edges) + kernel bandwidth
# screen, in one pass since LA is large (37863) and fits are fast per-node.
source("R/airbnb-core-chi-la.R")

cli_args <- commandArgs(trailingOnly = TRUE)
n_daemons <- if (length(cli_args) >= 1) as.integer(cli_args[1]) else 14L

seeds <- 1001:1015
screen_seeds <- 1001:1005
thresholds <- c(0.02, 0.1, 0.25, 0.5, 1, 2, 4)

start_daemons(n_daemons)

baseline_jobs <- unlist(
  lapply(names(arms), function(a) {
    use <- if (a %in% stochastic) seeds else seeds[1]
    lapply(use, function(s) list(arm = a, seed = s, kspec = list(kernel = "uniform", threshold_mult = 1, adaptive = FALSE)))
  }),
  recursive = FALSE
)

kernel_screen_jobs <- unlist(lapply(thresholds, function(t) {
  lapply(screen_seeds, function(s) {
    list(arm = "GraphSAGE + LayerNorm", seed = s, kspec = list(kernel = "gaussian", threshold_mult = t, adaptive = FALSE))
  })
}), recursive = FALSE)

jobs <- c(baseline_jobs, kernel_screen_jobs)
cat(sprintf("\nChicago -> LA: %d fits (baseline + kernel screen)\n", length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, la_task)[.progress, .stop])
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

baseline <- res[res$kernel == "uniform", ]
screen <- res[res$kernel == "gaussian", ]

cat("\n=== Baseline (uniform edges), mean of independent single-model fits ===\n")
bsum <- baseline |>
  group_by(arm) |>
  summarise(n_fits = dplyr::n(), mae = mean(mae), rmse = mean(rmse),
            rsq_trad_sd = sd(rsq_trad), rsq_trad = mean(rsq_trad),
            bias = mean(bias), cal_slope = mean(cal_slope), .groups = "drop") |>
  as.data.frame()
print(bsum[order(-bsum$rsq_trad), ], row.names = FALSE, digits = 3)

cat("\n=== Kernel screen (5 seeds), GraphSAGE + LayerNorm ===\n")
ssum <- screen |>
  group_by(threshold_mult) |>
  summarise(mae = mean(mae), rmse = mean(rmse), rsq_trad = mean(rsq_trad),
            bias = mean(bias), cal_slope = mean(cal_slope), .groups = "drop") |>
  as.data.frame()
print(ssum[order(-ssum$rsq_trad), ], row.names = FALSE, digits = 3)

saveRDS(list(folds = res, baseline_summary = bsum, screen_summary = ssum), "data/airbnb-chi-la-results.rds")
cat("\nSaved to data/airbnb-chi-la-results.rds\n")

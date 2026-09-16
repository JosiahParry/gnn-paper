# CA -> WV census-tract diabetes-prevalence transfer, same protocol as
# R/kc-to-ames.R. Every arm scored as a single deployable model; seeds
# characterize typical performance and spread, never combined.
#
# Run: Rscript R/cdc-places.R [n_daemons]

source("R/cdc-places-core.R")

cli_args <- commandArgs(trailingOnly = TRUE)
n_daemons <- if (length(cli_args) >= 1) as.integer(cli_args[1]) else 10L

seeds <- 1001:1015

start_daemons(n_daemons)

jobs <- unlist(
  lapply(names(arms), function(a) {
    use <- if (a %in% stochastic) seeds else seeds[1]
    lapply(use, function(s) list(arm = a, seed = s))
  }),
  recursive = FALSE
)

cat(sprintf("\nCA -> WV transfer: %d fits\n", length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, wv_task)[.progress, .stop])
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

summary <- res |>
  group_by(arm) |>
  summarise(
    n_fits = dplyr::n(),
    mae = mean(mae), rmse = mean(rmse), rsq = mean(rsq),
    rsq_trad_sd = sd(rsq_trad), rsq_trad = mean(rsq_trad),
    bias = mean(bias), cal_slope = mean(cal_slope), .groups = "drop"
  ) |>
  as.data.frame()

cat("\n=== CA -> WV, diabetes prevalence: mean of independent single-model fits ===\n")
print(summary[order(-summary$rsq_trad), ], row.names = FALSE, digits = 3)

dir.create("data", showWarnings = FALSE)
saveRDS(list(folds = res, summary = summary), "data/cdc-places-results.rds")
cat("\nSaved to data/cdc-places-results.rds\n")

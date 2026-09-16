# Queens -> Brooklyn 311 response-time transfer: baseline + kernel screen.
# Mirrors R/trees-test.R exactly so the two non-price targets are directly
# comparable.
source("R/nyc311-core.R")
cli <- commandArgs(trailingOnly = TRUE)
nd <- if (length(cli) >= 1) as.integer(cli[1]) else 14L

seeds <- 1001:1010
screen_seeds <- 1001:1005
thresholds <- c(0.1, 0.25, 0.5, 1, 2)

start_daemons(nd)

baseline <- unlist(lapply(names(arms), function(a) {
  use <- if (a %in% stochastic) seeds else seeds[1]
  lapply(use, function(s) list(arm = a, seed = s, kspec = list(kernel = "uniform", threshold_mult = 1)))
}), recursive = FALSE)

screen <- unlist(lapply(thresholds, function(t) {
  lapply(screen_seeds, function(s) {
    list(arm = "GraphSAGE + LayerNorm", seed = s, kspec = list(kernel = "gaussian", threshold_mult = t))
  })
}), recursive = FALSE)

jobs <- c(baseline, screen)
cat(sprintf("\nQueens -> Brooklyn (311): %d fits\n", length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, bkn_task)[.progress, .stop])
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

b <- res[res$kernel == "uniform", ] |> group_by(arm) |>
  summarise(n = dplyr::n(), mae = mean(mae), rsq_trad_sd = sd(rsq_trad),
            rsq_trad = mean(rsq_trad), bias = mean(bias), cal = mean(cal_slope), .groups = "drop") |>
  as.data.frame()
cat("\n=== Baseline (uniform edges) ===\n")
print(b[order(-b$rsq_trad), ], row.names = FALSE, digits = 3)

# Paired test: GraphSAGE + LayerNorm vs the best tabular arm, same seeds.
u <- res[res$kernel == "uniform", ]
best_tab <- b$arm[b$arm %in% c("XGBoost", "XGBoost + lags")][
  which.max(b$rsq_trad[b$arm %in% c("XGBoost", "XGBoost + lags")])]
g <- u$rsq_trad[u$arm == "GraphSAGE + LayerNorm"]
t_ <- u$rsq_trad[u$arm == best_tab]
if (length(g) == length(t_)) {
  tt <- t.test(g, t_, paired = TRUE)
  cat(sprintf("\nGraphSAGE + LayerNorm vs %s: diff=%+.4f p=%.3f CI[%+.4f, %+.4f]\n",
              best_tab, unname(tt$estimate), tt$p.value, tt$conf.int[1], tt$conf.int[2]))
}

s <- res[res$kernel == "gaussian", ] |> group_by(threshold_mult) |>
  summarise(mae = mean(mae), rsq_trad = mean(rsq_trad), cal = mean(cal_slope), .groups = "drop") |>
  as.data.frame()
cat("\n=== Kernel screen, GraphSAGE + LayerNorm (5 seeds) ===\n")
print(s[order(-s$rsq_trad), ], row.names = FALSE, digits = 3)

saveRDS(list(folds = res, baseline = b, screen = s), "data/nyc311-results.rds")
cat("\nSaved to data/nyc311-results.rds\n")

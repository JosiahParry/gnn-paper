# Generic transfer test: baseline arms + gaussian-kernel bandwidth screen,
# for any (dataset, source, target) triple. Used to run each transfer in BOTH
# directions, so a win can be attributed to the method rather than to which
# region happened to be the source.
#
# Usage:
#   Rscript R/pair-test.R <data.rds> <source> <target> [n_daemons] [out.rds]

cli <- commandArgs(trailingOnly = TRUE)
stopifnot(length(cli) >= 3)
options(pair_data = cli[1], pair_src = cli[2], pair_tgt = cli[3])
nd <- if (length(cli) >= 4) as.integer(cli[4]) else 14L
out_path <- if (length(cli) >= 5) cli[5] else
  sprintf("data/pair-%s-to-%s.rds", cli[2], cli[3])

source("R/pair-core.R")

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
cat(sprintf("\n%s -> %s: %d fits\n", SRC, TGT, length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, target_task)[.progress, .stop])
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res),
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))

b <- res[res$kernel == "uniform", ] |> group_by(arm) |>
  summarise(n = dplyr::n(), mae = mean(mae), rsq_trad_sd = sd(rsq_trad),
            rsq_trad = mean(rsq_trad), bias = mean(bias), cal = mean(cal_slope),
            .groups = "drop") |> as.data.frame()
cat(sprintf("\n=== Baseline, %s -> %s (uniform edges) ===\n", SRC, TGT))
print(b[order(-b$rsq_trad), ], row.names = FALSE, digits = 3)

# Paired test vs the better tabular arm, on identical seeds.
u <- res[res$kernel == "uniform", ]
tab <- b[b$arm %in% c("XGBoost", "XGBoost + lags"), ]
best_tab <- tab$arm[which.max(tab$rsq_trad)]
g <- u$rsq_trad[u$arm == "GraphSAGE + LayerNorm"]
t_ <- u$rsq_trad[u$arm == best_tab]
if (length(g) == length(t_) && length(g) > 1) {
  tt <- t.test(g, t_, paired = TRUE)
  cat(sprintf("\nGraphSAGE + LayerNorm vs %s: diff=%+.4f p=%.4f CI[%+.4f, %+.4f]\n",
              best_tab, unname(tt$estimate), tt$p.value, tt$conf.int[1], tt$conf.int[2]))
}

s <- res[res$kernel == "gaussian", ] |> group_by(threshold_mult) |>
  summarise(mae = mean(mae), rsq_trad = mean(rsq_trad), cal = mean(cal_slope),
            .groups = "drop") |> as.data.frame()
cat("\n=== Kernel screen, GraphSAGE + LayerNorm (5 seeds) ===\n")
print(s[order(-s$rsq_trad), ], row.names = FALSE, digits = 3)

saveRDS(list(folds = res, baseline = b, screen = s, src = SRC, tgt = TGT), out_path)
cat(sprintf("\nSaved to %s\n", out_path))

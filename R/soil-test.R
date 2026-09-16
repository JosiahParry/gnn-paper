# Soil-carbon raster transfer: full arm comparison + neighbour-definition test.
#
# Two questions in one run:
#   1. Does GraphSAGE beat the non-graph arms on a raster transfer?
#   2. On a grid, is the natural adjacency (touching cells) better than the
#      KNN-30 we have used everywhere else by necessity?
#
# Run: Rscript R/soil-test.R [n_daemons]

# Usage: Rscript R/soil-test.R [n_daemons] [data.rds] [source] [target] [cell_m]
cli <- commandArgs(trailingOnly = TRUE)
nd <- if (length(cli) >= 1) as.integer(cli[1]) else 14L
if (length(cli) >= 2) options(soil_data = cli[2])
if (length(cli) >= 4) options(soil_src = cli[3], soil_tgt = cli[4])
if (length(cli) >= 5) options(soil_cell = as.numeric(cli[5]))

source("R/soil-core.R")

seeds <- 1001:1010
gseeds <- 1001:1005
graphs <- c("queen", "queen2", "knn30")
gspec <- function(gr) list(graph = gr, kernel = "uniform", threshold_mult = 1)

start_daemons(nd)

# (a) every arm at the natural raster adjacency
baseline <- unlist(lapply(names(arms), function(a) {
  use <- if (a %in% stochastic) seeds else seeds[1]
  lapply(use, function(s) list(arm = a, seed = s, g = gspec("queen")))
}), recursive = FALSE)

# (b) the two graph-using arms at the wider and KNN neighbourhoods
gtest <- unlist(lapply(setdiff(graphs, "queen"), function(gr) {
  unlist(lapply(c("GraphSAGE + LayerNorm", "XGBoost + lags"), function(a) {
    lapply(gseeds, function(s) list(arm = a, seed = s, g = gspec(gr)))
  }), recursive = FALSE)
}), recursive = FALSE)

jobs <- c(baseline, gtest)
cat(sprintf("\n%s -> %s (raster): %d fits\n", SRC, TGT, length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, target_task)[.progress, .stop])
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res),
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))

b <- res[res$graph == "queen", ] |> group_by(arm) |>
  summarise(n = dplyr::n(), mae = mean(mae), rsq_sd = sd(rsq_trad),
            rsq_trad = mean(rsq_trad), bias = mean(bias), cal = mean(cal_slope),
            .groups = "drop") |> as.data.frame()
cat("\n=== Arms, queen adjacency (8 touching cells) ===\n")
print(b[order(-b$rsq_trad), ], row.names = FALSE, digits = 3)

u <- res[res$graph == "queen", ]
tab <- b[b$arm %in% c("XGBoost", "OLS"), ]
best_tab <- tab$arm[which.max(tab$rsq_trad)]
g <- u$rsq_trad[u$arm == "GraphSAGE + LayerNorm"]
t_ <- u$rsq_trad[u$arm == best_tab]
if (length(g) > 1 && length(t_) == length(g)) {
  tt <- t.test(g, t_, paired = TRUE)
  cat(sprintf("\nGraphSAGE + LayerNorm vs %s: diff=%+.4f p=%.4f CI[%+.4f, %+.4f]\n",
              best_tab, unname(tt$estimate), tt$p.value, tt$conf.int[1], tt$conf.int[2]))
} else if (length(t_) == 1) {
  cat(sprintf("\nGraphSAGE + LayerNorm mean %.4f vs %s (deterministic) %.4f -> %+.4f\n",
              mean(g), best_tab, t_, mean(g) - t_))
}

s <- res |> group_by(arm, graph) |>
  summarise(n = dplyr::n(), rsq_trad = mean(rsq_trad), cal = mean(cal_slope),
            .groups = "drop") |> as.data.frame()
cat("\n=== Neighbour definition (graph-using arms) ===\n")
print(s[s$arm %in% c("GraphSAGE + LayerNorm", "XGBoost + lags"), ][
  order(s$arm[s$arm %in% c("GraphSAGE + LayerNorm", "XGBoost + lags")]), ],
  row.names = FALSE, digits = 3)

outf <- sprintf("data/soil-results-%s-%s.rds", SRC, TGT)
saveRDS(list(folds = res, baseline = b, graphs = s), outf)
cat(sprintf("\nSaved to %s\n", outf))

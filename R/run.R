# Rscript R/run.R <data.rds> <src> <tgt> [crs] [graph] [cell] [mode] [seeds] [out]

a <- commandArgs(trailingOnly = TRUE)
stopifnot(length(a) >= 3)
arg <- function(i, d) if (length(a) >= i && nzchar(a[i])) a[i] else d

options(gnn_data = a[1], gnn_src = a[2], gnn_tgt = a[3],
        gnn_crs = if (arg(4, "4326") == "NA") NA else as.numeric(arg(4, "4326")),
        gnn_graph = arg(5, "knn30"), gnn_cell = as.numeric(arg(6, "60")),
        gnn_mode = arg(7, "reg"))
n_seeds <- as.integer(arg(8, "10"))
out <- arg(9, sprintf("data/res-%s-to-%s.rds", a[2], a[3]))

source("R/core.R")
seeds <- seq(1001, length.out = n_seeds)
metric <- if (MODE == "clf") "auc" else "rsq_trad"

start_daemons()
jobs <- unlist(lapply(names(arms), function(x)
  lapply(if (x %in% stochastic) seeds else seeds[1],
         function(s) list(arm = x, seed = s))), recursive = FALSE)

cat(sprintf("\n%s -> %s [%s, %s]: %d fits\n", SRC, TGT, GRAPH, MODE, length(jobs)))
res <- do.call(rbind, mirai_map(jobs, task)[.progress, .stop])
daemons(0)

s <- res |> group_by(arm) |>
  summarise(n = dplyr::n(), across(where(is.numeric) & !seed, mean), .groups = "drop") |>
  as.data.frame()
s <- s[order(-s[[metric]]), ]
print(s, row.names = FALSE, digits = 3)

g <- res[[metric]][res$arm == "GraphSAGE + LayerNorm"]
for (o in setdiff(names(arms), "GraphSAGE + LayerNorm")) {
  v <- res[[metric]][res$arm == o]
  if (length(v) == length(g) && length(g) > 1) {
    t <- t.test(g, v, paired = TRUE)
    cat(sprintf("vs %-22s %+.4f  p=%.4f  (%d/%d seeds)\n", o,
                unname(t$estimate), t$p.value, sum(g > v), length(g)))
  } else if (length(v) == 1) {
    cat(sprintf("vs %-22s %+.4f  (deterministic)\n", o, mean(g) - v))
  }
}

saveRDS(list(folds = res, summary = s), out)
cat(sprintf("\nSaved %s\n", out))

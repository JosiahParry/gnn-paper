# Land cover classification transfer: Des Moines -> Cedar Rapids.
# Usage: Rscript R/clf-test.R [n_daemons]

cli <- commandArgs(trailingOnly = TRUE)
nd <- if (length(cli) >= 1) as.integer(cli[1]) else 14L

source("R/clf-core.R")

seeds <- 1001:1010
gseeds <- 1001:1005

start_daemons(nd)

baseline <- unlist(lapply(names(arms), function(a) {
  use <- if (a %in% stochastic) seeds else seeds[1]
  lapply(use, function(s) list(arm = a, seed = s, graph = "queen"))
}), recursive = FALSE)

gtest <- unlist(lapply(c("queen2", "knn30"), function(gr) {
  unlist(lapply(c("GraphSAGE + LayerNorm", "XGBoost + lags"), function(a) {
    lapply(gseeds, function(s) list(arm = a, seed = s, graph = gr))
  }), recursive = FALSE)
}), recursive = FALSE)

jobs <- c(baseline, gtest)
cat(sprintf("\n%s -> %s (classification): %d fits\n", SRC, TGT, length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, target_task)[.progress, .stop])
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res),
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))

b <- res[res$graph == "queen", ] |> group_by(arm) |>
  summarise(n = dplyr::n(), accuracy = mean(accuracy), bal_acc = mean(bal_accuracy),
            auc = mean(auc), auc_sd = sd(auc), sens = mean(sensitivity),
            spec = mean(specificity), brier = mean(brier), .groups = "drop") |>
  as.data.frame()
cat(sprintf("\n=== Classification, queen adjacency (target positive rate %.3f) ===\n",
            mean(y_of("tgt"))))
print(b[order(-b$auc), ], row.names = FALSE, digits = 3)

u <- res[res$graph == "queen", ]
g <- u$auc[u$arm == "GraphSAGE + LayerNorm"]
for (opp in c("XGBoost + lags", "XGBoost", "Logistic")) {
  o <- u$auc[u$arm == opp]
  if (length(o) == length(g) && length(g) > 1) {
    tt <- t.test(g, o, paired = TRUE)
    cat(sprintf("GraphSAGE+LN vs %-15s AUC %+.4f  p=%.4f  (%d/%d seeds)\n",
                opp, unname(tt$estimate), tt$p.value, sum(g > o), length(g)))
  } else if (length(o) == 1) {
    cat(sprintf("GraphSAGE+LN vs %-15s AUC %+.4f  (deterministic)\n", opp, mean(g) - o))
  }
}

s <- res |> group_by(arm, graph) |>
  summarise(n = dplyr::n(), auc = mean(auc), bal_acc = mean(bal_accuracy), .groups = "drop") |>
  as.data.frame()
cat("\n=== Neighbour definition ===\n")
print(s[s$arm %in% c("GraphSAGE + LayerNorm", "XGBoost + lags"), ], row.names = FALSE, digits = 3)

saveRDS(list(folds = res, baseline = b, graphs = s), "data/clf-results.rds")
cat("\nSaved to data/clf-results.rds\n")

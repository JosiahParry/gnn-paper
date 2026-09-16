# Does choosing the neighbourhood on source-region data change the verdict?
#
# Usage: Rscript R/tuned-test.R <data.rds> <source> <target> [n_daemons] [out.rds]

cli <- commandArgs(trailingOnly = TRUE)
stopifnot(length(cli) >= 3)
options(pair_data = cli[1], pair_src = cli[2], pair_tgt = cli[3])
nd <- if (length(cli) >= 4) as.integer(cli[4]) else 14L
out_path <- if (length(cli) >= 5) cli[5] else
  sprintf("data/tuned-%s-to-%s.rds", cli[2], cli[3])

source("R/tuned-core.R")

seeds <- 1001:1008
arms_tuned <- c("GraphSAGE + LayerNorm (tuned)", "XGBoost + lags (tuned)")

start_daemons_tuned(nd)

jobs <- unlist(lapply(arms_tuned, function(a) {
  lapply(seeds, function(s) list(arm = a, seed = s))
}), recursive = FALSE)

cat(sprintf("\n%s -> %s, neighbourhood tuned on source: %d fits\n", SRC, TGT, length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, tuned_task)[.progress, .stop])
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res),
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))

s <- res |> dplyr::group_by(arm) |>
  dplyr::summarise(n = dplyr::n(), rsq_trad = mean(rsq_trad), sd = sd(rsq_trad),
                   mae = mean(mae), cal = mean(cal_slope), .groups = "drop") |>
  as.data.frame()
cat(sprintf("\n=== %s -> %s, neighbourhood chosen on source data ===\n", SRC, TGT))
print(s[order(-s$rsq_trad), ], row.names = FALSE, digits = 3)

cat("\n=== which bandwidth the source data chose ===\n")
print(table(res$arm, res$chosen))

saveRDS(list(folds = res, summary = s), out_path)
cat(sprintf("\nSaved to %s\n", out_path))

# Trees Queens -> Brooklyn: extend the gaussian bandwidth screen tighter.
#
# The first screen (R/trees-test.R) was monotonic over 2.0x .. 0.1x of the
# median 30-NN distance and had NOT bottomed out -- rsq_trad rose from 0.119
# at 2.0x to 0.237 at 0.1x, the tightest setting tested. That means the
# uniform-edge baseline reported in that run (0.132) reflects a
# mis-specified neighbourhood, not the method's ceiling on this problem.
#
# This run continues the sweep to 0.01x and re-runs 0.1x as an in-run anchor
# so every setting is compared under identical conditions. XGBoost + lags is
# swept over the same bandwidths: if the neighbourhood was too broad, it was
# too broad for BOTH graph-based arms, and treating that as a GraphSAGE-
# specific problem would be wrong.
#
# Note what this run cannot do: OLS scored 0.276 here with no graph at all.
# Finding a bandwidth that beats 0.276 would be a genuine result; failing to
# is the conclusion, and the sweep is reported either way.

source("R/trees-core.R")
cli <- commandArgs(trailingOnly = TRUE)
nd <- if (length(cli) >= 1) as.integer(cli[1]) else 14L

seeds <- 1001:1005
thresholds <- c(0.01, 0.02, 0.05, 0.1)

start_daemons(nd)

jobs <- unlist(lapply(thresholds, function(t) {
  unlist(lapply(c("GraphSAGE + LayerNorm", "XGBoost + lags"), function(a) {
    lapply(seeds, function(s) {
      list(arm = a, seed = s, kspec = list(kernel = "gaussian", threshold_mult = t))
    })
  }), recursive = FALSE)
}), recursive = FALSE)

cat(sprintf("\nTrees tight kernel screen: %d fits\n", length(jobs)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, bkn_task)[.progress, .stop])
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res),
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))

s <- res |> group_by(arm, threshold_mult) |>
  summarise(n = dplyr::n(), mae = mean(mae), rsq_trad = mean(rsq_trad),
            rsq_sd = sd(rsq_trad), cal = mean(cal_slope), .groups = "drop") |>
  as.data.frame()
cat("\n=== Tight gaussian bandwidths (5 seeds each) ===\n")
print(s[order(s$arm, -s$rsq_trad), ], row.names = FALSE, digits = 3)

cat("\nReference from R/trees-test.R on the same split:\n")
cat("  OLS 0.276 | XGBoost 0.270 | GraphSAGE+LN uniform 0.132 | 0.1x 0.237\n")
cat(sprintf("\nBest here: %s at %.2fx -> %.3f (OLS is 0.276)\n",
            s$arm[which.max(s$rsq_trad)], s$threshold_mult[which.max(s$rsq_trad)],
            max(s$rsq_trad)))

saveRDS(list(folds = res, summary = s), "data/trees-kernel-tight.rds")
cat("\nSaved to data/trees-kernel-tight.rds\n")

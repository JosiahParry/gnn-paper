# Architecture ablation for the KC -> Ames transfer.
#
# Baseline (already run, 30 seeds): GraphSAGE + LayerNorm(mode="graph",
# implicit default), concat=TRUE, self-loops, unweighted KNN-30.
#   rsq_trad = 0.199, cal_slope = 0.774
#
# layer_layer_norm's mode defaults to "graph" (first level of
# c("graph","node")) whenever it's passed bare as `norm = layer_layer_norm`,
# since model_sage calls it as norm(hidden_dim) with no mode override. That
# collapses every node in the current graph to one shared scalar mean/var per
# layer, rather than the per-node normalization "LayerNorm" usually means.
#
# SAGELayer's concat=TRUE (the default, never overridden at any call site)
# passes each node's own raw features straight into every layer's linear
# transform via cat([x, neighbor_agg]), on top of add_graph_self_loops()
# already folding x into neighbor_agg itself. The focal node's own signal is
# carried twice and can dominate over genuine neighbor aggregation.
#
# This script isolates both fixes against the same baseline, seeds 1001:1015.
# variants / fit_sage_variant / sweep_task live in the helpers file so daemons
# can source them too.

source("R/kc-to-ames-core.R")
source("R/kc-to-ames-arch-sweep-helpers.R")

sweep_seeds <- 1001:1015

start_daemons(14)
everywhere(source("R/kc-to-ames-arch-sweep-helpers.R", local = FALSE))

jobs <- unlist(
  lapply(names(variants), function(v) lapply(sweep_seeds, function(s) list(variant = v, seed = s))),
  recursive = FALSE
)

cat(sprintf("\nArchitecture sweep: %d fits (%d variants x %d seeds)\n", length(jobs), length(variants), length(sweep_seeds)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, sweep_task)[.progress, .stop])
daemons(0)

var_ames <- var(ames$y)
res$rsq_trad <- 1 - res$rmse^2 / var_ames

summary <- res |>
  group_by(variant) |>
  summarise(
    mae = mean(mae), rmse = mean(rmse),
    rsq = mean(rsq),
    rsq_trad = 1 - mean(rmse)^2 / var_ames,
    bias = mean(bias), cal_slope = mean(cal_slope),
    .groups = "drop"
  ) |>
  as.data.frame()

cat("\n=== Architecture sweep, Ames transfer, mean across 15 seeds ===\n")
cat("baseline (already run, 30 seeds): rsq_trad=0.199 cal_slope=0.774\n")
print(summary, row.names = FALSE, digits = 3)

cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

saveRDS(res, "data/kc-to-ames-arch-sweep.rds")
cat("\nSaved to data/kc-to-ames-arch-sweep.rds\n")

# Random CV: six folds of scattered nodes.
#
# The permissive design. Held-out nodes are interspersed among training nodes,
# so a test node's neighbours in the full lattice are mostly training nodes --
# though the model never sees them, since the test set still gets its own
# subgraph.
#
# Same fold count as the block run and the same scenario draws, so the two are
# directly comparable and the difference between them is attributable to the
# split alone.
#
# Run from the repo root, after R/calibrate.R:
#   R -f R/simulate-random.R

library(mirai)

source("R/scenarios.R")

start_daemons(6)

set.seed(1)
fold_id <- sample(rep_len(seq_len(k_folds), n))

splits_random <- lapply(seq_len(k_folds), function(i) {
  make_split(which(fold_id == i), fold = i)
})

cat("Random fold sizes (test nodes):\n")
print(vapply(splits_random, \(s) length(s$test_id), integer(1)))

results <- lapply(scenarios, function(s) {
  run_scenario(s$label, s$df, splits_random, sem = s$sem)
})

daemons(0)

folds <- do.call(rbind, lapply(results, \(r) r$folds))
summary <- do.call(rbind, lapply(results, \(r) r$summary))

dir.create("data", showWarnings = FALSE)
saveRDS(
  list(
    split = "random", folds = folds, summary = summary,
    morans = morans, e_params = e_params
  ),
  "data/scenario-results-random.rds"
)

cat("\n\n=== Random CV, mean out-of-sample R2 across folds ===\n")
print(
  summary |>
    select(label, arm, rsq_mean, rsq_sd) |>
    tidyr::pivot_wider(names_from = arm, values_from = c(rsq_mean, rsq_sd)) |>
    as.data.frame(),
  row.names = FALSE,
  digits = 3
)

cat("\nSaved to data/scenario-results-random.rds\n")

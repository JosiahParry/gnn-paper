# Block CV: six contiguous square blocks, each held out in turn.
#
# The demanding design. Held-out nodes form a contiguous region, so the model
# must predict somewhere it has never seen rather than interpolating between
# training neighbours.
#
# Run from the repo root, after R/calibrate.R:
#   R -f R/simulate-block.R

library(blockCV)
library(mirai)

source("R/scenarios.R")

# Six vertical strips, each spanning the full height of the lattice.
block_cv <- cv_spatial(
  grid,
  k = k_folds,
  hexagon = FALSE,
  rows_cols = c(1, k_folds),
  selection = "systematic",
  progress = FALSE,
  report = FALSE
)

start_daemons(6)

splits_block <- lapply(seq_len(k_folds), function(i) {
  make_split(which(block_cv$folds_ids == i), fold = i)
})

cat("Block fold sizes (test nodes):\n")
print(vapply(splits_block, \(s) length(s$test_id), integer(1)))

results <- lapply(scenarios, function(s) {
  run_scenario(s$label, s$df, splits_block, sem = s$sem)
})

daemons(0)

folds <- do.call(rbind, lapply(results, \(r) r$folds))
summary <- do.call(rbind, lapply(results, \(r) r$summary))

dir.create("data", showWarnings = FALSE)
saveRDS(
  list(
    split = "block", folds = folds, summary = summary,
    morans = morans, e_params = e_params
  ),
  "data/scenario-results-block.rds"
)

cat("\n\n=== Block CV, mean out-of-sample R2 across folds ===\n")
print(
  summary |>
    select(label, arm, rsq_mean, rsq_sd) |>
    tidyr::pivot_wider(names_from = arm, values_from = c(rsq_mean, rsq_sd)) |>
    as.data.frame(),
  row.names = FALSE,
  digits = 3
)

cat("\nSaved to data/scenario-results-block.rds\n")

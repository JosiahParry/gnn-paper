# Every state held out in turn, all six arms, one seed.
#
# Run from the repo root:
#   R -f R/states.R

source("R/states-core.R")

cat(sprintf("%d states\n", length(all_fips)))

t0 <- Sys.time()

state_res <- lapply(seq_along(all_fips), function(i) {
  cat(sprintf("[%2d/%d] %s\n", i, length(all_fips), all_fips[i]))
  state_task(all_fips[i])
})

results <- do.call(rbind, state_res)

cat(sprintf(
  "\n%d states in %.1f min\n",
  length(all_fips),
  as.numeric(difftime(Sys.time(), t0, units = "mins"))
))

dir.create("data", showWarnings = FALSE)
saveRDS(results, "data/state-results.rds")

cat(sprintf(
  "\n\n=== Held-out states (%d) ===\n",
  dplyr::n_distinct(results$STUSPS)
))

print(
  results |>
    group_by(arm) |>
    summarise(
      mae = mean(mae),
      rmse = mean(rmse),
      rsq = mean(rsq),
      rsq_trad = mean(rsq_trad),
      mean_abs_bias = mean(abs(mean_bias)),
      median_abs_bias = median(abs(mean_bias)),
      max_abs_bias = max(abs(mean_bias)),
      resid_moran = mean(resid_moran, na.rm = TRUE),
      .groups = "drop"
    ) |>
    as.data.frame(),
  row.names = FALSE,
  digits = 3
)

cat("\n=== Per-state R2 (squared correlation) ===\n")
print(
  results |>
    select(STUSPS, arm, rsq) |>
    tidyr::pivot_wider(names_from = arm, values_from = rsq) |>
    arrange(STUSPS) |>
    as.data.frame(),
  row.names = FALSE,
  digits = 3
)

cat("\n=== Per-state R2 (traditional, 1 - SSE/SST) ===\n")
print(
  results |>
    select(STUSPS, arm, rsq_trad) |>
    tidyr::pivot_wider(names_from = arm, values_from = rsq_trad) |>
    arrange(STUSPS) |>
    as.data.frame(),
  row.names = FALSE,
  digits = 3
)

cat("\n=== GraphSAGE + LayerNorm vs GraphSAGE, states won (rsq_trad) ===\n")
sage_wide <- results |>
  filter(arm %in% c("GraphSAGE", "GraphSAGE + LayerNorm")) |>
  select(STUSPS, arm, rsq_trad) |>
  tidyr::pivot_wider(names_from = arm, values_from = rsq_trad)
n_layernorm_wins <- sum(sage_wide[["GraphSAGE + LayerNorm"]] > sage_wide[["GraphSAGE"]])
cat(sprintf(
  "LayerNorm wins %d / %d states\n",
  n_layernorm_wins, nrow(sage_wide)
))

cat("\nSaved to data/state-results.rds\n")

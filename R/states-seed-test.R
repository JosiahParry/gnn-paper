# Is plain GraphSAGE genuinely better than GraphSAGE+LayerNorm on the states
# holdout, or is the production 0.645 vs 0.617 gap seed noise from a single
# fixed-seed fit per state? Also tests whether the full recipe
# (LayerNorm + dropout 0.1 + wd 1e-4), which is what "GraphSAGE + LayerNorm"
# means everywhere else in this project, changes the answer.
#
# Run: Rscript R/states-seed-test.R [n_daemons]

source("R/states-seed-test-helpers.R")

cli_args <- commandArgs(trailingOnly = TRUE)
n_daemons <- if (length(cli_args) >= 1) as.integer(cli_args[1]) else 14L

seeds <- 1001:1005

daemons(n_daemons)
core <- normalizePath("R/states-seed-test-helpers.R", mustWork = TRUE)
everywhere(
  { source(core_path, local = FALSE); torch_set_num_threads(1L) },
  .args = list(core_path = core), .min = n_daemons
)
cat(sprintf("%d daemons up\n", n_daemons))

jobs <- unlist(unlist(lapply(names(sage_configs), function(cfg) {
  lapply(all_fips, function(f) {
    lapply(seeds, function(s) list(fips = f, seed = s, config = cfg))
  })
}), recursive = FALSE), recursive = FALSE)

cat(sprintf("\nStates seed test: %d fits (%d states x %d configs x %d seeds)\n",
            length(jobs), length(all_fips), length(sage_configs), length(seeds)))
t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, state_seed_task)[.progress, .stop])
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

# Per-seed national average, then spread across seeds: this is the quantity
# the production single-seed run reports exactly one draw of.
per_seed <- res |>
  group_by(config, seed) |>
  summarise(rsq_trad = mean(rsq_trad), mae = mean(mae), .groups = "drop")

cat("\n=== National mean rsq_trad, per seed ===\n")
print(tidyr::pivot_wider(per_seed[, c("config","seed","rsq_trad")],
                          names_from = config, values_from = rsq_trad),
      row.names = FALSE, digits = 4)

cat("\n=== Across-seed summary ===\n")
summ <- per_seed |>
  group_by(config) |>
  summarise(rsq_trad_mean = mean(rsq_trad), rsq_trad_sd = sd(rsq_trad),
            rsq_trad_min = min(rsq_trad), rsq_trad_max = max(rsq_trad),
            mae_mean = mean(mae), .groups = "drop") |>
  as.data.frame()
print(summ, row.names = FALSE, digits = 4)

cat("\n=== Paired per-state comparison, plain vs layernorm (all seeds pooled) ===\n")
w <- res |>
  filter(config %in% c("plain", "layernorm")) |>
  select(stusps, seed, config, rsq_trad) |>
  tidyr::pivot_wider(names_from = config, values_from = rsq_trad)
cat(sprintf("plain better in %d of %d state-seed pairs (%.1f%%)\n",
            sum(w$plain > w$layernorm), nrow(w), 100 * mean(w$plain > w$layernorm)))
cat(sprintf("mean paired difference (plain - layernorm): %.4f (sd %.4f)\n",
            mean(w$plain - w$layernorm), sd(w$plain - w$layernorm)))
print(t.test(w$plain, w$layernorm, paired = TRUE))

saveRDS(list(folds = res, per_seed = per_seed, summary = summ), "data/states-seed-test.rds")
cat("\nSaved to data/states-seed-test.rds\n")

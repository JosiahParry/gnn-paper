# Does kernel-weighting the queen-contiguity edges help on the states
# holdout? Weights existing contiguity edges by inter-centroid distance; it
# does not prune them (see make_state_graph_kernel).
#
# Uniform is included as an in-run baseline so every setting is compared
# under identical conditions (same seeds, same parallel execution).
# The noise floor here is ~0.01 on the national mean (5 seeds), so anything
# smaller than that is not a finding.
#
# Run: Rscript R/states-kernel-test.R [n_daemons]

source("R/states-seed-test-helpers.R")

cli_args <- commandArgs(trailingOnly = TRUE)
n_daemons <- if (length(cli_args) >= 1) as.integer(cli_args[1]) else 14L

seeds <- 1001:1005

settings <- list(
  list(kernel = "uniform",  threshold_mult = 1,    adaptive = FALSE, label = "uniform"),
  list(kernel = "gaussian", threshold_mult = 0.25, adaptive = FALSE, label = "gauss_0.25x"),
  list(kernel = "gaussian", threshold_mult = 0.5,  adaptive = FALSE, label = "gauss_0.5x"),
  list(kernel = "gaussian", threshold_mult = 1,    adaptive = FALSE, label = "gauss_1x"),
  list(kernel = "gaussian", threshold_mult = 2,    adaptive = FALSE, label = "gauss_2x"),
  list(kernel = "gaussian", threshold_mult = 4,    adaptive = FALSE, label = "gauss_4x"),
  list(kernel = "gaussian", threshold_mult = 1,    adaptive = TRUE,  label = "gauss_adaptive")
)

daemons(n_daemons)
core <- normalizePath("R/states-seed-test-helpers.R", mustWork = TRUE)
everywhere(
  { source(core_path, local = FALSE); torch_set_num_threads(1L) },
  .args = list(core_path = core), .min = n_daemons
)
cat(sprintf("%d daemons up\n", n_daemons))

jobs <- unlist(unlist(lapply(settings, function(st) {
  lapply(all_fips, function(f) {
    lapply(seeds, function(s) {
      list(fips = f, seed = s, config = "layernorm_full",
           kernel = st$kernel, threshold_mult = st$threshold_mult,
           adaptive = st$adaptive, label = st$label)
    })
  })
}), recursive = FALSE), recursive = FALSE)

cat(sprintf("\nStates kernel test: %d fits (%d settings x %d states x %d seeds)\n",
            length(jobs), length(settings), length(all_fips), length(seeds)))

kernel_task <- function(spec) {
  r <- state_kernel_task(spec)
  cbind(label = spec$label, r)
}

t0 <- Sys.time()
res <- do.call(rbind, mirai_map(jobs, kernel_task)[.progress, .stop])
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", nrow(res), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

per_seed <- res |>
  group_by(label, seed) |>
  summarise(rsq_trad = mean(rsq_trad), mae = mean(mae), .groups = "drop")

summ <- per_seed |>
  group_by(label) |>
  summarise(rsq_trad_mean = mean(rsq_trad), rsq_trad_sd = sd(rsq_trad),
            mae_mean = mean(mae), .groups = "drop") |>
  as.data.frame()

cat("\n=== National mean rsq_trad by kernel setting (5 seeds) ===\n")
print(summ[order(-summ$rsq_trad_mean), ], row.names = FALSE, digits = 4)

base_lab <- "uniform"
cat(sprintf("\n=== Paired per-state tests vs %s ===\n", base_lab))
for (lab in setdiff(summ$label, base_lab)) {
  w <- res |>
    filter(label %in% c(base_lab, lab)) |>
    select(stusps, seed, label, rsq_trad) |>
    tidyr::pivot_wider(names_from = label, values_from = rsq_trad)
  tt <- t.test(w[[lab]], w[[base_lab]], paired = TRUE)
  cat(sprintf("%-16s diff=%+.4f  p=%.3f  CI[%+.4f, %+.4f]  better in %d/%d\n",
              lab, unname(tt$estimate), tt$p.value, tt$conf.int[1], tt$conf.int[2],
              sum(w[[lab]] > w[[base_lab]]), nrow(w)))
}

saveRDS(list(folds = res, per_seed = per_seed, summary = summ), "data/states-kernel-test.rds")
cat("\nSaved to data/states-kernel-test.rds\n")

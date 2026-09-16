# Does prediction-level ensembling rescue GraphSAGE + LayerNorm in the
# domain-shift simulation the way it did at KC -> Ames (rsq_trad 0.38 ->
# 0.583)? The single-fit sweep showed both GraphSAGE+LayerNorm and
# XGBoost+lags degrading catastrophically at shift=2/3, with XGBoost actually
# less bad -- opposite of the real-data result, but that sweep never
# ensembled predictions. This checks shift=2 and shift=3 with more seeds and
# genuine prediction averaging.

source("R/domain-shift-sim-core.R")

check_seeds <- 1:15
shifts_to_check <- c(2, 3)
arms_to_check <- c("GraphSAGE + LayerNorm", "XGBoost + lags")

jobs <- unlist(unlist(lapply(shifts_to_check, function(s) {
  lapply(arms_to_check, function(a) lapply(check_seeds, function(sd) list(shift = s, arm = a, seed = sd)))
}), recursive = FALSE), recursive = FALSE)

cat(sprintf("\nDomain-shift ensemble check: %d fits\n", length(jobs)))
daemons(10)
core_path <- normalizePath("R/domain-shift-sim-core.R", mustWork = TRUE)
everywhere({ source(core_path, local = FALSE) }, .args = list(core_path = core_path), .min = 10)

t0 <- Sys.time()
raw <- mirai_map(jobs, domain_shift_task_preds)[.progress, .stop]
daemons(0)
cat(sprintf("\n%d fits in %.1f min\n", length(raw), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

metrics <- do.call(rbind, lapply(raw, `[[`, "metrics"))

results <- list()
for (s in shifts_to_check) {
  for (a in arms_to_check) {
    sub <- Filter(function(r) r$shift == s && r$arm == a, raw)
    truth <- sub[[1]]$truth
    pred_mat <- do.call(rbind, lapply(sub, `[[`, "preds"))
    ens_pred <- colMeans(pred_mat)
    ens_score <- score(data.frame(truth = truth, estimate = ens_pred))
    mean_metric <- metrics[metrics$shift == s & metrics$arm == a, ]
    cat(sprintf("\n--- shift=%s, arm=%s ---\n", s, a))
    cat(sprintf(
      "  mean-of-metrics (n=%d):  rsq_trad=%.3f  mae=%.3f  cal_slope=%.3f\n",
      nrow(mean_metric), mean(mean_metric$rsq_trad), mean(mean_metric$mae), mean(mean_metric$cal_slope)
    ))
    cat(sprintf(
      "  prediction ensemble:     rsq_trad=%.3f  mae=%.3f  cal_slope=%.3f\n",
      ens_score$rsq_trad, ens_score$mae, ens_score$cal_slope
    ))
    results[[paste(s, a)]] <- list(mean_metric = mean_metric, ens_score = ens_score)
  }
}

saveRDS(results, "data/domain-shift-ensemble-check.rds")
cat("\nSaved to data/domain-shift-ensemble-check.rds\n")

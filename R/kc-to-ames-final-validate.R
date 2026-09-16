# Full 30-seed validation of the winning architecture found by the sweep
# waves, plus a prediction-level ensemble (mean of the 30 raw prediction
# vectors, scored once) to see whether ensembling stabilizes further beyond
# whatever averaging the per-seed metrics already does.
#
# Set WINNING_VARIANT / WINNING_LIST below before running -- filled in once
# wave 3 finishes.
#
# Run: Rscript R/kc-to-ames-final-validate.R

source("R/kc-to-ames-core.R")
source("R/kc-to-ames-arch-sweep2-helpers.R")

WINNING_VARIANT <- Sys.getenv("KC_AMES_WINNING_VARIANT", unset = "dropout")
WINNING_LIST <- Sys.getenv("KC_AMES_WINNING_LIST", unset = "variants2")

final_seeds <- 1001:1030

start_daemons(10)
everywhere(source("R/kc-to-ames-arch-sweep2-helpers.R", local = FALSE))

jobs <- lapply(final_seeds, function(s) list(variant = WINNING_VARIANT, seed = s))

cat(sprintf(
  "\nFinal validation: variant '%s' (from %s), %d seeds\n",
  WINNING_VARIANT, WINNING_LIST, length(final_seeds)
))
t0 <- Sys.time()
raw <- mirai_map(
  jobs, final_task,
  .args = list(variant_list = get(WINNING_LIST))
)[.progress, .stop]
daemons(0)

metrics <- do.call(rbind, lapply(raw, `[[`, "metrics"))
pred_mat <- do.call(rbind, lapply(raw, `[[`, "preds"))  # 30 x n_ames

var_ames <- var(ames$y)
metrics$rsq_trad <- 1 - metrics$rmse^2 / var_ames

cat(sprintf("\n%d fits in %.1f min\n", nrow(metrics), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

cat("\n=== Final validation, mean-of-metrics across 30 seeds ===\n")
summ <- metrics |>
  summarise(
    mae = mean(mae), rmse = mean(rmse), rsq = mean(rsq),
    rsq_sd = sd(rsq), rsq_trad_sd = sd(rsq_trad),
    rsq_trad = 1 - mean(rmse)^2 / var_ames,
    bias = mean(bias), cal_slope = mean(cal_slope)
  )
print(summ, digits = 3)

cat("\n=== Prediction-level ensemble (mean of 30 raw prediction vectors, scored once) ===\n")
ensemble_pred <- colMeans(pred_mat)
ens_score <- score(ames$y, ensemble_pred)
print(ens_score, digits = 3)

cat("\n=== Reference: XGBoost + lags (from data/kc-to-ames-results.rds) ===\n")
xgb_ref <- readRDS("data/kc-to-ames-results.rds")$ames_summary
print(xgb_ref[xgb_ref$arm == "XGBoost + lags", c("mae", "rmse", "rsq_trad")], digits = 3)

dir.create("data", showWarnings = FALSE)
saveRDS(
  list(variant = WINNING_VARIANT, metrics = metrics, pred_mat = pred_mat,
       ensemble_pred = ensemble_pred, ensemble_score = ens_score, summary = summ),
  "data/kc-to-ames-final-validation.rds"
)
cat("\nSaved to data/kc-to-ames-final-validation.rds\n")

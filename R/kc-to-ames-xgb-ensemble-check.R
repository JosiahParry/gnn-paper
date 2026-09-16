# Fairness check: does ensembling XGBoost + lags across seeds also improve
# it, the way it did for GraphSAGE + LayerNorm + dropout=0.1 (rsq_trad 0.38 ->
# 0.583)? XGBoost + lags is stochastic too (unseeded internal validation
# split), so if ensembling is "free" for any stochastic arm this wouldn't be
# a genuine GraphSAGE result -- it needs to hold up against an equally
# ensembled competitor.

source("R/kc-to-ames-core.R")

xgb_task <- function(spec) {
  predictor <- arms[["XGBoost + lags"]](final_train, final_val, spec$seed)
  preds <- predictor(seq_len(nrow(ames)), "ames")
  list(metrics = cbind(seed = spec$seed, score(ames$y, preds)), preds = preds)
}

start_daemons(10)
jobs <- lapply(1001:1030, function(s) list(seed = s))

cat("\nXGBoost + lags ensemble check: 30 fits\n")
t0 <- Sys.time()
raw <- mirai_map(jobs, xgb_task)[.progress, .stop]
daemons(0)

metrics <- do.call(rbind, lapply(raw, `[[`, "metrics"))
pred_mat <- do.call(rbind, lapply(raw, `[[`, "preds"))

var_ames <- var(ames$y)
metrics$rsq_trad <- 1 - metrics$rmse^2 / var_ames

cat(sprintf("\n%d fits in %.1f min\n", nrow(metrics), as.numeric(difftime(Sys.time(), t0, units = "mins"))))

cat("\n=== XGBoost + lags, mean-of-metrics across 30 seeds ===\n")
print(
  data.frame(
    mae = mean(metrics$mae), rmse = mean(metrics$rmse),
    rsq_trad = 1 - mean(metrics$rmse)^2 / var_ames,
    bias = mean(metrics$bias), cal_slope = mean(metrics$cal_slope)
  ), digits = 3
)

cat("\n=== XGBoost + lags, prediction-level ensemble ===\n")
ensemble_pred <- colMeans(pred_mat)
print(score(ames$y, ensemble_pred), digits = 3)

cat("\n=== Reference: GraphSAGE + LayerNorm + dropout=0.1 ensemble ===\n")
gs <- readRDS("data/kc-to-ames-final-validation.rds")
print(gs$ensemble_score, digits = 3)

saveRDS(list(metrics = metrics, pred_mat = pred_mat, ensemble_pred = ensemble_pred),
        "data/kc-to-ames-xgb-ensemble-check.rds")
cat("\nSaved to data/kc-to-ames-xgb-ensemble-check.rds\n")

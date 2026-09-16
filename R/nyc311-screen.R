# Pre-compute screen for the NYC 311 Queens <-> Brooklyn pair.
#
# Two cheap questions, asked before spending any GNN compute:
#
#   1. DGP compatibility -- does an OLS fit on one borough still predict the
#      other? If coefficients differ, no amount of normalisation saves the
#      transfer (this is what killed LA->NYC Airbnb and Chicago->Nashville).
#
#   2. Is there anything for a graph to exploit -- is residual spatial
#      autocorrelation large relative to covariate spatial autocorrelation?
#      If covariates already carry the spatial signal, a plain tabular model
#      sees it too and the graph adds nothing (this is what screened out the
#      Chicago/Seattle building-energy pair).

suppressPackageStartupMessages({
  library(dplyr); library(sf); library(sfdep); library(spdep)
})

k_nb <- 30L

cleaned <- readRDS("data/nyc311-clean.rds")
qns <- cleaned$queens; bkn <- cleaned$brooklyn
feature_cols <- setdiff(names(qns), c("lat", "lon", "y"))

cat(sprintf("Queens   n = %d\nBrooklyn n = %d\nfeatures (%d): %s\n\n",
            nrow(qns), nrow(bkn), length(feature_cols),
            paste(feature_cols, collapse = ", ")))

frame <- function(d) d[, c(feature_cols, "y")]

# 1. OLS cross-prediction -------------------------------------------------

rsq_trad <- function(truth, est) 1 - sum((truth - est)^2) / sum((truth - mean(truth))^2)

cross <- function(src, tgt, src_nm, tgt_nm) {
  m <- lm(y ~ ., data = frame(src))
  p_in  <- predict(m, newdata = frame(src))
  p_out <- predict(m, newdata = frame(tgt))
  data.frame(
    fit = src_nm, predict = tgt_nm,
    rsq_in  = rsq_trad(src$y, p_in),
    rsq_out = rsq_trad(tgt$y, p_out),
    bias    = mean(p_out - tgt$y),
    cal_slope = unname(coef(lm(tgt$y ~ p_out))[2])
  )
}

cat("=== OLS cross-prediction (DGP compatibility) ===\n")
ols <- rbind(cross(qns, bkn, "queens", "brooklyn"),
             cross(bkn, qns, "brooklyn", "queens"))
print(ols, row.names = FALSE, digits = 3)
cat("\nPass criterion: rsq_out clearly positive in BOTH directions.\n\n")

# Coefficient comparison: where do the two boroughs actually disagree?
cq <- coef(lm(y ~ ., data = frame(qns)))
cb <- coef(lm(y ~ ., data = frame(bkn)))
cmp <- data.frame(term = names(cq), queens = unname(cq), brooklyn = unname(cb[names(cq)]))
cmp$abs_diff <- abs(cmp$queens - cmp$brooklyn)
cat("=== Largest coefficient disagreements ===\n")
print(head(cmp[order(-cmp$abs_diff), ], 8), row.names = FALSE, digits = 3)

# 2. Moran's I: covariates vs OLS residuals -------------------------------

moran_profile <- function(d, nm) {
  g <- st_as_sf(d, coords = c("lon", "lat"), crs = 4326) |> st_transform(3857)
  nb <- sfdep::st_knn(st_geometry(g), k = k_nb)
  lw <- nb2listw(nb, style = "W")

  mi <- function(v) if (length(unique(v)) < 2) NA_real_ else
    unname(moran.test(v, lw, zero.policy = TRUE)$estimate[1])

  cov_i <- vapply(feature_cols, function(f) mi(d[[f]]), numeric(1))
  y_i   <- mi(d$y)
  res_i <- mi(residuals(lm(y ~ ., data = frame(d))))

  cat(sprintf("\n=== Moran's I (%s, KNN-%d) ===\n", nm, k_nb))
  cat(sprintf("  target y            : %+.3f\n", y_i))
  cat(sprintf("  OLS residuals       : %+.3f   <- what a graph could still exploit\n", res_i))
  cat(sprintf("  covariates: median %+.3f, max %+.3f (%s)\n",
              median(cov_i, na.rm = TRUE), max(cov_i, na.rm = TRUE),
              names(which.max(cov_i))))
  print(round(sort(cov_i, decreasing = TRUE), 3))
  data.frame(region = nm, y = y_i, resid = res_i,
             cov_med = median(cov_i, na.rm = TRUE), cov_max = max(cov_i, na.rm = TRUE))
}

mor <- rbind(moran_profile(qns, "queens"), moran_profile(bkn, "brooklyn"))

cat("\n=== Verdict ===\n")
print(mor, row.names = FALSE, digits = 3)
cat("\nA graph is worth trying when residual I is substantial AND not already\n",
    "dominated by covariate I. Residual I below ~0.05 means there is no\n",
    "spatial signal left over for neighbours to supply.\n", sep = "")

saveRDS(list(ols = ols, coefs = cmp, moran = mor), "data/nyc311-screen.rds")
cat("\nSaved to data/nyc311-screen.rds\n")

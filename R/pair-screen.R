# Generic pre-compute screen for a (dataset, source, target) triple.
# Generalises R/nyc311-screen.R so any new pair gets the same two checks:
#
#   1. DGP compatibility -- does OLS fit on one region still predict the
#      other? A pair that fails measures the pair, not the architecture.
#   2. Is there spatial signal left for a graph -- Moran's I of the OLS
#      residuals, against Moran's I of the covariates.
#
# Both are reported for the SOURCE region (available at deployment time, no
# target labels) and the target region (oracle), because section 2a of
# REPORT.md claims the source-only version is sufficient.
#
# Usage: Rscript R/pair-screen.R <data.rds> <source> <target>

suppressPackageStartupMessages({
  library(dplyr); library(sf); library(sfdep); library(spdep)
})

cli <- commandArgs(trailingOnly = TRUE)
stopifnot(length(cli) >= 3)
DATA <- cli[1]; SRC <- cli[2]; TGT <- cli[3]
k_nb <- 30L

cleaned <- readRDS(DATA)
src <- cleaned[[SRC]]; tgt <- cleaned[[TGT]]
feature_cols <- setdiff(intersect(names(src), names(tgt)), c("boro", "lat", "lon", "y"))

cat(sprintf("%s  n = %d\n%s  n = %d\nfeatures (%d): %s\n\n",
            SRC, nrow(src), TGT, nrow(tgt), length(feature_cols),
            paste(feature_cols, collapse = ", ")))

frame <- function(d) d[, c(feature_cols, "y")]
rsq_trad <- function(truth, est) 1 - sum((truth - est)^2) / sum((truth - mean(truth))^2)

cross <- function(a, b, an, bn) {
  m <- lm(y ~ ., data = frame(a))
  p_out <- predict(m, newdata = frame(b))
  data.frame(fit = an, predict = bn,
             rsq_in = rsq_trad(a$y, predict(m, newdata = frame(a))),
             rsq_out = rsq_trad(b$y, p_out),
             bias = mean(p_out - b$y),
             cal_slope = unname(coef(lm(b$y ~ p_out))[2]))
}

cat("=== OLS cross-prediction (DGP compatibility) ===\n")
ols <- rbind(cross(src, tgt, SRC, TGT), cross(tgt, src, TGT, SRC))
print(ols, row.names = FALSE, digits = 3)

listw_of <- function(d) {
  g <- st_geometry(st_as_sf(d, coords = c("lon", "lat"), crs = 4326) |> st_transform(3857))
  nb2listw(sfdep::st_knn(g, k = k_nb), style = "W")
}
mi <- function(v, lw) if (length(unique(v)) < 2) NA_real_ else
  unname(moran.test(v, lw, zero.policy = TRUE)$estimate[1])

lw_s <- listw_of(src); lw_t <- listw_of(tgt)
m <- lm(y ~ ., data = frame(src))
res_s <- mi(residuals(m), lw_s)
res_t <- mi(tgt$y - predict(m, newdata = frame(tgt)), lw_t)
cov_s <- vapply(feature_cols, function(f) mi(src[[f]], lw_s), numeric(1))
cov_t <- vapply(feature_cols, function(f) mi(tgt[[f]], lw_t), numeric(1))

cat(sprintf("\n=== Moran's I (KNN-%d) ===\n", k_nb))
cat(sprintf("%-10s target y %+.3f | OLS residuals %+.3f | covariates med %+.3f max %+.3f | frac exceeding %.2f\n",
            SRC, mi(src$y, lw_s), res_s, median(cov_s, na.rm = TRUE), max(cov_s, na.rm = TRUE),
            mean(cov_s > res_s, na.rm = TRUE)))
cat(sprintf("%-10s target y %+.3f | OLS residuals %+.3f | covariates med %+.3f max %+.3f | frac exceeding %.2f\n",
            TGT, mi(tgt$y, lw_t), res_t, median(cov_t, na.rm = TRUE), max(cov_t, na.rm = TRUE),
            mean(cov_t > res_t, na.rm = TRUE)))

cat("\ncovariate Moran's I, source region:\n")
print(round(sort(cov_s, decreasing = TRUE), 3))

cat("\n=== Section 2a rule applied ===\n")
cat(sprintf("source residual Moran's I = %.3f\n", res_s))
cat(sprintf("  wins so far sit at 0.221-0.338; losses at 0.116-0.214\n"))
cat(sprintf("  -> rule predicts: %s\n",
            if (res_s > 0.22) "GraphSAGE WINS" else if (res_s < 0.15) "GraphSAGE LOSES" else "ambiguous"))

saveRDS(list(ols = ols, res_src = res_s, res_tgt = res_t,
             cov_src = cov_s, cov_tgt = cov_t),
        sprintf("data/screen-%s-%s.rds", SRC, TGT))
cat(sprintf("\nSaved to data/screen-%s-%s.rds\n", SRC, TGT))

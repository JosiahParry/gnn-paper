# Calibrate the pre-compute diagnostic across every transfer pair with a
# known outcome.
#
# Section 2 of REPORT.md states the rule qualitatively: "a graph helps when
# residual spatial signal is large relative to what the covariates already
# carry." Applied by eye it just failed a recorded ex-ante prediction on the
# street-tree transfer (predicted a modest win, GraphSAGE lost to OLS by
# 0.144). So the rule needs either a number or a retraction.
#
# Two questions here:
#
#   1. Is there a single statistic over the Moran profile that separates the
#      wins from the losses across all pairs?
#
#   2. Can it be computed WITHOUT target labels? The rule as written uses
#      residuals in the target region, but in a genuine inductive deployment
#      the target's labels are exactly what you do not have. If the rule only
#      works with oracle residuals, it is not a deployment-time screen and
#      the paper must say so.
#
# Cheap: Moran's I only, no model fitting beyond OLS.

suppressPackageStartupMessages({
  library(dplyr); library(sf); library(sfdep); library(spdep)
})

k_nb <- 30L

as_proj <- function(d) st_as_sf(d, coords = c("lon", "lat"), crs = 4326) |> st_transform(3857)

listw_of <- function(d) {
  g <- st_geometry(as_proj(d))
  nb2listw(sfdep::st_knn(g, k = k_nb), style = "W")
}

mi <- function(v, lw) {
  if (length(unique(v)) < 2) return(NA_real_)
  unname(moran.test(v, lw, zero.policy = TRUE)$estimate[1])
}

# Pairs with a known outcome ----------------------------------------------
# outcome = margin of GraphSAGE+LayerNorm over the best non-graph arm.

ab <- readRDS("data/airbnb-all-cities-clean.rds")
tr <- readRDS("data/trees-clean.rds")
n311 <- readRDS("data/nyc311-clean.rds")

drop_cols <- c("boro", "price", "y")
pairs <- list(
  list(nm = "Broward -> San Diego (Airbnb price)",   src = ab$broward,  tgt = ab$sandiego,  outcome = "WIN  +0.030"),
  list(nm = "Chicago -> LA (Airbnb price)",          src = ab$chicago,  tgt = ab$la,        outcome = "WIN  +0.022"),
  list(nm = "LA -> NYC (Airbnb price)",              src = ab$la,       tgt = ab$nyc,       outcome = "FAIL (all arms)"),
  list(nm = "Chicago -> Nashville (Airbnb price)",   src = ab$chicago,  tgt = ab$nashville, outcome = "FAIL (all arms)"),
  list(nm = "Queens -> Brooklyn (tree DBH)",         src = tr$queens,   tgt = tr$brooklyn,  outcome = "LOSS -0.144"),
  list(nm = "Queens -> Brooklyn (311 response)",     src = n311$queens, tgt = n311$brooklyn, outcome = "pending")
)

profile_pair <- function(p) {
  feats <- setdiff(intersect(names(p$src), names(p$tgt)), c(drop_cols, "lat", "lon"))
  fs <- p$src[, c(feats, "y")]; ft <- p$tgt[, c(feats, "y")]

  lw_s <- listw_of(p$src); lw_t <- listw_of(p$tgt)
  m <- lm(y ~ ., data = fs)

  res_s <- residuals(m)                                   # deployment-time
  res_t <- ft$y - predict(m, newdata = ft)                # oracle (needs labels)

  i_res_s <- mi(res_s, lw_s)
  i_res_t <- mi(res_t, lw_t)
  cov_s <- vapply(feats, function(f) mi(p$src[[f]], lw_s), numeric(1))
  cov_t <- vapply(feats, function(f) mi(p$tgt[[f]], lw_t), numeric(1))

  data.frame(
    pair = p$nm, outcome = p$outcome, n_feat = length(feats),
    # deployment-time (source region only)
    res_src = i_res_s,
    cov_src_med = median(cov_s, na.rm = TRUE),
    frac_src_exceed = mean(cov_s > i_res_s, na.rm = TRUE),
    ratio_src = i_res_s / median(cov_s, na.rm = TRUE),
    # oracle (uses target labels)
    res_tgt = i_res_t,
    cov_tgt_med = median(cov_t, na.rm = TRUE),
    frac_tgt_exceed = mean(cov_t > i_res_t, na.rm = TRUE),
    ratio_tgt = i_res_t / median(cov_t, na.rm = TRUE)
  )
}

out <- do.call(rbind, lapply(pairs, function(p) {
  cat(sprintf("profiling %s ...\n", p$nm)); profile_pair(p)
}))

cat("\n=== Deployment-time statistics (source region only, no target labels) ===\n")
print(out[, c("pair", "outcome", "res_src", "cov_src_med", "frac_src_exceed", "ratio_src")],
      row.names = FALSE, digits = 3)

cat("\n=== Oracle statistics (residuals evaluated in the target region) ===\n")
print(out[, c("pair", "outcome", "res_tgt", "cov_tgt_med", "frac_tgt_exceed", "ratio_tgt")],
      row.names = FALSE, digits = 3)

cat("\nratio = residual Moran's I / median covariate Moran's I.\n",
    "frac_exceed = fraction of covariates whose own I exceeds the residual I.\n",
    "A usable rule must separate WIN from LOSS using the deployment-time\n",
    "columns alone -- the oracle columns are not available when it matters.\n", sep = "")

saveRDS(out, "data/moran-calibrate.rds")
cat("\nSaved to data/moran-calibrate.rds\n")

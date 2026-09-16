# Can transferability be predicted WITHOUT the target region's answers?
#
# Our current transferability screen fits OLS on X, predicts Y, and scores
# against Y's true values. That uses exactly the information a deployed model
# does not have, so it is a benchmarking tool, not a deployment tool.
#
# This tests three label-free signals against the pairs where the answer is
# already known. Two use only the columns; one additionally uses the SOURCE
# region's answers, which we always have.
#
#   auc        - domain classifier. Can a model tell X rows from Y rows using
#                covariates alone? 0.5 = the regions look alike; 1.0 = disjoint
#                support, i.e. the model would be extrapolating.
#   coverage   - share of Y rows inside X's per-column 1st-99th percentile range.
#   beta_cv    - within-source coefficient instability. Split X into spatial
#                blocks, fit OLS in each, and measure how much the standardised
#                coefficients move. High = the relationship is already unstable
#                inside X, so expecting it to hold in Y is optimistic.
#
# Ground truth is `transfers`: does OLS fit on X achieve positive R2 on Y?
# That is the thing we are trying to predict without looking at it.
#
# EXPECTED LIMIT, stated before running: all three of our known failures are
# relationship shifts (same inputs, different meaning), and no statistic on
# inputs alone can see those. auc and coverage should therefore FAIL to
# separate. beta_cv is the only one with a mechanism for catching them, and
# only indirectly. If nothing separates, that is the finding.

suppressPackageStartupMessages({library(dplyr); library(sf)})

set.seed(1)

rsq_trad <- function(truth, est) 1 - sum((truth - est)^2) / sum((truth - mean(truth))^2)

# --- signals -------------------------------------------------------------

# Domain classifier AUC via logistic regression on standardised covariates.
domain_auc <- function(Xs, Xt) {
  n <- min(nrow(Xs), nrow(Xt), 4000)
  a <- Xs[sample(nrow(Xs), n), , drop = FALSE]
  b <- Xt[sample(nrow(Xt), n), , drop = FALSE]
  d <- as.data.frame(rbind(a, b))
  mu <- colMeans(d); sdv <- apply(d, 2, sd); sdv[sdv == 0] <- 1
  d <- as.data.frame(scale(d, mu, sdv))
  d$lab <- rep(0:1, each = n)
  # half for fitting, half for scoring, so AUC is out-of-sample
  i <- sample(nrow(d), floor(nrow(d) / 2))
  m <- suppressWarnings(glm(lab ~ ., data = d[i, ], family = binomial()))
  p <- suppressWarnings(predict(m, newdata = d[-i, ], type = "response"))
  y <- d$lab[-i]
  r <- rank(p)
  (sum(r[y == 1]) - sum(y == 1) * (sum(y == 1) + 1) / 2) / (sum(y == 1) * sum(y == 0))
}

# Share of target rows inside the source's 1st-99th percentile box.
coverage <- function(Xs, Xt) {
  lo <- apply(Xs, 2, quantile, 0.01, na.rm = TRUE)
  hi <- apply(Xs, 2, quantile, 0.99, na.rm = TRUE)
  mean(apply(Xt, 1, function(r) all(r >= lo & r <= hi)))
}

# Within-source coefficient instability: spatial blocks via a k-means split on
# coordinates, OLS in each, then the median across coefficients of
# (sd across blocks / sd of that coefficient's own scale).
beta_instability <- function(X, y, coords, nblocks = 8) {
  ok <- stats::complete.cases(X, y)
  X <- X[ok, , drop = FALSE]; y <- y[ok]; coords <- coords[ok, , drop = FALSE]
  km <- kmeans(scale(coords), centers = nblocks, nstart = 5, iter.max = 50)
  Z <- scale(X)                                   # standardised, so coefficients are comparable
  Z[!is.finite(Z)] <- 0
  d <- as.data.frame(Z); d$y <- y
  B <- do.call(rbind, lapply(seq_len(nblocks), function(k) {
    idx <- which(km$cluster == k)
    if (length(idx) < 10 * ncol(X)) return(NULL)
    coef(lm(y ~ ., data = d[idx, ]))[-1]
  }))
  if (is.null(B) || nrow(B) < 3) return(NA_real_)
  # Scale-free: spread across blocks relative to the typical magnitude.
  s <- apply(B, 2, sd, na.rm = TRUE)
  m <- apply(abs(B), 2, median, na.rm = TRUE)
  median(s / pmax(m, 1e-8), na.rm = TRUE)
}

# --- assemble the pairs --------------------------------------------------

mk <- function(nm, src, tgt, feats, coordcols = c("lon", "lat")) {
  list(nm = nm, src = src, tgt = tgt, feats = feats, cc = coordcols)
}

ab <- readRDS("data/airbnb-all-cities-clean.rds")
abf <- setdiff(names(ab$la), c("lat", "lon", "price", "y"))
tr <- readRDS("data/trees-clean.rds")
trf <- setdiff(names(tr$queens), c("boro", "lat", "lon", "y"))
n311 <- readRDS("data/nyc311-clean.rds")
n3f <- setdiff(names(n311$queens), c("lat", "lon", "y"))
pl <- readRDS("data/pluto-clean.rds")
plf <- setdiff(names(pl$queens), c("lat", "lon", "y"))
rv <- readRDS("data/airbnb-reviews-clean.rds")
rvf <- setdiff(names(rv$broward), c("lat", "lon", "y"))
av <- readRDS("data/airbnb-avail-clean.rds")
avf <- setdiff(names(av$broward), c("lat", "lon", "y"))

pairs <- list(
  mk("Broward -> San Diego (price)",   ab$broward,  ab$sandiego,  abf),
  mk("Chicago -> LA (price)",          ab$chicago,  ab$la,        abf),
  mk("Queens -> Brooklyn (trees)",     tr$queens,   tr$brooklyn,  trf),
  mk("Queens -> Brooklyn (311)",       n311$queens, n311$brooklyn, n3f),
  mk("Broward -> San Diego (reviews)", rv$broward,  rv$sandiego,  rvf),
  mk("LA -> NYC (price)",              ab$la,       ab$nyc,       abf),
  mk("Chicago -> Nashville (price)",   ab$chicago,  ab$nashville, abf),
  mk("Queens -> Brooklyn (bldg age)",  pl$queens,   pl$brooklyn,  plf),
  mk("Bronx -> Brooklyn (bldg age)",   pl$bronx,    pl$brooklyn,  plf),
  mk("Broward -> San Diego (avail)",   av$broward,  av$sandiego,  avf)
)

run_pair <- function(p) {
  cat(sprintf("  %s\n", p$nm))
  Xs <- as.matrix(p$src[, p$feats]); Xt <- as.matrix(p$tgt[, p$feats])
  ys <- p$src$y; yt <- p$tgt$y
  # ground truth (uses target labels -- this is what we are trying to predict)
  m <- lm(y ~ ., data = p$src[, c(p$feats, "y")])
  truth <- rsq_trad(yt, predict(m, newdata = p$tgt[, c(p$feats, "y")]))

  data.frame(
    pair = p$nm,
    ols_out = truth,
    transfers = truth > 0.05,
    auc = domain_auc(Xs, Xt),
    coverage = coverage(Xs, Xt),
    beta_cv = beta_instability(Xs, ys, as.matrix(p$src[, p$cc]))
  )
}

cat("Computing label-free signals...\n")
res <- do.call(rbind, lapply(pairs, run_pair))
res <- res[order(-res$ols_out), ]

cat("\n=== Label-free signals vs the ground truth they must predict ===\n")
print(res, row.names = FALSE, digits = 3)

cat("\n=== Separation (ground truth: does OLS transfer at all?) ===\n")
for (v in c("auc", "coverage", "beta_cv")) {
  a <- res[[v]][res$transfers]; b <- res[[v]][!res$transfers]
  a <- a[is.finite(a)]; b <- b[is.finite(b)]
  verdict <- if (min(a) > max(b)) "SEPARATES (higher = transfers)" else
             if (max(a) < min(b)) "SEPARATES (lower = transfers)" else "OVERLAPS"
  cat(sprintf("%-9s transfers [%.3f, %.3f]   fails [%.3f, %.3f]   %s\n",
              v, min(a), max(a), min(b), max(b), verdict))
}

cat("\nauc      ~0.5 means the two regions' inputs look alike; ~1.0 means disjoint.\n")
cat("coverage share of target rows inside the source's input range.\n")
cat("beta_cv  how much the relationship already moves WITHIN the source region.\n")

saveRDS(res, "data/transfer-screen-labelfree.rds")
cat("\nSaved to data/transfer-screen-labelfree.rds\n")

# Which NYC borough pair can carry the building-age transfer?
#
# Queens <-> Brooklyn fails the DGP screen outright (out-of-region R2 is
# negative both ways, ~9 years of bias) because the two boroughs were
# developed in different eras. Other pairs may not: this ranks all 20 ordered
# pairs by OLS cross-prediction before any model compute is spent, exactly as
# R/airbnb-screen-pairs.R does for the Airbnb cities.
#
# Cheap: OLS only, no Moran, no graph.

suppressPackageStartupMessages(library(dplyr))

d <- readRDS("data/pluto-clean.rds")
boros <- names(d)
feature_cols <- setdiff(names(d[[1]]), c("lat", "lon", "y"))
frame <- function(x) x[, c(feature_cols, "y")]
rsq_trad <- function(truth, est) 1 - sum((truth - est)^2) / sum((truth - mean(truth))^2)

fits <- lapply(d, function(x) lm(y ~ ., data = frame(x)))

res <- do.call(rbind, lapply(boros, function(a) {
  do.call(rbind, lapply(setdiff(boros, a), function(b) {
    p <- predict(fits[[a]], newdata = frame(d[[b]]))
    data.frame(fit = a, predict = b,
               rsq_in = rsq_trad(d[[a]]$y, fitted(fits[[a]])),
               rsq_out = rsq_trad(d[[b]]$y, p),
               bias = mean(p - d[[b]]$y),
               cal_slope = unname(coef(lm(d[[b]]$y ~ p))[2]))
  }))
}))

cat("=== All ordered borough pairs, ranked by out-of-region R2 ===\n")
print(res[order(-res$rsq_out), ], row.names = FALSE, digits = 3)

# A usable pair needs BOTH directions positive, as in every other transfer
# in this project.
sym <- do.call(rbind, lapply(seq_len(nrow(res)), function(i) {
  j <- which(res$fit == res$predict[i] & res$predict == res$fit[i])
  if (length(j) != 1 || res$fit[i] > res$predict[i]) return(NULL)
  data.frame(a = res$fit[i], b = res$predict[i],
             rsq_ab = res$rsq_out[i], rsq_ba = res$rsq_out[j],
             worst = min(res$rsq_out[i], res$rsq_out[j]),
             bias_ab = res$bias[i], bias_ba = res$bias[j])
}))
cat("\n=== Symmetric pairs, ranked by the WORSE direction ===\n")
print(sym[order(-sym$worst), ], row.names = FALSE, digits = 3)

best <- sym[which.max(sym$worst), ]
cat(sprintf("\nBest candidate: %s <-> %s (worse direction R2 = %.3f)\n",
            best$a, best$b, best$worst))
cat(if (best$worst > 0.05)
      "-> passes the DGP screen; proceed to the full screen and transfer.\n"
    else
      "-> NO borough pair passes. Building age is not transferable across NYC boroughs.\n")

saveRDS(list(all = res, sym = sym), "data/pluto-pair-screen.rds")

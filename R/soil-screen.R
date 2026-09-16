# Pre-fit screen for the soil-carbon raster transfer.
#
# Same two checks as every other pair, but on grid geometry, and with the
# neighbour definition varied -- on a raster there is a natural one (the
# touching cells) and we have never checked whether our KNN-30 default was
# the right choice.
#
# Coordinates are metres in an equal-area CRS, so distances are plain
# Euclidean; crs = NA keeps sf from treating metres as degrees.

suppressPackageStartupMessages({
  library(dplyr); library(sf); library(sfdep); library(spdep)
})

# Configurable so v1 (5 km climate grid) and v2 (1 km terrain grid) share one
# validated code path.
# Usage: Rscript R/soil-screen.R [data.rds] [source] [target] [cell_metres]
cli <- commandArgs(trailingOnly = TRUE)
DATA <- if (length(cli) >= 1) cli[1] else "data/soil-clean.rds"
SRC  <- if (length(cli) >= 2) cli[2] else "cornbelt"
TGT  <- if (length(cli) >= 3) cli[3] else "ohiovalley"
CELL <- if (length(cli) >= 4) as.numeric(cli[4]) else 5000

d <- readRDS(DATA)
src <- d[[SRC]]; tgt <- d[[TGT]]
feature_cols <- setdiff(names(src), c("lon", "lat", "y"))

cat(sprintf("%s n = %d\n%s n = %d\nfeatures (%d): %s\n\n", SRC, nrow(src), TGT, nrow(tgt),
            length(feature_cols), paste(feature_cols, collapse = ", ")))

frame <- function(x) x[, c(feature_cols, "y")]
rsq_trad <- function(truth, est) 1 - sum((truth - est)^2) / sum((truth - mean(truth))^2)

cross <- function(a, b, an, bn) {
  m <- lm(y ~ ., data = frame(a))
  p <- predict(m, newdata = frame(b))
  data.frame(fit = an, predict = bn,
             rsq_in = rsq_trad(a$y, fitted(m)), rsq_out = rsq_trad(b$y, p),
             bias = mean(p - b$y), cal_slope = unname(coef(lm(b$y ~ p))[2]))
}

cat("=== OLS cross-prediction (compatibility of the two regions) ===\n")
print(rbind(cross(src, tgt, SRC, TGT), cross(tgt, src, TGT, SRC)),
      row.names = FALSE, digits = 3)

geom_of <- function(x) st_geometry(st_as_sf(x, coords = c("lon", "lat"), crs = NA))

nb_of <- function(x, graph) {
  g <- geom_of(x)
  if (graph == "knn30") return(sfdep::st_knn(g, k = 30))
  r <- c(queen = 1.45, queen2 = 2.9)[[graph]]
  nb <- spdep::dnearneigh(st_coordinates(g), 0, CELL * r)
  for (i in which(vapply(nb, function(z) identical(z, 0L), logical(1)))) nb[[i]] <- i
  class(nb) <- "nb"; nb
}

mi <- function(v, lw) if (length(unique(v)) < 2) NA_real_ else
  unname(moran.test(v, lw, zero.policy = TRUE)$estimate[1])

cat("\n=== Moran's I by neighbour definition ===\n")
prof <- do.call(rbind, lapply(c("queen", "queen2", "knn30"), function(gr) {
  lw_s <- nb2listw(nb_of(src, gr), style = "W", zero.policy = TRUE)
  m <- lm(y ~ ., data = frame(src))
  res_s <- mi(residuals(m), lw_s)
  cov_s <- vapply(feature_cols, function(f) mi(src[[f]], lw_s), numeric(1))
  cat(sprintf("%-7s avg neighbours %.1f | residual I %+.3f | covariates med %+.3f max %+.3f | frac exceeding %.2f\n",
              gr, mean(lengths(nb_of(src, gr))), res_s,
              median(cov_s, na.rm = TRUE), max(cov_s, na.rm = TRUE),
              mean(cov_s > res_s, na.rm = TRUE)))
  data.frame(graph = gr, res_src = res_s, cov_med = median(cov_s, na.rm = TRUE),
             frac_exceed = mean(cov_s > res_s, na.rm = TRUE))
}))

cat("\ncovariate Moran's I (queen, source):\n")
lw_q <- nb2listw(nb_of(src, "queen"), style = "W", zero.policy = TRUE)
print(round(sort(vapply(feature_cols, function(f) mi(src[[f]], lw_q), numeric(1)),
                 decreasing = TRUE), 3))

cat("\n=== Rule from section 2a ===\n")
f <- prof$frac_exceed[prof$graph == "queen"]
cat(sprintf("frac exceeding (queen) = %.2f\n", f))
cat(sprintf("  wins so far <= 0.25; losses >= 0.33\n  -> predicts: %s\n",
            if (f <= 0.25) "GraphSAGE WINS" else if (f >= 0.33) "GraphSAGE LOSES" else "in the gap"))

outf <- sprintf("data/screen-%s-%s.rds", SRC, TGT)
saveRDS(prof, outf)
cat(sprintf("\nSaved to %s\n", outf))

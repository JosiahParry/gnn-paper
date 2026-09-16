# Pre-fit screens. Rscript R/screen.R <data.rds> <src> <tgt> [crs] [graph] [cell]
#
# 1. Will anything transfer? Fit on each region, predict the other.
#    A pair that fails this measures the pair, not the method.
# 2. Is there spatial structure left over, and do neighbouring covariates
#    explain it? Neither statistic predicted the outcome reliably -- see
#    FINDINGS.md -- but both are cheap and worth looking at.

suppressPackageStartupMessages({library(dplyr); library(sf); library(sfdep); library(spdep)})

a <- commandArgs(trailingOnly = TRUE)
stopifnot(length(a) >= 3)
arg <- function(i, d) if (length(a) >= i && nzchar(a[i])) a[i] else d
DATA <- a[1]; SRC <- a[2]; TGT <- a[3]
CRS <- if (arg(4, "4326") == "NA") NA else as.numeric(arg(4, "4326"))
GRAPH <- arg(5, "knn30"); CELL <- as.numeric(arg(6, "60"))

d <- readRDS(DATA)
src <- d[[SRC]]; tgt <- d[[TGT]]
drop_cols <- c("y", "lulc_raw", "boro", "price", "soc", "lst")
feats <- setdiff(intersect(names(src), names(tgt)), c(drop_cols, "lon", "lat"))
feats <- feats[vapply(src[feats], is.numeric, logical(1))]

cat(sprintf("%s n=%d, %s n=%d, %d features\n\n", SRC, nrow(src), TGT, nrow(tgt), length(feats)))

fr <- function(x) x[, c(feats, "y")]
r2 <- function(t, e) 1 - sum((t - e)^2) / sum((t - mean(t))^2)

cross <- function(x, y, xn, yn) {
  m <- lm(y ~ ., data = fr(x))
  p <- predict(m, newdata = fr(y))
  data.frame(fit = xn, predict = yn, rsq_in = summary(m)$r.squared, rsq_out = r2(y$y, p),
             bias = mean(p - y$y), cal_slope = unname(coef(lm(y$y ~ p))[2]))
}
cat("=== will anything transfer ===\n")
print(rbind(cross(src, tgt, SRC, TGT), cross(tgt, src, TGT, SRC)), row.names = FALSE, digits = 3)

geom <- function(x) {
  g <- st_as_sf(x, coords = c("lon", "lat"), crs = CRS)
  st_geometry(if (!is.na(CRS) && CRS == 4326) st_transform(g, 3857) else g)
}
nb_of <- function(x) {
  g <- geom(x)
  if (GRAPH == "knn30") return(sfdep::st_knn(g, k = 30))
  r <- c(queen = 1.45, queen2 = 2.9)[[GRAPH]]
  nb <- spdep::dnearneigh(st_coordinates(g), 0, CELL * r)
  for (i in which(vapply(nb, function(z) identical(z, 0L), logical(1)))) nb[[i]] <- i
  `class<-`(nb, "nb")
}
mi <- function(v, lw) if (length(unique(v)) < 2) NA_real_ else
  unname(moran.test(v, lw, zero.policy = TRUE)$estimate[1])

lw <- nb2listw(nb_of(src), style = "W", zero.policy = TRUE)
m <- lm(y ~ ., data = fr(src))
res <- residuals(m)
cov_i <- vapply(feats, function(f) mi(src[[f]], lw), numeric(1))
lagX <- as.data.frame(lag.listw(lw, as.matrix(src[, feats]), zero.policy = TRUE))
names(lagX) <- paste0("lag_", feats); lagX$r <- res

cat(sprintf("\n=== spatial structure in the source (%s) ===\n", GRAPH))
cat(sprintf("residual Moran's I        %+.3f\n", mi(res, lw)))
cat(sprintf("covariate Moran's I       median %+.3f  max %+.3f\n",
            median(cov_i, na.rm = TRUE), max(cov_i, na.rm = TRUE)))
cat(sprintf("covariates above residual %.2f\n", mean(cov_i > mi(res, lw), na.rm = TRUE)))
cat(sprintf("residual explained by neighbouring covariates  %.3f\n",
            summary(lm(r ~ ., data = lagX))$r.squared))

saveRDS(list(cov_i = cov_i, resid_i = mi(res, lw),
             lag_r2 = summary(lm(r ~ ., data = lagX))$r.squared),
        sprintf("data/screen-%s-%s.rds", SRC, TGT))

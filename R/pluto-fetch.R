# NYC PLUTO building age, all five boroughs (pair chosen by the DGP screen).
#
# THE DECISIVE TEST. After six transfer pairs, "GraphSAGE wins" is perfectly
# confounded with "the target is a price": all three wins are prices, all
# three losses are not. Section 2a of REPORT.md proposes that what actually
# matters is the source region's residual Moran's I, not the target's name.
# Those two explanations make opposite predictions on exactly one kind of
# problem: a NON-price target with HIGH residual spatial autocorrelation.
#
# Building age is that problem. Year built is not a price, and it is about as
# spatially clustered as an urban variable gets -- city blocks are developed
# in cohorts, so neighbouring lots share a construction era -- while the
# physical covariates available here (lot area, floor area, units, land use)
# explain it only weakly. That combination is what should leave a large
# residual spatial signal for a graph to supply.
#
# Predictions are recorded in reports/predictions.md BEFORE the run:
#   - if GraphSAGE wins here, the rule is about spatial signal
#   - if GraphSAGE loses here, the rule is really about prices, and the
#     six-point threshold in section 2a is a coincidence
#
# Same region pair as the tree and 311 tests, so geometry is held constant
# across all three non-price targets.

suppressPackageStartupMessages({library(dplyr); library(jsonlite)})

fetch_paged <- function(base, where, select, page = 50000) {
  offset <- 0; out <- list()
  repeat {
    url <- sprintf("%s?$select=%s&$where=%s&$limit=%d&$offset=%d",
                   base, select, where, page, offset)
    d <- fromJSON(URLencode(url))
    if (length(d) == 0 || nrow(d) == 0) break
    out[[length(out) + 1]] <- d
    cat(sprintf("  ...%d\n", offset + nrow(d)))
    if (nrow(d) < page) break
    offset <- offset + page
  }
  bind_rows(out)
}

sel <- paste(c("borough", "lotarea", "bldgarea", "comarea", "resarea",
               "numbldgs", "numfloors", "unitsres", "unitstotal", "yearbuilt",
               "landuse", "latitude", "longitude"), collapse = ",")
whr <- paste0("borough in('QN','BK','BX','MN','SI') AND yearbuilt>1850 AND yearbuilt<2024 ",
              "AND latitude IS NOT NULL AND lotarea>0 AND bldgarea>0")

cat("Fetching NYC PLUTO (all boroughs)...\n")
raw <- fetch_paged("https://data.cityofnewyork.us/resource/64uk-42ks.json", whr, sel)
cat(sprintf("%d raw rows\n", nrow(raw)))

num <- function(x) suppressWarnings(as.numeric(x))

d <- raw |>
  transmute(
    boro = c(QN = "queens", BK = "brooklyn", BX = "bronx",
             MN = "manhattan", SI = "statenisland")[borough],
    lat = num(latitude), lon = num(longitude),
    yearbuilt = num(yearbuilt),
    lotarea = num(lotarea), bldgarea = num(bldgarea),
    comarea = num(comarea), resarea = num(resarea),
    numbldgs = num(numbldgs), numfloors = num(numfloors),
    unitsres = num(unitsres), unitstotal = num(unitstotal),
    landuse = landuse
  ) |>
  filter(!is.na(lat), !is.na(lon), !is.na(yearbuilt), !is.na(bldgarea),
         !is.na(numfloors), numfloors > 0, numfloors < 100)

cat(sprintf("%d rows after filters\n", nrow(d)))

# Land-use dummies present in EVERY borough, so the feature space is shared
# whichever pair is eventually used as source and target.
lu_keep <- names(which(tapply(d$landuse, d$landuse, length) > 2000))
for (b in unique(d$boro)) lu_keep <- intersect(lu_keep, unique(d$landuse[d$boro == b]))
cat(sprintf("land-use classes kept: %s\n", paste(lu_keep, collapse = ", ")))

feat <- data.frame(
  log_lotarea  = log1p(d$lotarea),
  log_bldgarea = log1p(d$bldgarea),
  log_comarea  = log1p(d$comarea),
  log_resarea  = log1p(d$resarea),
  numbldgs     = d$numbldgs,
  numfloors    = d$numfloors,
  unitsres     = d$unitsres,
  unitstotal   = d$unitstotal,
  far          = d$bldgarea / d$lotarea
)
for (l in lu_keep) feat[[paste0("lu_", l)]] <- as.integer(d$landuse == l)

out <- bind_cols(data.frame(boro = d$boro, lat = d$lat, lon = d$lon), feat)
out$y_raw <- d$yearbuilt

# Drop non-finite rows: comarea/resarea are missing on a few hundred lots and
# landuse is occasionally blank. lm() silently drops these, which then makes
# residuals() shorter than the spatial weights and breaks moran.test.
ok <- Reduce(`&`, lapply(out[, setdiff(names(out), "boro")], is.finite))
cat(sprintf("dropping %d rows with non-finite values\n", sum(!ok)))
out <- out[ok, ]

N_KEEP <- 25000L
split_boro <- function(nm) {
  o <- out[out$boro == nm, ]
  # Demean within borough and scale to decades so the target is on a sane
  # numeric range; this is a linear transform, so R2 is unaffected.
  o$y <- (o$y_raw - mean(o$y_raw)) / 10
  o$y_raw <- NULL; o$boro <- NULL
  # Density-preserving subsample (the N closest to the borough centroid), as
  # in the tree and 311 datasets: a random draw would thin the graph and
  # dilute exactly the spatial structure under test.
  if (nrow(o) > N_KEEP) {
    cx <- median(o$lon); cy <- median(o$lat)
    d2 <- ((o$lon - cx) * cos(cy * pi / 180))^2 + (o$lat - cy)^2
    o <- o[order(d2)[seq_len(N_KEEP)], ]
  }
  o
}

boros <- sort(unique(out$boro))
res <- setNames(lapply(boros, split_boro), boros)
for (b in boros) cat(sprintf("%-14s n = %6d  sd(y) = %.2f decades\n", b, nrow(res[[b]]), sd(res[[b]]$y)))
cat(sprintf("features %d\n", ncol(res[[1]]) - 2))

saveRDS(res, "data/pluto-clean.rds")
cat("Saved to data/pluto-clean.rds\n")

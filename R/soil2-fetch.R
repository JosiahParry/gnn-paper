# Raster transfer test, version 2: small regions, measured terrain.
#
# WHY VERSION 1 FAILED (both screens, for two separate reasons):
#
#  1. Covariates were WorldClim climate at 2.5 arcmin. That is not measured
#     at 5 km -- it is a smooth surface interpolated from sparse weather
#     stations, so every climate column came back with Moran's I of 0.997+.
#     The interpolation had already done the spatial smoothing a graph is
#     supposed to do, leaving nothing to add. The screen said so (0.89).
#  2. The two regions differed too much: Ohio Valley soil carbon varies more
#     than twice as much as the Corn Belt's, and the reverse transfer scored
#     R2 = -46.
#
# WHAT CHANGES HERE:
#
#  - Covariates are SRTM terrain at 3 arcsec (~90 m), which is MEASURED, not
#    interpolated, and retains genuine fine-scale variability. Climate is
#    dropped entirely: over a region this small it is nearly constant, so it
#    would contribute nothing but smoothness.
#  - One covariate is deliberately rough: the standard deviation of 90 m
#    elevation inside each 1 km cell. Sub-grid roughness cannot be smooth by
#    construction, which is the property version 1 lacked.
#  - Regions are two ~1 degree boxes in the same landscape (the Driftless
#    Area and the ridge country just east of it), so the soil-forming process
#    is shared and the DGP screen has a chance of passing.
#  - 1 km cells, ~10k per region: small and fast, as requested.
#
# The science is standard digital soil mapping: at hillslope scale, soil
# carbon follows topography -- water and sediment accumulate in hollows and
# toeslopes, and erode from convex crests. Terrain should explain a lot, and
# what it misses (land use, parent material) is what could leave structure
# for a graph.

suppressPackageStartupMessages({library(terra); library(geodata)})

PATH <- "data/geo"
dir.create(PATH, showWarnings = FALSE, recursive = TRUE)
AGG_SOC <- 4      # 250 m SoilGrids -> 1 km
FINE_M  <- 100    # terrain is computed at this resolution, then aggregated
AGG_DEM <- 10     # 100 m -> 1 km

# BOXSET "across": two different landscapes. Rejected by the DGP screen --
# terrain explains soil carbon well in the dissected Driftless (in-sample 0.44)
# and poorly in the ridge country east of it (0.15), so the relationship does
# not carry across. Kept for the record.
# BOXSET "within": two boxes INSIDE the Driftless, ~100 km apart north-south,
# same landform and same soil-forming process, differing only in location.
BOXSET <- getOption("soil2_boxes", Sys.getenv("SOIL2_BOXES", "within"))

boxes <- switch(BOXSET,
  across = list(
    driftless = c(-91.4, -90.4, 42.7, 43.6),   # SW Wisconsin / NE Iowa
    ridges    = c(-89.6, -88.6, 42.7, 43.6)    # S central Wisconsin, ~140 km east
  ),
  within = list(
    drift_n = c(-91.5, -90.7, 43.0, 43.7),     # Driftless, north
    drift_s = c(-91.5, -90.7, 42.0, 42.7)      # Driftless, south, ~110 km away
  )
)
cat(sprintf("box set: %s\n", BOXSET))

cat("Opening SoilGrids (remote, windowed read)...\n")
soc <- soil_world_vsi(var = "soc", depth = 5, stat = "mean")

build <- function(nm, bb) {
  cat(sprintf("\n=== %s ===\n", nm))
  ll <- ext(bb[1], bb[2], bb[3], bb[4])

  # 1 km analysis grid from SoilGrids, in its own equal-area CRS.
  cache <- file.path(PATH, sprintf("soc1km_%s_%s.tif", BOXSET, nm))
  if (file.exists(cache)) {
    tmpl <- rast(cache); cat("  soc: cached\n")
  } else {
    bv <- project(vect(ll, crs = "EPSG:4326"), crs(soc))
    t0 <- Sys.time()
    s <- crop(soc, bv)
    tmpl <- aggregate(s, fact = AGG_SOC, fun = mean, na.rm = TRUE)
    writeRaster(tmpl, cache, overwrite = TRUE)
    cat(sprintf("  soc crop+aggregate %.0fs\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  }
  names(tmpl) <- "soc"
  cat(sprintf("  grid %s at %s m\n", paste(dim(tmpl)[1:2], collapse = "x"),
              paste(round(res(tmpl)), collapse = "x")))

  # SRTM 90 m, mosaicked over the box.
  lons <- unique(c(floor(bb[1] / 5) * 5 + 2.5, floor(bb[2] / 5) * 5 + 2.5))
  lats <- unique(c(floor(bb[3] / 5) * 5 + 2.5, floor(bb[4] / 5) * 5 + 2.5))
  tiles <- list()
  for (lo in lons) for (la in lats) {
    t <- try(elevation_3s(lon = lo, lat = la, path = PATH), silent = TRUE)
    if (!inherits(t, "try-error")) tiles[[length(tiles) + 1]] <- t
  }
  stopifnot(length(tiles) > 0)
  dem_ll <- if (length(tiles) == 1) tiles[[1]] else do.call(merge, tiles)
  dem_ll <- crop(dem_ll, ll)
  cat(sprintf("  srtm %s cells at ~90 m\n", ncell(dem_ll)))

  # Work in the analysis CRS at 100 m, so terrain describes real ground
  # distances rather than degrees.
  dem <- project(dem_ll, crs(tmpl), res = FINE_M, method = "bilinear")
  slope <- terrain(dem, v = "slope", unit = "degrees")
  asp   <- terrain(dem, v = "aspect", unit = "radians")
  tpi   <- terrain(dem, v = "TPI")
  tri   <- terrain(dem, v = "TRI")
  # Aspect is circular, so average its sine and cosine, never the angle.
  northness <- cos(asp); eastness <- sin(asp)
  names(dem) <- "elevation"; names(slope) <- "slope"
  names(tpi) <- "tpi"; names(tri) <- "tri"
  names(northness) <- "northness"; names(eastness) <- "eastness"

  fine <- c(dem, slope, tpi, tri, northness, eastness)
  coarse <- aggregate(fine, fact = AGG_DEM, fun = mean, na.rm = TRUE)
  # The deliberately rough covariate: within-cell relief at 100 m. This one
  # cannot be smooth by construction, which is what version 1 lacked.
  relief <- aggregate(dem, fact = AGG_DEM, fun = sd, na.rm = TRUE)
  names(relief) <- "relief_sd"

  cov <- resample(c(coarse, relief), tmpl, method = "bilinear")
  d <- as.data.frame(c(tmpl, cov), xy = TRUE, na.rm = TRUE)

  # Rename coordinates BEFORE creating the target: as.data.frame(xy = TRUE)
  # calls them x/y, so a target named y would overwrite the latitude column.
  names(d)[names(d) == "x"] <- "lon"
  names(d)[names(d) == "y"] <- "lat"
  stopifnot(!"y" %in% names(d))

  d <- d[is.finite(d$soc) & d$soc > 0, ]
  d$y <- log(d$soc) - mean(log(d$soc))
  d$soc <- NULL
  stopifnot(all(is.finite(d$y)))
  cat(sprintf("  %d complete cells, %d covariates, sd(y) = %.3f\n",
              nrow(d), ncol(d) - 3, sd(d$y)))
  d
}

out <- lapply(names(boxes), function(n) build(n, boxes[[n]]))
names(out) <- names(boxes)

cat("\n=== summary ===\n")
for (n in names(out)) cat(sprintf("%-10s n = %5d  sd(y) = %.3f\n", n, nrow(out[[n]]), sd(out[[n]]$y)))
cat(sprintf("features: %s\n", paste(setdiff(names(out[[1]]), c("lon", "lat", "y")), collapse = ", ")))

outfile <- sprintf("data/soil2-clean-%s.rds", BOXSET)
saveRDS(out, outfile)
cat(sprintf("\nSaved to %s\n", outfile))

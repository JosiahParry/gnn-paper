# Raster transfer test v3: MODIS NDVI from terrain, Idaho -> Montana.
#
# WHY THIS TARGET, AFTER THREE SOIL FAILURES.
#
# The soil attempts failed for a reason that had nothing to do with rasters:
# SoilGrids is not measured, it is the output of a global ML model fitted to
# terrain and climate -- the same covariate families we were feeding in. We
# were reverse-engineering someone else's model, and its regional quirks
# showed up as transfer failures that said nothing about spatial transfer.
#
# NDVI has none of that. It is a direct satellite measurement of surface
# reflectance, at 231 m, with no model in between. Terrain is SRTM, also
# measured. Both sides of the equation are observations.
#
# WHY TERRAIN SHOULD PREDICT NDVI. In semi-arid mountains, topography
# controls water: north-facing slopes hold moisture and carry denser
# vegetation, south-facing slopes dry out, hollows and toeslopes accumulate
# water and soil. Aspect, slope and hillslope position are real drivers, not
# proxies. What terrain CANNOT see -- fire history, logging, land ownership,
# soil parent material -- is patchy and spatially clustered, which is exactly
# the leftover structure a graph is supposed to exploit.
#
# REGIONS: two forested mountain blocks in the Northern Rockies, ~150 km
# apart, same ecoregion and same climate regime, so the terrain->vegetation
# relationship has a chance of being shared. A single mid-summer date is used
# for both, so no phenological offset is introduced.
#
# Prediction is recorded in reports/predictions.md BEFORE the model runs.

suppressPackageStartupMessages({
  library(terra); library(geodata); library(MODISTools)
})

PATH <- "data/geo"
dir.create(PATH, showWarnings = FALSE, recursive = TRUE)
DATE <- "2023-07-12"     # peak growing season, one 16-day composite
KM <- 10                 # half-width; 20 km boxes, 9 km gap between the Iowa farms
FINE_M <- 100            # terrain computed at this resolution

# REGIONSET "mountains": two Northern Rockies blocks ~150 km apart. REJECTED
# by the DGP screen -- and rejected for a real ecological reason, not a data
# problem: the elevation and aspect coefficients INVERT between them
# (calibration slopes -0.56 and -1.19). In Idaho higher ground is drier and
# browner; 130 km north in Montana higher ground is wetter and greener.
#
# REGIONSET "iowa": the design this project should have been testing all
# along. Two corn farms in west-central Iowa, 29 km apart, same soil
# association, same climate, same crop, same management era. The premise of
# inductive spatial transfer is "measure the process in A, apply it in a
# place where the process is the SAME" -- not "apply it somewhere different
# and see what happens". Every earlier raster pair tested the latter.
#
# In flat till plains, within-field vegetation vigour tracks micro-
# topography: hollows and toeslopes hold water, convex knobs shed it and burn
# out in July. That is a genuine, shared, local mechanism.
REGIONSET <- Sys.getenv("NDVI_REGIONS", "iowa")

regions <- switch(REGIONSET,
  mountains = list(
    idaho   = c(lat = 44.00, lon = -115.00),   # Idaho batholith, Boise NF
    montana = c(lat = 45.20, lon = -114.30)    # Bitterroot Range, ~150 km NE
  ),
  iowa = list(
    summit = c(lat = 41.9985, lon = -94.8384), # Summit Genetics, Carroll Co.
    lewis  = c(lat = 41.8269, lon = -94.5711)  # Lewis Farms, 29 km SE
  )
)
cat(sprintf("region set: %s\n", REGIONSET))

build <- function(nm, ctr) {
  cat(sprintf("\n=== %s ===\n", nm))

  cache <- file.path(PATH, sprintf("ndvi_%s_%s.tif", REGIONSET, nm))
  if (file.exists(cache)) {
    ndvi <- rast(cache); cat("  ndvi: cached\n")
  } else {
    t0 <- Sys.time()
    sub <- mt_subset(product = "MOD13Q1", band = "250m_16_days_NDVI",
                     lat = ctr[["lat"]], lon = ctr[["lon"]],
                     km_lr = KM, km_ab = KM,
                     start = DATE, end = DATE, internal = TRUE, progress = FALSE)
    ndvi <- mt_to_terra(sub, reproject = FALSE)
    writeRaster(ndvi, cache, overwrite = TRUE)
    cat(sprintf("  ndvi download %.0fs\n", as.numeric(difftime(Sys.time(), t0, units = "secs"))))
  }
  names(ndvi) <- "ndvi"
  cat(sprintf("  grid %s at %.0f m (MODIS sinusoidal)\n",
              paste(dim(ndvi)[1:2], collapse = "x"), res(ndvi)[1]))

  # SRTM covering the box, with a margin so terrain at the edges is valid.
  ll <- project(ext(ndvi), from = crs(ndvi), to = "EPSG:4326")
  pad <- 0.15
  llp <- ext(ll$xmin - pad, ll$xmax + pad, ll$ymin - pad, ll$ymax + pad)
  lons <- unique(floor(c(llp$xmin, llp$xmax) / 5) * 5 + 2.5)
  lats <- unique(floor(c(llp$ymin, llp$ymax) / 5) * 5 + 2.5)
  tiles <- list()
  for (lo in lons) for (la in lats) {
    t <- try(elevation_3s(lon = lo, lat = la, path = PATH), silent = TRUE)
    if (!inherits(t, "try-error")) tiles[[length(tiles) + 1]] <- t
  }
  stopifnot(length(tiles) > 0)
  dem_ll <- if (length(tiles) == 1) tiles[[1]] else do.call(merge, tiles)
  dem_ll <- crop(dem_ll, llp)

  # Terrain in the analysis CRS, at 100 m, so slopes are real gradients.
  dem <- project(dem_ll, crs(ndvi), res = FINE_M, method = "bilinear")
  slope <- terrain(dem, v = "slope", unit = "degrees")
  asp   <- terrain(dem, v = "aspect", unit = "radians")
  tpi   <- terrain(dem, v = "TPI")
  tri   <- terrain(dem, v = "TRI")
  # Aspect is circular: average its sine and cosine, never the angle itself.
  # northness is the moisture axis in the northern hemisphere.
  northness <- cos(asp); eastness <- sin(asp)
  # Sub-cell roughness, computed at 100 m before averaging: cannot be smooth
  # by construction, unlike an interpolated surface.
  relief <- focal(dem, w = 5, fun = sd, na.rm = TRUE)

  names(dem) <- "elevation"; names(slope) <- "slope"
  names(tpi) <- "tpi"; names(tri) <- "tri"
  names(northness) <- "northness"; names(eastness) <- "eastness"
  names(relief) <- "relief_sd"

  fine <- c(dem, slope, tpi, tri, northness, eastness, relief)
  cov <- resample(fine, ndvi, method = "average")

  d <- as.data.frame(c(ndvi, cov), xy = TRUE, na.rm = TRUE)

  # Rename coordinates BEFORE creating the target: as.data.frame(xy = TRUE)
  # names them x/y, so a target called y would overwrite latitude.
  names(d)[names(d) == "x"] <- "lon"
  names(d)[names(d) == "y"] <- "lat"
  stopifnot(!"y" %in% names(d))

  # Water and cloud give NDVI at or below zero; drop them rather than model
  # them, since they are not a vegetation signal.
  d <- d[is.finite(d$ndvi) & d$ndvi > 0.05, ]
  d$y <- d$ndvi - mean(d$ndvi)    # demeaned within region, as everywhere else
  d$ndvi <- NULL
  stopifnot(all(is.finite(d$y)))

  cat(sprintf("  %d cells, %d covariates, sd(y) = %.4f\n", nrow(d), ncol(d) - 3, sd(d$y)))
  d
}

out <- lapply(names(regions), function(n) build(n, regions[[n]]))
names(out) <- names(regions)

cat("\n=== summary ===\n")
for (n in names(out)) cat(sprintf("%-8s n = %5d  sd(y) = %.4f\n", n, nrow(out[[n]]), sd(out[[n]]$y)))
cat(sprintf("features: %s\n", paste(setdiff(names(out[[1]]), c("lon", "lat", "y")), collapse = ", ")))

outfile <- sprintf("data/ndvi-clean-%s.rds", REGIONSET)
saveRDS(out, outfile)
cat(sprintf("\nSaved to %s\n", outfile))
cat("Coordinates are metres in the MODIS sinusoidal CRS; cell size ~231 m.\n")

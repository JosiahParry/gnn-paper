# Raster transfer test: soil organic carbon, US Corn Belt -> Ohio Valley.
#
# Two gaps this closes at once.
#
# 1. RASTER. Every real test so far is irregular points (houses, listings,
#    trees, service requests, tracts). Our only grid evidence is synthetic,
#    and it is the most favourable evidence we have -- GraphSAGE wins
#    decisively on the nonlinear lattice scenarios. An ArcGIS Pro tool has to
#    work on rasters, so that gap matters for the deliverable, not just the
#    paper.
#
# 2. THE PRICE CONFOUND. All three wins so far are prices; all four losses
#    are not. Soil carbon is not a price, and it has the profile our pre-fit
#    screen says should favour a graph: strong spatial structure, driven
#    partly by land use and soil parent material that our covariates do not
#    observe. If GraphSAGE wins here, "it only works on prices" is dead.
#
# Design choices worth stating:
#
#  - Everything is resampled to ONE 5 km grid. If the climate covariates were
#    left coarser than the target, they would look artificially smooth, which
#    would inflate covariate Moran's I and bias the exact diagnostic under
#    test.
#  - The two boxes are ~330 km apart with no overlap, so the target region's
#    graph is genuinely disjoint, as in every other transfer here.
#  - Land cover is deliberately NOT included. Soil carbon depends heavily on
#    cultivation history, so omitting it is what leaves spatial structure the
#    covariates cannot express -- the situation a graph is supposed to
#    exploit. Including it would test a different question.

suppressPackageStartupMessages({library(terra); library(geodata)})

PATH <- "data/geo"
dir.create(PATH, showWarnings = FALSE, recursive = TRUE)
RES_M <- 5000        # target grid, metres
AGG <- 20            # 250 m SoilGrids -> 5 km

# Default pair: two agricultural regions ~330 km apart. They differ in soil
# order (Mollisols west, Alfisols east), so the DGP screen may reject them --
# as it rejected the NYC boroughs for building age. If it does, fall back to
# `options(soil_boxes = "within")`, two disjoint boxes inside the Corn Belt,
# where the soil-forming processes are the same and only the location differs.
BOXSET <- getOption("soil_boxes", "across")

boxes <- switch(BOXSET,
  across = list(
    cornbelt   = c(-98, -90, 39, 45),   # Iowa / S Minnesota / E Nebraska
    ohiovalley = c(-86, -78, 37, 43)    # Indiana / Ohio / W Pennsylvania
  ),
  within = list(
    cornbelt   = c(-98, -94, 40, 45),   # W Iowa / Nebraska / S Minnesota
    ohiovalley = c(-91, -87, 39, 44)    # E Iowa / Illinois / S Wisconsin
  )
)
cat(sprintf("box set: %s\n", BOXSET))

cat("Opening SoilGrids (remote, windowed read)...\n")
soc <- soil_world_vsi(var = "soc", depth = 5, stat = "mean")

cat("Downloading WorldClim bioclim (2.5 arcmin)...\n")
bio <- worldclim_global(var = "bio", res = 2.5, path = PATH)
cat("Downloading global elevation (2.5 arcmin)...\n")
elev <- elevation_global(res = 2.5, path = PATH)

# Bioclim layers we keep, named for legibility downstream. geodata names them
# "wc2.1_2.5m_bio_1" (underscore before the index), not "...bio1".
bio_keep <- c(bio_1 = "mean_temp", bio_4 = "temp_seasonality",
              bio_5 = "max_temp_warm", bio_6 = "min_temp_cold",
              bio_12 = "annual_precip", bio_15 = "precip_seasonality")
stopifnot(all(paste0("wc2.1_2.5m_", names(bio_keep)) %in% names(bio)))

build_region <- function(nm, bb) {
  cat(sprintf("\n=== %s ===\n", nm))
  ll <- ext(bb[1], bb[2], bb[3], bb[4])

  # 1. target grid: crop SoilGrids in its own equal-area CRS, aggregate to 5 km.
  #    The remote windowed read is ~2 min per region, so cache it -- otherwise
  #    every downstream fix pays for it again.
  # Cache key includes the box set: the same region name covers different
  # extents under "across" and "within", so a name-only key would silently
  # reuse the wrong crop.
  cache <- file.path(PATH, sprintf("soc5km_%s_%s.tif", BOXSET, nm))
  if (file.exists(cache)) {
    tmpl <- rast(cache)
    cat("  soc: using cached 5 km grid\n")
  } else {
    bv <- project(vect(ll, crs = "EPSG:4326"), crs(soc))
    t0 <- Sys.time()
    s <- crop(soc, bv)
    cat(sprintf("  soc crop %.0fs, %s cells at 250m\n",
                as.numeric(difftime(Sys.time(), t0, units = "secs")), ncell(s)))
    tmpl <- aggregate(s, fact = AGG, fun = mean, na.rm = TRUE)
    writeRaster(tmpl, cache, overwrite = TRUE)
  }
  names(tmpl) <- "soc"
  cat(sprintf("  template %s at %s m\n", paste(dim(tmpl)[1:2], collapse = "x"),
              paste(round(res(tmpl)), collapse = "x")))

  # 2. covariates: crop in lon/lat, then project ONTO the template grid so
  #    every layer shares one resolution and alignment.
  bl <- bio[[paste0("wc2.1_2.5m_", names(bio_keep))]]
  names(bl) <- unname(bio_keep)
  cl <- project(crop(bl, ll), tmpl, method = "bilinear")

  el <- project(crop(elev, ll), tmpl, method = "bilinear")
  names(el) <- "elevation"
  # Terrain derivatives are computed AFTER projection, on the analysis grid,
  # so they describe the same neighbourhood the graph will use.
  slope <- terrain(el, v = "slope", unit = "degrees")
  tri   <- terrain(el, v = "TRI")
  names(slope) <- "slope"; names(tri) <- "roughness"

  stk <- c(tmpl, cl, el, slope, tri)
  d <- as.data.frame(stk, xy = TRUE, na.rm = TRUE)
  cat(sprintf("  %d complete cells, %d covariates\n", nrow(d), ncol(d) - 3))

  # Rename the coordinate columns FIRST. as.data.frame(xy = TRUE) calls them
  # "x" and "y", so creating a target named "y" before renaming would
  # overwrite the latitude column and then relabel the target as "lat".
  names(d)[names(d) == "x"] <- "lon"
  names(d)[names(d) == "y"] <- "lat"
  stopifnot(!"y" %in% names(d))

  # Target: log soil organic carbon, demeaned within region (as everywhere
  # else in this project -- the level is not identified across regions).
  d <- d[is.finite(d$soc) & d$soc > 0, ]
  d$y <- log(d$soc) - mean(log(d$soc))
  d$soc <- NULL
  stopifnot(all(is.finite(d$y)), all(c("lon", "lat") %in% names(d)))
  d
}

out <- lapply(names(boxes), function(n) build_region(n, boxes[[n]]))
names(out) <- names(boxes)

cat("\n=== summary ===\n")
for (n in names(out)) {
  cat(sprintf("%-12s n = %6d  sd(y) = %.3f\n", n, nrow(out[[n]]), sd(out[[n]]$y)))
}
cat(sprintf("features: %s\n",
            paste(setdiff(names(out[[1]]), c("lon", "lat", "y")), collapse = ", ")))

outfile <- if (BOXSET == "across") {
  "data/soil-clean.rds"
} else {
  sprintf("data/soil-clean-%s.rds", BOXSET)
}
saveRDS(out, outfile)
cat(sprintf("\nSaved to %s\n", outfile))
cat("NOTE: coordinates are metres in an equal-area CRS, not degrees.\n")

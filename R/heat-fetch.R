# Urban heat: land surface temperature from Landsat, two similar Iowa cities.
#
# WHY THIS IS THE BEST-DESIGNED TEST IN THE PROJECT.
#
# Everything learned so far points here:
#
#  1. SIMILAR REGIONS. Every "different region" pair failed catastrophically
#     (reverse R2 of -46, -8.4, -3.97, with coefficients inverting). The only
#     raster pair that transferred was two Iowa corn farms 29 km apart. So:
#     two mid-size Iowa cities, same climate, same development era, same
#     building stock and same vegetation.
#
#  2. MEASURED, NOT MODELLED. SoilGrids was an ML model's output and we were
#     reverse-engineering it. Land surface temperature is read off Landsat's
#     thermal sensor; the optical covariates are read off the same satellite.
#     Both sides are observations.
#
#  3. THE MECHANISM MATCHES THE ONE SCENARIO WHERE GRAPHSAGE WINS. Heat moves.
#     A park cools the blocks downwind of it; a parking lot heats its
#     surroundings. So a cell's temperature depends on its NEIGHBOURS' surface
#     cover, not only its own -- a genuine spatial lag in the data-generating
#     process. That is simulation Scenario E, the only row of the grid whose
#     DGP contains a lag term, and the row where GraphSAGE beat everything
#     (0.429 against OLS 0.283).
#
#     This is the first real dataset whose physics matches that scenario. In
#     every earlier real test the spatial structure was either already in the
#     covariates (CDC, energy) or unreachable because it lived in the target
#     (Iowa NDVI field boundaries, simulation Scenario C).
#
#  4. NOT A PRICE. All three wins so far are prices. This breaks that.
#
# LEAKAGE NOTE: the thermal bands (10, 11) define the target and are excluded
# from the covariates. Only the optical bands and indices derived from them
# are used.
#
# Two targets are built from one fetch:
#   heat  - regression: surface temperature in Celsius, demeaned per city
#   lulc  - classification: Esri Sentinel-2 land cover class, majority per cell

suppressPackageStartupMessages({
  library(terra); library(sf); library(geodata)
})

PATH <- "data/geo"
dir.create(PATH, showWarnings = FALSE, recursive = TRUE)

LS_URL <- "https://landsat2.arcgis.com/arcgis/rest/services/Landsat/MS/ImageServer"
LC_URL <- "https://ic.imagery1.arcgis.com/arcgis/rest/services/Sentinel2_10m_LandCover/ImageServer"

SPAN <- 10000    # 10 km box
N    <- 167      # -> ~60 m cells, ~28k per city

# CRITICAL. The Landsat service is a mosaic of "best available" scenes, and
# left to itself it handed us a JULY scene for Des Moines (20.7-45.9 C) and a
# JANUARY scene for Cedar Rapids (-1.0 to 5.5 C). Vegetation cools a surface
# in summer and does nothing in winter, so that is not the same physical
# process in the two cities -- the transfer would have been meaningless and
# the failure would have looked like "regions do not transfer".
#
# This rule pins BOTH cities to clear summer 2023 scenes, and is applied to
# the optical bands as well as the thermal one so that vegetation index and
# temperature are read from the same acquisition.
MOSAIC_RULE <- paste0(
  '{"mosaicMethod":"esriMosaicAttribute","sortField":"CloudCover",',
  '"sortValue":0,"ascending":true,',
  '"where":"CloudCover < 0.05 AND AcquisitionDate BETWEEN ',
  "timestamp '2023-06-15 00:00:00' AND timestamp '2023-09-10 00:00:00'\"}")

cities <- list(
  desmoines   = c(lon = -93.6250, lat = 41.5868),
  cedarrapids = c(lon = -91.6656, lat = 41.9779)
)

# Landsat optical bands only. Thermal (9, 10) is the target; QA (8) and
# Cirrus (7) are quality flags, not surface properties.
BANDS <- c(CoastalAerosol = 0, Blue = 1, Green = 2, Red = 3,
           NearInfrared = 4, SWIR1 = 5, SWIR2 = 6)

export_tif <- function(url, bbox, size, dest, rendering = NULL, band_ids = NULL,
                       pixel_type = "F32", interp = NULL, mosaic = NULL) {
  # system2() does NOT quote its arguments: the shell splits them on spaces.
  # The rendering rule is JSON containing both spaces and quotes, so every
  # value must be shQuote()d or curl receives fragments. Coordinates go
  # through sprintf("%.2f") because as.character() turns round numbers like
  # 5000000 into "5e+06", which the service rejects.
  q <- function(...) c("--data-urlencode", shQuote(sprintf(...)))
  # The service root only describes the service; pixels come from /exportImage.
  endpoint <- paste0(sub("/+$", "", url), "/exportImage")
  args <- c("-G", shQuote(endpoint),
            q("bbox=%s", paste(sprintf("%.2f", bbox), collapse = ",")),
            q("bboxSR=3857"),
            q("size=%d,%d", size, size),
            q("format=tiff"),
            q("pixelType=%s", pixel_type),
            q("f=image"))
  if (!is.null(rendering))
    args <- c(args, q('renderingRule={"rasterFunction":"%s"}', rendering))
  if (!is.null(band_ids))
    args <- c(args, q("bandIds=%s", paste(band_ids, collapse = ",")))
  if (!is.null(interp)) args <- c(args, q("interpolation=%s", interp))
  if (!is.null(mosaic)) args <- c(args, c("--data-urlencode", shQuote(paste0("mosaicRule=", mosaic))))
  args <- c(args, "-o", shQuote(dest), "-s", "--max-time", "120")

  st <- system2("curl", args)
  if (st != 0 || !file.exists(dest) || file.size(dest) < 5000) {
    msg <- if (file.exists(dest)) paste(readLines(dest, n = 8, warn = FALSE), collapse = " ") else "no file"
    unlink(dest)   # never leave an error page cached as if it were a raster
    stop(sprintf("export failed (%s): %s", basename(dest), substr(msg, 1, 300)))
  }
  rast(dest)
}

to_merc <- function(lon, lat) {
  p <- st_transform(st_sfc(st_point(c(lon, lat)), crs = 4326), 3857)
  as.numeric(st_coordinates(p))
}

build <- function(nm, ctr) {
  cat(sprintf("\n=== %s ===\n", nm))
  ctr_m <- to_merc(ctr[["lon"]], ctr[["lat"]])
  bbox <- c(ctr_m[1] - SPAN/2, ctr_m[2] - SPAN/2, ctr_m[1] + SPAN/2, ctr_m[2] + SPAN/2)

  f_lst <- file.path(PATH, sprintf("heat_lst_%s.tif", nm))
  f_ms  <- file.path(PATH, sprintf("heat_ms_%s.tif", nm))
  f_lc  <- file.path(PATH, sprintf("heat_lc_%s.tif", nm))

  lst <- if (file.exists(f_lst)) rast(f_lst) else
    export_tif(LS_URL, bbox, N, f_lst, rendering = "Band 10 Surface Temperature in Celsius",
               mosaic = MOSAIC_RULE)
  names(lst) <- "lst"
  cat(sprintf("  lst  %s, range %.1f-%.1f C\n", paste(dim(lst)[1:2], collapse="x"),
              min(values(lst), na.rm=TRUE), max(values(lst), na.rm=TRUE)))

  ms <- if (file.exists(f_ms)) rast(f_ms) else
    export_tif(LS_URL, bbox, N, f_ms, band_ids = unname(BANDS), mosaic = MOSAIC_RULE)
  names(ms) <- names(BANDS)
  cat(sprintf("  ms   %d bands\n", nlyr(ms)))

  # Land cover: nearest-neighbour so classes are not blended into nonsense.
  lc <- if (file.exists(f_lc)) rast(f_lc) else
    export_tif(LC_URL, bbox, N, f_lc, pixel_type = "U8", interp = "RSP_NearestNeighbor")
  names(lc) <- "lulc"
  cat(sprintf("  lulc classes: %s\n", paste(sort(unique(values(lc))), collapse=",")))

  # Spectral indices: the standard surface descriptors for heat studies.
  eps <- 1e-6
  ndvi <- (ms$NearInfrared - ms$Red) / (ms$NearInfrared + ms$Red + eps)        # vegetation
  ndbi <- (ms$SWIR1 - ms$NearInfrared) / (ms$SWIR1 + ms$NearInfrared + eps)    # built-up
  ndwi <- (ms$Green - ms$NearInfrared) / (ms$Green + ms$NearInfrared + eps)    # water
  names(ndvi) <- "ndvi"; names(ndbi) <- "ndbi"; names(ndwi) <- "ndwi"

  # Terrain: mild in Iowa but cold air pools in valleys, so keep it.
  llb <- project(ext(lst), from = "EPSG:3857", to = "EPSG:4326")
  pad <- 0.2
  tiles <- list()
  for (lo in unique(floor(c(llb$xmin-pad, llb$xmax+pad)/5)*5+2.5))
    for (la in unique(floor(c(llb$ymin-pad, llb$ymax+pad)/5)*5+2.5)) {
      t <- try(elevation_3s(lon = lo, lat = la, path = PATH), silent = TRUE)
      if (!inherits(t, "try-error")) tiles[[length(tiles)+1]] <- t
    }
  dem_ll <- if (length(tiles) == 1) tiles[[1]] else do.call(merge, tiles)
  dem <- project(crop(dem_ll, ext(llb$xmin-pad, llb$xmax+pad, llb$ymin-pad, llb$ymax+pad)),
                 crs(lst), res = 60, method = "bilinear")
  elev <- resample(dem, lst, method = "average"); names(elev) <- "elevation"
  tpi <- terrain(elev, v = "TPI"); names(tpi) <- "tpi"

  stk <- c(lst, ms, ndvi, ndbi, ndwi, elev, tpi, lc)
  d <- as.data.frame(stk, xy = TRUE, na.rm = TRUE)

  # Rename coords BEFORE any target is created (the x/y collision bug).
  names(d)[names(d) == "x"] <- "lon"
  names(d)[names(d) == "y"] <- "lat"
  stopifnot(!"y" %in% names(d))
  d
}

raw <- lapply(names(cities), function(n) build(n, cities[[n]]))
names(raw) <- names(cities)

feat <- c(names(BANDS), "ndvi", "ndbi", "ndwi", "elevation", "tpi")

# --- regression: surface temperature -------------------------------------
heat <- lapply(raw, function(d) {
  o <- d[, c("lon", "lat", feat)]
  o$y <- d$lst - mean(d$lst)        # demeaned per city: acquisition date differs
  o[Reduce(`&`, lapply(o, is.finite)), ]
})
saveRDS(heat, "data/heat-clean.rds")

# --- classification: land cover ------------------------------------------
# Built (7) against everything else: the dominant urban-fringe distinction,
# and the one where a mixed pixel is genuinely ambiguous on its own spectra.
lulc <- lapply(raw, function(d) {
  o <- d[, c("lon", "lat", feat)]
  o$y <- as.integer(d$lulc == 7)
  o$lulc_raw <- d$lulc
  o[Reduce(`&`, lapply(o[, c("lon","lat",feat,"y")], is.finite)), ]
})
saveRDS(lulc, "data/lulc-clean.rds")

sds <- vapply(heat, function(d) sd(d$y), numeric(1))
if (max(sds) / min(sds) > 2) {
  warning(sprintf(paste0("Temperature spread differs %.1fx between cities (%s). ",
                         "Check the scenes are from the same season."),
                  max(sds)/min(sds), paste(sprintf("%.2f", sds), collapse = " vs ")))
}

cat("\n=== summary ===\n")
for (n in names(heat)) {
  cat(sprintf("%-12s n = %5d | temp sd = %.2f C | built share = %.2f\n",
              n, nrow(heat[[n]]), sd(heat[[n]]$y), mean(lulc[[n]]$y)))
}
cat(sprintf("covariates (%d): %s\n", length(feat), paste(feat, collapse = ", ")))
cat("\nSaved data/heat-clean.rds and data/lulc-clean.rds\n")

# Data acquisition. Each target writes a named list of regions with
# lon, lat, y and numeric features.
# Rscript R/fetch.R <airbnb|nonprice|cdc|cdcclean|nyc311|trees|heat|kcames>

MODE <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(MODE)) stop("pick a target")

MODE <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(MODE)) stop("pick one: airbnb nonprice cdc cdcclean nyc311 trees heat kcames")

if (MODE == "airbnb") {

  suppressPackageStartupMessages(library(dplyr))

  clean_one <- function(path, city) {
    d <- read.csv(path, stringsAsFactors = FALSE)
    bathrooms <- as.numeric(gsub("[^0-9.]", "", d$bathrooms_text))
    price <- as.numeric(gsub("[$,]", "", d$price))

    out <- data.frame(
      city = city,
      lat = d$latitude,
      lon = d$longitude,
      price = price,
      accommodates = d$accommodates,
      bathrooms = bathrooms,
      bedrooms = d$bedrooms,
      beds = d$beds,
      minimum_nights = pmin(d$minimum_nights, 365),
      availability_365 = d$availability_365,
      number_of_reviews = d$number_of_reviews,
      is_entire_home = as.integer(d$room_type == "Entire home/apt")
    )

    out <- out |> filter(
      !is.na(price), price > 0, !is.na(bathrooms), !is.na(lat), !is.na(accommodates),
      !is.na(minimum_nights), !is.na(availability_365), !is.na(number_of_reviews)
    )

    for (col in c("bedrooms", "beds")) {
      med_by_acc <- out |> group_by(accommodates) |> summarise(med = median(.data[[col]], na.rm = TRUE))
      global_med <- median(out[[col]], na.rm = TRUE)
      idx <- is.na(out[[col]])
      fill <- med_by_acc$med[match(out$accommodates[idx], med_by_acc$accommodates)]
      fill[is.na(fill)] <- global_med
      out[[col]][idx] <- fill
    }

    out
  }

  cat("Cleaning LA...\n")
  la <- clean_one("data/airbnb-raw/la_listings.csv", "la")
  cat(sprintf("  %d rows\n", nrow(la)))

  cat("Cleaning NYC...\n")
  nyc <- clean_one("data/airbnb-raw/nyc_listings.csv", "nyc")
  cat(sprintf("  %d rows\n", nrow(nyc)))

  saveRDS(list(la = la, nyc = nyc), "data/airbnb-clean.rds")
  cat("\nSaved to data/airbnb-clean.rds\n")
}

if (MODE == "nonprice") {

  suppressPackageStartupMessages(library(dplyr))

  ab <- readRDS("data/airbnb-all-cities-clean.rds")

  # Cities used in the two transfers that passed the DGP screen on price.
  keep <- c("broward", "sandiego", "chicago", "la")

  build <- function(target) {
    out <- lapply(ab[keep], function(d) {
      y_raw <- switch(target,
        reviews = log1p(d$number_of_reviews),
        avail   = d$availability_365 / 365
      )
      feats <- d |>
        transmute(
          log_price = log(price),
          accommodates, bathrooms, bedrooms, beds,
          minimum_nights = log1p(minimum_nights),
          is_entire_home,
          availability = availability_365 / 365,
          log_reviews = log1p(number_of_reviews)
        )
      feats[[switch(target, reviews = "log_reviews", avail = "availability")]] <- NULL

      o <- bind_cols(data.frame(lat = d$lat, lon = d$lon), feats)
      o$y <- y_raw - mean(y_raw)
      o[Reduce(`&`, lapply(o, is.finite)), ]
    })
    names(out) <- keep
    out
  }

  for (tg in c("reviews", "avail")) {
    d <- build(tg)
    f <- sprintf("data/airbnb-%s-clean.rds", tg)
    saveRDS(d, f)
    cat(sprintf("\n=== target: %s -> %s ===\n", tg, f))
    for (n in names(d)) cat(sprintf("%-10s n = %6d  sd(y) = %.3f\n", n, nrow(d[[n]]), sd(d[[n]]$y)))
    cat(sprintf("features (%d): %s\n", ncol(d[[1]]) - 3,
                paste(setdiff(names(d[[1]]), c("lat", "lon", "y")), collapse = ", ")))
  }
}

if (MODE == "cdc") {

  suppressPackageStartupMessages({library(dplyr); library(jsonlite)})

  measures <- c(
    "DIABETES",    # target
    "BINGE", "CSMOKING", "LPA", "SLEEP",          # risk behaviors
    "ACCESS2", "CHECKUP",                          # access to care
    "FOODINSECU", "HOUSINSECU", "LACKTRPT",        # social needs
    "GHLTH"                                        # self-rated health
  )

  fetch_state <- function(state, measures, page_size = 50000) {
    measure_list <- paste(sprintf("'%s'", measures), collapse = ",")
    offset <- 0
    out <- list()
    repeat {
      url <- sprintf(
        "https://data.cdc.gov/resource/cwsq-ngmh.json?$where=stateabbr='%s' AND measureid in (%s)&$limit=%d&$offset=%d",
        state, measure_list, page_size, offset
      )
      d <- fromJSON(URLencode(url))
      if (length(d) == 0 || nrow(d) == 0) break
      out[[length(out) + 1]] <- d
      if (nrow(d) < page_size) break
      offset <- offset + page_size
    }
    bind_rows(out)
  }

  cat("Fetching CA...\n")
  ca <- fetch_state("CA", measures)
  cat(sprintf("  %d rows\n", nrow(ca)))

  cat("Fetching WV...\n")
  wv <- fetch_state("WV", measures)
  cat(sprintf("  %d rows\n", nrow(wv)))

  raw <- bind_rows(ca, wv)
  saveRDS(raw, "data/cdc-places-raw.rds")
  cat(sprintf("\nSaved %d raw rows to data/cdc-places-raw.rds\n", nrow(raw)))
}

if (MODE == "nyc311") {

  suppressPackageStartupMessages({library(dplyr); library(jsonlite)})

  AGENCY <- "DSNY"
  FROM <- "2024-01-01T00:00:00"
  TO   <- "2025-01-01T00:00:00"

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

  sel <- paste(c("created_date", "closed_date", "complaint_type", "descriptor",
                 "borough", "latitude", "longitude", "open_data_channel_type",
                 "location_type"), collapse = ",")

  whr <- sprintf(paste0("agency='%s' AND created_date>='%s' AND created_date<'%s' ",
                        "AND closed_date IS NOT NULL AND latitude IS NOT NULL ",
                        "AND borough in('QUEENS','BROOKLYN')"),
                 AGENCY, FROM, TO)

  cat("Fetching NYC 311 (DSNY, 2024, Queens + Brooklyn)...\n")
  raw <- fetch_paged("https://data.cityofnewyork.us/resource/erm2-nwe9.json", whr, sel)
  cat(sprintf("%d raw rows\n", nrow(raw)))

  d <- raw |>
    transmute(
      boro = tolower(borough),
      lat = as.numeric(latitude), lon = as.numeric(longitude),
      created = as.POSIXct(created_date, format = "%Y-%m-%dT%H:%M:%OS", tz = "UTC"),
      closed  = as.POSIXct(closed_date,  format = "%Y-%m-%dT%H:%M:%OS", tz = "UTC"),
      complaint = complaint_type, descriptor = descriptor,
      channel = open_data_channel_type, loctype = location_type
    ) |>
    filter(!is.na(lat), !is.na(lon), !is.na(created), !is.na(closed)) |>
    mutate(hours = as.numeric(difftime(closed, created, units = "hours"))) |>
    filter(hours > 0, hours < 24 * 30)

  cat(sprintf("%d rows after basic filters\n", nrow(d)))

  keep_shared <- function(x, min_n = 500) {
    tb <- table(d$boro, x)
    colnames(tb)[apply(tb, 2, min) >= min_n]
  }
  top_complaint <- keep_shared(d$complaint)
  top_desc <- keep_shared(d$descriptor)
  cat(sprintf("shared complaint types: %s\n", paste(top_complaint, collapse = " | ")))
  cat(sprintf("shared descriptors: %d\n", length(top_desc)))

  d <- d |> filter(complaint %in% top_complaint)

  dummies <- function(x, levs, prefix) {
    m <- sapply(levs, function(l) as.integer(x == l))
    colnames(m) <- paste0(prefix, make.names(tolower(levs)))
    as.data.frame(m)
  }

  feat <- bind_cols(
    dummies(d$complaint, top_complaint, "cx_"),
    dummies(d$descriptor, head(top_desc, 12), "dx_"),
    data.frame(
      ch_online = as.integer(d$channel == "ONLINE"),
      ch_phone  = as.integer(d$channel == "PHONE"),
      hour_sin  = sin(2 * pi * as.integer(format(d$created, "%H")) / 24),
      hour_cos  = cos(2 * pi * as.integer(format(d$created, "%H")) / 24),
      dow       = as.integer(format(d$created, "%u")),
      is_weekend = as.integer(as.integer(format(d$created, "%u")) >= 6),
      month     = as.integer(format(d$created, "%m"))
    )
  )
  # Drop degenerate columns (all-zero in either borough).
  ok <- sapply(feat, function(v) length(unique(v)) > 1 &&
                 min(tapply(v, d$boro, function(z) length(unique(z)))) > 1)
  feat <- feat[, ok, drop = FALSE]

  out <- bind_cols(data.frame(boro = d$boro, lat = d$lat, lon = d$lon), feat)
  out$y_raw <- log1p(d$hours)

  N_KEEP <- 25000L
  split_boro <- function(nm) {
    o <- out[out$boro == nm, ]
    o$y <- o$y_raw - mean(o$y_raw)   # demean within borough
    o$y_raw <- NULL; o$boro <- NULL
    if (nrow(o) > N_KEEP) {
      cx <- median(o$lon); cy <- median(o$lat)
      d2 <- ((o$lon - cx) * cos(cy * pi / 180))^2 + (o$lat - cy)^2
      o <- o[order(d2)[seq_len(N_KEEP)], ]
    }
    o
  }

  qns <- split_boro("queens"); bkn <- split_boro("brooklyn")
  cat(sprintf("\nQueens %d, Brooklyn %d, features %d\n",
              nrow(qns), nrow(bkn), ncol(qns) - 2))

  saveRDS(list(queens = qns, brooklyn = bkn), "data/nyc311-clean.rds")
  cat("Saved to data/nyc311-clean.rds\n")
}

if (MODE == "trees") {

  suppressPackageStartupMessages({library(dplyr); library(jsonlite)})

  fields <- "tree_id,tree_dbh,latitude,longitude,boroname,spc_common,health,curb_loc,sidewalk,guards,steward"

  fetch_boro <- function(boro, page = 50000) {
    offset <- 0; out <- list()
    repeat {
      url <- sprintf(
        "https://data.cityofnewyork.us/resource/uvpi-gqnh.json?$select=%s&$where=status='Alive' AND boroname='%s'&$limit=%d&$offset=%d",
        fields, boro, page, offset)
      d <- fromJSON(URLencode(url))
      if (length(d) == 0 || nrow(d) == 0) break
      out[[length(out) + 1]] <- d
      if (nrow(d) < page) break
      offset <- offset + page
    }
    bind_rows(out)
  }

  N_KEEP <- 25000

  cat("Fetching Queens...\n");   q <- fetch_boro("Queens");   cat(sprintf("  %d\n", nrow(q)))
  cat("Fetching Brooklyn...\n"); b <- fetch_boro("Brooklyn"); cat(sprintf("  %d\n", nrow(b)))

  # Shared species vocabulary so both boroughs get identical columns.
  all_spc <- c(q$spc_common, b$spc_common)
  top_spc <- names(sort(table(all_spc), decreasing = TRUE))[1:8]
  cat("\nshared species:", paste(top_spc, collapse = ", "), "\n")

  prep <- function(d, label) {
    out <- d |>
      transmute(
        boro = label,
        lat = as.numeric(latitude), lon = as.numeric(longitude),
        dbh = as.numeric(tree_dbh),
        health_good = as.integer(health == "Good"),
        health_poor = as.integer(health == "Poor"),
        oncurb = as.integer(curb_loc == "OnCurb"),
        sidewalk_damage = as.integer(sidewalk == "Damage"),
        has_guard = as.integer(guards %in% c("Helpful", "Harmful", "Unsure")),
        has_steward = as.integer(steward != "None"),
        spc = spc_common
      ) |>
      filter(!is.na(lat), !is.na(lon), !is.na(dbh), dbh > 0, dbh < 60)

    for (s in top_spc) {
      out[[paste0("spc_", gsub("[^a-z]", "_", tolower(s)))]] <- as.integer(!is.na(out$spc) & out$spc == s)
    }
    out <- out |> select(-spc)

    # Density-preserving: keep the N_KEEP trees closest to the borough centre.
    if (nrow(out) > N_KEEP) {
      cx <- median(out$lon); cy <- median(out$lat)
      d2 <- ((out$lon - cx) * cos(cy * pi / 180))^2 + (out$lat - cy)^2
      out <- out[order(d2)[seq_len(N_KEEP)], ]
    }
    out$y <- log(out$dbh) - mean(log(out$dbh))
    out |> select(-dbh)
  }

  qc <- prep(q, "queens"); bc <- prep(b, "brooklyn")
  cat(sprintf("\nKept: Queens %d, Brooklyn %d\n", nrow(qc), nrow(bc)))
  saveRDS(list(queens = qc, brooklyn = bc), "data/trees-clean.rds")
  cat("Saved to data/trees-clean.rds\n")
}

if (MODE == "heat") {

  suppressPackageStartupMessages({
    library(terra); library(sf); library(geodata)
  })

  PATH <- "data/geo"
  dir.create(PATH, showWarnings = FALSE, recursive = TRUE)

  LS_URL <- "https://landsat2.arcgis.com/arcgis/rest/services/Landsat/MS/ImageServer"
  LC_URL <- "https://ic.imagery1.arcgis.com/arcgis/rest/services/Sentinel2_10m_LandCover/ImageServer"

  SPAN <- 10000    # 10 km box
  N    <- 167      # -> ~60 m cells, ~28k per city

  MOSAIC_RULE <- paste0(
    '{"mosaicMethod":"esriMosaicAttribute","sortField":"CloudCover",',
    '"sortValue":0,"ascending":true,',
    '"where":"CloudCover < 0.05 AND AcquisitionDate BETWEEN ',
    "timestamp '2023-06-15 00:00:00' AND timestamp '2023-09-10 00:00:00'\"}")

  cities <- list(
    desmoines   = c(lon = -93.6250, lat = 41.5868),
    cedarrapids = c(lon = -91.6656, lat = 41.9779)
  )

  BANDS <- c(CoastalAerosol = 0, Blue = 1, Green = 2, Red = 3,
             NearInfrared = 4, SWIR1 = 5, SWIR2 = 6)

  export_tif <- function(url, bbox, size, dest, rendering = NULL, band_ids = NULL,
                         pixel_type = "F32", interp = NULL, mosaic = NULL) {
    q <- function(...) c("--data-urlencode", shQuote(sprintf(...)))
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
}

if (MODE == "kcames") {

  suppressPackageStartupMessages({library(dplyr); library(sf)})

  ab <- readRDS("data/airbnb-all-cities-clean.rds")
  ab_feats <- setdiff(names(ab$la), c("lat", "lon", "price", "y"))
  stopifnot(!"price" %in% ab_feats)

  mk_ab <- function(a, b, an, bn, file) {
    out <- setNames(lapply(list(a, b), function(d) {
      o <- d[, c("lon", "lat", ab_feats, "y")]
      o[Reduce(`&`, lapply(o, is.finite)), ]
    }), c(an, bn))
    saveRDS(out, file)
    cat(sprintf("%-34s %s %d / %s %d, %d features\n", file, an, nrow(out[[an]]),
                bn, nrow(out[[bn]]), length(ab_feats)))
  }

  mk_ab(ab$broward, ab$sandiego, "broward", "sandiego", "data/rev-brow-sd.rds")
  mk_ab(ab$chicago, ab$la,       "chicago", "la",       "data/rev-chi-la.rds")

  kc_feats <- c("lot_area", "living_area", "above_grade", "basement",
                "bedrooms", "bathrooms", "yr_built", "renovated")

  kc <- readr::read_csv("data/kc_house_data.csv", show_col_types = FALSE) |>
    transmute(price, lot_area = sqft_lot, living_area = sqft_living,
              above_grade = sqft_above, basement = sqft_basement, bedrooms,
              bathrooms, yr_built, renovated = as.integer(yr_renovated > 0),
              lon = long, lat) |>
    mutate(y = log(price) - mean(log(price))) |>
    select(lon, lat, all_of(kc_feats), y) |> as.data.frame()

  ames <- modeldata::ames |>
    transmute(price = Sale_Price, lot_area = Lot_Area, living_area = Gr_Liv_Area,
              above_grade = First_Flr_SF + Second_Flr_SF, basement = Total_Bsmt_SF,
              bedrooms = Bedroom_AbvGr,
              bathrooms = Full_Bath + Bsmt_Full_Bath + 0.5 * (Half_Bath + Bsmt_Half_Bath),
              yr_built = Year_Built,
              renovated = as.integer(Year_Remod_Add > Year_Built),
              lon = Longitude, lat = Latitude) |>
    mutate(y = log(price) - mean(log(price))) |>
    select(lon, lat, all_of(kc_feats), y) |> as.data.frame()

  saveRDS(list(kc = kc, ames = ames), "data/rev-kc-ames.rds")
  cat(sprintf("%-34s kc %d / ames %d, %d features\n",
              "data/rev-kc-ames.rds", nrow(kc), nrow(ames), length(kc_feats)))

  cat("\nNote: Ames has only 2,930 rows, so ames -> kc trains on a tenth of the\n")
  cat("data the forward run used. It is a weaker mirror than the other pairs\n")
  cat("and should be read as such rather than as a like-for-like reversal.\n")
}

if (MODE == "cdcclean") {
  suppressPackageStartupMessages({library(dplyr); library(sf)})
  feats <- c("BINGE","CSMOKING","LPA","SLEEP","ACCESS2","CHECKUP",
             "FOODINSECU","HOUSINSECU","LACKTRPT","GHLTH")
  raw <- readRDS("data/cdc-places-raw.rds")
  w <- raw |> mutate(data_value = as.numeric(data_value)) |>
    select(locationname, stateabbr, measureid, data_value, geolocation) |>
    tidyr::pivot_wider(names_from = measureid, values_from = data_value)
  xy <- do.call(rbind, lapply(w$geolocation$coordinates, function(c) c(lon = c[1], lat = c[2])))
  w <- w |> mutate(lon = xy[, "lon"], lat = xy[, "lat"]) |> select(-geolocation) |>
    filter(complete.cases(across(all_of(c("DIABETES", feats)))))
  out <- lapply(c(ca = "CA", wv = "WV"), function(st) {
    d <- w |> filter(stateabbr == st)
    o <- as.data.frame(d[, c("lon", "lat", feats)])
    o$y <- d$DIABETES
    o
  })
  saveRDS(out, "data/cdc-clean.rds")
  cat(sprintf("ca %d, wv %d, %d features -> data/cdc-clean.rds\n",
              nrow(out$ca), nrow(out$wv), length(feats)))
}

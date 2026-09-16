# Diagnostics and figures.
# Rscript R/analysis.R <moran|lagged|labelfree|figures>

MODE <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(MODE)) {
  MODE <- "moran"
}

MODE <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(MODE)) {
  MODE <- "moran"
}

if (MODE == "moran") {
  suppressPackageStartupMessages({
    library(dplyr)
    library(sf)
    library(sfdep)
    library(spdep)
  })

  k_nb <- 30L

  listw_of <- function(g) nb2listw(sfdep::st_knn(g, k = k_nb), style = "W")
  mi <- function(v, lw) {
    if (length(unique(v)) < 2) {
      return(NA_real_)
    }
    unname(moran.test(v, lw, zero.policy = TRUE)$estimate[1])
  }

  # Build every pair as (sf source, sf target, feature names) ----------------

  as_proj <- function(d) {
    st_as_sf(d, coords = c("lon", "lat"), crs = 4326) |> st_transform(3857)
  }

  ab <- readRDS("data/airbnb-all-cities-clean.rds")
  ab_feats <- setdiff(names(ab$la), c("lat", "lon", "price", "y"))

  tr <- readRDS("data/trees-clean.rds")
  tr_feats <- setdiff(names(tr$queens), c("boro", "lat", "lon", "y"))

  n311 <- readRDS("data/nyc311-clean.rds")
  n311_feats <- setdiff(names(n311$queens), c("lat", "lon", "y"))

  # King County -> Ames
  kc_feats <- c(
    "lot_area",
    "living_area",
    "above_grade",
    "basement",
    "bedrooms",
    "bathrooms",
    "yr_built",
    "renovated"
  )
  kc <- readr::read_csv("data/kc_house_data.csv", show_col_types = FALSE) |>
    transmute(
      price,
      lot_area = sqft_lot,
      living_area = sqft_living,
      above_grade = sqft_above,
      basement = sqft_basement,
      bedrooms,
      bathrooms,
      yr_built,
      renovated = as.integer(yr_renovated > 0),
      long,
      lat
    ) |>
    st_as_sf(coords = c("long", "lat"), crs = 4326) |>
    st_transform(3857) |>
    mutate(y = log(price) - mean(log(price)))
  ames <- modeldata::ames |>
    transmute(
      price = Sale_Price,
      lot_area = Lot_Area,
      living_area = Gr_Liv_Area,
      above_grade = First_Flr_SF + Second_Flr_SF,
      basement = Total_Bsmt_SF,
      bedrooms = Bedroom_AbvGr,
      bathrooms = Full_Bath +
        Bsmt_Full_Bath +
        0.5 * (Half_Bath + Bsmt_Half_Bath),
      yr_built = Year_Built,
      renovated = as.integer(Year_Remod_Add > Year_Built),
      Longitude,
      Latitude
    ) |>
    st_as_sf(coords = c("Longitude", "Latitude"), crs = 4326) |>
    st_transform(3857) |>
    mutate(y = log(price) - mean(log(price)))

  # CDC PLACES CA -> WV
  cdc_feats <- c(
    "BINGE",
    "CSMOKING",
    "LPA",
    "SLEEP",
    "ACCESS2",
    "CHECKUP",
    "FOODINSECU",
    "HOUSINSECU",
    "LACKTRPT",
    "GHLTH"
  )
  cdc_raw <- readRDS("data/cdc-places-raw.rds")
  cdc_wide <- cdc_raw |>
    mutate(data_value = as.numeric(data_value)) |>
    select(locationname, stateabbr, measureid, data_value, geolocation) |>
    tidyr::pivot_wider(names_from = measureid, values_from = data_value)
  cdc_xy <- do.call(
    rbind,
    lapply(cdc_wide$geolocation$coordinates, function(c) {
      c(lon = c[1], lat = c[2])
    })
  )
  cdc_wide <- cdc_wide |>
    mutate(lon = cdc_xy[, "lon"], lat = cdc_xy[, "lat"]) |>
    select(-geolocation) |>
    filter(complete.cases(across(all_of(c("DIABETES", cdc_feats)))))
  cdc <- st_as_sf(cdc_wide, coords = c("lon", "lat"), crs = 4326) |>
    st_transform(3857) |>
    mutate(y = DIABETES)

  pairs <- list(
    list(
      nm = "King County -> Ames (house price)",
      src = kc,
      tgt = ames,
      feats = kc_feats,
      outcome = "WIN  +0.218",
      passes_ols = TRUE
    ),
    list(
      nm = "Broward -> San Diego (Airbnb price)",
      src = as_proj(ab$broward),
      tgt = as_proj(ab$sandiego),
      feats = ab_feats,
      outcome = "WIN  +0.030",
      passes_ols = TRUE
    ),
    list(
      nm = "Chicago -> LA (Airbnb price)",
      src = as_proj(ab$chicago),
      tgt = as_proj(ab$la),
      feats = ab_feats,
      outcome = "WIN  +0.022",
      passes_ols = TRUE
    ),
    list(
      nm = "CA -> WV (diabetes prevalence)",
      src = cdc |> filter(stateabbr == "CA"),
      tgt = cdc |> filter(stateabbr == "WV"),
      feats = cdc_feats,
      outcome = "LOSS -0.040",
      passes_ols = TRUE
    ),
    list(
      nm = "Queens -> Brooklyn (tree DBH)",
      src = as_proj(tr$queens),
      tgt = as_proj(tr$brooklyn),
      feats = tr_feats,
      outcome = "LOSS -0.144",
      passes_ols = TRUE
    ),
    list(
      nm = "Queens -> Brooklyn (311 response)",
      src = as_proj(n311$queens),
      tgt = as_proj(n311$brooklyn),
      feats = n311_feats,
      outcome = "LOSS -0.057",
      passes_ols = TRUE
    ),
    list(
      nm = "LA -> NYC (Airbnb price)",
      src = as_proj(ab$la),
      tgt = as_proj(ab$nyc),
      feats = ab_feats,
      outcome = "FAIL: DGP",
      passes_ols = FALSE
    ),
    list(
      nm = "Chicago -> Nashville (Airbnb price)",
      src = as_proj(ab$chicago),
      tgt = as_proj(ab$nashville),
      feats = ab_feats,
      outcome = "FAIL: DGP",
      passes_ols = FALSE
    )
  )

  profile_pair <- function(p) {
    fs <- st_drop_geometry(p$src)[, c(p$feats, "y")]
    ft <- st_drop_geometry(p$tgt)[, c(p$feats, "y")]
    lw_s <- listw_of(st_geometry(p$src))
    lw_t <- listw_of(st_geometry(p$tgt))
    m <- lm(y ~ ., data = fs)

    i_res_s <- mi(residuals(m), lw_s)
    i_res_t <- mi(ft$y - predict(m, newdata = ft), lw_t)
    cov_s <- vapply(p$feats, function(f) mi(fs[[f]], lw_s), numeric(1))
    cov_t <- vapply(p$feats, function(f) mi(ft[[f]], lw_t), numeric(1))

    data.frame(
      pair = p$nm,
      outcome = p$outcome,
      ols = p$passes_ols,
      n_feat = length(p$feats),
      res_src = i_res_s,
      cov_src_med = median(cov_s, na.rm = TRUE),
      frac_src_exceed = mean(cov_s > i_res_s, na.rm = TRUE),
      res_tgt = i_res_t,
      cov_tgt_med = median(cov_t, na.rm = TRUE),
      frac_tgt_exceed = mean(cov_t > i_res_t, na.rm = TRUE)
    )
  }

  out <- do.call(
    rbind,
    lapply(pairs, function(p) {
      cat(sprintf("profiling %s ...\n", p$nm))
      profile_pair(p)
    })
  )

  cat(
    "\n=== Deployment-time statistics (SOURCE region only; no target labels) ===\n"
  )
  print(
    out[, c("pair", "outcome", "res_src", "cov_src_med", "frac_src_exceed")],
    row.names = FALSE,
    digits = 3
  )

  cat(
    "\n=== Oracle statistics (residuals in the target region; needs labels) ===\n"
  )
  print(
    out[, c("pair", "outcome", "res_tgt", "cov_tgt_med", "frac_tgt_exceed")],
    row.names = FALSE,
    digits = 3
  )

  # Separation among pairs that pass the DGP screen -------------------------
  g <- out[out$ols, ]
  win <- grepl("^WIN", g$outcome)
  cat("\n=== Separation among the six pairs that pass the OLS/DGP screen ===\n")
  for (v in c("res_src", "frac_src_exceed", "res_tgt", "frac_tgt_exceed")) {
    cat(sprintf(
      "%-16s wins [%.3f, %.3f]   losses [%.3f, %.3f]   %s\n",
      v,
      min(g[[v]][win]),
      max(g[[v]][win]),
      min(g[[v]][!win]),
      max(g[[v]][!win]),
      if (min(g[[v]][win]) > max(g[[v]][!win])) {
        "SEPARATES (higher = win)"
      } else if (max(g[[v]][win]) < min(g[[v]][!win])) {
        "SEPARATES (lower = win)"
      } else {
        "overlaps"
      }
    ))
  }

  saveRDS(out, "data/moran-calibrate2.rds")
  cat("\nSaved to data/moran-calibrate2.rds\n")
}

if (MODE == "lagged") {
  suppressPackageStartupMessages({
    library(dplyr)
    library(sf)
    library(sfdep)
    library(spdep)
  })

  rsq_trad <- function(truth, est) {
    1 - sum((truth - est)^2) / sum((truth - mean(truth))^2)
  }

  lag_r2 <- function(d, feats, coords_are_metres = FALSE, k = 30) {
    g <- if (coords_are_metres) {
      st_geometry(st_as_sf(d, coords = c("lon", "lat"), crs = NA))
    } else {
      st_geometry(
        st_as_sf(d, coords = c("lon", "lat"), crs = 4326) |> st_transform(3857)
      )
    }
    nb <- sfdep::st_knn(g, k = min(k, length(g) - 1L))
    lw <- nb2listw(nb, style = "W", zero.policy = TRUE)

    fr <- d[, c(feats, "y")]
    m <- lm(y ~ ., data = fr)
    res <- residuals(m)

    X <- as.matrix(d[, feats])
    lagX <- as.data.frame(lag.listw(lw, X, zero.policy = TRUE))
    names(lagX) <- paste0("lag_", feats)
    lagX$r <- res

    ml <- lm(r ~ ., data = lagX)
    # Also report residual Moran's I for the direct comparison.
    mi <- unname(moran.test(res, lw, zero.policy = TRUE)$estimate[1])
    c(
      lag_r2 = summary(ml)$r.squared,
      resid_moran = mi,
      insample = summary(m)$r.squared
    )
  }

  ab <- readRDS("data/airbnb-all-cities-clean.rds")
  abf <- setdiff(names(ab$la), c("lat", "lon", "price", "y"))
  tr <- readRDS("data/trees-clean.rds")
  trf <- setdiff(names(tr$queens), c("boro", "lat", "lon", "y"))
  n311 <- readRDS("data/nyc311-clean.rds")
  n3f <- setdiff(names(n311$queens), c("lat", "lon", "y"))
  rv <- readRDS("data/airbnb-reviews-clean.rds")
  rvf <- setdiff(names(rv$broward), c("lat", "lon", "y"))
  iowa <- readRDS("data/ndvi-clean-iowa.rds")
  iof <- setdiff(names(iowa$summit), c("lon", "lat", "y"))

  kcf <- c(
    "lot_area",
    "living_area",
    "above_grade",
    "basement",
    "bedrooms",
    "bathrooms",
    "yr_built",
    "renovated"
  )
  kc <- readr::read_csv("data/kc_house_data.csv", show_col_types = FALSE) |>
    transmute(
      price,
      lot_area = sqft_lot,
      living_area = sqft_living,
      above_grade = sqft_above,
      basement = sqft_basement,
      bedrooms,
      bathrooms,
      yr_built,
      renovated = as.integer(yr_renovated > 0),
      lon = long,
      lat
    ) |>
    mutate(y = log(price) - mean(log(price))) |>
    select(-price) |>
    as.data.frame()

  cdcf <- c(
    "BINGE",
    "CSMOKING",
    "LPA",
    "SLEEP",
    "ACCESS2",
    "CHECKUP",
    "FOODINSECU",
    "HOUSINSECU",
    "LACKTRPT",
    "GHLTH"
  )
  craw <- readRDS("data/cdc-places-raw.rds")
  cw <- craw |>
    mutate(data_value = as.numeric(data_value)) |>
    select(locationname, stateabbr, measureid, data_value, geolocation) |>
    tidyr::pivot_wider(names_from = measureid, values_from = data_value)
  cxy <- do.call(
    rbind,
    lapply(cw$geolocation$coordinates, function(c) c(lon = c[1], lat = c[2]))
  )
  cw <- cw |>
    mutate(lon = cxy[, "lon"], lat = cxy[, "lat"]) |>
    select(-geolocation) |>
    filter(complete.cases(across(all_of(c("DIABETES", cdcf)))))
  ca <- cw |>
    filter(stateabbr == "CA") |>
    mutate(y = DIABETES) |>
    as.data.frame()

  jobs <- list(
    list(
      nm = "Broward -> San Diego (price)",
      d = ab$broward,
      f = abf,
      m = FALSE,
      out = "WIN  +0.030"
    ),
    list(
      nm = "Chicago -> LA (price)",
      d = ab$chicago,
      f = abf,
      m = FALSE,
      out = "WIN  +0.022"
    ),
    list(
      nm = "Queens -> Brooklyn (311)",
      d = n311$queens,
      f = n3f,
      m = FALSE,
      out = "LOSS -0.057"
    ),
    list(
      nm = "Queens -> Brooklyn (trees)",
      d = tr$queens,
      f = trf,
      m = FALSE,
      out = "LOSS -0.144"
    ),
    list(
      nm = "Broward -> SD (reviews)",
      d = rv$broward,
      f = rvf,
      m = FALSE,
      out = "LOSS -0.086"
    ),
    list(
      nm = "Iowa farms (NDVI)",
      d = iowa$summit,
      f = iof,
      m = TRUE,
      out = "not run"
    ),
    list(
      nm = "King County -> Ames (price)",
      d = kc,
      f = kcf,
      m = FALSE,
      out = "WIN  +0.218"
    ),
    list(
      nm = "CA -> WV (diabetes)",
      d = ca,
      f = cdcf,
      m = FALSE,
      out = "LOSS -0.040"
    )
  )

  res <- do.call(
    rbind,
    lapply(jobs, function(j) {
      cat(sprintf("  %s\n", j$nm))
      v <- lag_r2(j$d, j$f, j$m)
      data.frame(
        pair = j$nm,
        outcome = j$out,
        insample = v[["insample"]],
        resid_moran = v[["resid_moran"]],
        lag_r2 = v[["lag_r2"]]
      )
    })
  )

  cat("\n=== Do NEIGHBOURS' COVARIATES explain what the model missed? ===\n")
  print(res[order(-res$lag_r2), ], row.names = FALSE, digits = 3)

  cat(
    "\nlag_r2      share of the OLS residual recoverable from neighbours' covariates.\n"
  )
  cat("            This is what GraphSAGE actually has access to.\n")
  cat(
    "resid_moran the old statistic: how spatially structured the residual is,\n"
  )
  cat("            regardless of whether anything observable explains it.\n")

  w <- grepl("^WIN", res$outcome)
  l <- grepl("^LOSS", res$outcome)
  for (v in c("lag_r2", "resid_moran")) {
    cat(sprintf(
      "\n%-12s wins [%.3f, %.3f]  losses [%.3f, %.3f]  %s",
      v,
      min(res[[v]][w]),
      max(res[[v]][w]),
      min(res[[v]][l]),
      max(res[[v]][l]),
      if (min(res[[v]][w]) > max(res[[v]][l])) "SEPARATES" else "overlaps"
    ))
  }
  cat("\n")

  saveRDS(res, "data/lagged-covariate-screen.rds")
  cat("\nSaved to data/lagged-covariate-screen.rds\n")
}

if (MODE == "labelfree") {
  suppressPackageStartupMessages({
    library(dplyr)
    library(sf)
  })

  set.seed(1)

  rsq_trad <- function(truth, est) {
    1 - sum((truth - est)^2) / sum((truth - mean(truth))^2)
  }

  # --- signals -------------------------------------------------------------

  # Domain classifier AUC via logistic regression on standardised covariates.
  domain_auc <- function(Xs, Xt) {
    n <- min(nrow(Xs), nrow(Xt), 4000)
    a <- Xs[sample(nrow(Xs), n), , drop = FALSE]
    b <- Xt[sample(nrow(Xt), n), , drop = FALSE]
    d <- as.data.frame(rbind(a, b))
    mu <- colMeans(d)
    sdv <- apply(d, 2, sd)
    sdv[sdv == 0] <- 1
    d <- as.data.frame(scale(d, mu, sdv))
    d$lab <- rep(0:1, each = n)
    # half for fitting, half for scoring, so AUC is out-of-sample
    i <- sample(nrow(d), floor(nrow(d) / 2))
    m <- suppressWarnings(glm(lab ~ ., data = d[i, ], family = binomial()))
    p <- suppressWarnings(predict(m, newdata = d[-i, ], type = "response"))
    y <- d$lab[-i]
    r <- rank(p)
    (sum(r[y == 1]) - sum(y == 1) * (sum(y == 1) + 1) / 2) /
      (sum(y == 1) * sum(y == 0))
  }

  # Share of target rows inside the source's 1st-99th percentile box.
  coverage <- function(Xs, Xt) {
    lo <- apply(Xs, 2, quantile, 0.01, na.rm = TRUE)
    hi <- apply(Xs, 2, quantile, 0.99, na.rm = TRUE)
    mean(apply(Xt, 1, function(r) all(r >= lo & r <= hi)))
  }

  beta_instability <- function(X, y, coords, nblocks = 8) {
    ok <- stats::complete.cases(X, y)
    X <- X[ok, , drop = FALSE]
    y <- y[ok]
    coords <- coords[ok, , drop = FALSE]
    km <- kmeans(scale(coords), centers = nblocks, nstart = 5, iter.max = 50)
    Z <- scale(X) # standardised, so coefficients are comparable
    Z[!is.finite(Z)] <- 0
    d <- as.data.frame(Z)
    d$y <- y
    B <- do.call(
      rbind,
      lapply(seq_len(nblocks), function(k) {
        idx <- which(km$cluster == k)
        if (length(idx) < 10 * ncol(X)) {
          return(NULL)
        }
        coef(lm(y ~ ., data = d[idx, ]))[-1]
      })
    )
    if (is.null(B) || nrow(B) < 3) {
      return(NA_real_)
    }
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
    mk("Broward -> San Diego (price)", ab$broward, ab$sandiego, abf),
    mk("Chicago -> LA (price)", ab$chicago, ab$la, abf),
    mk("Queens -> Brooklyn (trees)", tr$queens, tr$brooklyn, trf),
    mk("Queens -> Brooklyn (311)", n311$queens, n311$brooklyn, n3f),
    mk("Broward -> San Diego (reviews)", rv$broward, rv$sandiego, rvf),
    mk("LA -> NYC (price)", ab$la, ab$nyc, abf),
    mk("Chicago -> Nashville (price)", ab$chicago, ab$nashville, abf),
    mk("Queens -> Brooklyn (bldg age)", pl$queens, pl$brooklyn, plf),
    mk("Bronx -> Brooklyn (bldg age)", pl$bronx, pl$brooklyn, plf),
    mk("Broward -> San Diego (avail)", av$broward, av$sandiego, avf)
  )

  run_pair <- function(p) {
    cat(sprintf("  %s\n", p$nm))
    Xs <- as.matrix(p$src[, p$feats])
    Xt <- as.matrix(p$tgt[, p$feats])
    ys <- p$src$y
    yt <- p$tgt$y
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
    a <- res[[v]][res$transfers]
    b <- res[[v]][!res$transfers]
    a <- a[is.finite(a)]
    b <- b[is.finite(b)]
    verdict <- if (min(a) > max(b)) {
      "SEPARATES (higher = transfers)"
    } else if (max(a) < min(b)) {
      "SEPARATES (lower = transfers)"
    } else {
      "OVERLAPS"
    }
    cat(sprintf(
      "%-9s transfers [%.3f, %.3f]   fails [%.3f, %.3f]   %s\n",
      v,
      min(a),
      max(a),
      min(b),
      max(b),
      verdict
    ))
  }

  cat(
    "\nauc      ~0.5 means the two regions' inputs look alike; ~1.0 means disjoint.\n"
  )
  cat("coverage share of target rows inside the source's input range.\n")
  cat(
    "beta_cv  how much the relationship already moves WITHIN the source region.\n"
  )

  saveRDS(res, "data/transfer-screen-labelfree.rds")
  cat("\nSaved to data/transfer-screen-labelfree.rds\n")
}

if (MODE == "figures") {
  suppressPackageStartupMessages({
    library(ggplot2)
    library(dplyr)
  })

  d <- readRDS("data/moran-calibrate2.rds")

  margin <- c(
    "King County -> Ames (house price)" = 0.218,
    "Broward -> San Diego (Airbnb price)" = 0.030,
    "Chicago -> LA (Airbnb price)" = 0.022,
    "CA -> WV (diabetes prevalence)" = -0.040,
    "Queens -> Brooklyn (311 response)" = -0.057,
    "Queens -> Brooklyn (tree DBH)" = -0.144
  )

  p <- d |>
    filter(ols) |>
    mutate(
      margin = unname(margin[pair]),
      result = ifelse(margin > 0, "GraphSAGE wins", "GraphSAGE loses"),
      # Two pairs share the region names (trees and 311), so keep the target.
      label = ifelse(
        grepl("Queens", pair),
        sub("Queens -> Brooklyn \\((.*)\\)", "Queens->Brooklyn: \\1", pair),
        sub(" \\(.*", "", pair)
      ),
      kind = ifelse(grepl("price", pair), "price target", "non-price target")
    )

  lo <- max(p$frac_src_exceed[p$margin > 0])
  hi <- min(p$frac_src_exceed[p$margin < 0])
  thr <- (lo + hi) / 2

  g <- ggplot(p, aes(frac_src_exceed, margin)) +
    annotate(
      "rect",
      xmin = lo,
      xmax = hi,
      ymin = -Inf,
      ymax = Inf,
      fill = "grey85",
      alpha = 0.6
    ) +
    annotate(
      "text",
      x = thr,
      y = min(p$margin) * 0.55,
      label = sprintf("no pair lands\nin this gap\n(%.2f - %.2f)", lo, hi),
      size = 2.9,
      colour = "grey30"
    ) +
    geom_hline(yintercept = 0, linewidth = 0.3, colour = "grey40") +
    geom_point(aes(colour = result, shape = kind), size = 3.4) +
    geom_text(
      aes(label = label),
      hjust = -0.12,
      size = 2.9,
      colour = "grey20"
    ) +
    scale_colour_manual(
      values = c("GraphSAGE wins" = "#1b7837", "GraphSAGE loses" = "#b2182b")
    ) +
    scale_shape_manual(
      values = c("price target" = 16, "non-price target" = 17)
    ) +
    scale_x_continuous(expand = expansion(mult = c(0.08, 0.35))) +
    labs(
      title = "A pre-fit statistic separates the wins from the losses",
      subtitle = paste0(
        "Share of covariates more spatially autocorrelated than the OLS residual, source region only.\n",
        "Six pairs that pass the OLS/DGP screen. The threshold is fitted to these six points."
      ),
      x = "fraction of covariates with Moran's I above the residual's, source region (KNN-30)",
      y = "GraphSAGE + LayerNorm margin over best non-graph arm",
      colour = NULL,
      shape = NULL
    ) +
    theme_minimal(base_size = 11) +
    theme(legend.position = "bottom", panel.grid.minor = element_blank())

  dir.create("images", showWarnings = FALSE)
  ggsave("images/fig-moran-rule.png", g, width = 8, height = 5.5, dpi = 200)
  cat("Saved images/fig-moran-rule.png\n")
  print(
    p[, c("pair", "res_src", "frac_src_exceed", "margin")],
    row.names = FALSE,
    digits = 3
  )
  cat(sprintf(
    "\nGap between highest loss (%.3f) and lowest win (%.3f)\n",
    lo,
    hi
  ))
}

if (MODE == "export") {
  # CSV copies of every results table, for sharing outside the repo.
  library(nanoparquet)
  dir.create("reports/csv", showWarnings = FALSE)
  for (f in list.files(
    "data/results",
    pattern = "[.]parquet$",
    full.names = TRUE
  )) {
    d <- read_parquet(f)
    out <- file.path("reports/csv", sub("[.]parquet$", ".csv", basename(f)))
    write.csv(d, out, row.names = FALSE)
    cat(sprintf("%-28s %5d x %2d -> %s\n", basename(f), nrow(d), ncol(d), out))
  }
}

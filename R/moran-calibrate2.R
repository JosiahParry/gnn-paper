# Moran diagnostic calibration, all six transfer pairs with a known outcome.
#
# Extends R/moran-calibrate.R with King County -> Ames and CDC CA -> WV, the
# two pairs whose data is assembled in their own cores rather than a shared
# clean rds. That brings the calibration set to three wins and three losses,
# which is the minimum worth drawing a threshold on -- and even then the
# threshold is fitted to six points and must be reported as such.
#
# The question: is there a statistic, computable from the SOURCE region
# alone (no target labels, as in a real inductive deployment), that
# separates the pairs where GraphSAGE wins from the pairs where it loses?

suppressPackageStartupMessages({
  library(dplyr); library(sf); library(sfdep); library(spdep)
})

k_nb <- 30L

listw_of <- function(g) nb2listw(sfdep::st_knn(g, k = k_nb), style = "W")
mi <- function(v, lw) {
  if (length(unique(v)) < 2) return(NA_real_)
  unname(moran.test(v, lw, zero.policy = TRUE)$estimate[1])
}

# Build every pair as (sf source, sf target, feature names) ----------------

as_proj <- function(d) st_as_sf(d, coords = c("lon", "lat"), crs = 4326) |> st_transform(3857)

ab <- readRDS("data/airbnb-all-cities-clean.rds")
ab_feats <- setdiff(names(ab$la), c("lat", "lon", "price", "y"))

tr <- readRDS("data/trees-clean.rds")
tr_feats <- setdiff(names(tr$queens), c("boro", "lat", "lon", "y"))

n311 <- readRDS("data/nyc311-clean.rds")
n311_feats <- setdiff(names(n311$queens), c("lat", "lon", "y"))

# King County -> Ames
kc_feats <- c("lot_area", "living_area", "above_grade", "basement",
              "bedrooms", "bathrooms", "yr_built", "renovated")
kc <- readr::read_csv("data/kc_house_data.csv", show_col_types = FALSE) |>
  transmute(price, lot_area = sqft_lot, living_area = sqft_living,
            above_grade = sqft_above, basement = sqft_basement, bedrooms,
            bathrooms, yr_built, renovated = as.integer(yr_renovated > 0),
            long, lat) |>
  st_as_sf(coords = c("long", "lat"), crs = 4326) |> st_transform(3857) |>
  mutate(y = log(price) - mean(log(price)))
ames <- modeldata::ames |>
  transmute(price = Sale_Price, lot_area = Lot_Area, living_area = Gr_Liv_Area,
            above_grade = First_Flr_SF + Second_Flr_SF, basement = Total_Bsmt_SF,
            bedrooms = Bedroom_AbvGr,
            bathrooms = Full_Bath + Bsmt_Full_Bath + 0.5 * (Half_Bath + Bsmt_Half_Bath),
            yr_built = Year_Built,
            renovated = as.integer(Year_Remod_Add > Year_Built),
            Longitude, Latitude) |>
  st_as_sf(coords = c("Longitude", "Latitude"), crs = 4326) |> st_transform(3857) |>
  mutate(y = log(price) - mean(log(price)))

# CDC PLACES CA -> WV
cdc_feats <- c("BINGE", "CSMOKING", "LPA", "SLEEP", "ACCESS2", "CHECKUP",
               "FOODINSECU", "HOUSINSECU", "LACKTRPT", "GHLTH")
cdc_raw <- readRDS("data/cdc-places-raw.rds")
cdc_wide <- cdc_raw |>
  mutate(data_value = as.numeric(data_value)) |>
  select(locationname, stateabbr, measureid, data_value, geolocation) |>
  tidyr::pivot_wider(names_from = measureid, values_from = data_value)
cdc_xy <- do.call(rbind, lapply(cdc_wide$geolocation$coordinates, function(c) c(lon = c[1], lat = c[2])))
cdc_wide <- cdc_wide |>
  mutate(lon = cdc_xy[, "lon"], lat = cdc_xy[, "lat"]) |> select(-geolocation) |>
  filter(complete.cases(across(all_of(c("DIABETES", cdc_feats)))))
cdc <- st_as_sf(cdc_wide, coords = c("lon", "lat"), crs = 4326) |>
  st_transform(3857) |> mutate(y = DIABETES)

pairs <- list(
  list(nm = "King County -> Ames (house price)", src = kc, tgt = ames,
       feats = kc_feats, outcome = "WIN  +0.218", passes_ols = TRUE),
  list(nm = "Broward -> San Diego (Airbnb price)", src = as_proj(ab$broward), tgt = as_proj(ab$sandiego),
       feats = ab_feats, outcome = "WIN  +0.030", passes_ols = TRUE),
  list(nm = "Chicago -> LA (Airbnb price)", src = as_proj(ab$chicago), tgt = as_proj(ab$la),
       feats = ab_feats, outcome = "WIN  +0.022", passes_ols = TRUE),
  list(nm = "CA -> WV (diabetes prevalence)", src = cdc |> filter(stateabbr == "CA"),
       tgt = cdc |> filter(stateabbr == "WV"),
       feats = cdc_feats, outcome = "LOSS -0.040", passes_ols = TRUE),
  list(nm = "Queens -> Brooklyn (tree DBH)", src = as_proj(tr$queens), tgt = as_proj(tr$brooklyn),
       feats = tr_feats, outcome = "LOSS -0.144", passes_ols = TRUE),
  list(nm = "Queens -> Brooklyn (311 response)", src = as_proj(n311$queens), tgt = as_proj(n311$brooklyn),
       feats = n311_feats, outcome = "LOSS -0.057", passes_ols = TRUE),
  list(nm = "LA -> NYC (Airbnb price)", src = as_proj(ab$la), tgt = as_proj(ab$nyc),
       feats = ab_feats, outcome = "FAIL: DGP", passes_ols = FALSE),
  list(nm = "Chicago -> Nashville (Airbnb price)", src = as_proj(ab$chicago), tgt = as_proj(ab$nashville),
       feats = ab_feats, outcome = "FAIL: DGP", passes_ols = FALSE)
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
    pair = p$nm, outcome = p$outcome, ols = p$passes_ols, n_feat = length(p$feats),
    res_src = i_res_s, cov_src_med = median(cov_s, na.rm = TRUE),
    frac_src_exceed = mean(cov_s > i_res_s, na.rm = TRUE),
    res_tgt = i_res_t, cov_tgt_med = median(cov_t, na.rm = TRUE),
    frac_tgt_exceed = mean(cov_t > i_res_t, na.rm = TRUE)
  )
}

out <- do.call(rbind, lapply(pairs, function(p) {
  cat(sprintf("profiling %s ...\n", p$nm)); profile_pair(p)
}))

cat("\n=== Deployment-time statistics (SOURCE region only; no target labels) ===\n")
print(out[, c("pair", "outcome", "res_src", "cov_src_med", "frac_src_exceed")],
      row.names = FALSE, digits = 3)

cat("\n=== Oracle statistics (residuals in the target region; needs labels) ===\n")
print(out[, c("pair", "outcome", "res_tgt", "cov_tgt_med", "frac_tgt_exceed")],
      row.names = FALSE, digits = 3)

# Separation among pairs that pass the DGP screen -------------------------
g <- out[out$ols, ]
win <- grepl("^WIN", g$outcome)
cat("\n=== Separation among the six pairs that pass the OLS/DGP screen ===\n")
for (v in c("res_src", "frac_src_exceed", "res_tgt", "frac_tgt_exceed")) {
  cat(sprintf("%-16s wins [%.3f, %.3f]   losses [%.3f, %.3f]   %s\n", v,
              min(g[[v]][win]), max(g[[v]][win]),
              min(g[[v]][!win]), max(g[[v]][!win]),
              if (min(g[[v]][win]) > max(g[[v]][!win])) "SEPARATES (higher = win)"
              else if (max(g[[v]][win]) < min(g[[v]][!win])) "SEPARATES (lower = win)"
              else "overlaps"))
}

saveRDS(out, "data/moran-calibrate2.rds")
cat("\nSaved to data/moran-calibrate2.rds\n")

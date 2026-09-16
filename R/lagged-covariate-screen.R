# Fixing the pre-test rule.
#
# THE PROBLEM. Section 2a says a graph helps when the OLS residual is more
# spatially structured than the covariates. Our own simulation contradicts
# that. Scenario C is spatially correlated NOISE with a correctly specified
# mean: residual structure is high, covariate structure is lower, so the rule
# predicts a win -- and GraphSAGE scored 0.006 against OLS's 0.280, the most
# decisive loss in the whole grid.
#
# Why the rule is wrong: GraphSAGE aggregates neighbours' COVARIATES, never
# their labels -- that is what makes it inductive. Spatial structure that
# lives in the target and is not reflected in any covariate is therefore
# unreachable to it. Residual Moran's I cannot tell those two cases apart:
#
#   (a) residual structure caused by an unobserved field that neighbours'
#       covariates proxy for  -> a graph CAN recover it
#   (b) residual structure that is pure spatial noise                -> nothing can
#
# THE FIX. Ask directly whether neighbours' covariates explain the residual:
#
#   1. Fit OLS in the source region.
#   2. Build the spatial lag of every covariate.
#   3. Regress the OLS residual on those lagged covariates.
#   4. R2 of that regression = how much of what the model missed is
#      recoverable from neighbours' observable features.
#
# This is exactly the quantity GraphSAGE has access to, it is computed on the
# SOURCE region alone (no target labels), and unlike residual Moran's I it
# separates (a) from (b) by construction.

suppressPackageStartupMessages({
  library(dplyr); library(sf); library(sfdep); library(spdep)
})

rsq_trad <- function(truth, est) 1 - sum((truth - est)^2) / sum((truth - mean(truth))^2)

# lag_r2: the proposed statistic. deg_free_note -- lagged covariates are
# smooth, so this is optimistic in absolute terms; what matters is the
# ordering across problems, all computed identically.
lag_r2 <- function(d, feats, coords_are_metres = FALSE, k = 30) {
  g <- if (coords_are_metres) {
    st_geometry(st_as_sf(d, coords = c("lon", "lat"), crs = NA))
  } else {
    st_geometry(st_as_sf(d, coords = c("lon", "lat"), crs = 4326) |> st_transform(3857))
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
  c(lag_r2 = summary(ml)$r.squared, resid_moran = mi, insample = summary(m)$r.squared)
}

ab   <- readRDS("data/airbnb-all-cities-clean.rds")
abf  <- setdiff(names(ab$la), c("lat", "lon", "price", "y"))
tr   <- readRDS("data/trees-clean.rds")
trf  <- setdiff(names(tr$queens), c("boro", "lat", "lon", "y"))
n311 <- readRDS("data/nyc311-clean.rds")
n3f  <- setdiff(names(n311$queens), c("lat", "lon", "y"))
rv   <- readRDS("data/airbnb-reviews-clean.rds")
rvf  <- setdiff(names(rv$broward), c("lat", "lon", "y"))
iowa <- readRDS("data/ndvi-clean-iowa.rds")
iof  <- setdiff(names(iowa$summit), c("lon", "lat", "y"))

# The two pairs that destroyed the PREVIOUS version of this rule. Any new
# statistic has to face them before it is believed.
kcf <- c("lot_area","living_area","above_grade","basement","bedrooms",
         "bathrooms","yr_built","renovated")
kc <- readr::read_csv("data/kc_house_data.csv", show_col_types = FALSE) |>
  transmute(price, lot_area = sqft_lot, living_area = sqft_living,
            above_grade = sqft_above, basement = sqft_basement, bedrooms,
            bathrooms, yr_built, renovated = as.integer(yr_renovated > 0),
            lon = long, lat) |>
  mutate(y = log(price) - mean(log(price))) |> select(-price) |> as.data.frame()

cdcf <- c("BINGE","CSMOKING","LPA","SLEEP","ACCESS2","CHECKUP",
          "FOODINSECU","HOUSINSECU","LACKTRPT","GHLTH")
craw <- readRDS("data/cdc-places-raw.rds")
cw <- craw |> mutate(data_value = as.numeric(data_value)) |>
  select(locationname, stateabbr, measureid, data_value, geolocation) |>
  tidyr::pivot_wider(names_from = measureid, values_from = data_value)
cxy <- do.call(rbind, lapply(cw$geolocation$coordinates, function(c) c(lon = c[1], lat = c[2])))
cw <- cw |> mutate(lon = cxy[,"lon"], lat = cxy[,"lat"]) |> select(-geolocation) |>
  filter(complete.cases(across(all_of(c("DIABETES", cdcf)))))
ca <- cw |> filter(stateabbr == "CA") |> mutate(y = DIABETES) |> as.data.frame()

jobs <- list(
  list(nm = "Broward -> San Diego (price)", d = ab$broward,  f = abf, m = FALSE, out = "WIN  +0.030"),
  list(nm = "Chicago -> LA (price)",        d = ab$chicago,  f = abf, m = FALSE, out = "WIN  +0.022"),
  list(nm = "Queens -> Brooklyn (311)",     d = n311$queens, f = n3f, m = FALSE, out = "LOSS -0.057"),
  list(nm = "Queens -> Brooklyn (trees)",   d = tr$queens,   f = trf, m = FALSE, out = "LOSS -0.144"),
  list(nm = "Broward -> SD (reviews)",      d = rv$broward,  f = rvf, m = FALSE, out = "LOSS -0.086"),
  list(nm = "Iowa farms (NDVI)",            d = iowa$summit, f = iof, m = TRUE,  out = "not run"),
  list(nm = "King County -> Ames (price)",   d = kc,          f = kcf, m = FALSE, out = "WIN  +0.218"),
  list(nm = "CA -> WV (diabetes)",           d = ca,          f = cdcf, m = FALSE, out = "LOSS -0.040")
)

res <- do.call(rbind, lapply(jobs, function(j) {
  cat(sprintf("  %s\n", j$nm))
  v <- lag_r2(j$d, j$f, j$m)
  data.frame(pair = j$nm, outcome = j$out, insample = v[["insample"]],
             resid_moran = v[["resid_moran"]], lag_r2 = v[["lag_r2"]])
}))

cat("\n=== Do NEIGHBOURS' COVARIATES explain what the model missed? ===\n")
print(res[order(-res$lag_r2), ], row.names = FALSE, digits = 3)

cat("\nlag_r2      share of the OLS residual recoverable from neighbours' covariates.\n")
cat("            This is what GraphSAGE actually has access to.\n")
cat("resid_moran the old statistic: how spatially structured the residual is,\n")
cat("            regardless of whether anything observable explains it.\n")

w <- grepl("^WIN", res$outcome); l <- grepl("^LOSS", res$outcome)
for (v in c("lag_r2", "resid_moran")) {
  cat(sprintf("\n%-12s wins [%.3f, %.3f]  losses [%.3f, %.3f]  %s", v,
              min(res[[v]][w]), max(res[[v]][w]), min(res[[v]][l]), max(res[[v]][l]),
              if (min(res[[v]][w]) > max(res[[v]][l])) "SEPARATES" else "overlaps"))
}
cat("\n")

saveRDS(res, "data/lagged-covariate-screen.rds")
cat("\nSaved to data/lagged-covariate-screen.rds\n")

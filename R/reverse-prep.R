# Build clean two-region files so every transfer can be run in BOTH
# directions through the shared R/pair-core.R.
#
# The three price wins were only ever run one way. The two losses were run
# both ways and replicated, so the evidence was stronger for the results that
# went against us than for the results the paper rests on. This fixes that.
#
# LEAKAGE GUARD: the Airbnb tables carry a `price` column and the target is
# log(price). pair-core.R takes every non-y column as a feature, so `price`
# must be dropped here or the model trivially predicts itself.

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

# King County / Ames. Rebuilt from source so both regions share one feature
# set and plain lon/lat, which is what pair-core.R expects.
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

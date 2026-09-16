# Building energy benchmarking: Chicago and Seattle, data year 2023.
#
# Why this dataset: every transfer win so far is a PRICE target (house,
# Airbnb x2), so the "individual-record targets favour a GNN, aggregate
# targets don't" claim rests on a thin contrast. This is an individual-record
# target that is not a price -- site energy use intensity (kBtu/sq ft), a
# physical quantity -- so it tests the claim on a genuinely different target
# type, on the same kind of geometry (individual buildings with lat/lon).
#
# Both cities report through ENERGY STAR Portfolio Manager, so the target
# definition (site EUI) and the property-type vocabulary are shared.
#
# LEAKAGE NOTE: energy_star_score, source EUI, weather-normalized EUI, GHG
# intensity and total GHG are all computed FROM the same metered energy as
# site EUI. None of them are used as covariates.

suppressPackageStartupMessages({library(dplyr); library(jsonlite)})

YEAR <- 2023

fetch_paged <- function(base, where, page = 50000) {
  offset <- 0; out <- list()
  repeat {
    url <- sprintf("%s?$where=%s&$limit=%d&$offset=%d", base, where, page, offset)
    d <- fromJSON(URLencode(url))
    if (length(d) == 0 || nrow(d) == 0) break
    out[[length(out) + 1]] <- d
    if (nrow(d) < page) break
    offset <- offset + page
  }
  bind_rows(out)
}

cat("Fetching Chicago...\n")
chi_raw <- fetch_paged(
  "https://data.cityofchicago.org/resource/xq83-jr8c.json",
  sprintf("data_year=%d AND site_eui_kbtu_sq_ft IS NOT NULL", YEAR)
)
cat(sprintf("  %d rows\n", nrow(chi_raw)))

cat("Fetching Seattle...\n")
sea_raw <- fetch_paged(
  "https://data.seattle.gov/resource/teqw-tu6e.json",
  sprintf("datayear='%d' AND siteeui_kbtu_sf IS NOT NULL", YEAR)
)
cat(sprintf("  %d rows\n", nrow(sea_raw)))

# Shared property-type flags. Both cities use ENERGY STAR category names.
type_flags <- function(x) {
  x <- ifelse(is.na(x), "", x)
  data.frame(
    is_multifamily = as.integer(grepl("Multifamily", x, fixed = TRUE)),
    is_office      = as.integer(grepl("Office", x, fixed = TRUE)),
    is_school      = as.integer(grepl("K-12 School", x, fixed = TRUE)),
    is_hotel       = as.integer(grepl("Hotel", x, fixed = TRUE)),
    is_warehouse   = as.integer(grepl("Warehouse", x, fixed = TRUE)),
    is_retail      = as.integer(grepl("Retail", x, fixed = TRUE))
  )
}

chi <- chi_raw |>
  transmute(
    city = "chicago",
    lat = as.numeric(latitude), lon = as.numeric(longitude),
    eui = as.numeric(site_eui_kbtu_sq_ft),
    gfa = as.numeric(gross_floor_area_buildings_sq_ft),
    year_built = as.numeric(year_built),
    n_buildings = as.numeric(of_buildings),
    ptype = primary_property_type
  )

sea <- sea_raw |>
  transmute(
    city = "seattle",
    lat = as.numeric(latitude), lon = as.numeric(longitude),
    eui = as.numeric(siteeui_kbtu_sf),
    gfa = as.numeric(propertygfabuildings),
    year_built = as.numeric(yearbuilt),
    n_buildings = as.numeric(numberofbuildings),
    ptype = largestpropertyusetype
  )

clean <- function(d) {
  d <- bind_cols(d, type_flags(d$ptype)) |> select(-ptype)
  d |>
    filter(
      !is.na(lat), !is.na(lon), !is.na(eui), !is.na(gfa), !is.na(year_built),
      !is.na(n_buildings), eui > 0, gfa > 0, year_built > 1800
    ) |>
    # EUI has a long right tail (data centres, hospitals); trim the extreme
    # 0.5% at each end so a handful of outliers don't dominate RMSE.
    filter(eui >= quantile(eui, 0.005), eui <= quantile(eui, 0.995)) |>
    mutate(
      log_gfa = log(gfa),
      y = log(eui) - mean(log(eui))   # demeaned within city, as elsewhere
    ) |>
    select(-gfa, -eui)
}

chi_c <- clean(chi)
sea_c <- clean(sea)
cat(sprintf("\nAfter cleaning: Chicago %d, Seattle %d\n", nrow(chi_c), nrow(sea_c)))

saveRDS(list(chicago = chi_c, seattle = sea_c), "data/energy-clean.rds")
cat("Saved to data/energy-clean.rds\n")

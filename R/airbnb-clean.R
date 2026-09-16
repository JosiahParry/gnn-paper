# Clean the raw Inside Airbnb listings CSVs (data.insideairbnb.com, free,
# no auth) down to what the transfer pipeline needs, and save a small RDS.
#
# LA -> NYC transfer: individual listing records (not aggregates, unlike
# CDC PLACES), real lat/lon, genuine local-neighborhood pricing effects
# (transit access, block desirability) -- the same character of problem as
# KC -> Ames, but both regions clear 20k+ rows after cleaning.

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

  # bedrooms/beds are missing for some listings (often studios or
  # under-specified ones); impute from this region's own observed median at
  # the same `accommodates` value -- no target information used.
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

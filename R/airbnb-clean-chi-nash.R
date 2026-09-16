# Same cleaning as R/airbnb-clean.R, second city pair: Chicago -> Nashville.
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

cat("Cleaning Chicago...\n")
chi <- clean_one("data/airbnb-raw/chicago_listings.csv", "chicago")
cat(sprintf("  %d rows\n", nrow(chi)))

cat("Cleaning Nashville...\n")
nash <- clean_one("data/airbnb-raw/nashville_listings.csv", "nashville")
cat(sprintf("  %d rows\n", nrow(nash)))

saveRDS(list(chicago = chi, nashville = nash), "data/airbnb-chi-nash-clean.rds")
cat("\nSaved to data/airbnb-chi-nash-clean.rds\n")

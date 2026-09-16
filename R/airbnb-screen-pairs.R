# Cross-predict every ordered pair of 13 Airbnb cities with OLS.

suppressPackageStartupMessages(library(dplyr))

feature_cols <- c(
  "accommodates",
  "bathrooms",
  "bedrooms",
  "beds",
  "minimum_nights",
  "availability_365",
  "number_of_reviews",
  "is_entire_home"
)

clean_one <- function(path) {
  d <- read.csv(path, stringsAsFactors = FALSE)
  bathrooms <- as.numeric(gsub("[^0-9.]", "", d$bathrooms_text))
  price <- as.numeric(gsub("[$,]", "", d$price))

  out <- data.frame(
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
  out <- out |>
    filter(
      !is.na(price),
      price > 0,
      !is.na(bathrooms),
      !is.na(lat),
      !is.na(accommodates),
      !is.na(minimum_nights),
      !is.na(availability_365),
      !is.na(number_of_reviews)
    )
  for (col in c("bedrooms", "beds")) {
    med_by_acc <- out |>
      group_by(accommodates) |>
      summarise(med = median(.data[[col]], na.rm = TRUE))
    global_med <- median(out[[col]], na.rm = TRUE)
    idx <- is.na(out[[col]])
    fill <- med_by_acc$med[match(
      out$accommodates[idx],
      med_by_acc$accommodates
    )]
    fill[is.na(fill)] <- global_med
    out[[col]][idx] <- fill
  }
  out$y <- log(out$price) - mean(log(out$price))
  out
}

cities <- c(
  la = "data/airbnb-raw/la_listings.csv",
  nyc = "data/airbnb-raw/nyc_listings.csv",
  chicago = "data/airbnb-raw/chicago_listings.csv",
  nashville = "data/airbnb-raw/nashville_listings.csv",
  dallas = "data/airbnb-raw/dallas_listings.csv",
  fortworth = "data/airbnb-raw/fortworth_listings.csv",
  columbus = "data/airbnb-raw/columbus_listings.csv",
  twincities = "data/airbnb-raw/twincities_listings.csv",
  denver = "data/airbnb-raw/denver_listings.csv",
  dc = "data/airbnb-raw/dc_listings.csv",
  sandiego = "data/airbnb-raw/sandiego_listings.csv",
  broward = "data/airbnb-raw/broward_listings.csv",
  vegas = "data/airbnb-raw/vegas_listings.csv"
)

cat("Cleaning all cities...\n")
dat <- lapply(cities, clean_one)
for (nm in names(dat)) {
  cat(sprintf("  %-12s n=%d\n", nm, nrow(dat[[nm]])))
}
saveRDS(dat, "data/airbnb-all-cities-clean.rds")

cat("\nFitting per-city OLS and cross-predicting every pair...\n")
models <- lapply(dat, function(d) lm(reformulate(feature_cols, "y"), data = d))

reg_metrics_rsq_trad <- function(truth, estimate) {
  var_truth <- var(truth)
  rmse <- sqrt(mean((truth - estimate)^2))
  bias <- mean(estimate - truth)
  cal_slope <- unname(coef(lm(truth ~ estimate))[2])
  data.frame(
    rsq_trad = 1 - rmse^2 / var_truth,
    bias = bias,
    cal_slope = cal_slope
  )
}

results <- list()
for (a in names(dat)) {
  for (b in names(dat)) {
    if (a == b) {
      next
    }
    pred <- predict(models[[a]], newdata = dat[[b]])
    m <- reg_metrics_rsq_trad(dat[[b]]$y, pred)
    results[[length(results) + 1]] <- cbind(source = a, target = b, m)
  }
}
results <- do.call(rbind, results)
results <- results[order(-results$rsq_trad), ]

cat("\n=== OLS cross-region transfer screen, all pairs, ranked ===\n")
print(results, row.names = FALSE, digits = 3)

saveRDS(results, "data/airbnb-pair-screen.rds")
cat("\nSaved to data/airbnb-pair-screen.rds\n")

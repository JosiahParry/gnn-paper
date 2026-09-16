# The controlled decisive test: same rows, same regions, same graph -- only
# the TARGET changes from a price to something that is not a price.
#
# After eight pairs, "GraphSAGE wins" is perfectly confounded with "the
# target is a price". Every attempt to break that with a new dataset has run
# into a different obstacle (building energy had no spatial signal; NYC PLUTO
# building age fails the DGP screen in all 20 borough pairs). A new dataset
# also changes the region pair, the covariates and the geometry at the same
# time as the target, so even a clean result would be hard to attribute.
#
# The Airbnb tables already contain non-price columns on region pairs that
# ALREADY pass the DGP screen. Holding everything else fixed and moving one
# variable from the target slot to the covariate slot is the cleanest
# available version of the experiment:
#
#   target  = log1p(number_of_reviews), demeaned within city
#   or      = availability_365 / 365, demeaned within city
#   price   = becomes an ordinary covariate
#
# Reviews are a demand/desirability proxy, so if the mechanism behind the
# price wins really is "a latent, spatially smooth location-value field that
# the covariates do not proxy", reviews should behave like price and win.
# If instead GraphSAGE only ever wins on literal prices, it should lose here
# despite a favourable Moran profile. Either way the confound moves.

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
    # Price moves into the covariate set; the former target column and the
    # column now used as the target are both removed from the features.
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

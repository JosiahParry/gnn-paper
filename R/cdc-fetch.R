# Pull CDC PLACES census-tract-level data for the CA -> WV transfer test.
#
# All 11 measures come from the same source (CDC PLACES, data.cdc.gov), so
# this needs no external merge: DIABETES is the target, the other ten are
# behavioral / access-to-care / social-need predictors, deliberately avoiding
# other Health Outcomes measures as predictors to keep them conceptually
# distinct from the target rather than restating it.
#
# Source data.cdc.gov/resource/cwsq-ngmh.json (PLACES: Census Tract Data,
# public, no auth required).

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

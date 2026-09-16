# NYC 311 service-request response time: Queens -> Brooklyn.
#
# Why this dataset: a second NON-PRICE individual-record target, on the same
# region pair as the street-tree test, so the two share geography and differ
# only in target. Three of the four existing transfer wins are price targets;
# the "individual-record targets favour a GNN" claim needs contrast.
#
# Target: log1p(hours from created to closed). Restricted to one agency
# (DSNY, sanitation) so the service process -- and therefore the data
# generating process -- is the same in both boroughs. DSNY is route-based:
# response time depends on where a request sits relative to collection routes
# and depots, which is exactly the kind of structure neighbours share and a
# graph method can exploit.
#
# LEAKAGE NOTE: closed_date defines the target, so nothing downstream of
# closure is a covariate -- no status, no resolution_description,
# no resolution_action_updated_date.

suppressPackageStartupMessages({library(dplyr); library(jsonlite)})

AGENCY <- "DSNY"
FROM <- "2024-01-01T00:00:00"
TO   <- "2025-01-01T00:00:00"

fetch_paged <- function(base, where, select, page = 50000) {
  offset <- 0; out <- list()
  repeat {
    url <- sprintf("%s?$select=%s&$where=%s&$limit=%d&$offset=%d",
                   base, select, where, page, offset)
    d <- fromJSON(URLencode(url))
    if (length(d) == 0 || nrow(d) == 0) break
    out[[length(out) + 1]] <- d
    cat(sprintf("  ...%d\n", offset + nrow(d)))
    if (nrow(d) < page) break
    offset <- offset + page
  }
  bind_rows(out)
}

sel <- paste(c("created_date", "closed_date", "complaint_type", "descriptor",
               "borough", "latitude", "longitude", "open_data_channel_type",
               "location_type"), collapse = ",")

whr <- sprintf(paste0("agency='%s' AND created_date>='%s' AND created_date<'%s' ",
                      "AND closed_date IS NOT NULL AND latitude IS NOT NULL ",
                      "AND borough in('QUEENS','BROOKLYN')"),
               AGENCY, FROM, TO)

cat("Fetching NYC 311 (DSNY, 2024, Queens + Brooklyn)...\n")
raw <- fetch_paged("https://data.cityofnewyork.us/resource/erm2-nwe9.json", whr, sel)
cat(sprintf("%d raw rows\n", nrow(raw)))

d <- raw |>
  transmute(
    boro = tolower(borough),
    lat = as.numeric(latitude), lon = as.numeric(longitude),
    created = as.POSIXct(created_date, format = "%Y-%m-%dT%H:%M:%OS", tz = "UTC"),
    closed  = as.POSIXct(closed_date,  format = "%Y-%m-%dT%H:%M:%OS", tz = "UTC"),
    complaint = complaint_type, descriptor = descriptor,
    channel = open_data_channel_type, loctype = location_type
  ) |>
  filter(!is.na(lat), !is.na(lon), !is.na(created), !is.na(closed)) |>
  mutate(hours = as.numeric(difftime(closed, created, units = "hours"))) |>
  filter(hours > 0, hours < 24 * 30)

cat(sprintf("%d rows after basic filters\n", nrow(d)))

# Keep only categories present in BOTH boroughs with decent volume, so the
# feature space is shared rather than borough-specific.
keep_shared <- function(x, min_n = 500) {
  tb <- table(d$boro, x)
  colnames(tb)[apply(tb, 2, min) >= min_n]
}
top_complaint <- keep_shared(d$complaint)
top_desc <- keep_shared(d$descriptor)
cat(sprintf("shared complaint types: %s\n", paste(top_complaint, collapse = " | ")))
cat(sprintf("shared descriptors: %d\n", length(top_desc)))

d <- d |> filter(complaint %in% top_complaint)

dummies <- function(x, levs, prefix) {
  m <- sapply(levs, function(l) as.integer(x == l))
  colnames(m) <- paste0(prefix, make.names(tolower(levs)))
  as.data.frame(m)
}

feat <- bind_cols(
  dummies(d$complaint, top_complaint, "cx_"),
  dummies(d$descriptor, head(top_desc, 12), "dx_"),
  data.frame(
    ch_online = as.integer(d$channel == "ONLINE"),
    ch_phone  = as.integer(d$channel == "PHONE"),
    hour_sin  = sin(2 * pi * as.integer(format(d$created, "%H")) / 24),
    hour_cos  = cos(2 * pi * as.integer(format(d$created, "%H")) / 24),
    dow       = as.integer(format(d$created, "%u")),
    is_weekend = as.integer(as.integer(format(d$created, "%u")) >= 6),
    month     = as.integer(format(d$created, "%m"))
  )
)
# Drop degenerate columns (all-zero in either borough).
ok <- sapply(feat, function(v) length(unique(v)) > 1 &&
               min(tapply(v, d$boro, function(z) length(unique(z)))) > 1)
feat <- feat[, ok, drop = FALSE]

out <- bind_cols(data.frame(boro = d$boro, lat = d$lat, lon = d$lon), feat)
out$y_raw <- log1p(d$hours)

N_KEEP <- 25000L
split_boro <- function(nm) {
  o <- out[out$boro == nm, ]
  o$y <- o$y_raw - mean(o$y_raw)   # demean within borough
  o$y_raw <- NULL; o$boro <- NULL
  # Density-preserving subsample: take the N closest to the borough centroid
  # rather than a random draw, which would thin the graph and dilute exactly
  # the spatial structure under test.
  if (nrow(o) > N_KEEP) {
    cx <- median(o$lon); cy <- median(o$lat)
    d2 <- ((o$lon - cx) * cos(cy * pi / 180))^2 + (o$lat - cy)^2
    o <- o[order(d2)[seq_len(N_KEEP)], ]
  }
  o
}

qns <- split_boro("queens"); bkn <- split_boro("brooklyn")
cat(sprintf("\nQueens %d, Brooklyn %d, features %d\n",
            nrow(qns), nrow(bkn), ncol(qns) - 2))

saveRDS(list(queens = qns, brooklyn = bkn), "data/nyc311-clean.rds")
cat("Saved to data/nyc311-clean.rds\n")

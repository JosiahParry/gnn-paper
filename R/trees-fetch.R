# NYC 2015 Street Tree Census: Queens -> Brooklyn transfer.
#
# Target: tree_dbh (trunk diameter, inches) -- individual-record, non-price,
# continuous. Expected profile is the opposite of building energy: weak
# covariates but strong spatial structure (blocks are planted in one era
# with one species, so neighbouring trees are similar sizes). That is where
# the Moran's I diagnostic predicts a GNN wins.
#
# SUBSAMPLING: taken as a contiguous dense disk (the N trees nearest a
# centre point), NOT a random sample. A random 10% sample would triple the
# distance between "nearest neighbours" and dilute exactly the hyperlocal
# structure under test.
#
# CAVEAT: health/sidewalk are field observations partly reflecting tree size
# (big roots crack sidewalks). Not arithmetic leakage, but noted.

suppressPackageStartupMessages({library(dplyr); library(jsonlite)})

fields <- "tree_id,tree_dbh,latitude,longitude,boroname,spc_common,health,curb_loc,sidewalk,guards,steward"

fetch_boro <- function(boro, page = 50000) {
  offset <- 0; out <- list()
  repeat {
    url <- sprintf(
      "https://data.cityofnewyork.us/resource/uvpi-gqnh.json?$select=%s&$where=status='Alive' AND boroname='%s'&$limit=%d&$offset=%d",
      fields, boro, page, offset)
    d <- fromJSON(URLencode(url))
    if (length(d) == 0 || nrow(d) == 0) break
    out[[length(out) + 1]] <- d
    if (nrow(d) < page) break
    offset <- offset + page
  }
  bind_rows(out)
}

N_KEEP <- 25000

cat("Fetching Queens...\n");   q <- fetch_boro("Queens");   cat(sprintf("  %d\n", nrow(q)))
cat("Fetching Brooklyn...\n"); b <- fetch_boro("Brooklyn"); cat(sprintf("  %d\n", nrow(b)))

# Shared species vocabulary so both boroughs get identical columns.
all_spc <- c(q$spc_common, b$spc_common)
top_spc <- names(sort(table(all_spc), decreasing = TRUE))[1:8]
cat("\nshared species:", paste(top_spc, collapse = ", "), "\n")

prep <- function(d, label) {
  out <- d |>
    transmute(
      boro = label,
      lat = as.numeric(latitude), lon = as.numeric(longitude),
      dbh = as.numeric(tree_dbh),
      health_good = as.integer(health == "Good"),
      health_poor = as.integer(health == "Poor"),
      oncurb = as.integer(curb_loc == "OnCurb"),
      sidewalk_damage = as.integer(sidewalk == "Damage"),
      has_guard = as.integer(guards %in% c("Helpful", "Harmful", "Unsure")),
      has_steward = as.integer(steward != "None"),
      spc = spc_common
    ) |>
    filter(!is.na(lat), !is.na(lon), !is.na(dbh), dbh > 0, dbh < 60)

  for (s in top_spc) {
    out[[paste0("spc_", gsub("[^a-z]", "_", tolower(s)))]] <- as.integer(!is.na(out$spc) & out$spc == s)
  }
  out <- out |> select(-spc)

  # Density-preserving: keep the N_KEEP trees closest to the borough centre.
  if (nrow(out) > N_KEEP) {
    cx <- median(out$lon); cy <- median(out$lat)
    d2 <- ((out$lon - cx) * cos(cy * pi / 180))^2 + (out$lat - cy)^2
    out <- out[order(d2)[seq_len(N_KEEP)], ]
  }
  out$y <- log(out$dbh) - mean(log(out$dbh))
  out |> select(-dbh)
}

qc <- prep(q, "queens"); bc <- prep(b, "brooklyn")
cat(sprintf("\nKept: Queens %d, Brooklyn %d\n", nrow(qc), nrow(bc)))
saveRDS(list(queens = qc, brooklyn = bc), "data/trees-clean.rds")
cat("Saved to data/trees-clean.rds\n")

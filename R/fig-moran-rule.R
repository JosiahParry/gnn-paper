# The decision figure: does a statistic computable BEFORE fitting anything
# predict whether GraphSAGE will beat the best non-graph model?
#
# x = fraction of covariates whose own Moran's I exceeds the Moran's I of the
#     OLS residuals, in the SOURCE region. No target labels are needed, so it
#     is available at deployment time.
# y = GraphSAGE + LayerNorm margin over the best non-graph arm
#
# The ABSOLUTE residual Moran's I does NOT work: over these six pairs the
# wins span [0.221, 0.615] and the losses [0.116, 0.446], which overlap --
# CA -> WV has a strongly autocorrelated residual (0.446) and still loses,
# because its covariates are more autocorrelated still. What separates the
# pairs is the comparison, not the level.
#
# Pairs that fail the OLS/DGP screen are drawn separately: that screen gates
# first, and those pairs measure the region pair rather than the method.

suppressPackageStartupMessages({library(ggplot2); library(dplyr)})

d <- readRDS("data/moran-calibrate2.rds")

margin <- c(
  "King County -> Ames (house price)"    = 0.218,
  "Broward -> San Diego (Airbnb price)"  = 0.030,
  "Chicago -> LA (Airbnb price)"         = 0.022,
  "CA -> WV (diabetes prevalence)"       = -0.040,
  "Queens -> Brooklyn (311 response)"    = -0.057,
  "Queens -> Brooklyn (tree DBH)"        = -0.144
)

p <- d |>
  filter(ols) |>
  mutate(
    margin = unname(margin[pair]),
    result = ifelse(margin > 0, "GraphSAGE wins", "GraphSAGE loses"),
    # Two pairs share the region names (trees and 311), so keep the target.
    label = ifelse(grepl("Queens", pair),
                   sub("Queens -> Brooklyn \\((.*)\\)", "Queens->Brooklyn: \\1", pair),
                   sub(" \\(.*", "", pair)),
    kind = ifelse(grepl("price", pair), "price target", "non-price target")
  )

# Threshold drawn across the empty gap. Lower = win, so the gap runs from the
# highest win up to the lowest loss.
lo <- max(p$frac_src_exceed[p$margin > 0]); hi <- min(p$frac_src_exceed[p$margin < 0])
thr <- (lo + hi) / 2

g <- ggplot(p, aes(frac_src_exceed, margin)) +
  annotate("rect", xmin = lo, xmax = hi, ymin = -Inf, ymax = Inf,
           fill = "grey85", alpha = 0.6) +
  annotate("text", x = thr, y = min(p$margin) * 0.55,
           label = sprintf("no pair lands\nin this gap\n(%.2f - %.2f)", lo, hi),
           size = 2.9, colour = "grey30") +
  geom_hline(yintercept = 0, linewidth = 0.3, colour = "grey40") +
  geom_point(aes(colour = result, shape = kind), size = 3.4) +
  geom_text(aes(label = label), hjust = -0.12, size = 2.9, colour = "grey20") +
  scale_colour_manual(values = c("GraphSAGE wins" = "#1b7837", "GraphSAGE loses" = "#b2182b")) +
  scale_shape_manual(values = c("price target" = 16, "non-price target" = 17)) +
  scale_x_continuous(expand = expansion(mult = c(0.08, 0.35))) +
  labs(
    title = "A pre-fit statistic separates the wins from the losses",
    subtitle = paste0(
      "Share of covariates more spatially autocorrelated than the OLS residual, source region only.\n",
      "Six pairs that pass the OLS/DGP screen. The threshold is fitted to these six points."),
    x = "fraction of covariates with Moran's I above the residual's, source region (KNN-30)",
    y = "GraphSAGE + LayerNorm margin over best non-graph arm",
    colour = NULL, shape = NULL
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom", panel.grid.minor = element_blank())

dir.create("images", showWarnings = FALSE)
ggsave("images/fig-moran-rule.png", g, width = 8, height = 5.5, dpi = 200)
cat("Saved images/fig-moran-rule.png\n")
print(p[, c("pair", "res_src", "frac_src_exceed", "margin")], row.names = FALSE, digits = 3)
cat(sprintf("\nGap between highest loss (%.3f) and lowest win (%.3f)\n", lo, hi))

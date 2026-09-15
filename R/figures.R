# Figures for the paper and the slide deck.
#
# Run from the repo root, after the four result scripts:
#   R -f R/figures.R

library(dplyr)
library(tidyr)
library(ggplot2)

dir.create("images", showWarnings = FALSE)

SURFACE <- "#fcfcfb"
INK <- "#0b0b0b"
INK2 <- "#52514e"
MUTED <- "#898781"
GRID <- "#e1e0d9"
BASE <- "#c3c2b7"
BLUE <- "#2a78d6"
ORANGE <- "#eb6834"
AQUA <- "#1baf7a"

theme_gnn <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.background = element_rect(fill = SURFACE, colour = NA),
      panel.background = element_rect(fill = SURFACE, colour = NA),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = GRID, linewidth = 0.3),
      axis.line.x = element_line(colour = BASE, linewidth = 0.4),
      axis.ticks = element_blank(),
      axis.text = element_text(colour = INK2),
      axis.title = element_text(colour = INK2, size = rel(0.9)),
      plot.title = element_text(colour = INK, face = "bold", size = rel(1.15)),
      plot.subtitle = element_text(colour = INK2, size = rel(0.92), lineheight = 1.2),
      plot.caption = element_text(colour = MUTED, size = rel(0.78), hjust = 0),
      strip.text = element_text(colour = INK, face = "bold", size = rel(0.92)),
      legend.position = "top",
      legend.justification = "left",
      legend.title = element_blank(),
      legend.text = element_text(colour = INK2),
      plot.title.position = "plot",
      plot.caption.position = "plot"
    )
}

save_fig <- function(plot, file, width, height) {
  ggsave(
    file.path("images", file), plot,
    width = width, height = height, dpi = 200, bg = SURFACE
  )
  cat("wrote images/", file, "\n", sep = "")
}

arm_order <- c(
  "OLS", "SEM", "XGBoost", "XGBoost + lags",
  "GraphSAGE", "GraphSAGE + LayerNorm"
)

# 1. Scenario E, block versus random -------------------------------------
#
# The headline protocol result: only the arms that consult the graph respond
# to the holdout design.

block <- readRDS("data/scenario-results-block.rds")$summary
random <- readRDS("data/scenario-results-random.rds")$summary

e_rows <- bind_rows(
  block |> mutate(design = "Spatial block"),
  random |> mutate(design = "Random")
) |>
  filter(grepl("^E", label)) |>
  mutate(
    label = recode(label, "E | Ix=0.0" = "Ix = 0", "E | Ix=0.4" = "Ix = 0.4"),
    arm = factor(arm, levels = arm_order),
    design = factor(design, levels = c("Spatial block", "Random")),
    uses_graph = arm %in% c("XGBoost + lags", "GraphSAGE", "GraphSAGE + LayerNorm")
  ) |>
  filter(!is.na(arm))

p1 <- ggplot(e_rows, aes(rsq_mean, arm, colour = design)) +
  geom_line(aes(group = arm), colour = BASE, linewidth = 1.6, lineend = "round") +
  geom_point(size = 3.4) +
  # Block above the point, random below, so the labels stay legible where the
  # two designs give nearly the same score.
  geom_text(
    aes(
      label = sprintf("%.2f", rsq_mean),
      vjust = ifelse(design == "Spatial block", -1.3, 2.1)
    ),
    size = 3.1, show.legend = FALSE
  ) +
  facet_wrap(~label, ncol = 1) +
  scale_colour_manual(values = c("Spatial block" = BLUE, "Random" = ORANGE)) +
  scale_x_continuous(limits = c(0, 0.62), expand = expansion(mult = c(0.01, 0.06))) +
  scale_y_discrete(expand = expansion(add = 0.75)) +
  labs(
    title = "Random CV makes the GNN look half as good",
    subtitle = "Same model, same data. Only the way we split it changed.",
    x = expression(paste("Out-of-sample ", R^2, ", mean of six folds")),
    y = NULL,
    caption = "Random folds build the held-out graph at one sixth the density."
  ) +
  theme_gnn()

save_fig(p1, "fig-scenario-e.png", 8.5, 6)

# 2. King County to Ames, across initialisations --------------------------

kc <- readRDS("data/kc-to-ames-results.rds")

if (!is.null(kc$ames_seeds)) {
  seeds_df <- kc$ames_seeds |>
    mutate(arm = factor(arm, levels = arm_order)) |>
    filter(!is.na(arm))

  means <- seeds_df |>
    group_by(arm) |>
    summarise(rsq_mean = mean(rsq), n = dplyr::n(), .groups = "drop")

  p2 <- ggplot(seeds_df, aes(rsq, arm)) +
    geom_point(
      colour = BLUE, alpha = 0.45, size = 2.6,
      position = position_jitter(height = 0.10, width = 0)
    ) +
    geom_point(
      data = means, aes(rsq_mean, arm),
      shape = 124, size = 9, colour = INK
    ) +
    geom_text(
      data = means,
      aes(rsq_mean, arm, label = sprintf("%.2f", rsq_mean)),
      vjust = -1.5, size = 3.2, colour = INK
    ) +
    scale_x_continuous(limits = c(0, 0.75), expand = expansion(mult = c(0.01, 0.04))) +
    labs(
      title = "Plain GraphSAGE gives a different answer every run",
      subtitle = "LayerNorm's worst run still beats its average.",
      x = expression(paste("Ames ", R^2)),
      y = NULL,
      caption = "One dot per initialization. OLS and XGBoost are deterministic here, one fit each."
    ) +
    theme_gnn()

  save_fig(p2, "fig-ames-seeds.png", 8.5, 5)

  cat("\nAmes transfer, across initialisations:\n")
  print(
    seeds_df |>
      group_by(arm) |>
      summarise(
        fits = dplyr::n(), mean = mean(rsq), sd = sd(rsq),
        min = min(rsq), max = max(rsq), .groups = "drop"
      ) |>
      as.data.frame(),
    row.names = FALSE, digits = 3
  )
} else {
  cat("data/kc-to-ames-results.rds has no per-seed results; skipping figure 2\n")
}

# 3. Held-out states, GraphSAGE against engineered lags -------------------

states <- readRDS("data/state-results.rds")

pair <- states |>
  filter(arm %in% c("GraphSAGE", "XGBoost + lags")) |>
  select(STUSPS, arm, rsq) |>
  pivot_wider(names_from = arm, values_from = rsq) |>
  mutate(
    gap = GraphSAGE - `XGBoost + lags`,
    STUSPS = reorder(STUSPS, gap)
  )

wins <- sum(pair$gap > 0)

p3 <- pair |>
  pivot_longer(c(GraphSAGE, `XGBoost + lags`), names_to = "arm", values_to = "rsq") |>
  ggplot(aes(rsq, STUSPS)) +
  geom_line(aes(group = STUSPS), colour = BASE, linewidth = 1.1, lineend = "round") +
  geom_point(aes(colour = arm), size = 2.6) +
  scale_colour_manual(
    values = c("GraphSAGE" = BLUE, "XGBoost + lags" = ORANGE)
  ) +
  scale_x_continuous(limits = c(0, 1), expand = expansion(mult = c(0.01, 0.03))) +
  labs(
    title = sprintf("GraphSAGE wins %d of %d held-out states", wins, nrow(pair)),
    subtitle = "Giving XGBoost the neighbor averages doesn't close the gap.",
    x = expression(paste("Held-out state ", R^2)),
    y = NULL,
    caption = "Neighbor averages beat plain XGBoost in only 22 of 39 states. Each state gets its own graph."
  ) +
  theme_gnn()

save_fig(p3, "fig-states-paired.png", 8, 9)

# 4. Every scenario, both holdout designs ---------------------------------

all_sims <- bind_rows(
  block |> mutate(design = "Spatial block"),
  random |> mutate(design = "Random")
) |>
  mutate(
    arm = factor(arm, levels = arm_order),
    design = factor(design, levels = c("Spatial block", "Random")),
    label = factor(label, levels = rev(unique(block$label)))
  ) |>
  filter(!is.na(arm), !is.na(rsq_mean))

p4 <- ggplot(all_sims, aes(rsq_mean, label, colour = design)) +
  geom_point(size = 2.3, alpha = 0.9) +
  facet_wrap(~arm, nrow = 1) +
  scale_colour_manual(values = c("Spatial block" = BLUE, "Random" = ORANGE)) +
  scale_x_continuous(
    limits = c(0, 1), breaks = c(0, 0.5, 1),
    expand = expansion(mult = c(0.02, 0.05))
  ) +
  labs(
    title = "GNNs only win when neighbors actually matter",
    subtitle = "Everywhere else, simpler models match or beat them.",
    x = expression(paste("Out-of-sample ", R^2)),
    y = NULL,
    caption = "Scenario E is the only one where a location's neighbors affect its outcome. SEM is fit only where errors cluster."
  ) +
  theme_gnn(base_size = 11) +
  theme(panel.spacing.x = unit(0.8, "lines"))

save_fig(p4, "fig-simulation-overview.png", 12, 5.5)

cat("\nDone.\n")

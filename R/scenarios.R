# The twelve scenario rows, built once.
#
# Both drivers source this, so the block and random runs score identical data
# and differ only in which nodes are held out.
#
# Lambda values come from data/calibration.rds, so every chart is labelled with
# realised Moran's I.

source("R/sim-core.R")

calib <- readRDS("data/calibration.rds")$calibrated

lam <- function(role, target) {
  calib$lambda[calib$role == role & calib$target_moran == target]
}

lam_ix_040 <- lam("covariate (Ix)", 0.4)
lam_ix_070 <- lam("covariate (Ix)", 0.7)
lam_iu_040 <- lam("error (Iu)", 0.4)
lam_iu_080 <- lam("error (Iu)", 0.8)

set.seed(0)

# Covariates -------------------------------------------------------------

x_iid <- make_x(0)
x_ix040 <- make_x(lam_ix_040)
x_ix070 <- make_x(lam_ix_070)

# Scenario D: iid covariates plus one spatially structured covariate that
# enters the mean nonlinearly.
x_sp_040 <- make_field(lam_ix_040)
x_sp_070 <- make_field(lam_ix_070)

# Error fields -----------------------------------------------------------

u_iid <- rnorm(n, sd = sigma)
u_iu040 <- make_field(lam_iu_040)
u_iu080 <- make_field(lam_iu_080)

# Scenarios --------------------------------------------------------------

# A: linear mean, iid errors. Negative control -- the graph carries nothing.
y_A <- xb(x_iid) + rnorm(n, sd = sigma)

# B: nonlinear mean, iid errors, X itself spatially structured at three levels.
# The mean function is identical across the three; only the spatial arrangement
# of X changes.
y_B_000 <- m_nonlinear(x_iid) + rnorm(n, sd = sigma)
y_B_040 <- m_nonlinear(x_ix040) + rnorm(n, sd = sigma)
y_B_070 <- m_nonlinear(x_ix070) + rnorm(n, sd = sigma)

# C: linear mean, spatially autocorrelated errors.
y_C_000 <- xb(x_iid) + u_iid
y_C_040 <- xb(x_iid) + u_iu040
y_C_080 <- xb(x_iid) + u_iu080

# D: nonlinear mean with a spatial covariate, plus spatial errors.
y_D_000_040 <- m_nonlinear_sp(x_iid, x_sp_040) + rnorm(n, sd = sigma)
y_D_040_040 <- m_nonlinear_sp(x_iid, x_sp_040) + u_iu040
y_D_080_070 <- m_nonlinear_sp(x_iid, x_sp_070) + u_iu080

# E: neighbourhood covariate effect -- y responds to the average covariate value
# among a node's neighbours, not only to the node's own.
#
# This is the only scenario in the set whose DGP contains a lag term, so it is
# the only one that can speak to whether lag features or a GNN recover
# neighbourhood structure. In A and C the mean is linear in the node's own X; in
# B and D it is nonlinear but still a function of x_i alone. WX is noise by
# construction in all four, so a null there is a property of the equations, not
# a result.
#
# Two rows. With an iid covariate field, Wx1 is close to orthogonal to x1 and
# the neighbourhood effect is information a node genuinely cannot get from
# itself. With a moderately autocorrelated field, Wx1 correlates with x1 and a
# model can recover part of the effect from the node's own covariate -- the
# situation real covariate surfaces are usually in, and the one Renato asks
# about at main.tex:223.
#
# Drawn last, so A-D are untouched by anything added or removed here.
wx1_iid <- lag.listw(lw_full, x_iid[, 1])
wx1_040 <- lag.listw(lw_full, x_ix040[, 1])

theta_iid <- theta_for_share(wx1_iid, x_iid)
theta_040 <- theta_for_share(wx1_040, x_ix040)

y_E_iid <- xb(x_iid) + theta_iid * wx1_iid + rnorm(n, sd = sigma)
y_E_040 <- xb(x_ix040) + theta_040 * wx1_040 + rnorm(n, sd = sigma)

# Assembly ---------------------------------------------------------------
#
# sem = TRUE only where the DGP has a spatially autocorrelated error term.

scenarios <- list(
  list(label = "A", sem = FALSE, df = cbind(y = y_A, x_iid)),
  list(label = "B | Ix=0.0", sem = FALSE, df = cbind(y = y_B_000, x_iid)),
  list(label = "B | Ix=0.4", sem = FALSE, df = cbind(y = y_B_040, x_ix040)),
  list(label = "B | Ix=0.7", sem = FALSE, df = cbind(y = y_B_070, x_ix070)),
  list(label = "C | Iu=0.0", sem = TRUE, df = cbind(y = y_C_000, x_iid)),
  list(label = "C | Iu=0.4", sem = TRUE, df = cbind(y = y_C_040, x_iid)),
  list(label = "C | Iu=0.8", sem = TRUE, df = cbind(y = y_C_080, x_iid)),
  list(label = "D | Iu=0.0 Ix=0.4", sem = TRUE, df = cbind(y = y_D_000_040, x_iid, x_sp = x_sp_040)),
  list(label = "D | Iu=0.4 Ix=0.4", sem = TRUE, df = cbind(y = y_D_040_040, x_iid, x_sp = x_sp_040)),
  list(label = "D | Iu=0.8 Ix=0.7", sem = TRUE, df = cbind(y = y_D_080_070, x_iid, x_sp = x_sp_070)),
  list(label = "E | Ix=0.0", sem = FALSE, df = cbind(y = y_E_iid, x_iid)),
  list(label = "E | Ix=0.4", sem = FALSE, df = cbind(y = y_E_040, x_ix040))
)

# What Scenario E actually contains. share_wx is theta^2 Var(Wx1) / Var(y),
# the spillover's realised share of the response variance; it drifts from the
# 0.2 target for the autocorrelated row because Xb and Wx1 covary there.
# cor_x_wx says how much of the neighbourhood effect a model could recover from
# the node's own x1 without ever consulting the graph -- near zero for the iid
# row, substantial for the autocorrelated one.
e_params <- data.frame(
  label = c("E | Ix=0.0", "E | Ix=0.4"),
  theta = c(theta_iid, theta_040),
  var_wx = c(var(wx1_iid), var(wx1_040)),
  share_wx = c(
    theta_iid^2 * var(wx1_iid) / var(y_E_iid),
    theta_040^2 * var(wx1_040) / var(y_E_040)
  ),
  cor_x_wx = c(
    cor(x_iid[, 1], wx1_iid),
    cor(x_ix040[, 1], wx1_040)
  )
)

cat("\nScenario E parameters:\n")
print(e_params, row.names = FALSE, digits = 3)

# Realised Moran's I of every field and response, for labelling the charts.
morans <- data.frame(
  variable = c(
    "x_ix040", "x_ix070", "x_sp_040", "x_sp_070", "u_iu040", "u_iu080",
    "y_A", "y_B_000", "y_B_040", "y_B_070",
    "y_C_000", "y_C_040", "y_C_080",
    "y_D_000_040", "y_D_040_040", "y_D_080_070",
    "y_E_iid", "y_E_040"
  ),
  moran_i = c(
    moran_of(x_ix040[, 1]), moran_of(x_ix070[, 1]),
    moran_of(x_sp_040), moran_of(x_sp_070),
    moran_of(u_iu040), moran_of(u_iu080),
    moran_of(y_A), moran_of(y_B_000), moran_of(y_B_040), moran_of(y_B_070),
    moran_of(y_C_000), moran_of(y_C_040), moran_of(y_C_080),
    moran_of(y_D_000_040), moran_of(y_D_040_040), moran_of(y_D_080_070),
    moran_of(y_E_iid), moran_of(y_E_040)
  )
)

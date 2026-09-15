# Pins the DGP parameters to realised Moran's I and emits the parameter table
# for the paper.
#
# Measures a lambda -> Moran's I curve and inverts it at the Ix and Iu targets.
# Covariate fields and error fields are built by the same call --
# sim_error(make_error(...), lw, lambda) -- so a single curve serves both.
#
# Fits no models. Run from the repo root:
#   R -f R/calibrate.R

source("R/sim-core.R")

n_draws <- 5L
lambda_grid <- c(0, 0.3, 0.5, 0.65, 0.75, 0.82, 0.87, 0.9, 0.93, 0.95, 0.97, 0.99)

target_ix <- c(0.4, 0.7)
target_iu <- c(0.4, 0.8)

set.seed(0)

# lambda -> Moran's I ----------------------------------------------------

cat(sprintf(
  "Measuring Moran's I at %d values of lambda, %d draws each (n = %d, KNN-%d)\n",
  length(lambda_grid),
  n_draws,
  n,
  k_nb
))

curve <- do.call(
  rbind,
  lapply(lambda_grid, function(lam) {
    i_vals <- vapply(seq_len(n_draws), \(d) moran_of(make_field(lam)), numeric(1))
    cat(sprintf(
      "  lambda = %.2f  ->  I = %.3f (sd %.3f)\n",
      lam,
      mean(i_vals),
      sd(i_vals)
    ))
    data.frame(
      lambda = lam,
      moran_mean = mean(i_vals),
      moran_sd = sd(i_vals),
      moran_min = min(i_vals),
      moran_max = max(i_vals)
    )
  })
)

# Invert the curve. Moran's I is monotone in lambda, so linear interpolation
# between grid points is enough to land the targets.
solve_lambda <- function(target) {
  approx(x = curve$moran_mean, y = curve$lambda, xout = target)$y
}

calibrated <- data.frame(
  role = c(rep("covariate (Ix)", length(target_ix)), rep("error (Iu)", length(target_iu))),
  target_moran = c(target_ix, target_iu),
  lambda = c(solve_lambda(target_ix), solve_lambda(target_iu))
)

cat("\n=== Calibrated parameters ===\n")
print(calibrated, row.names = FALSE, digits = 3)

# Confirm the solved lambdas actually land on target, since interpolation
# between grid points is an approximation and the fields are random.
cat("\nVerifying at the solved lambdas:\n")

verify <- do.call(
  rbind,
  lapply(seq_len(nrow(calibrated)), function(i) {
    lam <- calibrated$lambda[i]
    i_vals <- vapply(seq_len(n_draws), \(d) moran_of(make_field(lam)), numeric(1))
    data.frame(
      role = calibrated$role[i],
      target = calibrated$target_moran[i],
      lambda = lam,
      realised = mean(i_vals),
      realised_sd = sd(i_vals)
    )
  })
)

print(verify, row.names = FALSE, digits = 3)

# Parameter table --------------------------------------------------------
#
# Fold sizes follow from k_folds and val_prop, both fixed in sim-core.R.

n_test <- floor(n / k_folds)
n_fit <- n - n_test
n_val <- floor(val_prop * n_fit)
n_train <- n_fit - n_val

params <- rbind(
  data.frame(parameter = "n", value = as.character(n)),
  data.frame(parameter = "lattice", value = sprintf("%d x %d", grid_cols, grid_rows)),
  data.frame(parameter = "graph", value = sprintf("KNN, k = %d, row-standardised", k_nb)),
  data.frame(parameter = "folds", value = sprintf("%d, inductive", k_folds)),
  data.frame(parameter = "n train / val / test per fold",
             value = sprintf("%d / %d / %d", n_train, n_val, n_test)),
  data.frame(parameter = "beta", value = paste(beta, collapse = ", ")),
  data.frame(parameter = "Wx1 target share of Var(y), scenario E",
             value = as.character(formals(theta_for_share)$share)),
  data.frame(parameter = "sigma", value = as.character(sigma)),
  data.frame(parameter = "lambda for Ix = 0.4 / 0.7",
             value = sprintf("%.3f / %.3f", calibrated$lambda[1], calibrated$lambda[2])),
  data.frame(parameter = "rho for Iu = 0.4 / 0.8",
             value = sprintf("%.3f / %.3f", calibrated$lambda[3], calibrated$lambda[4])),
  data.frame(parameter = "GraphSAGE hidden dims", value = paste(sage_hidden, collapse = ", ")),
  data.frame(parameter = "epochs / lr / patience",
             value = sprintf("%d / %s / %d", n_epochs, format(lr), patience)),
  data.frame(parameter = "XGBoost trees", value = "500")
)

cat("\n=== Parameter table (for main.tex) ===\n")
print(params, row.names = FALSE, right = FALSE)

dir.create("data", showWarnings = FALSE)
saveRDS(
  list(curve = curve, calibrated = calibrated, verify = verify, params = params),
  "data/calibration.rds"
)

cat("\nSaved to data/calibration.rds\n")

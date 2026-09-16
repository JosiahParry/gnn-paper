# Simulation results, for Renato

Both simulation designs (block and random holdout, six scenarios A-E, six
inductive folds each) are complete and rerun with two corrections since the
version this was last discussed:

1. **The R² metric was wrong.** `yardstick::rsq` (squared Pearson
   correlation) is invariant to a model's predictions being systematically
   too spread out or offset -- it can't detect miscalibration. Every table
   below reports **`rsq_trad`** (1 - SSE/SST, the textbook R²) alongside the
   old `rsq` for comparison. `rsq_trad` is the one to cite in the paper.
2. **LayerNorm was silently running in the wrong mode.** The GNN library's
   `layer_layer_norm()` defaults to normalizing per-*graph* (one shared
   scalar for the whole lattice) rather than per-*node*, which is not what
   "LayerNorm" is supposed to do. Fixed and every GraphSAGE+LayerNorm number
   below is refit with the correct per-node normalization.

Summary table (12 scenario rows x 5-6 arms x 2 designs, mean and sd across
the 6 folds): **`reports/renato-simulation-results.csv`**.

**Underlying per-fold results** (every individual fold, not just the
mean/sd -- 792 rows = 12 scenarios x 6 folds x 5-6 arms x 2 designs):
- `reports/renato-scenario-folds-all.csv` -- both designs combined
- `reports/renato-scenario-folds-block.csv` -- block design only
- `reports/renato-scenario-folds-random.csv` -- random design only

Columns: `design, label, fold, arm, mae, rmse, rsq, rsq_trad`. `label` is
the scenario (A, B\|Ix=0.0, ..., E\|Ix=0.4); `fold` is 1-6.

## Calibration parameters (closes `main.tex:155`, "models are not completely
specified", and gives the realized Moran's I for `main.tex:276-279`)

| parameter | value |
| --- | --- |
| n | 7000 |
| lattice | 100 x 70 |
| graph | KNN, k = 15, row-standardised |
| folds | 6, inductive |
| n train / val / test per fold | 5251 / 583 / 1166 |
| beta | 1, 0.3, 0.3, 0.3, 0.3, 0.3 |
| sigma | 1 |
| Wx1 target share of Var(y), Scenario E | 0.2 |
| lambda for Ix = 0.4 / 0.7 | 0.857 / 0.961 |
| rho for Iu = 0.4 / 0.8 | 0.857 / 0.975 |
| GraphSAGE hidden dims | 56, 32, 16 |
| epochs / lr / patience | 500 / 0.01 / 20 |
| XGBoost trees | 500 |

**Realized Moran's I** (averaged over calibration draws -- use these
numbers in the text, not the nominal targets):

| role | target | lambda/rho | realized I | sd |
| --- | --- | --- | --- | --- |
| covariate (Ix) | 0.4 | 0.857 | **0.379** | 0.020 |
| covariate (Ix) | 0.7 | 0.961 | **0.686** | 0.050 |
| error (Iu) | 0.4 | 0.857 | **0.390** | 0.030 |
| error (Iu) | 0.8 | 0.975 | **0.770** | 0.067 |

## Headline results by scenario (rsq_trad, block design, mean of 6 folds)

| scenario | GraphSAGE | +LayerNorm | OLS | SEM | XGBoost | XGBoost+lags |
| --- | --- | --- | --- | --- | --- | --- |
| A (linear, iid) | 0.345 | 0.330 | 0.358 | -- | 0.156 | 0.188 |
| B \| Ix=0.0 (nonlinear) | 0.968 | 0.965 | 0.778 | -- | 0.957 | 0.955 |
| B \| Ix=0.4 | 0.964 | 0.961 | 0.771 | -- | 0.952 | 0.951 |
| B \| Ix=0.7 | 0.954 | 0.947 | 0.734 | -- | 0.940 | 0.940 |
| C \| Iu=0.0 (spatial error) | 0.338 | 0.325 | 0.355 | 0.355 | 0.155 | 0.203 |
| C \| Iu=0.4 | 0.299 | 0.297 | 0.342 | 0.343 | 0.144 | 0.157 |
| **C \| Iu=0.8** | **0.006** | **0.057** | 0.280 | 0.285 | 0.037 | 0.074 |
| D \| Iu=0.0 Ix=0.4 | 0.937 | 0.936 | 0.455 | 0.455 | 0.912 | 0.914 |
| D \| Iu=0.4 Ix=0.4 | 0.934 | 0.933 | 0.456 | 0.456 | 0.911 | 0.912 |
| D \| Iu=0.8 Ix=0.7 | 0.928 | 0.921 | 0.386 | 0.398 | 0.905 | 0.904 |
| E \| Ix=0.0 (lag effect) | 0.429 | 0.411 | 0.283 | -- | 0.067 | 0.294 |
| E \| Ix=0.4 | 0.509 | 0.496 | 0.418 | -- | 0.231 | 0.381 |

Random-design numbers are in the CSV; the pattern is the same except
Scenario C at Iu=0.8, where block gives LayerNorm a large win (0.006 ->
0.057) and random shows the opposite (LayerNorm *loses* to plain GraphSAGE,
0.181 -> 0.161) -- a genuine block-vs-random asymmetry, not noise, and worth
a sentence in the methods section since it echoes the existing Scenario E
holdout-design point at `main.tex:493`.

## What this settles for the paper text (section 8 of PLAN.md)

- **Scenario C confirms the prediction**: correctly-specified OLS/SEM
  clearly win at high spatial-error autocorrelation (Iu=0.8, ~0.28-0.29)
  over GraphSAGE (0.006-0.057) -- `main.tex:379`'s claim that GraphSAGE "may
  partially recover a smooth residual surface" does not hold and should be
  cut or reversed.
- **Scenario E is the only row with a real lag effect in the DGP** and is
  where XGBoost+lags shows its clearest, most defensible gain over plain
  XGBoost (0.067->0.294 at Ix=0.0; 0.231->0.381 at Ix=0.4) -- consistent
  with the abstract's claim about lag features, and the *only* simulation
  scenario that can be cited as evidence for it.
- Scenario F (`main.tex:413-417`) is confirmed out of the design; not run.

## Not yet done, flagged rather than silently skipped

Whether `dropout=0.1` (found to help the real-data KC -> Ames transfer
substantially) also helps these simulation scenarios has **not** been
tested -- it was tuned for a genuine distribution-shift transfer problem,
and the states/simulation settings here are same-distribution holdouts,
where the mechanism it fixes may not apply. Recommend not adding it here
without testing first.

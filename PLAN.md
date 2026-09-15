# Rerun plan: simulations and empirical analyses

Working plan for regenerating every result in `Optimal-Design-layout/main.tex`
using the LayerNorm GraphSAGE architecture. Source material is the
`2026-04-09-gnn-proposal` repo; nothing there is modified, everything is
rewritten here.

## 1. Why the current results have to be rerun

Renato's margin notes in `main.tex` are all symptoms of five defects in the
existing simulation code. Recording them here so the rewrite is checked
against a list rather than a memory.

**No LayerNorm.** Both `2026-06-29/R/simulate-scenarios.R:164` (block CV) and
`2026-07-27/R/simulate-scenarios-random.R:168` (random split) call
`model_sage(in_features, hidden_dims = c(56,32,16), out_features = 1)` with no
`norm` argument. The architecture the paper is being rebuilt around appears
nowhere in the simulations.

**There is no plain-XGBoost arm.** `train_eval_xgb()`
(`simulate-scenarios-random.R:249-275`) always builds `lag_` columns. Every bar
labeled "XGBoost" in the slides is XGBoost + first-order lags. The paper's
central claim — that explicit spatial lag features recover much of the
predictive gain — is untestable without the un-lagged comparison.

**The random split has no cross-validation.** `simulate-scenarios-random.R:104-126`
builds a single 60/20/20 partition, and `run_models()` at line 316 states it:
"One split, so no averaging." The block script runs 6 folds; the random script
runs 1. They are not comparable, and the single split is the likely source of
`main.tex:365` ("the bar plot here is very suspicious").

**Three incompatible vocabularies for one knob.** The code uses `lambda = 0.85 /
0.95` for the covariate field and `0.85 / 0.97` for the error field, labels the
outputs `0.50 / 0.75` and `0.50 / 0.90`, and `main.tex:276-279` claims
`Ix ∈ {0, 0.4, 0.7}` and `Iu ∈ {0, 0.4, 0.8}`. None of these are the realized
Moran's I. The block script already computes the realized values at
`simulate-scenarios.R:535` — that is the number that should drive every label.

**No classical spatial model.** Confirmed by `main.tex:345`. There is no SEM in
the simulations or in the US-states analysis.

One further issue found while reading, decided below rather than inherited:

- Scenario B adds a `1.5*sin(x_sp)` term at the two spatially correlated
  settings (`simulate-scenarios-random.R:348-349`), so the three B settings are
  three different data-generating processes, not one process at three
  correlation levels. `main.tex:173` claims the latter.

Separately, the US-states figures are stitched from two different runs:
`2026-07-27/R/dumbbell-layernorm.R:16-24` reads GraphSAGE+LayerNorm from
`layernorm-results.rds` (August) and XGBoost from `all-states-results.rds`
(June), joined on state abbreviation across different seeds. The state scripts
also train a flat 500 epochs with no early stopping and no validation
checkpoint (`us-graphsage-layernorm.R:152-159`), while the simulations use
`patience = 20`.

## 2. Settled decisions

- **Inductive only.** Test nodes are always pulled out and given their own
  subgraph, with no edges back to training nodes. No transductive variant, no
  leakage.
- **No Monte Carlo replication.** Cross-validation supplies the replication.
- **The random split becomes 6-fold CV**, matching the block design, so the two
  split types are directly comparable and every scenario yields six numbers
  instead of one.
- **SEM comes from `sphet`** (`spreg(..., model = "error")`), GMM-estimated, so
  there is no eigenvalue decomposition and n ≈ 5,800 per fold is tractable.
- **SEM predicts as `X_test %*% beta`.** Under the inductive protocol there is
  no path for residual information to cross the boundary, so this is the whole
  of the prediction.
- **Scenario B is fixed**: `X` itself is generated as a spatially autocorrelated
  field at `Ix ∈ {0, 0.4, 0.7}`, with `m(.)` held fixed across the three levels.
- **GraphSAGE is fit twice per fold**, once with `norm = NULL` and once with
  `norm = layer_layer_norm`, seeded identically inside the pair so the
  normalization is the only difference. This is what makes "how much better does
  LayerNorm perform" answerable.

## 3. Model roster

| Arm | Features | Scenarios | Notes |
| --- | --- | --- | --- |
| OLS | `X` | all | unchanged |
| SEM | `X` | C, D only | `sphet::spreg`, model = "error" |
| XGBoost | `X` | all | new; does not exist in current code |
| XGBoost + lags | `X`, `WX` | all | `WX` is the neighbourhood mean of each covariate |
| GraphSAGE | `X` | all | `norm = NULL` |
| GraphSAGE + LayerNorm | `X` | all | `norm = layer_layer_norm` |

SEM is restricted to the two scenarios with spatially autocorrelated errors.
It is not meaningful in A, B, E or F and running it there buys nothing.

## 4. Scenario grid

Thirteen rows: the existing design plus the Scenario B fix and a second E row.

| Scenario | Rows | Mean | Errors |
| --- | --- | --- | --- |
| A | 1 | linear | iid |
| B | 3 (`Ix` = 0, 0.4, 0.7) | nonlinear | iid |
| C | 3 (`Iu` = 0, 0.4, 0.8) | linear | spatial |
| D | 3 (`Iu` x `Ix`) | nonlinear | spatial |
| E | 2 (`Ix` = 0, 0.4) | linear + `theta·Wx1` | iid |

Each row runs under both split types (block, random) at 6 folds each.

### What E is for, and why it was rewritten

E is the only row whose DGP contains a lag term. A and C are linear in the
node's own X; B and D are nonlinear but still functions of `x_i` alone. `WX` is
noise by construction in all of them, so an
"XGBoost + lags adds nothing" result there is a restatement of how the equations
were written and carries no evidence about lag features.

The inherited `theta = 0.5` (`2026-06-29/R/simulate-scenarios.R:446`) put the
spillover at roughly 1% of `Var(y)`: row-standardised KNN-15 gives
`Var(Wx1) = 1/15` for an iid `x1`, so `0.25 × 0.067` against a total near 1.55.
Nothing could detect it. `theta_for_share()` now sets `theta` from a target
share of response variance (0.2), so the term is large enough to find and the
two E rows stay comparable when the covariate field's autocorrelation changes
`Var(Wx1)`. Realised shares and `cor(x1, Wx1)` are recorded in `e_params`.

The two rows separate the cases that matter. At `Ix = 0`, `Wx1` is near
orthogonal to `x1`: the neighbourhood effect is information the node cannot
obtain from itself, so recovering it requires the graph. At `Ix = 0.4`, `Wx1`
correlates with `x1` and part of the effect is reachable from the node's own
covariate — the position real covariate surfaces are usually in, and Renato's
question at `main.tex:223`.

## 5. Phases

**Phase 1 — `R/sim-core.R`.** One shared engine holding grid and KNN-15
construction, `make_sub_graph()`, the DGP builders, one trainer per arm, and
the scoring function. The block and random drivers differ only in how folds are
constructed. Deliverable: the file, sourced but not yet run at scale.

**Phase 2 — calibration pass, no models.** Solve for the `lambda` producing
realized Moran's I of 0.4 and 0.7 on the KNN-15 graph, and the `rho` giving
`Iu ∈ {0.4, 0.8}`, averaged over draws. Cheap — no model fitting. Deliverable:
a parameter table (`beta`, `theta`, `sigma`, `lambda`, `rho`, `n`, grid
dimensions, per-fold train/val/test counts) that closes `main.tex:155`
("Models are not completely specified").

**Phase 3 — run both simulations.** `R/simulate-block.R` and
`R/simulate-random.R`, both sourcing `sim-core.R`. Report mean and spread
across the six folds per arm per row. Deliverable: `data/scenario-results.rds`.

**Phase 4 — unified US-states script.** All 48 states held out in turn, all six
arms inside one loop under one seed, with validation-based early stopping to
match the simulations. Reports MAE, RMSE, R2, mean bias and residual Moran's I
per state. Deliverable: `data/state-results.rds`, replacing the two-run stitch.

**Phase 5 — King County to Ames.** `2026-08-04/R/kc-to-ames-layernorm.R` already
uses the right architecture; it needs OLS and SEM added so the empirical section
uses the same roster as everything else.

**Phase 6 — figures, tables, paper.** Scenario charts with across-fold spread,
and the held-out block map requested at `main.tex:203` and `main.tex:336`.

## 6. Fit budget

| | current | planned |
| --- | --- | --- |
| Block, GraphSAGE fits | 72 | 144 |
| Random, GraphSAGE fits | 12 | 144 |
| **GraphSAGE total** | **84** | **288** |

XGBoost and OLS are negligible. SEM is 6 rows x 6 folds x 2 splits = 72 GMM
fits. Timing of one GraphSAGE fit and one `spreg` fit gets measured at the start
of Phase 3, before the full run is launched.

Twelve rows, six folds, two GraphSAGE fits per fold, two split types.

## 7. To verify before relying on it

- ~~Whether `sphet` fitted objects expose coefficients under stable names~~ —
  resolved. `coef()` returns a one-column matrix carrying names in `rownames`,
  so indexing by name without coercion yields `NA` and NaN predictions.
  `fit_predict_sem()` coerces and stops loudly if the names still miss.
- One GraphSAGE fit's wall time, to confirm 288 fits is an overnight job and not
  a multi-day one.

## 8. Expected consequences for the paper text

- Under the inductive protocol nothing observes the held-out region's residual
  field, and in Scenario C the errors are independent of `X` by construction, so
  correctly specified OLS should win there. `main.tex:379` currently claims
  GraphSAGE "may partially recover a smooth residual surface" — that sentence
  will need to match whatever the rerun produces.
- `main.tex:276-279` needs the realized Moran's I values from Phase 2.
- `main.tex` describes the lagged arm as `WX` and `W^2X` at lines 231, 303 and
  405. It is `X` and `WX`.
- `main.tex:413-417` (Scenario F) comes out.
- The abstract's claim that explicit spatial lag features recover much of the
  predictive gain can only be argued from Scenario E and from Ames/KC. The other
  rows have no neighbourhood term in their DGP, so their lag results are not
  evidence either way and must not be cited as such.
- The simulations are analytic constructions in which the true mean is a known
  function of each node's own covariates. That bounds what they can settle: they
  isolate mechanisms and validate the harness, but they cannot establish how lag
  features or a GNN behave on a surface with neighbour dependence running
  through it. The empirical sections carry that argument.

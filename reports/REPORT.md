# Inductive spatial transfer with GraphSAGE: findings

Eight real cross-region transfers, one land cover classification, one
simulation grid, and 39 held-out US states. Every number below is a **single
trained model** -- nothing is averaged across models -- and every R² is
`rsq_trad` (1 − SSE/SST), not squared correlation.

**For the one-page summary, read `FINDINGS.md`.** This document holds the
supporting detail, including the claims that were withdrawn.

---

## 1. GraphSAGE wins four of eight inductive spatial transfers

Train in one region, predict in a structurally disjoint one (own graph, no
edges across the boundary, target labels never seen).

| transfer | n source → target | GraphSAGE + LayerNorm | best non-graph arm | margin |
| --- | --- | --- | --- | --- |
| King County → Ames (house price) | 21,613 → 2,930 | **0.501** | 0.283 (XGBoost+lags) | **+0.218** |
| Des Moines → Cedar Rapids (**surface temperature**) | 27,225 → 27,225 | **0.822** | 0.778 (OLS) | **+0.044** |
| Broward → San Diego (Airbnb price) | 16,345 → 11,761 | **0.679** | 0.649 (XGBoost) | **+0.030** |
| Chicago → LA (Airbnb price) | 7,546 → 37,863 | **0.654** | 0.632 (OLS) | **+0.022** |
| California → West Virginia (diabetes prevalence) | 9,070 → 546 | 0.763 | **0.803** (XGBoost+lags) | −0.040 |
| Queens → Brooklyn (311 response time) | 25,000 → 25,000 | 0.232 | **0.301** (OLS) | −0.057 |
| Queens → Brooklyn (street-tree diameter) | 25,000 → 25,000 | 0.132 | **0.276** (OLS) | −0.144 |
| Broward → San Diego (Airbnb *reviews*) † | 16,345 → 11,761 | 0.185 | **0.271** (XGBoost) | −0.086 |

† Not an independent seventh problem: the same listings and graph as the
Broward → San Diego price transfer, with the target swapped to
log1p(number of reviews) and price demoted to a covariate. It is a
controlled target swap, not a new region pair.

Full arm-by-arm results:

| arm | KC→Ames | Brow→SD | Chi→LA | CA→WV | 311 | trees |
| --- | --- | --- | --- | --- | --- | --- |
| GraphSAGE + LayerNorm | **0.501** | **0.679** | **0.654** | 0.763 | 0.232 | 0.132 |
| GraphSAGE (no norm) | −0.873 | 0.673 | 0.550 | −0.120 | 0.254 | 0.028 |
| XGBoost + lags | 0.283 | 0.648 | 0.529 | **0.803** | 0.244 | 0.095 |
| XGBoost | −0.106 | 0.649 | 0.626 | 0.749 | 0.289 | 0.270 |
| OLS | 0.140 | 0.648 | 0.632 | 0.771 | **0.301** | **0.276** |

The two newest problems (311 and trees) were added specifically to test the
claim in section 2, and **both went against it**. Predictions for both were
recorded in `reports/predictions.md` before the runs, and both were wrong in
sign. The 311 loss is not noise: paired over ten shared seeds, GraphSAGE +
LayerNorm trails XGBoost by 0.057 (p < 0.001).

Calibration slopes for the GraphSAGE+LayerNorm arm are 1.03, 1.07, 1.08,
1.40, 0.81 and 0.62 (1.0 is perfect). On the three wins it is the
best-calibrated arm on its problem; on the two new losses it is the *worst*.

### Neither non-price loss is an artefact of which region was the source

Both were run backwards as well. In both cases the reverse direction is an
easier problem in absolute terms — Queens is more predictable than Brooklyn
on either target — but the **ordering of the arms is identical** and the
GraphSAGE deficit is, if anything, larger.

**311 response time:**

| arm | Queens → Brooklyn | Brooklyn → Queens |
| --- | --- | --- |
| OLS | **0.301** | 0.488 |
| XGBoost | 0.289 | **0.494** |
| GraphSAGE (no norm) | 0.254 | 0.439 |
| XGBoost + lags | 0.244 | 0.457 |
| GraphSAGE + LayerNorm | 0.232 | 0.427 |
| GraphSAGE+LN vs XGBoost | −0.057 (p < 0.001) | −0.067 (p < 0.0001) |

**Street-tree diameter:**

| arm | Queens → Brooklyn | Brooklyn → Queens |
| --- | --- | --- |
| OLS | **0.276** | 0.324 |
| XGBoost | 0.270 | **0.334** |
| GraphSAGE + LayerNorm | 0.132 | 0.224 |
| XGBoost + lags | 0.095 | 0.205 |
| GraphSAGE (no norm) | 0.028 | 0.197 |
| GraphSAGE+LN vs best | −0.144 | −0.110 (p < 0.0001) |

In all four runs the tightest bandwidth helps and never closes the gap
(311 reverse 0.471 at 0.25× against OLS 0.488; trees reverse 0.301 at 0.10×
against XGBoost 0.334).

Run with `R/pair-core.R`, a direction-generic core validated by reproducing
`R/trees-core.R`'s deterministic OLS arm exactly (0.2755 / −0.0852 / 0.8492).

## 1a. The losses are a dispersion failure, not a bias failure

On both new problems the ranking by calibration slope is the ranking by R²:

| arm | 311 R² | 311 cal | 311 bias | trees R² | trees cal | trees bias |
| --- | --- | --- | --- | --- | --- | --- |
| OLS | **0.301** | **1.016** | +0.026 | **0.276** | **0.849** | −0.085 |
| XGBoost | 0.289 | 0.929 | +0.030 | 0.270 | 0.818 | −0.081 |
| GraphSAGE | 0.254 | 0.831 | +0.012 | 0.028 | 0.526 | −0.006 |
| XGBoost + lags | 0.244 | 0.829 | −0.028 | 0.095 | 0.600 | −0.105 |
| GraphSAGE + LayerNorm | 0.232 | 0.806 | +0.009 | 0.132 | 0.619 | −0.014 |

Every arm that touches the graph has a calibration slope below 1, meaning
its predictions vary *more* than the truth does. The graph is supplying
neighbourhood variation that does not correspond to real variation in the
target, and that injected variance is what costs the R².

This also settles a separate question: **LayerNorm does fix transfer bias.**
The GraphSAGE arms have far the smallest bias on both problems (−0.014 and
−0.006 on trees, against −0.085 for OLS). Bias simply is not the binding
constraint here. Fixing it does not help when the problem is dispersion.

## 2. Deciding in advance whether a GNN will help

The losses are not tuning failures; they are properties of the target. We
made three attempts to predict them from summary statistics of the data and
all three failed. What did work was reasoning about whether the quantity
being modelled actually spreads between places. Both are recorded here: the
successful approach first, then the withdrawn statistics, so the failures
stay visible.

The first formulation, now retracted, was:

> Fit OLS. Compare Moran's I of the **residuals** against Moran's I of the
> **covariates**, on the target region's graph.

- **Covariates carry less spatial signal than the residuals** → there is
  spatial structure the covariates cannot express, and a graph can reach it.
  GraphSAGE wins. (Broward → San Diego: residual I = 0.31, covariates
  I = 0.07–0.23.)
- **Covariates already carry as much or more than the residuals** → there is
  no gap for a graph to close, and the extra machinery only adds variance.
  XGBoost+lags wins. (CA → WV: residual I = 0.214, but six of ten covariates
  exceed it, up to I = 0.563.)

### The one place a prediction worked: reason from the mechanism, not from a statistic

Three successive statistical rules were fitted to past results and all three
failed (archived in `reports/archive/pre-test-WITHDRAWN.md`). Two predictions
did succeed, and both were made on entirely different grounds.

**Urban surface temperature, Des Moines → Cedar Rapids.** Heat physically
spreads: a park cools the blocks around it, a parking lot warms them. So a
location's temperature genuinely depends on its *neighbours'* surface cover,
not only its own. That is a spatial spillover **in the data-generating
process itself** — precisely simulation Scenario E, the only row of the
grid whose DGP contains a lag term and the only row where GraphSAGE beat
everything.

Predicted in advance, before fitting: both graph-using arms beat both
non-graph arms, with GraphSAGE +0.02 to +0.08 over the best non-graph arm.

| arm | rsq_trad | calibration |
| --- | --- | --- |
| **GraphSAGE + LayerNorm** | **0.822** | 0.966 |
| XGBoost + lags | 0.802 | 1.020 |
| OLS | 0.778 | 0.896 |
| GraphSAGE (no norm) | 0.755 | 0.784 |
| XGBoost | 0.725 | 1.072 |

Correct on all three counts: +0.044 over OLS (p = 0.00013), inside the
predicted band; GraphSAGE over XGBoost+lags by +0.020 (p = 0.034, 9 of 10
seeds). **Adding neighbour information to XGBoost alone is worth +0.077
(p < 0.00001)**, which confirms the spillover directly rather than by
inference.

The withdrawn statistical rule predicted a *loss* here.

**The usable principle is therefore about the process, not about a summary
statistic:** a graph helps when the thing being modelled physically or
socially spills across locations — heat, congestion, contagion,
neighbourhood desirability — and not merely when the data looks spatially
clumpy. Clumpiness is a symptom of many things, only one of which a graph
can exploit.

### The principle predicts the classification result too — in the other direction

The same two cities, the same 60 m grid, the same covariates, but the target
is now **land cover class** (built-up vs not) rather than temperature.
Land cover is extremely clumpy — far clumpier than temperature — so a
clumpiness-based rule would predict a large graph advantage. The process
principle predicts the opposite: a pixel's class is not *caused* by its
neighbours. Buildings cluster, but one building does not make the adjacent
plot built. There is nothing to spill.

| arm | AUC | balanced acc. | sensitivity | specificity |
| --- | --- | --- | --- | --- |
| GraphSAGE + LayerNorm | 0.9271 | 0.828 | 0.889 | 0.766 |
| **Logistic regression** | **0.9267** | 0.792 | 0.942 | 0.642 |
| GraphSAGE (no norm) | 0.919 | 0.837 | 0.860 | 0.813 |
| XGBoost + lags | 0.887 | 0.703 | 0.949 | 0.457 |
| XGBoost | 0.877 | 0.653 | 0.961 | 0.345 |

**GraphSAGE ties logistic regression on AUC (+0.0005, p = 0.95).** AUC is
threshold-free, so an exact tie means the two models carry the *same
discriminative information*. GraphSAGE's better balanced accuracy (+0.035)
and specificity (+0.123) come with worse sensitivity (−0.053) and worse
overall accuracy (−0.023) on an 83%-positive problem: that is a different
operating point on the same curve, not extra information. Moving logistic's
threshold would reproduce it.

GraphSAGE *does* beat both tree models clearly (+0.040 AUC over XGBoost+lags,
p = 0.003, 9 of 10 seeds), so the graph is not useless — it is simply not
adding anything a correctly specified linear model does not already have.

Taken together the two tasks are a clean natural experiment: same cities,
same grid, same features, two targets. The graph wins decisively on the one
whose process spills across space and ties on the one whose process does
not, exactly as the principle says and opposite to what the clumpiness of
each target would suggest.

### Retracted: the "individual record vs aggregate rate" version of this rule

An earlier draft of this section explained the CDC loss by target *kind*:
aggregated rates are smooth and already explained by their own covariates,
whereas individual records carry hyperlocal structure a graph can reach.
That gave the practical rule *"GNNs pay off for individual-record targets;
they do not pay off for smooth aggregate rates."*

**That rule is wrong and is withdrawn.** It was formulated on four pairs, in
which every win happened to also be a price. Two further pairs were then run
to test it, with predictions recorded in advance:

| target | kind | predicted | actual |
| --- | --- | --- | --- |
| Queens → Brooklyn, street-tree diameter | individual record | modest win | **loss, −0.144** |
| Queens → Brooklyn, 311 response time | individual record | small win or wash | **loss, −0.057 (p < 0.001)** |

Both are individual-record targets with non-trivial residual spatial signal,
and a GNN lost to plain OLS on both. Being an individual record is therefore
not sufficient. What actually holds across all six pairs is narrower and
less comfortable:

| target | is a price? | result |
| --- | --- | --- |
| King County house price | yes | WIN +0.218 |
| Broward Airbnb price | yes | WIN +0.030 |
| Chicago → LA Airbnb price | yes | WIN +0.022 |
| CA → WV diabetes prevalence | no | LOSS −0.040 |
| Queens → Brooklyn 311 response | no | LOSS −0.057 |
| Queens → Brooklyn tree diameter | no | LOSS −0.144 |

Every win is a price; no loss is. That is a perfect split on six points,
which is worth stating plainly and worth trusting very little — with three
wins all from one domain, "price" and "whatever makes these three datasets
different" are not separable. Section 2a gives a measurable quantity that
does the same separation without appealing to the target's name, which is
the version that can actually be used.

### The tree loss is not a tuning failure: the tuner switches the graph off

The first bandwidth screen on the tree transfer was monotonic all the way to
the edge of its grid, so the uniform-edge result (0.132) was a mis-specified
neighbourhood rather than a ceiling. Extending the sweep settles it:

| gaussian bandwidth (× median 30-NN distance) | GraphSAGE + LayerNorm | XGBoost + lags |
| --- | --- | --- |
| 2.0× | 0.119 | — |
| 1.0× | 0.177 | — |
| 0.10× | 0.237 | 0.234 |
| 0.05× | 0.246 | 0.234 |
| 0.02× | 0.250 | 0.179 |
| 0.01× | **0.252** | 0.131 |

The curve flattens at 0.252, so tuning is converged and the honest
best-case for GraphSAGE here is 0.252 — still below OLS at 0.276.

What that optimum *is* matters more than its value. At 0.01× the gaussian
weight on a neighbour at the median distance is exp(−100²/2), i.e. zero to
machine precision; with self-loops every node sees only itself. **The
bandwidth search converges on discarding the graph**, and a GraphSAGE that
has switched its own graph off still loses to OLS. The graph is not
under-tuned on this problem, it is actively harmful, which is the same
conclusion the calibration slopes reach in section 1a from a different
direction.

The plausible mechanism, offered as a hypothesis and not established here:
a price carries a large unobserved, spatially smooth location-value
component that the covariates do not proxy, and neighbour aggregation is a
good estimator of exactly that. Tree diameter is driven by age and species,
whose spatial clustering comes from planting cohorts that the species
dummies already encode; 311 response time is driven by complaint type and
sanitation-district operations, likewise largely in the covariates.

## 2a. WITHDRAWN: the statistic that appeared to separate them

This section previously reported a rule — the share of covariates more
spatially clustered than the unexplained variation — that separated wins from
losses with no overlap. **It is withdrawn.** It was fitted to seven points,
and a third formulation built afterwards on better reasoning failed as well.

Three attempts, all withdrawn:

| version | basis | why it failed |
| --- | --- | --- |
| 1 | Individual records vs aggregate rates | Two individual-record targets lost to plain regression |
| 2 | How clustered the unexplained variation is | Wins 0.221-0.615, losses 0.116-0.446 — overlapping |
| 3 | How much of it neighbouring values recover | Wins 0.051-0.282, losses 0.005-0.075 — overlapping |

Version 3 had the best argument behind it: GraphSAGE reads its neighbours'
inputs and never their answers, so clustering that lives only in the answer
is unreachable to it. That argument is correct and is retained above. The
measurement built from it still did not predict, because it captures what is
recoverable without capturing whether recovering it is worth anything —
California to West Virginia already explains 96% of the variance, leaving no
headroom for any method.

The working replacement is the process-based reasoning in the section above,
which has made one correct prediction on a win and one on a tie. Full record
in `reports/archive/pre-test-WITHDRAWN.md`.


## 3. Transfer is only testable between regions with a compatible relationship

Two region pairs (LA → NYC, Chicago → Nashville) produced near-zero or
negative R² **for every arm including OLS**, with large systematic bias that
flipped sign between pairs. The cause is not architectural: the
covariate→price relationship genuinely differs between those markets. Fit
separately, `bedrooms` is worth 0.156 in LA and 0.030 in NYC.

Screen for this before spending any model compute: fit OLS on region A,
predict region B, check bias and R². A pair that fails this measures the
pair, not the architecture. Across 13 cities and 156 ordered pairs this
costs seconds and correctly ranked every pair we subsequently tested at
full scale (`R/airbnb-screen-pairs.R`).

### The two screens have now rejected three candidate datasets for free

Both screens are cheap enough to run on every candidate before committing
compute, and doing so has paid off three times:

| candidate | rejected by | evidence |
| --- | --- | --- |
| Chicago / Seattle building energy (site EUI) | Moran profile | Chicago target I = −0.011 — no spatial structure at all; Seattle covariates 0.21–0.31 against a residual of 0.079 |
| NYC PLUTO building age, **all 5 boroughs / 20 ordered pairs** | OLS/DGP | every pair has a direction with negative out-of-region R²; best is Bronx↔Brooklyn at −0.006, worst −2.40, biases of 1–3 decades. Each borough maps building form to a different era, so building age is not transferable across NYC boroughs at all |
| LA → NYC, Chicago → Nashville (Airbnb) | OLS/DGP | near-zero or negative R² for every arm including OLS |

The PLUTO rejection cost one data fetch and about two minutes of screening,
against the roughly twenty minutes of fitting a full arm comparison would
have taken — and would have produced a number that measured the borough
pair rather than the method.

## 4. Model specification

```
GraphSAGE, hidden dims (56, 32, 16)
  LayerNorm      mode = "node"          # not the mode="graph" default
  dropout        0.1                     # measured optimum, inverted-U
  weight decay   1e-4                    # Adam
  edges          KNN-k, Gaussian kernel via sfdep::st_kernel_weights()
  bandwidth      TUNE PER DATASET
  loss           L1, early stopping on a validation slice of the source
```

Each component is load-bearing:

- **Node-mode LayerNorm** is what prevents catastrophic failure under
  distribution shift. Without it, GraphSAGE scores −0.873 (KC→Ames) and
  −0.120 (CA→WV) — worse than predicting the target mean — and its
  seed-to-seed spread is enormous. `mode="graph"`, the default when the
  function is passed bare, normalizes the whole graph to one scalar and is
  not what "LayerNorm" means.
- **Dropout 0.1** is a measured optimum: 0.05 < 0.1 > 0.15 > 0.2 > 0.3 > 0.4.
- **Weight decay 1e-4** stacks with it (+0.087 on KC→Ames over dropout alone).
- **Gaussian kernel weighting** improved GraphSAGE on all four KNN-graph
  transfer problems. Gaussian is the only numerically stable shape tested —
  triangular and epanechnikov diverge at some bandwidths (epanechnikov
  reached −13.9).
- **The bandwidth does not transfer.** Optima were 0.02×, 0.75×, 2.0× and 4×
  the median neighbour distance on the four datasets — a 200-fold range,
  each a genuine peak with worse performance on both sides. Screen it every
  time; the cost is one short sweep.

Kernel tuning also buys stability, not just accuracy: on KC→Ames it cut
seed-to-seed spread from 0.353 to 0.112.

**Kernel tuning only applies to KNN graphs.** On the states problem, which
uses queen contiguity, no bandwidth beat uniform weighting and tight
bandwidths significantly hurt (1,365 fits, 7 settings, 5 seeds × 39 states,
paired per-state tests vs uniform):

| setting | diff vs uniform | p |
| --- | --- | --- |
| gaussian 2× | +0.003 | 0.52 |
| gaussian 4× | +0.003 | 0.58 |
| adaptive | −0.001 | 0.92 |
| gaussian 1× | −0.006 | 0.39 |
| gaussian 0.5× | −0.017 | 0.056 |
| gaussian 0.25× | **−0.056** | **<0.001** |

The mechanism is straightforward: a KNN graph contains arbitrary far
neighbours that exist only because *k* was fixed, and the kernel's job is to
suppress them. A contiguity graph has no such edges — every one is a real
shared border — so down-weighting by distance only discards information,
and the tighter the kernel the more it discards. **Tune the kernel when the
neighbour set is KNN; leave it uniform when the neighbour set is
contiguity-based.**

## 5. Simulation grid (Scenarios A–E, block and random holdout)

Six inductive folds, n = 7,000 on a 100×70 lattice, KNN-15. Realized (not
nominal) Moran's I: covariate Ix 0.379 / 0.686, error Iu 0.390 / 0.770.
Block design, `rsq_trad`:

| scenario | GraphSAGE | +LayerNorm | OLS | SEM | XGBoost | XGB+lags |
| --- | --- | --- | --- | --- | --- | --- |
| A linear, iid | 0.345 | 0.330 | 0.358 | — | 0.156 | 0.188 |
| B nonlinear (Ix 0 / 0.4 / 0.7) | 0.968 / 0.964 / 0.954 | 0.965 / 0.961 / 0.947 | 0.778 / 0.771 / 0.734 | — | 0.957 / 0.952 / 0.940 | 0.955 / 0.951 / 0.940 |
| C spatial error, Iu=0.0 | 0.338 | 0.325 | 0.355 | 0.355 | 0.155 | 0.203 |
| C, Iu=0.4 | 0.299 | 0.297 | 0.342 | 0.343 | 0.144 | 0.157 |
| C, Iu=0.8 | 0.006 | 0.057 | **0.280** | **0.285** | 0.037 | 0.074 |
| D nonlinear + spatial error | 0.928–0.937 | 0.921–0.936 | 0.386–0.456 | 0.398–0.456 | 0.905–0.912 | 0.904–0.914 |
| E lag effect, Ix=0.0 | **0.429** | 0.411 | 0.283 | — | 0.067 | 0.294 |
| E, Ix=0.4 | **0.509** | 0.496 | 0.418 | — | 0.231 | 0.381 |

Three conclusions the grid supports:

1. **Scenario C is a clean loss for GNNs and should be reported as one.**
   With spatially autocorrelated errors and a correctly specified mean,
   OLS/SEM win decisively (0.280/0.285 vs 0.006). Under an inductive
   protocol nothing observes the held-out region's error field, so there is
   nothing for any method to recover. Any claim that GraphSAGE partially
   recovers a residual surface is not supported.
2. **Scenario E is the only row whose DGP contains a lag term**, and it is
   the only place XGBoost+lags shows a defensible gain over plain XGBoost
   (0.067→0.294, 0.231→0.381). Lag results in A–D are a property of the
   equations, not evidence about lag features.
3. **Holdout design changes conclusions, not just precision.** Scenario E
   scores 0.429 under block and 0.211 under random holdout for the same
   method on the same data. The LayerNorm effect also reverses by design at
   C/Iu=0.8: block 0.006→0.057 (large gain), random 0.181→0.161 (loss).

Per-fold results: `reports/renato-scenario-folds-{all,block,random}.csv`.

## 6. Held-out US states (39 states, county-level vote share)

Every state held out in turn, trained on all counties outside it. GraphSAGE
arms measured over 5 seeds × 39 states (585 fits); the tabular arms are
deterministic or near-deterministic and are single fits.

| arm | MAE | rsq_trad | sd across seeds |
| --- | --- | --- | --- |
| GraphSAGE + LayerNorm + dropout + wd | 5.73 | **0.633** | 0.011 |
| GraphSAGE (no norm) | 5.69 | 0.622 | 0.013 |
| GraphSAGE + LayerNorm only | 5.89 | 0.612 | 0.008 |
| XGBoost + lags | 6.36 | 0.542 | — |
| XGBoost | 6.38 | 0.532 | — |
| OLS | 7.95 | 0.187 | — |
| SEM | 8.33 | 0.186 | — |

**Every GraphSAGE variant beats every tabular model by a wide margin**
(~0.07–0.10 over XGBoost+lags, ~0.43 over OLS/SEM); GraphSAGE beats
XGBoost+lags in 30 of 39 states.

**The GraphSAGE variants are not distinguishable from each other here.** The
three configurations span 0.612–0.633 with per-seed SD of 0.008–0.013, and
both paired per-state comparisons are non-significant (full recipe vs plain:
+0.011, p = 0.32; plain vs bare LayerNorm: +0.010, p = 0.34). Normalization
and regularization neither help nor hurt detectably on this problem — which
is the expected result under section 2, since a held-out state is drawn from
the same national distribution as the training states and there is no
distribution shift for them to protect against.

This is the one place in the project where the arms are close enough that a
single fit will mislead: the production `states.R` runs one fixed seed per
state, and that draw happened to put plain GraphSAGE (0.645) above every one
of the five seeds sampled here. **Quote the multi-seed means above, not the
single-seed run.**

## 7. Measurement requirements

- **Report `rsq_trad`, not `rsq`.** Squared correlation is invariant to
  affine rescaling and cannot see miscalibration. The gap is not cosmetic:
  plain GraphSAGE on KC→Ames scores 0.543 as squared correlation and −0.873
  as real R². Report calibration slope alongside it.
- **Report the spread across training runs, not just a point estimate.** A
  single fit's R² varies by 0.1–0.35 SD depending on the recipe. Repeated
  runs are for characterizing that distribution.
- **Ship and report one model.** Averaging several models' predictions
  inflates results (0.380 → 0.583 on KC→Ames) but is not a deployable
  artifact and is excluded from every number in this document.

## 8. What is not established

- **Predicting the benefit of a graph from summary statistics failed three
  times.** Judgement about whether the modelled quantity spreads between
  places has now called two problems correctly in advance (urban heat, a
  win; land cover, a tie), but two correct calls is a start, not a
  validated method. It has not yet been tested on a case where it predicts
  a loss.
- **The three price wins have each been run in one direction only.** Both
  losses were run both ways and replicate. The wins are what the paper
  rests on, so the asymmetry matters; `R/pair-core.R` makes fixing it cheap
  and it should come before any new datasets.
- **Bandwidth selection has no theory here**, only a search. The optimum is
  dataset-specific and worth looking for; we cannot predict it from
  properties of the data.
- **Two rental pairs failed outright** (LA→NYC, Chicago→Nashville) because
  the underlying relationship differs between those markets. We can detect
  this but not fix it. Fixing it would need features explaining *why*
  markets differ — transit access, tourism density, regulation.
- **Arm differences on the states problem sit below the noise floor.** With
  per-seed SD around 0.01 and gaps around 0.01, ranking the GraphSAGE
  variants there would need far more than the 5 seeds run. What is
  established is that all of them beat the tabular models; which is best is
  not.
- **Neighbour definition is barely explored.** On the raster problems the
  natural grid adjacency was competitive with the KNN-30 used everywhere
  else (heat: 0.822 vs 0.798; land cover: 0.927 vs 0.933), so the default
  cost nothing, but the wider settings rest on 5 seeds.
- **Resolved since the last draft:** every win used to be a price, leaving
  that perfectly confounded with success. Urban surface temperature broke
  it — a measured, non-price, raster target on which GraphSAGE wins by
  0.044 (p = 0.00013). Four earlier attempts to break the confound failed
  for unrelated reasons (building age and soil carbon failed the
  transferability screen; rental availability was unpredictable anywhere;
  rental reviews had no headroom).

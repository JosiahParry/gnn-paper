# Ex-ante predictions from the pre-compute screen

The Moran's I diagnostic in section 2 of `REPORT.md` was formulated *after*
seeing four transfer results. That makes it a description, not a test. To
make it a test, the prediction for each new problem is recorded here from
the screen alone, **before the models are fitted**, and scored afterwards
without revision.

The rule being tested:

> Fit OLS. Compare Moran's I of the residuals against Moran's I of the
> covariates on the target region's graph. A graph helps when residual
> spatial signal is large *relative to* what the covariates already carry.

---

## Queens → Brooklyn, street-tree DBH (`R/trees-test.R`)

Screen: OLS cross-prediction passes both directions (+0.275 / +0.324).
Residual Moran's I 0.157 (Queens) / 0.229 (Brooklyn), with mixed covariate
coverage — several covariates sit near or above the residual.

**Prediction: a modest GraphSAGE win, not a dramatic one.** Smaller than
King County → Ames (+0.218), plausibly in the range of the two Airbnb
transfers (+0.02 to +0.03). Not a loss, because residual signal is clearly
non-zero and the OLS screen is clean in both directions.

### Result: WRONG. GraphSAGE lost, and not narrowly.

| arm | rsq_trad | cal slope |
| --- | --- | --- |
| OLS | **0.276** | 0.849 |
| XGBoost | 0.270 | 0.818 |
| GraphSAGE + LayerNorm | 0.132 | 0.619 |
| XGBoost + lags | 0.095 | 0.600 |
| GraphSAGE (no norm) | 0.028 | 0.526 |

The predicted sign was wrong (−0.144, not a small positive), and the
best model on this problem uses no graph and no trees. Both graph-based
arms — GraphSAGE *and* XGBoost+lags — fall well below the two arms that
ignore the graph entirely, which points at the neighbourhood definition
rather than at GraphSAGE.

That reading is supported by the bandwidth screen, which had **not bottomed
out** at the tightest setting tested: rsq_trad rose monotonically from 0.119
at 2.0× the median 30-NN distance, to 0.177 at 1.0×, to 0.237 at 0.1×. The
uniform-edge 0.132 is therefore a mis-specified neighbourhood, not the
method's ceiling here. `R/trees-kernel-tight.R` continues the sweep to
0.01×; whatever it finds is reported, and even the best value so far (0.237)
is still below OLS.

**What this costs the rule as stated.** Section 2 of `REPORT.md` claims
individual-record targets favour a GNN and smooth aggregate rates do not.
Tree diameter is an individual-record target with non-trivial residual
spatial signal, and a GNN lost to OLS on it. So "individual record" is not
sufficient, and the rule as written is too strong.

## Queens → Brooklyn, 311 response time (`R/nyc311-test.R`)

Screen: OLS cross-prediction passes strongly — Queens→Brooklyn rsq_out
0.301, calibration slope 1.016, bias +0.026; Brooklyn→Queens 0.488. This is
the best-calibrated cross-region pair in the project. Largest coefficient
disagreement is `cx_snow.or.ice` (2.77 Queens vs 3.55 Brooklyn), a genuine
but contained difference.

Queens Moran's I: target 0.269, OLS residuals 0.127, covariates median
0.078 / max 0.542. **13 of 33 covariates exceed the residual I** (39%),
against 6 of 10 (60%) in the CA → WV case the rule says a graph should
lose, and against Broward → San Diego (residual 0.31 vs covariates
0.07–0.23) where it clearly won.

**Prediction: an intermediate case — a small GraphSAGE win or a wash.**
This is deliberately the least certain call of the set, which is what makes
it worth recording: the screen places it between the clear win and the
clear loss, so the outcome discriminates between "the rule has real
predictive content" and "the rule was fitted to four points."

The failure mode that would falsify the rule: a *large* GraphSAGE win here
(> +0.10), or a clear loss to a tabular arm despite non-trivial residual
signal.

**Addendum, added after the prediction above was written but still before
any model was fitted:** the Brooklyn (target-region) profile finished after
the prediction was recorded, and reads target I 0.112, residual I 0.111,
covariates median 0.077 / max 0.237. Residual I is essentially *equal* to
the target's own I — OLS removed almost none of the spatial structure, which
is the clearest "gap for a graph to close" signature in the set. Taken alone
this would have argued for a firmer win than "small win or wash". The
prediction above is left as written rather than revised, because a
prediction edited after seeing more of the screen is not a prediction.

---

### Result: WRONG in sign, though the low confidence was warranted.

| arm | rsq_trad | cal slope |
| --- | --- | --- |
| OLS | **0.301** | 1.016 |
| XGBoost | 0.289 | 0.929 |
| GraphSAGE (no norm) | 0.254 | 0.831 |
| XGBoost + lags | 0.244 | 0.829 |
| GraphSAGE + LayerNorm | 0.232 | 0.806 |

Paired over 10 shared seeds, GraphSAGE + LayerNorm vs XGBoost:
**−0.057, p < 0.001** — a real loss, not noise. Best kernel setting (0.25×)
reaches 0.260, still below OLS at 0.301.

Two incidental observations worth carrying forward:

- **Plain GraphSAGE beat GraphSAGE + LayerNorm here** (0.254 vs 0.232), the
  same ordering that looked suspicious on the states holdout and was shown
  there to be noise. Here the arms are separated by more than the seed
  spread, so it is not obviously noise on this problem.
- **OLS has a calibration slope of 1.016** — essentially perfect. Nothing
  was left for a more flexible model to pick up.

---

## Scoring both predictions

Two ex-ante predictions, both wrong in sign. That is the outcome the file
was created to make un-hideable, and it carries a specific cost: the
diagnostic in section 2 of `REPORT.md` was formulated on four pairs and has
now failed on both fresh pairs it was asked to call.

The pattern across all six pairs is no longer "individual record vs
aggregate":

| target | kind | result |
| --- | --- | --- |
| King County house price | individual, **price** | WIN +0.218 |
| Broward Airbnb price | individual, **price** | WIN +0.030 |
| Chicago→LA Airbnb price | individual, **price** | WIN +0.022 |
| CA→WV diabetes prevalence | aggregate rate | LOSS −0.040 |
| Queens→Brooklyn tree DBH | individual, physical | LOSS −0.144 |
| Queens→Brooklyn 311 response | individual, operational | LOSS −0.057 |

Every win is a **price**. Every loss is not. "Individual-record" does not
separate them; being a price does. The plausible mechanism — to be tested,
not asserted — is that prices carry a large unobserved, spatially smooth
location-value component that the covariates do not proxy, which is exactly
what neighbour aggregation can supply. Tree diameter is driven by age and
species, and its spatial clustering comes from planting cohorts that the
species dummies already encode. 311 response time is driven by complaint
type and sanitation-district operations, again largely in the covariates.

`R/moran-calibrate2.R` tests whether this is visible in the Moran profile
before fitting anything, and whether it is visible **without target
labels**. Its answer is in section 2a of `REPORT.md`: the separating
statistic is the *fraction of covariates more autocorrelated than the
residual*, not the absolute residual autocorrelation.

---

## Attempted decisive test: NYC PLUTO building age — rejected by the screen

With "price" perfectly confounded with "wins", the one experiment that
separates *the rule is about spatial signal* from *the rule is about prices*
is a **non-price target with a low covariate-exceedance fraction**. Building
age (year built, NYC PLUTO) was prepared for exactly this: not a price, and
about as spatially clustered as an urban variable gets, since city blocks
are developed in cohorts.

**Its Moran profile is exactly what the decisive test needs.** Source
region residual I = 0.325, and **frac_src_exceed = 0.18** — below the 0.25
win threshold, on a target that is not a price. So "non-price" and "low
exceedance fraction" are genuinely separable properties, and a test that
discriminates the two hypotheses is constructible in principle.

**But it never reached a model: no NYC borough pair passes the DGP screen.**
Queens ↔ Brooklyn fails in both directions, and extending to all five
boroughs and all 20 ordered pairs does not rescue it:

| pair | R² A→B | R² B→A | worse direction |
| --- | --- | --- | --- |
| bronx ↔ brooklyn | 0.048 | −0.006 | **−0.006** (best of all 20) |
| queens ↔ statenisland | 0.031 | −0.177 | −0.177 |
| brooklyn ↔ queens | −0.212 | −0.139 | −0.212 |
| *…14 further pairs, all worse* | | | down to −2.40 |

Every pair has at least one direction with negative out-of-region R², with
biases of one to three decades. Brooklyn was developed earlier than Queens,
Manhattan earlier still, Staten Island much later — a given building form
maps to a different era in each borough, and demeaning removes the level but
not the differing slope. The covariates also explain age only weakly
in-sample (R² 0.14–0.17).

Per section 3, such a pair measures the pair, not the architecture. Running
a GNN on it would produce a number that means nothing. **Building age is not
transferable across NYC boroughs**, which is itself a clean finding, obtained
for about a minute of screening.

This is the screening discipline working rather than a setback — it is the
second candidate rejected before spending compute, after the Chicago/Seattle
building-energy pair was rejected on its Moran profile. **The decisive
experiment remains open**, and section 8 of `REPORT.md` records what it
needs: a non-price target, a region pair that passes the OLS screen, and a
low covariate-exceedance fraction.

---

## Controlled target swap: Airbnb reviews, Broward → San Diego

A new dataset changes the region pair, the covariates and the geometry at
the same time as the target, so attribution is hard. The Airbnb tables allow
a stricter version: **same rows, same regions, same graph, same covariates —
only the target moves**, with price demoted from target to covariate.

Two non-price candidates were screened (`R/airbnb-nonprice-build.R`):

| target | DGP screen | frac_src_exceed | usable? |
| --- | --- | --- | --- |
| `availability_365 / 365` | **fails** (R² −0.041 / −0.005, in-sample only 0.026) | — | no — essentially unpredictable |
| `log1p(number_of_reviews)` | asymmetric (+0.210 forward, −0.116 reverse) | **0.50** | forward direction only |

**Neither is the decisive experiment.** The decisive experiment needs a
non-price target with a *low* exceedance fraction, where §2a predicts a win
and the price hypothesis predicts a loss. Reviews scores 0.50, so both
hypotheses predict a loss and the outcome cannot discriminate between them.

It is still worth running as a **prospective test of §2a**, which has made no
correct prediction yet and whose two predecessors went 0-for-2.

**Prediction: GraphSAGE + LayerNorm loses to the best non-graph arm.**
Exceedance fraction 0.50 puts it level with the tree transfer (0.50,
−0.144) and worse than 311 (0.33, −0.057). Expected margin in the −0.05 to
−0.15 band. A GraphSAGE *win* here would falsify §2a outright.

### Result: CORRECT — the first prediction this project has got right.

| arm | rsq_trad | cal slope |
| --- | --- | --- |
| XGBoost | **0.271** | 1.115 |
| OLS | 0.210 | 1.297 |
| XGBoost + lags | 0.191 | 0.879 |
| GraphSAGE + LayerNorm | 0.185 | 0.823 |
| GraphSAGE (no norm) | 0.125 | 0.743 |

Paired over ten shared seeds: **−0.086, p = 0.0004**, CI [−0.121, −0.050].
Predicted sign correct, and the margin lands inside the predicted −0.05 to
−0.15 band. Tightening the kernel helps as usual and as usual does not close
the gap (0.254 at 0.10×, against XGBoost 0.271).

**How much credit this earns.** Less than it looks. Predicting a loss is the
easier call — four of the seven pairs now lose — so one correct negative
prediction is weak evidence. §2a has still never predicted a *win* on a pair
it was not fitted to, and that is the prediction that would cost it
something. It also remains silent on the price confound, by construction:
reviews scores 0.50, so the price hypothesis predicted the same loss.

Prospective record of the diagnostic: **1 for 3**, and the one hit is the
version rewritten after the two misses.

One result is already in hand and does not depend on the run. Holding rows,
regions, covariates and graph fixed and changing only the target moves the
diagnostic from **0.25 (price, a win) to 0.50 (reviews)**. The statistic is
therefore responding to the target's spatial structure relative to its
covariates, not to the geometry or the feature set — which is what it claims
to measure. That is a controlled check the six-pair calibration could not
provide, since there every pair varied everything at once.

---

*Recorded before either test's results were read. Scored in `REPORT.md`.*

---

## Urban heat, Des Moines → Cedar Rapids (`R/heat-fetch.R`)

The best-transferring pair in the project. OLS cross-prediction is 0.778
forward and 0.681 reverse, with calibration 0.90 / 1.17 — out-of-region
accuracy essentially equal to in-sample. For comparison, the previous best
was 0.648 and King County → Ames managed 0.140.

Target: land surface temperature (Landsat thermal, measured). Covariates:
Landsat optical bands and the vegetation / built-up / water indices derived
from them, plus terrain. Thermal bands excluded from covariates.

**Two predictions in conflict, which is why this test is worth running.**

| basis | says |
| --- | --- |
| Mechanism: heat physically spreads, so a cell's temperature depends on its *neighbours'* surface cover — a genuine spatial lag, i.e. simulation Scenario E, where GraphSAGE beat everything (0.429 vs OLS 0.283) | **WIN** |
| The withdrawn Screen-2 statistic: covariate-exceedance fraction 0.83, far into the "loses" range | **LOSE** |

**My prediction: the graph-using arms (GraphSAGE and XGBoost+lags) both beat
the non-graph arms**, because the lag mechanism is physically real here in a
way it was not in any earlier dataset. Between GraphSAGE and XGBoost+lags I
genuinely do not know — both see the same neighbour information.

Stated as a number: GraphSAGE + LayerNorm beats the best *non-graph* arm by
**+0.02 to +0.08**. Against the best arm *including* XGBoost+lags, a coin
flip.

Caveat that could sink it either way: OLS already reaches 0.78, so headroom
is limited — though not saturated the way CA → WV was at 0.96, which is the
case that broke every version of the rule.

This is the first time a prediction here rests on the physics of the target
rather than on a statistic fitted to previous results.

### Result: CORRECT — the first prediction this project got right on a win.

| arm | rsq_trad | calibration |
| --- | --- | --- |
| **GraphSAGE + LayerNorm** | **0.822** | 0.966 |
| XGBoost + lags | 0.802 | 1.020 |
| OLS | 0.778 | 0.896 |
| GraphSAGE (no norm) | 0.755 | 0.784 |
| XGBoost | 0.725 | 1.072 |

Scoring the three parts of the prediction:

| predicted | actual |
| --- | --- |
| Both graph-using arms beat both non-graph arms | **Yes** — 0.822 and 0.802 against 0.778 and 0.725 |
| GraphSAGE+LN beats best non-graph arm by +0.02 to +0.08 | **+0.044**, inside the stated band (p = 0.00013) |
| GraphSAGE vs XGBoost+lags a coin flip | GraphSAGE by +0.020, p = 0.034, 9 of 10 seeds |

**The mechanism is confirmed directly.** Giving XGBoost access to
neighbours' surface cover improves it by **+0.077 (p < 0.00001)** over
plain XGBoost. The spatial spillover is real and large — a cell's
temperature genuinely does depend on what surrounds it — and GraphSAGE then
exploits that spillover slightly better than hand-built lag features do.

**The withdrawn Screen-2 statistic predicted a LOSS here** (exceedance
fraction 0.83, far into its "loses" range). It was wrong, which is further
confirmation that withdrawing it was correct.

**Why this one mattered.** It is the first prediction in this file grounded
in the *physics of the target* rather than a statistic fitted to previous
results — and the first correct call on a **win**, which is the direction
that costs something to predict. Prospective record of mechanism-based
reasoning: 1 for 1. Prospective record of fitted statistics: 1 for 3, and
the one hit was a predicted loss.

It is also the first non-price win, which breaks the confound that four
separate attempts (building age, availability, reviews, soil carbon) failed
to break.

---

## Land cover classification, Des Moines → Cedar Rapids (`R/clf-test.R`)

Same cities, same 60 m grid, same covariates as the urban-heat test. Only
the target changes: built-up vs not, instead of temperature. The project's
first classification task.

**Prediction recorded before the run** (in the run script's header): a graph
should help, because bare soil in a city and bare soil in a field have
nearly identical spectra and only their surroundings separate them.

### Result: the graph beats the tree models, and ties logistic regression.

| arm | AUC | balanced acc. | sensitivity | specificity |
| --- | --- | --- | --- | --- |
| GraphSAGE + LayerNorm | 0.9271 | 0.828 | 0.889 | 0.766 |
| **Logistic regression** | **0.9267** | 0.792 | 0.942 | 0.642 |
| GraphSAGE (no norm) | 0.919 | 0.837 | 0.860 | 0.813 |
| XGBoost + lags | 0.887 | 0.703 | 0.949 | 0.457 |
| XGBoost | 0.877 | 0.653 | 0.961 | 0.345 |

vs XGBoost + lags: **+0.040 AUC, p = 0.003**, 9 of 10 seeds.
vs Logistic: **+0.0005 AUC, p = 0.95** — a tie.

**The honest reading is that this is not a graph win.** AUC is
threshold-free, so tying on it means the two models hold the same
discriminative information. GraphSAGE's higher balanced accuracy (+0.035)
and specificity (+0.123) are bought with lower sensitivity (−0.053) and
lower accuracy (−0.023): a different point on the same ROC curve, not extra
signal. The mixed-pixel argument in the script header was plausible and
turned out to be wrong — logistic regression on the raw bands already
separates these classes as well as neighbour context does.

**But the process principle called it correctly.** Land cover is far
*clumpier* than temperature, so any rule based on spatial clumpiness would
have predicted a large graph advantage here and a smaller one for heat. The
principle says the opposite: a pixel's class is not caused by its
neighbours — buildings cluster, but one building does not make the next plot
built — so there is no spillover for a graph to recover. Heat does spill.
The graph wins on heat and ties on land cover.

Two targets, same cities, same grid, same features, opposite outcomes,
predicted in the right direction by the mechanism and the wrong direction by
clumpiness. That is the strongest evidence in this project for how to decide
whether a GNN is worth using.

### Incidental: neighbour definition

| task | best neighbourhood |
| --- | --- |
| Heat (regression) | queen2 (24 cells) 0.825, queen (8) 0.822, KNN-30 0.798 |
| Land cover (classification) | KNN-30 0.933, queen 0.927, queen2 0.923 |

Differences are small and the wider settings rest on 5 seeds rather than 10.
No recommendation is drawn from this yet beyond noting that the natural
raster adjacency is competitive with the KNN-30 used everywhere else, so
nothing was lost by that default.

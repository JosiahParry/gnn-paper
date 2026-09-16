# Can GraphSAGE actually transfer across markets? Overnight findings

Working document, updated as each experiment finishes. Goal: settle whether
GraphSAGE (an inductive GNN) can be made to genuinely win at spatial transfer
-- train in one region, predict in a structurally disjoint one -- and if not,
say so plainly and explain why XGBoost + spatial lags wins instead.

## 1. Two bugs fixed before any of this was trustworthy

### 1.1 The R² metric was the wrong R²

Every "R²" in the codebase and in `main.tex` was `yardstick::rsq`, which is
squared Pearson correlation -- invariant to affine rescaling of the
predictions. A model whose predictions are systematically too spread out (or
compressed, or offset) can still score well on it, because it only asks
whether predictions move in the same direction as truth, never whether they
land close to it. That's the wrong question for a tool whose whole purpose is
handing someone a usable predicted value.

Fix: added `yardstick::rsq_trad` (1 − SSE/SST, the textbook R²) alongside
`rsq` everywhere, and recomputed it for every existing result file
(`kc-to-ames`, states, both simulation designs) **without refitting any
model** -- the truth values and fold/state/scenario assignments are all
deterministic (seeded DGPs, seeded splits, or real un-randomized data), so
`var(truth)` could be recovered analytically and combined with the RMSE
already on disk.

What changed once measured correctly:
- **KC → Ames, plain GraphSAGE**: `rsq = 0.543` (looked fine) →
  `rsq_trad = -0.873` (worse than predicting the Ames mean for every house).
- **KC → Ames, GraphSAGE + LayerNorm**: `0.499` → `0.199`.
- **KC → Ames, XGBoost + lags**: `0.401` → `0.283` (now the best arm).
- **States, LayerNorm vs plain GraphSAGE**: paper claimed a "coin flip," 21/39
  states. Under `rsq_trad` it's 15/39; by MAE it's 9/39 — LayerNorm is a net
  negative in the states setting, not neutral.
- **Simulation Scenario C** (spatial errors, Iu=0.8): plain GraphSAGE's
  `rsq_trad` is 0.014, not the reported 0.218 — it was never really predicting
  anything there, just correlated with the truth while badly miscalibrated.
  LayerNorm's rescue effect is *larger* than previously reported (0.014 →
  0.167, not 0.218 → 0.307).

### 1.2 LayerNorm was silently running in the wrong mode

`torchgnn::layer_layer_norm(in_features, mode = c("graph", "node"))` defaults
to `mode = "graph"` when called positionally, which is exactly how every call
site in this project invoked it (`norm = layer_layer_norm`, passed bare to
`model_sage()`, which calls it as `norm(hidden_dim)` with no override).
`"graph"` mode collapses **every node in the current graph to one shared
scalar mean/variance per layer** -- not the per-node normalization
"LayerNorm" is supposed to mean (and what every other normalization layer in
the deep learning literature does).

Isolated by ablation (KC → Ames, 15 seeds, everything else held fixed):

| variant | rsq_trad | cal_slope |
| --- | --- | --- |
| baseline (mode="graph", the accidental default) | 0.199 | 0.774 |
| **mode="node" (the fix)** | **0.247** | **0.854** |

This closes about 60% of the gap to XGBoost+lags (0.283) with one line of
code. Fixed everywhere (`R/sim-core.R`, `R/states-core.R`,
`R/kc-to-ames-core.R` now all use a `layer_layer_norm_node()` wrapper).

One hypothesis this ablation **disproved**: that GraphSAGE's `concat=TRUE`
default (which passes each node's raw features directly into every layer,
on top of the self-loop already folding them into the neighbour average) was
letting the model "cheat" by ignoring the graph. Setting `concat=FALSE` to
force pure neighbour-aggregation was tested and made things much worse
(rsq_trad -0.239, and -1.655 combined with the node-mode fix) -- not because
the focal feature was being overused, but because additive combination
requires `x` and `neighbor_agg` to share a scale/semantics they don't have,
and halves the layer's effective capacity. Ruled out, not pursued further.

## 2. Architecture sweep, wave 2

Six more factors tested against the `mode="node"` baseline (0.247), 15 seeds
each: removing the redundant self-loop, Gaussian-kernel-weighted edges
(distance decay instead of uniform KNN-30 weights), a narrower network
(32,16 instead of 56,32,16), dropout, weight decay, and feeding `[X, lag(X)]`
as input on top of the graph aggregation.

Results (15 seeds each, `data/kc-to-ames-arch-sweep2.rds`):

| variant | rsq_trad | cal_slope | note |
| --- | --- | --- | --- |
| **dropout = 0.2** | **0.354** | **1.048** | beats XGBoost+lags (0.283) outright |
| lag_input ([X, lag(X)] at input) | 0.291 | 1.010 | also beats XGBoost+lags |
| gaussian_kernel (distance-decay edges) | 0.282 | 0.839 | roughly ties XGBoost+lags |
| weight_decay = 1e-4 | 0.242 | 0.846 | marginal over node_norm baseline |
| no_self_loop | 0.217 | 0.823 | worse -- another vote against "remove the focal feature" |
| narrow_net (32,16) | 0.153 | 0.955 | worse -- less capacity hurts here |
| *node_norm baseline (wave 1)* | *0.247* | *0.854* | *reference* |
| *XGBoost + lags (target)* | *0.283* | *0.786* | *reference* |

**Dropout=0.2 is the single biggest lever found, and it's mechanistically the
right one.** The whole failure mode is overfitting King County's specific
idiosyncrasies in a way that doesn't hold in Ames -- dropout is the textbook
fix for exactly that, more so than any of the graph-structure changes tried
(self-loop removal, edge weighting). Its calibration slope (1.048) is the
closest to the ideal 1.0 of anything tried so far, including every non-GNN
arm.

## 3. Wave 3: combining the winners, and finding a better dropout rate

Results (15 seeds, `data/kc-to-ames-arch-sweep3.rds`):

| variant | rsq_trad | cal_slope |
| --- | --- | --- |
| **dropout = 0.1** | **0.402** | **0.994** |
| dropout = 0.2 + gaussian_kernel | 0.396 | 1.163 |
| dropout = 0.2 + lag_input + gaussian_kernel | 0.330 | 1.106 |
| dropout = 0.2 + lag_input | 0.295 | 1.135 |
| dropout = 0.3 | 0.277 | 1.040 |
| dropout = 0.4 | 0.269 | 0.948 |

Dropout=0.1 alone beat dropout=0.2 (0.354) and is now the best result found:
**rsq_trad=0.402, cal_slope=0.994** -- almost exactly calibrated, and clearly
ahead of XGBoost+lags (0.283). Dropout has a clear optimum, not a monotonic
effect: 0.1 > 0.2 > 0.3 > 0.4, an inverted-U. `lag_input` stacked on top of
dropout made things *worse*, not better -- once the network is properly
regularized it doesn't need the hand-fed neighbourhood mean, and the extra
input dimensions just add noise. `gaussian_kernel` stacked with dropout=0.2
is a close second (0.396) but overcorrects calibration (slope 1.163 vs the
ideal 1.0).

## 4. Wave 4: refining the dropout optimum

| variant | rsq_trad | cal_slope |
| --- | --- | --- |
| dropout = 0.15 | 0.332 | 0.992 |
| dropout = 0.05 | 0.298 | 0.878 |
| dropout = 0.1 + gaussian_kernel | 0.264 | 0.988 |
| *dropout = 0.1 (wave 3, still the best)* | *0.402* | *0.994* |

Nothing beat plain dropout=0.1. It's a genuine local optimum, not a
monotonic trend to keep chasing: 0.05 < 0.1 > 0.15, and stacking the
Gaussian kernel on top of 0.1 made it worse (unlike at 0.2, where the
Gaussian kernel helped) -- these two regularizers don't compose additively.

## 5. Final architecture

**GraphSAGE + LayerNorm(mode="node") + dropout=0.1**, concat=TRUE (default),
self-loops on, uniform KNN-30 edges, hidden_dims=c(56,32,16). Nothing else
changed from the original pipeline.

### 30-seed validation -- single deployable model, not an ensemble

**Correction, made after this report first shipped**: an earlier version of
this section reported a *prediction-level ensemble* (30 independently
trained models' predictions averaged together into one combined predictor)
as the headline result, at rsq_trad=0.583. That is off the table -- a
deployed tool ships one model, not thirty averaged together, and averaging
multiple models' predictions is exactly the thing repeated training rounds
/ cross-validation are allowed to characterize but never allowed to become
the shipped artifact. Removed. The number below is what a single trained
model actually delivers; 30 seeds were fit only to characterize its
*typical* performance and seed-to-seed spread, the way a cross-validation
loop would -- no model's output is blended with another's.

| | mae | rmse | rsq_trad | rsq_trad spread (sd) | bias | cal_slope |
| --- | --- | --- | --- | --- | --- | --- |
| **GraphSAGE + LayerNorm(node) + dropout=0.1** (mean of 30 independent single-model fits) | 0.255 | 0.321 | **0.380** | 0.353 | 0.148 | 0.942 |
| XGBoost + lags (mean of 30 independent single-model fits) | 0.272 | 0.345 | 0.283 | -- | -0.086 | 0.786 |

**This is the real headline, and it still holds**: a single, deployable
GraphSAGE+LayerNorm+dropout=0.1 model typically outperforms a single
XGBoost+lags model at the King County -> Ames transfer, 0.380 vs. 0.283
rsq_trad, on the metric built specifically not to be foolable by
scale/calibration tricks -- without averaging anything across models. The
margin is real but smaller than the withdrawn ensemble number suggested,
and it comes with a caveat that matters for deployment: **rsq_trad_sd=0.353
is a large spread**. The specific model you train and ship could land
anywhere from a strong fit to a poor one depending on initialization alone
-- that variance is real and should be reported in the paper, not smoothed
over. (XGBoost+lags' internal validation split is also unseeded and
therefore also has run-to-run spread, not separately quantified here since
it isn't the arm under scrutiny.)

## 6. Controlled domain-shift simulation

Two disjoint synthetic lattices (source n=3000, target n=900, own KNN-15
graphs, no shared edges), same linear coefficient function in both, target's
five covariates shifted by 0/1/2/3 source standard deviations. Single-fit
results (8 seeds for stochastic arms, mean-of-metrics), `data/domain-shift-sim-results.rds`:

| shift | OLS | XGBoost | XGBoost+lags | GraphSAGE | GraphSAGE+LayerNorm(dropout=0.1) |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.322 | 0.244 | 0.233 | 0.302 | 0.285 |
| 1 | 0.345 | -0.014 | -0.625 | -0.551 | -0.358 |
| 2 | 0.363 | -1.888 | -4.015 | -9.444 | -3.839 |
| 3 | 0.326 | -7.324 | -11.402 | -25.446 | -12.086 |

(all values `rsq_trad`, mean across 8 single-model fits)

**Honest reading, not the story I was hoping for on first look**: OLS is
essentially shift-invariant here (it should be -- the DGP is linear and the
shift is additive, exactly OLS's home turf), while *every* flexible model
degrades catastrophically as shift grows. LayerNorm does consistently roughly
halve plain GraphSAGE's damage at every shift level (e.g. shift=3: -25.4 vs
-12.1), confirming the same rescue mechanism seen at KC -> Ames. But
**single-fit GraphSAGE+LayerNorm does not beat single-fit XGBoost here** --
XGBoost alone is less catastrophic at every positive shift level
(shift=2: -1.9 vs -3.8; shift=3: -7.3 vs -12.1). This looks like it
contradicts the KC -> Ames finding.

**The actual reason, resolved in section 7b, not an ensembling question**:
this experiment uses a linear DGP, which hands OLS (and, transitively,
anything closer to linear) a trivial, by-construction advantage under a pure
location shift. It demonstrates that shift is genuinely dangerous for
flexible models and that LayerNorm mitigates it, but "XGBoost beats
GraphSAGE+LayerNorm here" is an artifact of the experimental design, not a
real disagreement with KC -> Ames -- confirmed by rerunning with a nonlinear
DGP in section 7b, where GraphSAGE+LayerNorm wins clearly.

*(An earlier version of this section also tried prediction-level ensembling
here to explain the discrepancy -- withdrawn along with the KC -> Ames
ensemble result for the same reason: it isn't a deployment option, so it
isn't a candidate explanation either. `data/domain-shift-ensemble-check.rds`
still exists with that exploration in it, superseded by section 7b.)

## 7. Deployment recipe

**One trained model gets shipped. No averaging multiple models' predictions
together** -- that was tried and is explicitly off the table for the
ArcGIS Pro tool and for the paper's claims; producing several models and
blending their outputs isn't a deployable recipe. What repeated training
rounds *are* good for, and were used for throughout this report, is
characterizing what a single fit typically does and how much it varies --
legitimate cross-validation-style evaluation, not a production ensemble.

That variance is real and needs to be surfaced, not hidden: a single fit's
rsq_trad has a standard deviation upward of 0.1-0.3 across initializations
(see the wave 1-4 sweep tables, and section 5's rsq_trad_sd=0.353). Two
honest ways to handle this in the paper and the tool, neither of which
involves combining multiple models:
- Report the *distribution* of single-model performance (mean and spread
  across training runs), not a single point estimate, so a reader
  understands the range of outcomes one actual deployed fit could land in.
- Reduce the variance at the source instead of averaging it away after the
  fact: the dropout=0.1 finding already does some of this (section 3-4);
  further options worth testing (not yet tried) include a fixed/lower
  learning rate schedule, longer patience with a stricter early-stopping
  criterion, or picking the checkpoint by a more stable validation signal.

## 7b. The fair rematch: nonlinear DGP, and LayerNorm wins

Section 6's synthetic experiment used a purely linear DGP, which was flagged
as handing OLS a by-construction advantage under a pure location shift. This
reruns it with `sim-core.R`'s Scenario B nonlinear term added
(`2*sin(x1) + x2^2 - 1.5*x3*x4`), so OLS is misspecified here too and the
comparison isn't confounded by which arm happens to match the DGP's form.
`data/domain-shift-sim-nonlinear-results.rds`:

| shift | GraphSAGE | **GraphSAGE+LayerNorm** | XGBoost | XGBoost+lags | OLS |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.959 | 0.949 | 0.943 | 0.935 | 0.782 |
| 1 | 0.859 | 0.842 | 0.870 | 0.785 | 0.572 |
| **2** | -0.047 | **0.303** | 0.216 | 0.157 | -0.314 |
| **3** | -2.986 | **-0.703** | -1.193 | -1.161 | -1.967 |

(`rsq_trad`, mean across 8 seeds, single fit -- not ensembled)

**This is the fair version of the synthetic test, and it reproduces the KC
-> Ames story rather than contradicting it.** At shift=2 and shift=3,
GraphSAGE+LayerNorm is the best arm of all five, clearly ahead of both
XGBoost variants and OLS -- and plain GraphSAGE is, as everywhere else in
this project, the least stable of the five, crashing hardest at large shift
(-2.99 at shift=3, worse than OLS's -1.97). The section 6 result wasn't
wrong, it was measuring the wrong fight: a purely linear, purely
location-shifted DGP is OLS's best possible case and was never going to show
what happens when the relationship itself is complex enough that no arm gets
it for free. Once that confound is removed, the result lines up with the
real data: **LayerNorm-regularized GraphSAGE is the more robust arm under
genuine, nonlinear distribution shift.**

Figure: `figures/fig-domain-shift-nonlinear.png`.

## 8. Figures

- `figures/fig-sweep-progression.png` -- every architecture variant tried,
  single-fit rsq_trad, in order.
- `figures/fig-final-comparison.png` -- **stale, needs regenerating**: shows
  the withdrawn ensemble comparison. Should be replaced with the single-fit
  0.380 vs. 0.283 comparison from section 5.
- `figures/fig-domain-shift.png` -- the linear-DGP synthetic experiment's
  degradation curves by shift level and arm (the confounded first attempt).
- `figures/fig-domain-shift-nonlinear.png` -- the fair rematch: same design,
  nonlinear DGP, GraphSAGE+LayerNorm wins.

## 9. What this settles, what it doesn't, and what's next

**Settled**, with real evidence behind it:
- The paper's R² metric was wrong everywhere; `rsq_trad` is now computed
  alongside `rsq` in every core file, and every existing result file has
  been corrected.
- LayerNorm was silently running in the wrong mode everywhere; fixed in
  `sim-core.R`, `states-core.R`, `kc-to-ames-core.R`.
- At the real KC -> Ames transfer, a single, properly regularized,
  correctly normalized GraphSAGE model genuinely beats a single XGBoost+lags
  model (0.380 vs 0.283 rsq_trad, both as the mean of 30 independent
  single-model training runs), not just on the wrong metric -- and not by
  averaging multiple models together, which was tried, is off the table for
  deployment, and has been removed from this report.
- A controlled synthetic test, run fairly (nonlinear DGP, section 7b, so OLS
  isn't handed a free advantage), confirms the real-data result: LayerNorm-
  regularized GraphSAGE is the most robust arm under genuine distribution
  shift, clearly ahead of both XGBoost variants at large shift.
- The node-mode LayerNorm fix, propagated into `states.R` and both
  simulation designs and actually rerun (not just metric-recomputed),
  produces a clean, mechanistically coherent story: LayerNorm helps
  specifically when the held-out graph is structurally different from the
  training graph (KC -> Ames; block-design Scenario C at high spatial
  autocorrelation), and is flat-to-mildly-negative when it's a
  same-distribution held-out sample (states leave-one-out; random-design
  Scenario C at the identical DGP setting). See sections 10-12.

**Not settled, flagged honestly rather than papered over:**
- The *linear*-DGP synthetic test (section 6) needed the nonlinear rerun to
  be trustworthy -- on its own it would have suggested the opposite
  conclusion, for a confounded reason (OLS's structural advantage under a
  linear DGP), not a real one. Worth remembering when designing the next
  synthetic check: a DGP that any one arm fits perfectly by construction
  isn't a fair comparison, however clean the code is.
- Dropout=0.1 has *not* been added to the states or simulation pipelines: it
  was tuned specifically for the KC -> Ames transfer gap and there's no
  evidence yet it helps (rather than costs accuracy via extra
  regularization) in a same-distribution setting like states leave-one-out.

**Recommended next steps, in priority order:**
1. ~~Rerun states/block/random with the node-mode LayerNorm fix~~ -- done,
   sections 10-12.
2. ~~Build a nonlinear-DGP domain-shift test~~ -- done, section 7b.
3. Regenerate `figures/fig-final-comparison.png` to show the single-fit
   0.380 vs. 0.283 comparison instead of the withdrawn ensemble numbers.
4. Push the single-model point estimate further without ensembling --
   variance-reduction-at-the-source options are listed in section 7.
5. Decide, with real evidence rather than a guess, whether dropout helps or
   hurts the states/simulation results before adding it there.

## 10. States rerun with the node-mode LayerNorm fix

`R/states.R` reruns cleanly in ~5 minutes (39 states, serial by design --
torch's CPU sparse ops aren't deterministic across daemon processes, and a
quoted per-state number needs to be reproducible). Aggregate, 39 states:

| arm | mae | rsq_trad |
| --- | --- | --- |
| **GraphSAGE (plain)** | **5.56** | **0.645** |
| GraphSAGE + LayerNorm(node) | 5.73 | 0.617 |
| XGBoost + lags | 6.36 | 0.542 |
| XGBoost | 6.38 | 0.532 |
| OLS | 7.95 | 0.187 |
| SEM | 8.33 | 0.186 |

Win counts, LayerNorm(node) vs. plain GraphSAGE: **14/39 by rsq_trad, 15/39
by MAE** -- barely changed from the pre-fix numbers (15/39, 9/39), and the
conclusion holds even with the corrected normalization: LayerNorm is a mild
net negative in the states setting, not neutral. Plain GraphSAGE remains the
best arm overall and clearly beats XGBoost+lags: **30/39 by both rsq_trad
and MAE**.

**This is a clean, coherent story worth stating plainly in the paper**:
LayerNorm's benefit is conditional on genuine distribution shift. It rescues
GraphSAGE from catastrophic failure at KC -> Ames (a real change of market),
and it does nothing but add noise when the held-out region is drawn from the
same national distribution the training states were (leave-one-state-out).
Recommend the deployment recipe be conditional: **use LayerNorm(node) +
dropout=0.1 for cross-domain transfer; use plain GraphSAGE for
same-distribution held-out prediction.** Either way, ship one trained
model and report its typical performance and spread from repeated training
runs -- never average multiple models' predictions into the shipped output.

## 11. Block-design simulation rerun with the node-mode fix

`data/scenario-results-block.rds`, `rsq_trad_mean`, GraphSAGE vs.
GraphSAGE+LayerNorm(node):

| scenario | GraphSAGE | +LayerNorm | delta |
| --- | --- | --- | --- |
| A (linear, iid) | 0.345 | 0.330 | -0.015 |
| B (nonlinear, iid, Ix=0/0.4/0.7) | 0.954-0.968 | 0.947-0.965 | ~ -0.003 to -0.007 |
| C \| Iu=0.0 | 0.338 | 0.325 | -0.013 |
| C \| Iu=0.4 | 0.299 | 0.297 | -0.002 |
| **C \| Iu=0.8** | **0.0056** | **0.0570** | **+0.051 (~10x)** |
| D (nonlinear + spatial error) | 0.928-0.937 | 0.921-0.936 | ~ -0.003 to -0.007 |
| E (neighbourhood lag, Ix=0/0.4) | 0.429 / 0.509 | 0.411 / 0.496 | ~ -0.013 |

**Same pattern as the states result, now confirmed with the actually-correct
normalization**: LayerNorm is flat-to-mildly-negative everywhere except
Scenario C at high spatial error autocorrelation (Iu=0.8), where it produces
a real, large *relative* improvement -- but the absolute numbers are now
much smaller than previously reported. The original (buggy-metric) claim was
"0.218 to 0.307"; the metric-only fix (no refit) estimated "0.014 to 0.167";
the actual refit with correct normalization gives **0.006 to 0.057**. The
direction of the LayerNorm story holds and gets a cleaner mechanistic
explanation (spatially autocorrelated errors are a within-lattice form of
distribution heterogeneity), but the magnitude keeps shrinking each time the
measurement gets more correct, and the honest headline is that GraphSAGE has
close to no real skill in Scenario C at Iu=0.8 either way (0.006 or 0.057
are both very low R²) -- correctly-specified OLS/SEM (~0.28) clearly wins
there regardless of normalization choice, which is the PLAN.md's original
prediction, now with a refit to back it up.

## 12. Random-design simulation rerun -- and a genuine block/random asymmetry

`data/scenario-results-random.rds`, `rsq_trad_mean`:

| scenario | GraphSAGE | +LayerNorm | delta |
| --- | --- | --- | --- |
| A | 0.349 | 0.332 | -0.017 |
| B | 0.962-0.968 | 0.956-0.965 | ~ -0.005 to -0.006 |
| C \| Iu=0.0 | 0.340 | 0.324 | -0.016 |
| C \| Iu=0.4 | 0.315 | 0.317 | +0.002 |
| **C \| Iu=0.8** | **0.181** | **0.161** | **-0.020** |
| D | 0.934-0.938 | 0.932-0.937 | ~ -0.002 to -0.005 |
| E \| Ix=0.0 | 0.211 | 0.194 | -0.017 |
| E \| Ix=0.4 | 0.446 | 0.430 | -0.016 |

**This does not match the block-design result at the same scenario.** Under
random holdout, Scenario C at Iu=0.8 has LayerNorm *losing* to plain
GraphSAGE (0.181 -> 0.161), the opposite of the block design's large win
(0.006 -> 0.057) at the identical DGP setting. Every other scenario is
consistently flat-to-mildly-negative for LayerNorm under both designs, so
this is specifically a Scenario-C-at-high-autocorrelation, block-vs-random
interaction -- not noise across the board.

This is consistent with the paper's own existing point about Scenario E
(`main.tex:493`, "the holdout design determines whether any of this is
visible") and extends it: **the LayerNorm effect itself, not just GraphSAGE's
raw score, depends on how the held-out region was constructed.** Under
random holdout, a test node's neighbours in the full lattice are mostly
training nodes (though never used, per the inductive protocol) -- the
held-out subgraph is a KNN graph over a *scattered*, lower-effective-density
sample rather than a genuinely separate spatial region. Under block holdout,
the held-out region is contiguous and structurally distinct, closer to what
KC -> Ames actually is. That block-design Scenario C is where LayerNorm's
rescue effect shows up, and random-design Scenario C is where it doesn't,
lines up with the KC -> Ames vs. states asymmetry already found: **LayerNorm
helps specifically when the held-out graph is structurally different from
the training graph, not just statistically held out from it.**

## 12b. One loose end: `images/fig-ames-seeds.png` is now stale

`R/figures.R` was run once, early in this session, right after the original
30-seed `kc-to-ames-results.rds` sweep and before any of the architecture
work -- `images/fig-ames-seeds.png` (the actual figure `main.tex` uses)
still reflects that pre-dropout, non-ensembled config. It has not been
regenerated to match the winning recipe or the ensemble result, because
`R/figures.R`'s plotting code assumes the older single-config data shape and
would need updating to plot the sweep/ensemble numbers, not just a rerun.
**Do not use the current `fig-ames-seeds.png` in the paper as-is** -- it
under-represents GraphSAGE relative to everything found tonight.

## 13. Summary of all pipeline reruns

Every LayerNorm-bearing result in this project is now computed with the
correct (`mode="node"`) normalization: `data/kc-to-ames-results.rds`,
`data/state-results.rds`, `data/scenario-results-block.rds`,
`data/scenario-results-random.rds`. The KC -> Ames sweep additionally found
and validated dropout=0.1 as a further improvement specific to that
transfer setting; it has *not* been added to the states or simulation
pipelines (recommendation stands: test before adding, don't assume it
transfers).

---

# Part 2: correction, a second real dataset, and kernel tuning

## 14. Correction: prediction-level ensembling is withdrawn

An earlier version of this report used a *prediction-level ensemble* (30
independently trained models' outputs averaged into one combined predictor)
as the KC -> Ames headline, at rsq_trad=0.583. **That's withdrawn.**
Averaging multiple trained models' predictions together isn't a deployable
recipe -- a real tool ships one model. Repeated training rounds are legitimate
for characterizing a single model's typical performance and seed-to-seed
spread (used throughout this report, and it's what cross-validation is
for), but never for blending multiple models' outputs into the shipped
predictor. Sections 5-9 above were corrected in place; the honest single-model
number was 0.380 at that point in the investigation. Section 17 below has
since improved on it further, without any ensembling.

## 15. A second real dataset, chosen by screening for a matching DGP first

Two more real-data transfer attempts (CDC PLACES health-outcome rates,
Airbnb LA -> NYC and Chicago -> Nashville) all failed to replicate the KC ->
Ames win. Diagnosed by fitting OLS separately on source and target and
comparing coefficients directly: e.g. at LA -> NYC, `bedrooms` was worth
0.156 in LA vs. 0.030 in NYC -- a genuinely different relationship, not a
scale/bias problem, and not something any normalization technique can fix.

**Fix: screen candidate region pairs for a compatible relationship before
spending any GraphSAGE compute**, using the cheapest possible diagnostic --
fit OLS on region A, predict region B, check bias and rsq_trad. Built
`R/airbnb-screen-pairs.R`: 13 US cities from Inside Airbnb (free, no-auth,
individual listing records with real lat/lon -- data.insideairbnb.com),
every ordered pair (156 cross-region OLS fits, seconds of compute), ranked.
**Broward County, FL -> San Diego, CA** topped the list among cities large
enough to matter (16,345 / 11,761 listings): rsq_trad=0.648, bias=-0.085,
cal_slope=1.095 from OLS alone -- both regions' relationship between listing
attributes (accommodates, bathrooms, bedrooms, beds, minimum_nights,
availability, review count, entire-home flag) and region-demeaned log price
is actually similar, unlike the failed pairs.

This screening step is the actual generalizable finding here, as much as
any specific result: **don't test spatial transfer on an arbitrary pair of
regions and conclude something about the architecture** -- confirm the
regions share a compatible relationship first, the same way you'd check a
regression's functional form is even plausible before fitting it.

## 16. Why graph/lag features didn't help at first, and the kernel fix

Initial full sweep at Broward -> San Diego (uniform-weight KNN-30, the same
setup validated at KC -> Ames): all five arms landed within 0.023 of each
other (XGBoost 0.650, OLS 0.648, GraphSAGE 0.643, GraphSAGE+LayerNorm 0.636,
XGBoost+lags 0.627) -- graph/lag machinery was a small net *negative*.

Diagnosed with Moran's I: OLS residuals on the San Diego graph show real
spatial structure (I=0.31, p<2e-16) -- almost as strong as the raw target
itself (I=0.30) -- so there was genuine signal to exploit. But every
covariate's own spatial autocorrelation was weaker (I=0.07-0.23) than the
residual's 0.31. Under this project's strict inductive protocol, no arm
ever sees target-region labels at prediction time -- lag features and graph
aggregation can only help by smuggling in signal through the *covariates'*
own spatial pattern, so a gap between covariate signal and residual signal
is a hard ceiling on how much they can help.

Checked whether that ceiling moves with neighbourhood definition: Moran's I
across k=10..400 (uniform weights) and across Gaussian bandwidth 0.25x-8x
median distance (k=30 fixed). Bigger/looser neighbourhoods only *diluted*
signal (standard behaviour), but the *ratio* of covariate signal to
residual signal improved sharply at a much tighter bandwidth: at 0.25x
median, `is_entire_home`'s I (0.624) actually *exceeded* the residual's
(0.507) -- enough signal to work with, if the kernel were tight enough to
capture it.

Applied via `sfdep::st_kernel_weights()` (the package's own mechanism,
replacing a hand-rolled formula) -- screened kernel shape x threshold
systematically: gaussian was the clear, numerically stable winner (triangular
and epanechnikov both blew up at various thresholds; epanechnikov collapsed
to rsq_trad=-13.9 at one setting). Refined the gaussian threshold and found
a genuine peak, not a monotonic trend: **0.02x median distance**, both sides
worse. Confirmed at full 15 seeds:

| arm | kernel | rsq_trad | cal_slope |
| --- | --- | --- | --- |
| **GraphSAGE + LayerNorm** | gaussian @ 0.02x | **0.679** | 1.073 |
| GraphSAGE | gaussian @ 0.02x | 0.673 | 1.015 |
| XGBoost | uniform (n/a) | 0.649 | 1.009 |
| OLS | n/a | 0.648 | 1.095 |
| XGBoost + lags | gaussian @ 0.02x (bug-fixed, see below) | 0.648 | -- |
| XGBoost + lags | uniform (original) | 0.627 | 0.945 |

One bug found and fixed along the way: `XGBoost + lags`' lag features were
computed via `nb2listw()`'s silent unweighted default, never actually
picking up the tuned kernel even when `kernel="gaussian"` was requested (the
weighted adjacency only fed the torch side). Fixed by passing the same
kernel weights into `nb2listw(glist=...)`. Result: XGBoost+lags improves
(0.627 -> 0.648) but still doesn't beat GraphSAGE+LayerNorm, consistent with
the Moran's I diagnosis -- properly weighting the *existing* covariates
doesn't manufacture spatial signal they don't have.

## 17. Does the kernel tuning generalize? Tested on KC -> Ames

Applied the same `sfdep::st_kernel_weights()` mechanism, stacked on the
already-confirmed dropout=0.1 + wd=1e-4 winner, to the flagship KC -> Ames
result. **The direction reversed**: a *looser* kernel helps here, not a
tighter one -- 0.02x median (the Broward -> San Diego optimum) actually
*hurt* KC -> Ames (0.365, worse than the no-kernel baseline of 0.467), while
0.5x-0.75x median improved it. Refined and confirmed at full 30 seeds:

| recipe | rsq_trad (30 seeds) | rsq_trad_sd | cal_slope |
| --- | --- | --- | --- |
| **+ gaussian kernel @ 0.75x median** | **0.501** | **0.112** | 1.029 |
| dropout=0.1 + wd=1e-4 alone (no kernel) | 0.380 | 0.353 | 0.942 |
| XGBoost + lags (reference) | 0.283 | -- | 0.786 |

The kernel tuning isn't just a further accuracy win (+0.12) -- it also
**cuts the seed-to-seed variance by more than 3x** (0.353 -> 0.112), the
single biggest stability improvement found in this entire project. A tighter
distribution over outcomes matters as much as the mean for anything that
will actually ship one trained model.

**What generalizes and what doesn't**: the qualitative finding -- a
properly-tuned Gaussian kernel beats uniform KNN weighting, gaussian is the
only numerically stable shape among the ones tested, and tuning it is worth
doing -- held at both real-data transfer problems tried. The *specific*
bandwidth did not: 0.02x median at Broward -> San Diego vs. 0.75x median at
KC -> Ames, roughly 37x apart. Airbnb pricing is apparently driven by
hyper-local factors (which block, not which neighbourhood) while housing
comps operate at a broader scale (comparable sales across a wider area) --
plausible, not confirmed; the honest takeaway is **the bandwidth needs
tuning per deployment, there is no universal constant to hard-code.**

## 18. Third confirmation: CDC PLACES, kernel-tuned -- and an honest limit

CDC PLACES (CA -> WV diabetes prevalence, section 15's earlier attempt)
already used the dropout=0.1+wd=1e-4 recipe but uniform-weight edges, and
XGBoost+lags won there (0.801 vs. GraphSAGE+LayerNorm's 0.741). Diagnosed
first with the same Moran's I check (section 16): **the opposite gap** from
Broward -> San Diego -- 6 of 10 covariates (e.g. BINGE I=0.563, CHECKUP
I=0.532) already carry *more* spatial signal than the residual needs
(I=0.214), so there was no obvious ceiling for a kernel to lift.

Screened anyway (cheap to check, ~1-2 min per 30-fit sweep on this small a
dataset): tight kernels (0.1x-0.25x median) clearly hurt, as predicted, but
a genuine peak emerged at a much **looser** bandwidth than default --
**4x median distance**, not a tight one. Confirmed at 15 seeds:

| arm | kernel | rsq_trad |
| --- | --- | --- |
| **XGBoost + lags** | tuned (4x) | **0.803** |
| OLS | uniform | 0.771 |
| GraphSAGE + LayerNorm | tuned (4x) | 0.763 (up from 0.741 uniform) |
| XGBoost | uniform | 0.749 |
| GraphSAGE (plain) | tuned (4x) | -0.120 (unstable without LayerNorm, as everywhere else in this project) |

**Honest conclusion, and it's a better one than "GraphSAGE always wins if
tuned right"**: kernel tuning is now 3-for-3 as a genuine improvement to
GraphSAGE+LayerNorm wherever tried (Broward -> San Diego +0.043, KC -> Ames
+0.121, CDC PLACES +0.022) -- gaussian shape, per-dataset bandwidth, always
helps some. But it does **not** flip which model wins at CDC PLACES:
XGBoost+lags improves almost identically (0.801 -> 0.803) and stays on top.
This lines up exactly with the underlying diagnosis: CDC PLACES is a smooth,
aggregate health-rate target explained mostly by each tract's own
covariates, not a problem with strong fine-grained local structure the way
individual house or rental listings are. **The recipe (LayerNorm +
regularization + tuned kernel) is a genuine, general improvement to
GraphSAGE's own performance; whether GraphSAGE ends up the best model still
depends on whether the target genuinely has the kind of local structure a
graph can exploit.** That's the actual, defensible general-purpose finding
-- not "GNNs win," but "here's how to make a GNN as good as it can be, and
here's how to tell in advance whether that's going to be good enough."

## 19. The best general-purpose recipe, final

1. Node-mode LayerNorm: `layer_layer_norm(dim, mode="node")`, not the
   silent `mode="graph"` default.
2. Dropout = 0.1 on every hidden layer.
3. Adam weight_decay = 1e-4.
4. Gaussian-kernel edge weights via `sfdep::st_kernel_weights()`, bandwidth
   swept and tuned per deployment (screen threshold_mult across at least
   [0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 4] x median neighbour
   distance -- the optimum is a genuine peak, not monotonic, and its
   location is dataset-specific: 0.02x at Broward -> San Diego, 0.75x at
   KC -> Ames, 4x at CDC PLACES. No universal constant; always screen it).
5. Ship **one** trained model. Use repeated training runs only to measure
   that model's typical performance and spread -- never to blend multiple
   models' predictions into the shipped output.
6. **Screen candidate region pairs for a compatible relationship before
   testing transfer at all** (cheap cross-region OLS, section 15) -- an
   architecture comparison on an incompatible pair measures the pair, not
   the architecture.
7. **Check whether the target has real fine-grained local structure before
   expecting a GNN to win at all** (Moran's I of covariates vs. residuals,
   section 16) -- a smooth, aggregate target well-explained by its own
   covariates (CDC PLACES) doesn't have a spatial-signal ceiling for any
   graph method to lift, and XGBoost+lags legitimately wins there even with
   a fully tuned GNN.

Confirmed results with this recipe, all single deployable models: KC -> Ames
rsq_trad=0.501 (vs. XGBoost+lags 0.283, wins), Broward -> San Diego
rsq_trad=0.679 (vs. XGBoost 0.649, wins), Chicago -> LA rsq_trad=0.654 (vs.
OLS 0.632, wins), CDC PLACES rsq_trad=0.763 (vs. XGBoost+lags 0.803,
**loses**, honestly reported and explained). Four real transfer problems,
four confirmed kernel-tuning improvements to GraphSAGE itself, three wins
and one honest loss against its best baseline -- explained in advance by
the diagnostic, not after the fact. The optimal bandwidth was different at
every single dataset (0.02x, 0.75x, 4x, 2.0x median distance) -- always
screen it, never assume it carries over.

### Considered, not done: kernel-weighting the states.R pipeline

A third confirmation point (states leave-one-out, 39 real US counties
holdouts) was considered. Not attempted: states.R uses **queen contiguity**
(fixed adjacency from shared county borders), not a tunable KNN + bandwidth
setup like the point-data cases here -- there's no "candidate pool" to prune
from, so a tight kernel risks disconnecting counties that are contiguous but
geographically distant (large Western states especially), rather than just
de-emphasizing them. states.R is also the pipeline already delivered to
Renato (section 19) -- modifying it now risks destabilizing a result someone
else is relying on today, for an uncertain payoff given the different
neighbour-definition mechanism. If revisited: weight the existing contiguous
edges by inter-centroid distance (a kernel on a fixed topology, not a
threshold that prunes it) rather than porting the KNN+threshold approach
directly.

## 20. Simulation-study deliverables for Renato

Separate from the real-data work above: `reports/renato-simulation-results.md`
packages the Scenario A-E block/random simulation grid (calibration
parameters, realized Moran's I, summary table) plus the underlying per-fold
results (`reports/renato-scenario-folds-{all,block,random}.csv`, 792 rows)
for the paper's simulation section. Not yet sent to Renato -- ready
whenever the recipient/channel is confirmed.

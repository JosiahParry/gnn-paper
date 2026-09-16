# The pre-test: deciding before you train

Two questions have to be answered before fitting anything, and both can be
answered in about two minutes of compute:

1. **Will *any* model transfer from region X to region Y?**
   If the relationship between the columns and the answer genuinely differs
   between the regions, nothing bridges it. A result here measures the pair
   of regions, not the method.

2. **Is a graph worth it at all?**
   If the columns already carry the spatial signal, neighbour information
   adds variance and nothing else.

They are **ordered**. Question 1 gates question 2. A pair that fails
question 1 can score perfectly on question 2 and still be untestable — two
of ours do exactly that.

---

> **STATUS: Screen 2 is WITHDRAWN.** Three successive formulations were
> tried and all three fail. The section below is kept because the reasoning
> is worth preserving, but **do not use it and do not publish it as a rule.**
> See "Screen 2 is withdrawn" at the end for the evidence.

## Screen 2 first, because it is the one that is finished

**The rule.** In the source region: fit OLS, compute Moran's I of the
residuals, and Moran's I of each covariate. Take the **fraction of
covariates whose own Moran's I exceeds the residual's**.

| fraction | meaning | call |
| --- | --- | --- |
| ≤ 0.25 | the leftover spatial structure is stronger than what the columns carry | graph helps |
| ≥ 0.33 | the columns already carry the spatial signal | graph does not help |

Evidence across the seven evaluations run so far:

| transfer | fraction | result |
| --- | --- | --- |
| King County → Ames (house price) | 0.00 | GNN wins +0.218 |
| Chicago → LA (rental price) | 0.00 | GNN wins +0.022 |
| Broward → San Diego (rental price) | 0.25 | GNN wins +0.030 |
| *no pair lands between 0.25 and 0.33* | | |
| Queens → Brooklyn (311 response time) | 0.33 | GNN loses −0.057 |
| Queens → Brooklyn (tree diameter) | 0.50 | GNN loses −0.144 |
| Broward → San Diego (rental reviews) | 0.50 | GNN loses −0.086 |
| CA → WV (diabetes rate) | 0.90 | GNN loses −0.040 |

**Why the *fraction* and not the raw amount of leftover structure.** An
earlier version of this rule used the absolute residual Moran's I and looked
perfect on four pairs. It collapsed when two more were added: CA → WV has
strongly structured residuals (0.446) and still loses, because its
covariates are more structured still (0.590). Only the comparison separates
the cases. Wins span 0.221–0.615 on the raw number and losses span
0.116–0.446 — completely overlapping.

**This screen needs no answers from the target region.** It runs entirely on
the source. That is what makes it deployable.

**Standing of the rule.** The threshold is fitted to seven points and the
"gap" is the space between two adjacent observations. Prospectively it is
**1 for 3**: two earlier formulations each made a prediction and both were
wrong; the current one has made one correct call, and that call was a
predicted *loss*, which is the easy direction when most cases lose. It has
never correctly predicted a **win** on a pair it was not fitted to. Treat it
as a strong hint, not a decision procedure.

---

## Screen 1 is not finished, and the gap matters

**What we currently do:** fit OLS on X, predict Y, score against Y's true
values. Require both directions to be positive.

**Why that is not deployable:** it uses Y's answers. In real use, Y's
answers are the thing you are trying to produce. As written this is a
research tool for assembling a benchmark, not something a practitioner can
run.

It has been useful as a research tool — it rejected three candidate datasets
before any compute was spent:

| candidate | outcome |
| --- | --- |
| LA → NYC, Chicago → Nashville (rentals) | negative for every model including OLS |
| NYC building age | all 20 borough pairs fail; best −0.006 |
| Rental availability | essentially unpredictable anywhere (in-sample 2.6%) |

But every one of those verdicts used the target's answers.

---

## The label-free version of Screen 1

**The proposal:** train on X, know the properties of X, compare against the
properties of Y excluding the response — and decide from that alone.

This is right in outline, and it splits into two problems with very
different prospects.

### Problem A — different *inputs*. Solvable without labels.

If Y's rows sit outside the range of X's rows, the model is extrapolating
and will fail. This is entirely visible from the columns:

| check | what it measures |
| --- | --- |
| Domain-classifier AUC | train a classifier to tell X rows from Y rows using covariates only. ~0.5 means the two regions look alike; near 1.0 means they are disjoint and you are extrapolating |
| Coverage | share of Y rows falling inside the range / convex region of X |
| Standardised mean difference | per-column shift, in standard deviations |

### Problem B — same inputs, different *meaning*. **Not** solvable without labels.

This is the harder half and it needs to be said plainly: **you cannot detect
a change in the input→answer relationship by looking only at inputs.** If a
bedroom is worth 0.156 in LA and 0.030 in NYC, that difference lives
entirely in the relationship. The bedroom counts themselves can be
identically distributed. No statistic computed on Y's columns can see it,
because the quantity that changed is not a property of those columns.

This is not a gap in our method; it is a limit on what the information
allows. And it is the reason our failures failed:

| failure | cause | would a label-free input check have caught it? |
| --- | --- | --- |
| LA → NYC rentals | bedroom coefficient 0.156 vs 0.030 | **No** — relationship shift |
| NYC building age | same building form, different era per borough | **No** — relationship shift |
| Chicago → Nashville rentals | market relationship differs | **No** — relationship shift |

So a purely input-based screen would have passed all three.

### The partial answer: test the relationship's stability *inside* X

There is one thing we can do, and it uses only data we have — X's columns
**and** X's answers, never Y's:

> Split the source region into spatial blocks. Fit the model separately in
> each. Measure how much the coefficients move between blocks.

The logic: if the relationship is already unstable across short distances
*within* X, there is little reason to expect it to hold 500 km away in Y. If
it is rock-steady throughout X, it is a better bet that it travels.

This does not measure the X→Y shift directly — nothing label-free can. It
measures a **propensity** to shift, which is the most the information
permits. A stable source region is necessary, not sufficient.

### Proposed combined screen

| signal | needs Y's answers? | catches |
| --- | --- | --- |
| Domain-classifier AUC / coverage | no | extrapolation (different inputs) |
| Within-source coefficient instability | no | propensity to relationship shift |
| Current OLS cross-prediction | **yes** | actual relationship shift — the ground truth we validate against |

**Decision shape:** proceed if inputs overlap *and* the source relationship
is spatially stable. Refuse if either fails. Accept that some
relationship-shift failures will slip through — and that the honest
deliverable is a *risk flag*, not a guarantee.

---

## Tested: the label-free screen does not work as a green light

All three signals were computed for the ten pairs where the answer is
already known (`R/transfer-screen-labelfree.R`). Ground truth is whether OLS
fitted on X reaches a usable R² on Y — the thing we are trying to predict
without looking at it.

| pair | actual | classifier AUC | coverage | **source instability** |
| --- | --- | --- | --- | --- |
| Broward → San Diego (price) | **0.648** | 0.734 | 0.912 | 0.497 |
| Chicago → LA (price) | **0.632** | 0.641 | 0.941 | 0.663 |
| Queens → Brooklyn (311) | **0.301** | 0.654 | 0.965 | 0.747 |
| Queens → Brooklyn (trees) | **0.276** | 0.664 | 1.000 | 0.635 |
| Broward → San Diego (reviews) | **0.210** | 0.765 | 0.943 | 0.646 |
| LA → NYC (price) | 0.062 | 0.749 | 0.958 | **0.459** |
| Bronx → Brooklyn (building age) | 0.048 | 0.689 | 0.935 | 1.333 |
| Chicago → Nashville (price) | −0.035 | 0.776 | 0.873 | 0.663 |
| Broward → San Diego (availability) | −0.041 | 0.792 | 0.910 | 1.465 |
| Queens → Brooklyn (building age) | −0.140 | 0.703 | 0.915 | 1.480 |

**Every signal overlaps.** No threshold on any of them separates the
transfers from the failures.

### The counterexample, and why it was predictable

**LA → NYC has the most stable source relationship of all ten pairs
(0.459) and does not transfer.** The relationship is rock-steady throughout
Los Angeles — and New York simply disagrees with it. Internal consistency
within X carries no information about whether Y agrees, which is the
theoretical limit from Problem B arriving exactly where the argument said it
would.

The input-based signals fail the same way: LA and NYC listings *look* alike
(AUC 0.749, coverage 0.958). What differs is what a bedroom is worth, and
that is not a property of the bedroom counts.

### What does survive: a one-sided red flag

Source instability is useless as a green light but works as an alarm.

| instability | pairs | failures | false alarms |
| --- | --- | --- | --- |
| **> 1.0** | 3 | **3** | **0** |
| < 1.0 | 7 | 2 | — |

Above 1.0 it caught three of the five failures with no false alarms. Below
1.0 it means nothing — five transfer, two do not.

> **Rule: if the relationship is already unstable inside your source region,
> do not expect it to travel. If it is stable, you have learned nothing.**

This is worth having — it is free, and it caught both building-age pairs and
the availability target before any training — but it is a filter, not a
decision procedure.

### Conclusion

**Transferability cannot be screened without some labels from the target
region.** Whether a relationship holds in Y is a fact about Y's
relationship, and no amount of looking at Y's inputs recovers it.

The practical consequence is concrete and worth stating in the paper: before
trusting a transferred model, go and collect a **small pilot sample of
ground truth in the target region** — on our pairs, a few dozen points is
enough to estimate out-of-region R² and separate the usable transfers from
the failures decisively. That is a cheap, actionable requirement, and it is
more honest than a screen that would have waved LA → NYC through.

So the final shape of the pre-test is:

| step | needs target labels? | verdict |
| --- | --- | --- |
| 1. Source instability red flag | no | **reject** if unstable; otherwise inconclusive |
| 2. Pilot sample → out-of-region R² | **yes, a few dozen** | the only reliable transferability test |
| 3. Covariate-vs-residual Moran fraction | no | is a graph worth it |

---

## How often does transfer actually work? Rarely.

The screen was applied to every candidate pair considered for this project.
The hit rate is the headline number and deserves to be stated first, because
it reframes what the hard problem is:

| passed the transfer screen | rejected |
| --- | --- |
| King County → Ames (house price) | LA → NYC (rental price) |
| Broward → San Diego (rental price) | Chicago → Nashville (rental price) |
| Chicago → LA (rental price) | NYC building age — **all 20** borough pairs |
| CA → WV (diabetes rate) | Airbnb availability |
| Queens → Brooklyn (tree diameter) | Soil carbon: Corn Belt ↔ Ohio Valley |
| Queens → Brooklyn (311 response) | Soil carbon: Driftless ↔ Ridges |
| Broward → San Diego (reviews, one direction) | Soil carbon: two boxes *inside* the Driftless |

**Seven usable pairs against more than twenty-five rejected.** The failures
are not exotic; they are the pairs any practitioner would reasonably try.
Two boxes 110 km apart in the same landform, on the same target, still
produced a reverse-direction R² of −8.4.

The implication for the paper: the hard part of inductive spatial modelling
is not the architecture, it is establishing that two regions agree at all.
Combined with the result above — that agreement cannot be verified without
target labels — the single most useful thing this work can tell a
practitioner is: **collect a small pilot sample in the new region before
trusting any transferred model, whatever its architecture.**

## A trap worth naming: do not benchmark transfer on a modelled target

All three soil attempts used SoilGrids. SoilGrids is **not measurements** —
it is the output of a global machine-learning model fitted to covariates
that include terrain and climate, the same families we used as inputs.

That makes it unsuitable here, for three compounding reasons:

1. The target is a model prediction, so it carries that model's smoothness
   and its regional quirks rather than the field's.
2. Our covariates overlap with theirs, so the exercise partly reduces to
   reverse-engineering their model.
3. Where their model behaves differently in two regions — different training
   density, different covariate importance — that shows up as a transfer
   failure that says nothing about spatial transfer in general.

This is a generalisable caution. **A transfer benchmark needs a measured
target.** For soil that means point observations (ISRIC WoSIS, USDA NCSS),
which are sparse and irregular — which in turn means the raster framing
would have to be abandoned for that particular target.

---

## Screen 2 is withdrawn: we cannot predict in advance whether a GNN will help

Three formulations were tried. All three fail on the full set of pairs.

| version | statistic | outcome |
| --- | --- | --- |
| 1 | target kind (individual record vs aggregate rate) | **Wrong.** Two individual-record targets (trees, 311) lost to plain OLS |
| 2 | absolute Moran's I of the OLS residual | **Overlaps.** Wins 0.221–0.615, losses 0.116–0.446 |
| 3 | share of residual explained by neighbours' covariates | **Overlaps.** Wins 0.051–0.282, losses 0.005–0.075 |

Version 3 had the best reasoning behind it and is worth recording properly,
because the argument is correct even though the statistic is not.

**The argument.** GraphSAGE averages neighbours' *covariates*; it never sees
their labels, which is exactly what lets it work in a region with no labels.
So spatial structure splits in two:

- structure that neighbours' **inputs** reveal — a graph can use it;
- structure that lives only in the **answer** — a graph cannot.

Residual Moran's I cannot tell those apart, which is why version 2 failed.
Version 3 measured the first one directly. The Iowa farm data shows the
distinction is real: its residual structure is high (0.53) while neighbours'
terrain explains almost none of it (0.009), because the structure is field
boundaries — which live in the answer. That is the same shape as simulation
Scenario C, where GraphSAGE scored 0.006 against OLS's 0.280.

**Why it still failed.** CA → WV is a loss with a *higher* score (0.075)
than Broward → San Diego, a win (0.051). Its OLS already explains 96% of
the variance in-sample, so there is almost no headroom for any method to
win, regardless of how much of the remainder is spatially recoverable. The
statistic measures what is recoverable, not whether recovering it is worth
anything.

**Conclusion.** After three attempts on eight pairs, **no pre-fit statistic
we tried predicts whether a GNN will beat a tabular model.** This is the
honest result and it should be reported as such. It does not weaken the
project's other findings; it removes a claim we could not support.

### What still works, and should carry the paper instead

| finding | status |
| --- | --- |
| Similar regions transfer; different regions do not | Strong — Iowa farms pass, four "different region" pairs fail catastrophically |
| Transferability cannot be screened without target labels | Strong — three label-free signals tested, all overlap |
| Source instability > 1.0 is a reliable red flag | Modest — caught 3 of 5 failures, no false alarms |
| The simulation grid identifies the mechanism cleanly | Strong — GNNs win on nonlinear signal, lose on pure spatial error |

The simulation grid is the one place we *can* say when a GNN helps, because
there the data-generating process is known by construction. The gap is that
we have not found a way to measure the relevant property of a real dataset.
That is a well-posed open problem and a reasonable thing for a paper to
state rather than paper over.

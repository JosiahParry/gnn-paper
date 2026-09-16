# When does a graph neural network help with spatial prediction?

We trained GraphSAGE on one region and applied it unchanged to a second
region it had never seen, across eight such transfers plus one
classification. What decides the outcome is not how spatially clustered the
data looks, but whether the quantity being predicted spreads from place to
place.

## Results

Each row is one trained model applied to a disjoint region. R² is 1 − SSE/SST.
Where a transfer was run in both directions, both are shown; a result that
only holds one way is marked.

| Transfer | Target | Forward | Reverse |
| --- | --- | --- | --- |
| Des Moines ↔ Cedar Rapids | Surface temperature | **+0.044** | **+0.100** |
| Chicago ↔ Los Angeles | Rental price | **+0.022** | **+0.040** |
| King County ↔ Ames | House price | **+0.218** | +0.001 † |
| Broward ↔ San Diego | Rental price | **+0.030** | **−0.139** |
| California → West Virginia | Diabetes rate | −0.040 | — |
| Queens ↔ Brooklyn | 311 response time | −0.057 | −0.067 |
| Broward → San Diego | Rental reviews | −0.086 | — |
| Queens ↔ Brooklyn | Tree trunk diameter | −0.144 | −0.110 |

† Ames holds 2,930 rows against King County's 21,613, so the reverse trains
on a seventh of the data. It is not a fair mirror and should not be read as
one.

Margins are against the best non-graph model. Absolute scores for the two
strongest: surface temperature 0.822 against 0.778, house price 0.501
against 0.283.

A ninth test kept the same two cities and predictors as the temperature
transfer and changed only the target, to land cover class. GraphSAGE matched
logistic regression exactly (AUC 0.9271 against 0.9267, p = 0.95) and beat
gradient-boosted trees by 0.040 (p = 0.003).

## What separates the wins from the losses

Heat spreads. A park cools the blocks around it and a parking lot warms them,
so a location's temperature depends on its neighbours' surface as well as its
own. We measured this directly: giving gradient-boosted trees access to
neighbouring values raised their score by 0.077 (p < 0.00001), with no neural
network involved. House prices behave the same way — a desirable street lifts
every house on it.

Tree diameter does not spread; a large oak does not enlarge the next tree.
Nor does response time, nor land cover — buildings cluster, but one building
does not cause the next plot to be built. The graph lost or tied on all three.

The land cover test is the clearest, because everything but the target is held
fixed. Land cover is *more* spatially clustered than temperature, so any rule
based on clustering predicts the reverse of what happened. Asking whether the
process spreads gets both right.

## Direction matters, and we cannot yet say why

Two wins replicate when the regions are swapped and get larger. One fails
outright: Broward → San Diego gains 0.030, San Diego → Broward loses 0.139.
Both losses replicate. We have no account of what makes a pair asymmetric,
and the effect is large enough that single-direction results should not be
trusted on their own.

## What we could not do

**Predict the benefit from summary statistics.** We built three such measures.
Each looked convincing on the cases used to build it and each failed on cases
held back — one classified targets by kind, one measured how clustered the
unexplained variation was, one measured how much of it neighbouring values
recovered. All three are withdrawn.

**Judge transferability without answers from the destination.** Whether a
relationship holds in a new place is a fact about that place, and inspecting
its inputs does not recover it. Three such tests, checked against ten
transfers with known outcomes, all failed. Los Angeles and New York rental
listings look nearly identical on their inputs, yet a bedroom is worth 0.156
in one and 0.030 in the other. Collect a few dozen ground-truth points in the
new region before trusting any transferred model.

**Choose the neighbourhood size from source data.** Neighbourhood size should
be trained rather than assumed — GraphSAGE was tuned only by hand while its
rivals tuned themselves — but selecting it on held-out source data picked a
neighbourhood twenty times too wide on San Diego → Broward and degraded the
forward direction from 0.679 to 0.638. Bandwidth is an absolute distance and
the two cities differ in density, so what works at the source is wrong at the
destination. A neighbourhood defined by *count* rather than distance would
rescale itself and is the obvious next attempt; it is untested.

## How often transfer works at all

Of more than twenty-five candidate region pairs examined, seven were worth
modelling. The failures were ordinary pairings: two boxes 110 km apart in the
same landform gave a reverse-direction R² of −8.4, and building age failed
across all twenty ordered pairs of New York boroughs. Pairs chosen for
similarity succeeded — two Iowa corn farms 29 km apart transferred cleanly —
while pairs chosen for contrast had coefficients invert outright.

## Notes for practice

- Report R² as 1 − SSE/SST. Squared correlation cannot see miscalibration:
  the same King County → Ames predictions score 0.627 by that measure and
  0.501 by the correct one.
- Report one deployable model. Averaging across models raised a result from
  0.380 to 0.583 and produces nothing anyone can ship.
- Apply layer normalisation per node, not per graph. Without it the same
  architecture scored −0.873 on King County → Ames.
- Never select a hyperparameter on target performance. Our early kernel
  screens did, which made them unreproducible in deployment.
- Check satellite mosaics for season. Ours served July for one city and
  January for the other; undetected, that failure would have looked like two
  regions that do not transfer.
- Do not benchmark transfer against a modelled product. Three soil-carbon
  attempts failed before we noticed the target was itself a model's output,
  fitted to the same kinds of inputs we were supplying.

---

*Eight transfers, one classification, a five-scenario simulation grid and 39
held-out US states. Predictions for the five most recent tests were recorded
before the models ran and are scored — including the two that were wrong — in
`predictions.md`. Supporting detail and withdrawn claims are in `REPORT.md`.*

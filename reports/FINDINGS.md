# When does a graph neural network help with spatial prediction?

We trained GraphSAGE on one region and applied it unchanged to a second
region it had never seen. Across eight such transfers it won four times and
lost four. What decides the outcome is not how spatially clustered the data
looks, but whether the quantity being predicted spreads from place to place.

## Results

Each row is one trained model applied to a disjoint region. R² is 1 − SSE/SST.

| Transfer | Target | GraphSAGE | Best rival | Margin |
| --- | --- | --- | --- | --- |
| King County → Ames | House price | **0.501** | 0.283 | **+0.218** |
| Des Moines → Cedar Rapids | Surface temperature | **0.822** | 0.778 | **+0.044** |
| Broward → San Diego | Rental price | **0.679** | 0.649 | **+0.030** |
| Chicago → Los Angeles | Rental price | **0.654** | 0.632 | **+0.022** |
| California → West Virginia | Diabetes rate | 0.763 | **0.803** | −0.040 |
| Queens → Brooklyn | 311 response time | 0.232 | **0.301** | −0.057 |
| Broward → San Diego | Rental reviews | 0.185 | **0.271** | −0.086 |
| Queens → Brooklyn | Tree trunk diameter | 0.132 | **0.276** | −0.144 |

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

## What we could not do

**Predict the benefit from summary statistics.** We built three such measures.
Each looked convincing on the cases used to build it and each failed on cases
held back — one classified targets by kind, one measured how clustered the
unexplained variation was, one measured how much of it neighbouring values
recovered. All three are reported as failures.

**Judge transferability without answers from the destination.** Whether a
relationship holds in a new place is a fact about that place, and inspecting
its inputs does not recover it. Three such tests, checked against ten
transfers with known outcomes, all failed. Los Angeles and New York rental
listings look nearly identical on their inputs, yet a bedroom is worth 0.156
in one and 0.030 in the other. The practical course is to collect a few dozen
ground-truth points in the new region before trusting any transferred model.

## How often transfer works at all

Of more than twenty-five candidate region pairs examined, seven were worth
modelling. The failures were ordinary pairings, not exotic ones: two boxes
110 km apart in the same landform gave a reverse-direction R² of −8.4, and
building age failed across all twenty ordered pairs of New York boroughs.
Pairs chosen for similarity succeeded — two Iowa corn farms 29 km apart
transferred cleanly — while pairs chosen for contrast had coefficients invert
outright.

## Notes for practice

- Report R² as 1 − SSE/SST. Squared correlation cannot see miscalibration:
  one model here scores 0.543 by that measure and −0.873 by the correct one.
- Report one deployable model. Averaging across models raised a result from
  0.380 to 0.583 and produces nothing anyone can ship.
- Apply layer normalisation per node, not per graph. Without it the same
  architecture scored −0.873 on King County → Ames. It removes bias reliably,
  though bias was not what cost us the losses.
- Check satellite mosaics for season. Ours served July for one city and
  January for the other; undetected, that failure would have looked like two
  regions that do not transfer.
- Do not benchmark transfer against a modelled product. Three soil-carbon
  attempts failed before we noticed the target was itself a model's output,
  fitted to the same kinds of inputs we were supplying.

---

*Eight transfers, one classification, a five-scenario simulation grid and 39
held-out US states. Predictions for the four most recent tests were recorded
before the models ran and are scored — including the two that were wrong — in
`predictions.md`. Supporting detail and withdrawn claims are in `REPORT.md`.*

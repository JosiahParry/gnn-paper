# Handoff: run the King County to Ames seed sweep on the M4 Mini

## What we need to settle

Does LayerNorm actually help the Ames transfer, or is it noise?

Two runs of ten seeds each disagree:

| | run 1 (serial) | run 2 (8 daemons) |
| --- | --- | --- |
| GraphSAGE + LayerNorm | 0.582 (sd 0.044) | 0.545 (sd 0.075) |
| GraphSAGE | 0.512 (sd 0.122) | 0.526 (sd 0.095) |
| **gap** | **0.070** | **0.019** |

Same data, same settings. We need enough replicates to tell 0.02 from zero.
Thirty seeds per arm should do it.

## Why this machine

The job is CPU-bound and holds about 650 MB, so RAM is not the constraint --
core count is. The 16 GB MacBook has 10 cores and the last run at 8 daemons
made it unusable for 25 minutes.

Do not bother with the Ryzen's GPU. R's `torch` needs a CUDA build, and a
21,613-node sparse graph is too small to fill a GPU.

## Setup

```
git pull                      # or however the repo gets there
rv sync
```

`rproject.toml` pins R 4.6. If the Mini has a different version, change
`r_version` and re-sync, or rv drops to a temp library and `spdep` goes
missing.

Check it took:

```
R -q -e 'library(torchgnn); library(sphet); cat("ok\n")'
```

## The run

```
Rscript R/kc-to-ames.R 12
```

The argument is the daemon count. Leave two cores free. Expect roughly
45-60 minutes for 150 fits.

Everything needed is already committed:

- **All four stochastic arms are now seeded.** XGBoost was the bug in run 2 --
  its engine draws an internal validation split from R's RNG, unseeded, which
  moved its Ames R2 from 0.530 to 0.403 between runs. It was never
  deterministic and should not have been fit once.
- **`seeds` is 1001:1030**, so 30 fits per arm instead of 10.
- Only OLS is fit once, because only OLS is actually deterministic.

## What to send back

`data/kc-to-ames-results.rds`, plus the console block under
`=== Ames transfer, across initialisations ===`.

Then, on any machine:

```
R -f R/figures.R
```

which rewrites `images/fig-ames-seeds.png` from the new results.

## How to read the answer

- **If the gap holds near 0.07 with sd under 0.05**, LayerNorm is real and the
  paper can claim it.
- **If the gap is under 0.02**, it is noise. Cut the LayerNorm claim from the
  Ames section and keep LayerNorm only as the Scenario C result, where it
  raised 0.218 to 0.307 across six folds.

## One caveat

torch's CPU sparse operations are not deterministic across processes, so a
daemon run will not reproduce a serial run to the digit. That is fine for a
distribution over initialisations, which is what this reports. It is not fine
for a single quoted number, which is why `R/states.R` runs serially.

## Not blocked on this

These are stable and do not need rerunning -- they average over folds or
states, which damps the initialisation noise:

- Scenario E, block versus random (`data/scenario-results-*.rds`)
- The 39 held-out states (`data/state-results.rds`)
- King County within-county cross-validation

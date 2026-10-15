# Target-leakage audit — response to IEEE HONET reviewer

**Comment.** *"Possible target leakage. Some derived features (e.g. sm_change, rolling
means, lagged cumulative irrigation) may include information from the current interval
or carry irrigation persistence. This could inflate the gain from R² 0.55 to 0.79.
Please define the prediction horizon, compute the features causally, and report results
with and without irrigation-history features."*

Reproduced by `scripts/causal_ablation.py`, run inside the submitted notebook pipeline
so the baseline arm reproduces the published number.

## 1. Prediction horizon (was undocumented)

The task is **one-step-ahead**: predict `y[t]` (volume applied during interval `t`)
from inputs observed at `t-1`. At 10-minute sampling this is a 10-minute horizon.

This was already true of the submitted results but never stated. `make_sequences()`
builds window `X[i-lb:i]` with target `y[i]`, so the tabular feature vector
`X_tab = X_seq[:, -1, :]` is the row at `i-1`. **No model input is measured during the
interval being predicted.**

## 2. Current-interval leakage: NOT present

The feature *definitions* in §III-B read as concurrent (`sm_change = sm[t]-sm[t-1]`,
`rolling(6)` includes row `t`), which is why the concern was raised. The sequence
offset lags all of them by one interval before any model sees them.

Re-deriving every feature explicitly causally — rolling windows ending at `t-1`,
differences over `[t-2,t-1]`, agronomic indicators from the previous day, cumulative
irrigation reset at midnight and excluding `t` — **does not reduce** performance:

| Arm | Features | MAE | RMSE | R² |
|---|---|---|---|---|
| A. As submitted | 24 | 1.916 | 8.734 | **0.793** |
| B. Explicitly causal, + irrigation history | 24 | 1.970 | 8.250 | **0.815** |
| C. Explicitly causal, − irrigation history | 23 | 4.218 | 13.450 | **0.510** |

A reproduces the published 0.790. B is marginally *better* than A because it also
fixes a day-boundary defect: the original `groupby(date).cumsum().shift(1)` shifted
across midnight, so the first interval of each day inherited the previous day's total
instead of resetting to zero.

## 3. Irrigation persistence: CONFIRMED, and material

Removing `cum_daily_irr` costs **ΔR² = 0.306** (0.815 → 0.510). The reviewer is right
that the headline gain is substantially irrigation persistence rather than sensor fusion.

Sensor ladder, fully causal, **without** irrigation history:

| Configuration | #Feat | MAE | RMSE | R² |
|---|---|---|---|---|
| Soil moisture only | 1 | 5.401 | 18.284 | 0.094 |
| Soil suite (moisture + EC + temp) | 3 | 7.560 | 20.863 | −0.180 |
| All raw sensors | 7 | 5.878 | 14.832 | 0.404 |
| Raw + agronomic indicators | 10 | 5.315 | 14.634 | 0.419 |
| Full engineered (no irrigation history) | 23 | 4.218 | 13.450 | 0.510 |

So the defensible sensor-fusion claim is **0.094 → 0.510**, not 0.12 → 0.79.

## 4. Nuance worth keeping

Irrigation history is **not** sufficient on its own:

| Configuration | #Feat | R² |
|---|---|---|
| `cum_daily_irr` alone | 1 | **−0.098** |
| soil moisture + `cum_daily_irr` | 2 | **0.764** |

Alone it is worse than predicting the mean. Combined with a single soil-moisture
reading it reaches 0.764 — close to the full 24-feature model (0.815). The model is
therefore not memorising the schedule; it learns *when the daily budget is spent given
soil state*. That is a real signal, but it is an interaction with irrigation history,
not evidence that multi-sensor fusion alone drives the result.

## 5. Consequences for the manuscript

- State the 10-minute horizon explicitly (§III).
- State that features are computed causally (§III-B), and drop the misleading
  parenthetical implying only `cum_daily_irr` is lagged.
- Report Table III with and without irrigation history.
- Revise the "0.12 → 0.79 from sensor fusion" claim in the abstract, §IV-D and the
  conclusion: the sensor-fusion-only figure is 0.094 → 0.510.

Train/test split: 7,615 / 3,265 (temporal 70/30), 10,892 usable rows.

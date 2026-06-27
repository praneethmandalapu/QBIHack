# Genomics → simulation handoff (Praneeth → Vinesh)

How my XGBoost risk model plugs into the tumor-growth solver. The join key is the
**12-char TCGA barcode**; everything is precomputed, so consumption is a lookup.

## The one call you need

```python
from oncopulse import get_patient, growth_multiplier

get_patient("TCGA-AR-A1AX")        # -> {risk, expr(30), pam50, ...}
growth_multiplier("TCGA-AR-A1AX")  # -> risk_multiplier for the solver
```

`growth_multiplier(barcode, lo=0.8, hi=1.8)` maps genomic risk ∈ [0,1] linearly
to `lo + (hi-lo)*risk`. The solver uses it as `rho_eff = rho * risk_multiplier`
(`vinesh/tumor_pde_solver.py:87`).

## Where it goes

`vinesh/tumor_pde_solver.py:34` has the placeholder:

```python
"risk_multiplier": 1.0,  # <-- SWAP IN PRANEETH'S VALUE: scalar from XGBoost
```

Swap per patient at call time (don't hardcode):

```python
from oncopulse import growth_multiplier
params = {**DEFAULT_PARAMS, "risk_multiplier": growth_multiplier(tcga_id)}
frames = solve_growth(vol, timesteps=50, dt=0.1, params=params)
```

## Cohort values (lo=0.8, hi=1.8 → multiplier = 0.8 + risk)

| Barcode | PAM50 | risk | risk_multiplier |
|---|---|---|---|
| `TCGA-AR-A1AX` (primary) | Luminal A | 0.247 | **1.047** |
| `TCGA-AR-A1AQ` (primary) | Basal | 0.083 | **0.883** |
| `TCGA-E2-A15C` | Luminal A | 0.053 | 0.853 |
| `TCGA-BH-A0BQ` | Luminal A | 0.078 | 0.878 |
| `TCGA-A2-A04Q` | Basal | 0.252 | 1.052 |
| `TCGA-B6-A409` | Basal | 0.851 | 1.651 |
| `TCGA-A2-A04R` | Luminal B | 0.142 | 0.942 |

## ⚠️ Two things to reconcile with Vinesh

1. **The primary pair inverts the textbook subtype ordering.** `test_solver.py`
   assumes Luminal A ≈ 0.7 (slow) and Basal ≈ 1.6 (fast). My per-patient model
   says the opposite for *this* pair: the LumA `TCGA-AR-A1AX` (1.047) grows
   *faster* than the Basal `TCGA-AR-A1AQ` (0.883), because that Basal is an
   immune-infiltrated, better-prognosis TNBC. This is the precision-oncology
   story — but the demo narrative and the test's comments must agree on it.

2. **Two sources for `risk_multiplier`.** `vinesh/calibrate.py` fits the
   multiplier from the two imaging timepoints; this gives it from genomics. They
   answer different questions — calibration says how fast the tumor *did* grow,
   genomics says how aggressive the biology *is*. Suggested split: genomics sets
   the per-patient ratio (A1AX vs A1AQ), calibration sets the absolute lo/hi
   anchors. Worth a 5-minute chat before locking the demo.

risk is a relative rank, not a calibrated probability (`scale_pos_weight` in
training inflates absolutes). Use it monotonically; don't read 0.5 as "50%".

# HOW TO USE THE MODEL — OncoPulse genomic risk (Person 1 / Praneeth)

Complete reference for the genomics risk model, written for both humans and
coding agents. If you are an agent working in `vinesh/`, `philip-chandan/`, or
`app/` and you need a per-patient growth/risk number, **read this whole
file first** — it tells you exactly what to call, what you get back, and what it
means.

> One-sentence summary: a trained XGBoost model converts a breast-cancer
> patient's tumor gene-expression into a **risk score in [0, 1]**; that score is
> precomputed for every TCGA-BRCA patient and exposed through one Python module,
> `oncopulse.py`, keyed by **TCGA barcode**.

---

## 0. TL;DR — the only three lines most callers need

```python
import sys; sys.path.insert(0, "breast-cancer-sim")   # so `import oncopulse` works
from oncopulse import get_patient, growth_multiplier

get_patient("TCGA-AR-A1AX")          # -> {"risk": 0.247, "pam50": "BRCA_LumA", "expr": np.ndarray(30), ...}
growth_multiplier("TCGA-AR-A1AX")    # -> 1.047   (feed straight into Vinesh's solver)
```

`get_patient` and `growth_multiplier` are **pure pandas CSV lookups** — they do
**not** import xgboost and do **not** need the macOS libomp fix. Only
`score_expression` (scoring a brand-new expression vector) loads the model. See
§3 and §6.

---

## 1. What the model predicts (and what it does NOT mean)

| | |
|---|---|
| Model type | `xgboost.XGBClassifier`, binary, `objective=binary:logistic` |
| Target | **Disease-specific survival** — `1` = died of breast cancer, `0` = long-term / tumor-free survivor |
| Input | 30 driver genes as **z-scores**, in the exact order of `gene_list.json` |
| Output | `risk = model.predict_proba(X)[:, 1]` ∈ [0, 1] |
| Trained on | Pooled **METABRIC** (microarray) + **TCGA-BRCA** (RNA-seq), 2,481 samples |
| Performance | 5×4 repeated-CV **AUC 0.763 ± 0.019**; honest cross-cohort (train METABRIC → test TCGA) **AUC 0.666** |

**Read `risk` as a relative rank, NOT a calibrated probability.** Training used
`scale_pos_weight` to handle class imbalance, which inflates absolute
probabilities (an "average" patient scores ~0.7, not ~0.3). Use it for ordering
and monotonic mapping; do **not** tell a clinician "this patient has a 25% chance
of X." Ranking across patients is meaningful; the absolute number is not
clinically calibrated.

**The model encodes per-patient biology, not subtype clichés.** It will
sometimes rank a Luminal A patient above a Basal one (see §8). That is the
intended precision-oncology behavior, not a bug.

---

## 2. Environment setup

- Python **3.14**, virtualenv at repo `.venv` (or any env with the deps below).
- Dependencies actually used by the model: `xgboost`, `scikit-learn`, `shap`,
  `pandas`, `numpy`, `requests`.
- **macOS libomp gotcha (only matters if you load `model.pkl`):** xgboost's
  native lib needs `libomp.dylib`, which Homebrew-less Macs lack. scikit-learn
  ships a copy. Two ways to fix:

  ```python
  # Option A — in code, before importing xgboost (or before oncopulse.score_expression).
  # NB: the folder "models-praneeth" has a hyphen and is NOT an importable package,
  # so put it on sys.path and import the module by name:
  import sys; sys.path.insert(0, "breast-cancer-sim/models-praneeth")
  import _macos_omp_fix   # re-execs the interpreter once with DYLD_LIBRARY_PATH set
  ```

  ```bash
  # Option B — in the shell:
  SK=$(python -c "import sklearn,os;print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))")
  DYLD_LIBRARY_PATH="$SK" python your_script.py
  ```

  The re-exec in Option A **breaks stdin/heredoc invocation** (`python - <<EOF`).
  Always run from a real `.py` file. Linux/Windows need none of this.

---

## 3. The Python API — `breast-cancer-sim/oncopulse.py`

`oncopulse.py` lives at the `breast-cancer-sim/` root. To import it, that
directory must be on `sys.path` (it resolves all its data paths relative to its
own location, so you can import it from anywhere once it's on the path).

### `get_patient(barcode: str) -> dict`  *(CSV lookup, no xgboost)*
Look up a precomputed TCGA patient. `barcode` may be a full sample barcode
(`TCGA-AR-A1AX-01A-...`) or the 12-char patient barcode — it is truncated to
`barcode[:12]` internally. Raises `KeyError` if the patient was never scored.

```python
p = get_patient("TCGA-AR-A1AX")
p["barcode"]  # "TCGA-AR-A1AX"
p["risk"]     # 0.247           float in [0,1]
p["pam50"]    # "BRCA_LumA"     str: BRCA_LumA|BRCA_LumB|BRCA_Basal|BRCA_Her2|BRCA_Normal|NA
p["expr"]     # np.ndarray shape (30,), z-scored genes in GENE_LIST order
p["genes"]    # list[str], the 30 gene names (same order as expr)
```

### `growth_multiplier(barcode, lo=0.8, hi=1.8) -> float`  *(CSV lookup, no xgboost)*
Maps the patient's risk to the growth knob Vinesh's solver consumes:
`lo + (hi - lo) * risk`. Defaults give ~0.8–1.8, the ballpark of
`vinesh/test_solver.py`. See §7 for how it plugs into the PDE.

```python
growth_multiplier("TCGA-AR-A1AX")            # 1.047   (= 0.8 + 0.247)
growth_multiplier("TCGA-AR-A1AQ")            # 0.883
growth_multiplier("TCGA-AR-A1AX", lo=1.0, hi=2.0)  # retune the anchors
```

### `score_expression(expr) -> float`  *(LOADS model.pkl — needs xgboost + libomp)*
Escape hatch to score a raw z-scored expression vector for a patient **not** in
the precomputed table. `expr` is either a `{gene: zscore}` dict or a sequence in
`GENE_LIST` order. Missing genes default to 0.0 (cohort mean). See §5 for how to
compute the z-scores correctly.

```python
risk = score_expression({"MKI67": 1.8, "FOXM1": 1.2, "PGR": -1.0})   # others -> 0
```

### `gene_correlation() -> pd.DataFrame`  *(CSV lookup)*
30×30 gene–gene correlation matrix (for Vinesh's ODE gene coupling, if used).

### `list_barcodes() -> list[str]`  *(CSV lookup)*
All 1,082 scored TCGA patient barcodes.

### `GENE_LIST`  *(lazy module attribute)*
`from oncopulse import GENE_LIST` → the 30 gene names in canonical order. Lazy
(resolved on first access) so the module imports even before the model exists.

---

## 4. Data artifacts — every file, path, schema

All paths are under `breast-cancer-sim/`. Files under `data/processed/` and
`models-praneeth/saved/` are committed (the team handoff). The two heavy,
reproducible inputs are gitignored.

### Committed (use these)

| Path | Schema / contents |
|---|---|
| `models-praneeth/saved/model.pkl` | Pickled `XGBClassifier`. Unpickling needs xgboost installed (+ libomp on Mac). |
| `models-praneeth/saved/metrics.json` | `{cv_auc, cv_auc_std, best_rounds, best_params, cross_cohort, genes, n_samples, n_trials}` |
| `models-praneeth/saved/shap_importance.csv` | columns: `gene, mean_abs_shap` (ranked desc) — for Vinesh/Philip's EXPLAIN tab |
| `models-praneeth/saved/shap_values.pkl` | `{genes, base_value, values: ndarray(N,30), X: ndarray(N,30), index: list[barcode/sample]}` |
| `models-praneeth/saved/training_log.txt` | Full per-trial training log (evidence of the search) |
| `data/processed/gene_list.json` | `{genes: [30 names], n_genes: 30, input: "...", label: "..."}` — **single source of truth for gene order** |
| `data/processed/tcga_patient_features.csv` | **PRIMARY HANDOFF.** index `barcode`; columns: `risk, pam50, <30 z-scored gene cols>`. 1,082 rows. |
| `data/processed/patient_expression_top_genes.csv` | index `barcode`; the 30 z-scored gene columns only |
| `data/processed/gene_correlation_matrix.csv` | 30×30 correlation, gene names on both axes |
| `data/processed/zscore_reference_tcga.csv` | index = gene; columns `mean, sd` — TCGA per-gene stats (to z-score NEW TCGA patients) |
| `data/processed/zscore_reference_metabric.csv` | same, for METABRIC microarray |
| `data/processed/gene_candidates.json` | the 50 concordant candidate genes + their cross-cohort correlations |
| `data/processed/label_summary.json` | `{metabric:{pos,neg}, tcga:{pos,neg}, matrix}` |

### Gitignored (regenerate; see §9)

| Path | Why ignored |
|---|---|
| `data/raw/brca_metabric/`, `data/raw/brca_tcga_pan_can_atlas_2018/` | ~800 MB raw downloads |
| `data/processed/train_matrix.csv` | ~11 MB pooled training matrix |

---

## 5. The 30 genes & the z-score contract

The model input is **30 genes as z-scores, in `gene_list.json` order**. The genes
are a de-novo-selected proliferation + hormone + immune signature (e.g. `MKI67`,
`FOXM1`, `UBE2C`, `BUB1`, `PLK1` on the risk side; `PGR`, `BCL2`, `SCUBE2`,
`CD8A`, `IFNG` protective). Always read the order from `gene_list.json` — never
hardcode it.

**A z-score is relative to a cohort, not to a single patient.** This is the most
common way to misuse the model:

- If the patient is already in `tcga_patient_features.csv` → just `get_patient()`.
  The z-scores are already computed correctly. **Do this.**
- If you must score a NEW patient (advanced), z-score each gene against the saved
  reference, NOT against the new patient alone:

  ```python
  import json, pickle, numpy as np, pandas as pd
  base = "breast-cancer-sim"
  genes = json.load(open(f"{base}/data/processed/gene_list.json"))["genes"]
  ref   = pd.read_csv(f"{base}/data/processed/zscore_reference_tcga.csv", index_col=0)

  # raw_rsem: dict gene -> raw RSEM count for the new TCGA RNA-seq patient
  z = []
  for g in genes:
      x = np.log2(raw_rsem.get(g, 0) + 1.0)            # TCGA path: log2(RSEM+1)
      z.append((x - ref.loc[g, "mean"]) / ref.loc[g, "sd"])
  from oncopulse import score_expression
  risk = score_expression(z)
  ```

  METABRIC microarray values are already log-intensity — use
  `zscore_reference_metabric.csv` and skip the `log2`.

---

## 6. Which calls need what (dependency map for agents)

| Call | Reads | Needs xgboost? | Needs libomp (Mac)? |
|---|---|---|---|
| `get_patient` | `tcga_patient_features.csv` + `gene_list.json` | no | no |
| `growth_multiplier` | (calls `get_patient`) | no | no |
| `gene_correlation`, `list_barcodes`, `GENE_LIST` | CSV/JSON | no | no |
| `score_expression` | `model.pkl` | **yes** | **yes** |

**Practical consequence:** Philip, Vinesh, Vinesh/Philip almost always only need the
precomputed lookups → they can depend on just `pandas`, with no xgboost/libomp
headache. Only retraining or scoring novel expression touches the model.

---

## 7. Integration with Vinesh's PDE solver

Vinesh's `tumor_pde_solver.solve_growth()` uses Fisher–Kolmogorov growth where
the net proliferation rate is scaled by `risk_multiplier`:

```python
rho_eff = rho * risk_multiplier          # tumor_pde_solver.py
```

`DEFAULT_PARAMS["risk_multiplier"]` is the placeholder marked
`# <-- SWAP IN PRANEETH'S VALUE`. Wire it per patient at call time:

```python
import sys; sys.path.insert(0, "breast-cancer-sim")
from oncopulse import growth_multiplier
from tumor_pde_solver import solve_growth, DEFAULT_PARAMS

params = {**DEFAULT_PARAMS, "risk_multiplier": growth_multiplier(tcga_id)}
frames = solve_growth(baseline_volume, timesteps=50, dt=0.1, params=params)
```

### Calibration vs prediction — read this carefully

`vinesh/calibrate.py::calibrate_growth()` currently **back-solves**
`risk_multiplier` with `brentq` so the simulated burden matches the real
follow-up scan. That is a **fit to the answer**, not a prediction (Vinesh's own
docstring says so). If you fit the knob to the follow-up, you cannot then claim
to "predict" the follow-up.

The genomics model breaks that circularity because it **never sees the follow-up
scan**. Recommended design:

- **Validation mode (two timepoints):** set `risk_multiplier =
  growth_multiplier(barcode)`, simulate baseline→forward, then **compare**
  predicted burden to the real follow-up. Reuse `calibrate_growth`'s diagnostics
  (`target_burden`, `achieved_burden`, `burden_error_pct`) but **without** the
  `brentq` fit. This turns "calibration" into out-of-sample validation — the
  strongest demo claim.
- **Prediction mode (one timepoint — most patients):** there is no follow-up to
  calibrate against, so genomics is the **only** growth source.
- **Interventions (therapy sliders):** imaging cannot tell you drug response.
  Genomics can — the model's hormone genes (`PGR`, `BCL2`, …) should modulate the
  `delta` (death) term so endocrine therapy melts a luminal tumor more than a
  basal one.

### How calibration and genomics coexist (don't pick one)

```
risk_multiplier(patient) = calibration_scale × genomic_ratio(patient)
                           └ fit ONCE, sets    └ from the model, sets the
                             real-time units      per-patient aggressiveness
```

Calibrate the absolute scale on one patient; let genomics set everyone's ratio.

---

## 8. The demo cohort & the inverted-subtype caveat

Canonical cohort lives in
`simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort.json` (Philip's,
authoritative). The genomics scores for the primary pair:

| Barcode | PAM50 | risk | growth_multiplier |
|---|---|---|---|
| `TCGA-AR-A1AX` | Luminal A | 0.247 | **1.047** |
| `TCGA-AR-A1AQ` | Basal | 0.083 | **0.883** |

⚠️ **The model ranks this LumA ABOVE this Basal.** `TCGA-AR-A1AQ` is an
immune-infiltrated (high `CD8A`/`IFNG`), better-prognosis triple-negative tumor,
so its disease-specific risk is low. Consequence: the simulated LumA grows
faster than the Basal. This contradicts the textbook "Basal is aggressive"
assumption hardcoded in `vinesh/test_solver.py` (LumA 0.7 vs Basal 1.6). **The
demo narrative must commit to the per-patient genomic story, not the subtype
cliché.** Across all 1,082 TCGA patients the model does recover the aggressive
subtypes on average (Her2 0.29 > LumB 0.27 > LumA 0.16 ≈ Basal 0.16); Basal is
bimodal because immune-hot vs proliferative basals diverge.

If you need a subtype-aligned pair instead, the highest-risk available Basal is
`TCGA-B6-A409` (risk 0.851) — but only usable if Philip has imaging for it.

---

## 9. Regenerating everything from scratch

Run from `breast-cancer-sim/models-praneeth/` (set the libomp env per §2). Order
matters:

```bash
python download_data.py        # 1. METABRIC + TCGA from cBioPortal (~800MB) -> data/raw/
python build_features.py       # 2. z-score per cohort, curated gene universe,
                               #    concordance filter -> data/processed/ (matrix, refs, candidates)
python train_xgboost.py        # 3. 300-config repeated-CV search -> saved/model.pkl, gene_list.json, metrics.json
                               #    (N_TRIALS / N_REPEATS / N_SPLITS env vars tune the search)
python generate_shap.py        # 4. SHAP over the cohort -> saved/shap_importance.csv, shap_values.pkl
python build_patient_table.py  # 5. score ALL TCGA patients -> data/processed/tcga_patient_features.csv
```

Tuning knobs: gene universe is `CURATED_GENES` in `build_features.py`
(`USE_CURATED=0` to use the raw concordance filter — gives ~+5pts AUC but
biologically implausible top genes like olfactory receptors, so don't). Label
thresholds are constants at the top of `build_features.py`.

---

## 10. Gotchas checklist

- [ ] **Import path:** add `breast-cancer-sim/` to `sys.path` before `import oncopulse`.
- [ ] **Barcode:** always the 12-char patient barcode; `get_patient` truncates for you, but if you join CSVs yourself, do `barcode[:12]` on both sides.
- [ ] **libomp (Mac):** only needed to load `model.pkl` (i.e. `score_expression`/retraining). Lookups don't need it.
- [ ] **z-scores:** never z-score a lone new patient against itself — use the saved reference (§5).
- [ ] **risk is a relative rank**, not a calibrated probability (`scale_pos_weight`).
- [ ] **Subtype inversion:** the primary pair's LumA > Basal by design (§8).
- [ ] **Downloads** use `requests`, not urllib (Python 3.14 on Mac has no CA certs).
- [ ] **Re-exec quirk:** `_macos_omp_fix` breaks `python -` heredoc; run from a file.

---

## 11. File map (quick reference)

```
breast-cancer-sim/
├── oncopulse.py                      # THE API: get_patient, growth_multiplier, ...
├── models-praneeth/
│   ├── HOWTOUSEMODEL.md              # this file
│   ├── GENOMICS_HANDOFF.md           # focused genomics→solver handoff
│   ├── download_data.py build_features.py train_xgboost.py
│   ├── generate_shap.py build_patient_table.py _macos_omp_fix.py
│   └── saved/                        # model.pkl, metrics.json, shap_*, training_log.txt
└── data/processed/                   # gene_list.json, tcga_patient_features.csv,
                                      # zscore_reference_*.csv, gene_correlation_matrix.csv, ...
```

Questions a doc can't answer → ping Praneeth (Person 1).

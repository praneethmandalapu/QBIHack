# Philip-Chandan spike checklist (steps 1–4)

Parent plan: [`../HANDOFF_SPIKE.md`](../HANDOFF_SPIKE.md)

**Contract:** [`../handoff_contract.json`](../handoff_contract.json) (`version` **1.0.0**)

**Case:** `TCGA-AR-A1AX` · Luminal A · `2002-09-12`

---

## Your outputs

| Artifact | Path |
|----------|------|
| DICOM | `data/raw/tcia/luminal_a/TCGA-AR-A1AX/2002-09-12/` |
| Raw volume | `data/processed/raw-extract-philip-chandan/TCGA-AR-A1AX/baseline.npy` |
| Raw metadata | `data/processed/raw-extract-philip-chandan/TCGA-AR-A1AX/baseline.json` |
| QC plot | `data/qc/slice-plots-philip-chandan/luminal_a_TCGA-AR-A1AX_baseline_mid-z.png` |

Slug `luminal_a_TCGA-AR-A1AX_baseline` is unchanged for CLI, QC, and segmentation filenames.

---

## Step 1 — Download

```bash
cd breast-cancer-sim
python simulation-vinesh-philip-chandan/philip-chandan/download_tcia.py \
  --tcga-id TCGA-AR-A1AX --subtype "Luminal A" --longitudinal
```

Only the `2002-09-12` folder is required for the spike (follow-up can wait).

---

## Step 2 — Download QC

```bash
python -c "
from tcia_extractor import validate_series, resolve_study_dir
d = resolve_study_dir('TCGA-AR-A1AX', 'Luminal A', '2002-09-12')
r = validate_series(d)
print(r)
assert r['ok'], r['errors']
"
```

---

## Step 3 — Raw extract + export for Vinesh

```bash
python simulation-vinesh-philip-chandan/spike_paths.py
python simulation-vinesh-philip-chandan/philip-chandan/export_raw_extract.py
python simulation-vinesh-philip-chandan/philip-chandan/generate_manifest.py
```

Ping Vinesh when `TCGA-AR-A1AX/baseline.npy` and `.json` land in `raw-extract-philip-chandan/`.

**Out of scope for you (Option B):** normalize, resample, crop, `solve_growth`.

---

## Step 4 — Visual QC

```bash
python simulation-vinesh-philip-chandan/philip-chandan/qc_slice_plot.py
```

Open the PNG under `data/qc/slice-plots-philip-chandan/`. Confirm anatomy looks real; note series description in Slack if anything looks off.

---

## Handoff message template (Slack)

> Raw extract for spike is ready.
> `data/processed/raw-extract-philip-chandan/TCGA-AR-A1AX/baseline.npy`
> + `.json` (`contract_version` **1.0.0**, spacing_mm, shape). Your turn: `prepare_pde_input.py` → `pde-input-vinesh/TCGA-AR-A1AX/g64/baseline.npy`.

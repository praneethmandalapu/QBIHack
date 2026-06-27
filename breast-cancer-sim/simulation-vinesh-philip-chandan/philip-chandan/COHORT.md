# TCGA-BRCA Cohort — Longitudinal MRI Primary Pair

Patient IDs live in [`cohort.json`](cohort.json) (machine-readable) and this file (human-readable). Both pipelines should load `cohort.json` on Day 1 so imaging and genomics stay aligned.

**Selection criteria (rev2):** MRI on TCIA, at least two MR studies per patient, PAM50 subtype match, and meaningful time separation between studies.

---

## Primary Pair (start here)

| Subtype | TCGA barcode | Longitudinal span | PAM50 | Notes |
|---------|--------------|-------------------|-------|-------|
| **Luminal A** | `TCGA-AR-A1AX` | 2002-09-12 → 2003-09-24 (~12 mo) | Luminal A | Best LumA case with multi-month follow-up MR on TCIA |
| **Basal-like** | `TCGA-AR-A1AQ` | 2001-11-21 → 2003-05-07 (~17 mo) | Basal | Strong basal longitudinal case; contrast VIBRANT series |

### On-disk DICOM layout

```
data/raw/tcia/
├── luminal_a/TCGA-AR-A1AX/
│   ├── 2002-09-12/     # baseline MR
│   └── 2003-09-24/     # follow-up MR
└── basal/TCGA-AR-A1AQ/
    ├── 2001-11-21/
    └── 2003-05-07/
```

Download with:

```bash
python download_tcia.py --all-primary --longitudinal
```

Extract baseline volume for Vinesh:

```python
from tcia_extractor import extract_volume_with_spacing_for_timepoint

density, spacing = extract_volume_with_spacing_for_timepoint(
    "TCGA-AR-A1AX", "Luminal A", study_date="2002-09-12"
)
```

---

## Handoff for the Primary Pair

### Philip-Chandan (imaging)

1. Query the **TCIA API** for both barcodes (collection `TCGA-BRCA`, modality `MR`).
2. Download **both timepoints** per patient into dated subfolders.
3. Prefer post-contrast series (VIBRANT / +C) per timepoint.
4. Process baseline `.npy` for simulation; keep follow-up for validation / future growth comparison.

### Praneeth (genomics)

1. Query the **GDC Data Portal** for the same two barcodes.
2. Extract PAM50 subtype, ER/PR receptor status, and survival data.
3. Compute baseline risk scores for the Luminal A vs Basal-like comparison.

---

## Why we pivoted from rev1

| Old primary | Issue |
|-------------|-------|
| `TCGA-BH-A0BR` (Luminal A) | No MRI on TCIA |
| `TCGA-A2-A04P` (Basal) | No MRI on TCIA |

Only **19 / 139** TCGA-BRCA patients with MRI on TCIA have multiple MR studies. The new primaries are the best LumA + Basal pair with true follow-up imaging.

---

## Backup Roster

### Luminal A

| Priority | TCGA barcode | Longitudinal? | Notes |
|----------|--------------|---------------|-------|
| 1 | `TCGA-OL-A66N` | Yes (~3 mo) | PAM50 LumA |
| 2 | `TCGA-BH-A0DK` | Yes (~10 days) | PAM50 LumA; short interval |
| 3 | `TCGA-BH-A0BQ` | No | Single timepoint; good for quick tests |
| 4 | `TCGA-BH-A0BR` | No | Original pick; no TCIA MRI |

### Basal-like

| Priority | TCGA barcode | Longitudinal? | Notes |
|----------|--------------|---------------|-------|
| 1 | `TCGA-AR-A1AQ` | Yes (~17 mo) | Promoted to primary |
| 2 | `TCGA-A2-A04Q` | Verify TCIA | Original backup |
| 3 | `TCGA-BH-A0BW` | Verify TCIA | Original backup |
| 4 | `TCGA-A2-A04P` | No | Original pick; no TCIA MRI |

### Luminal B (later)

| TCGA barcode | Notes |
|--------------|-------|
| `TCGA-A2-A04R` | |
| `TCGA-E2-A15A` | |
| `TCGA-AR-A24S` | Longitudinal MRI; PAM50 LumB |

---

## Quick reference

```
PRIMARY (longitudinal MRI)
  Luminal A:   TCGA-AR-A1AX   (2002-09-12, 2003-09-24)
  Basal-like:  TCGA-AR-A1AQ   (2001-11-21, 2003-05-07)

BACKUP — Luminal A
  TCGA-OL-A66N
  TCGA-BH-A0DK
  TCGA-BH-A0BQ
  TCGA-BH-A0BR

BACKUP — Basal-like
  TCGA-AR-A1AQ
  TCGA-A2-A04Q
  TCGA-BH-A0BW
  TCGA-A2-A04P
```

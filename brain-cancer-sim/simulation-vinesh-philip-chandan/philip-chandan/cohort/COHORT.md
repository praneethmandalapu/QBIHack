# Brain imaging cohort (Philip-Chandan)

Patient picks for longitudinal glioma MRI + expert segmentation. Genomics alignment with Praneeth lives in `models-praneeth/` (stub).

## Selection criteria

1. **Longitudinal** — at least baseline + one follow-up (growth modeling).
2. **Expert segmentation** — radiologist or dataset-provided mask (required; no Otsu-only path in v1).
3. **Genomics overlap** — prefer cases Praneeth can label (IDH, grade, MGMT) for UI toggle.
4. **Download feasibility** — registration / access approved before locking IDs.

## Candidate datasets

See repo root [`DATASETS.md`](../../../DATASETS.md). Registry keys live in [`datasets.py`](datasets.py).

| Dataset key | Access | Notes |
|-------------|--------|-------|
| `ucsf_longitudinal_glioma` | UCSF portal | Primary spike target (UCSF-ALPTDG) |
| `mu_glioma_post` | TCIA NIfTI | **~11 GB** bulk download via Faspex; see `download_mu_glioma_post.py` |
| `lumiere` | Figshare | Longitudinal GBM + RANO + auto segmentations |
| `yale_brain_mets` | TBD | Metastases branch if glioma blocked |

## Discovery workflow

From `brain-cancer-sim/`:

```bash
# List candidate datasets from DATASETS.md
python simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort_discovery.py list-datasets

# After downloading NIfTI into data/raw/<dataset>/...
python simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort_discovery.py scan-local --dataset mu_glioma_post
python simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort_discovery.py find-longitudinal --dataset mu_glioma_post

# Suggest first spike patient once data is on disk
python simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort_discovery.py recommend-pair

# Validate cohort.json before notifying Praneeth / Vinesh
python simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort_discovery.py audit
python simulation-vinesh-philip-chandan/philip-chandan/cohort/cohort_discovery.py show PATIENT_ID --dataset mu_glioma_post
```

### On-disk layout (NIfTI datasets)

```
data/raw/mu_glioma_post/<patient_id>/
├── Timepoint_1/
│   ├── t1c.nii.gz
│   └── seg.nii.gz
└── Timepoint_2/
    ├── t1c.nii.gz
    └── seg.nii.gz

data/raw/lumiere/<patient_id>/   # Figshare extract
data/raw/ucsf_alptdg/<patient_id>/
```

Download MU-Glioma-Post:

```bash
python simulation-vinesh-philip-chandan/philip-chandan/download_mu_glioma_post.py --metadata-only
python simulation-vinesh-philip-chandan/philip-chandan/download_mu_glioma_post.py --imaging
```

Full imaging bundle is **~11 GB** via TCIA Faspex (requires `ascli` + `ascli config ascp install`).

## Slug convention

`{disease}_{dataset}_{patient_id}_{timepoint}` — e.g. `glioma_ucsf_P001_baseline`.

## After patient lock

1. Edit `cohort.json` — add real `patient_id`, timepoints, bump `version`.
2. Notify Praneeth (genomics), Vinesh (paths), Vinesh/Philip (manifest keys).

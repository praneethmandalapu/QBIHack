# Brain imaging cohort (Philip-Chandan)

Patient picks for longitudinal glioma MRI + expert segmentation. Genomics alignment with Praneeth lives in `models-praneeth/` (stub).

## Selection criteria

1. **Longitudinal** — at least baseline + one follow-up (growth modeling).
2. **Expert segmentation** — radiologist or dataset-provided mask (required; no Otsu-only path in v1).
3. **Genomics overlap** — prefer cases Praneeth can label (IDH, grade, MGMT) for UI toggle.
4. **Download feasibility** — registration / access approved before locking IDs.

## Candidate datasets

See repo root [`DATASETS.md`](../../../DATASETS.md).

| Dataset | Access | Notes |
|---------|--------|-------|
| UCSF Longitudinal Glioma | TBD | Primary spike target |
| MU-Glioma-Post | TBD | Fallback |
| Yale Brain Mets | TBD | Metastases branch if glioma blocked |

## Workflow

1. Run `cohort_discovery.py` *(TODO)* against local inventory or dataset manifest.
2. Edit `cohort.json` — bump `version` when IDs change.
3. Notify Praneeth (genomics), Vinesh (paths), Vihari (manifest keys).

## Slug convention

`{disease}_{dataset}_{patient_id}_{timepoint}` — e.g. `glioma_ucsf_P001_baseline`.

# Stretch plan — PyRadiomics (Philip-Chandan)

Living doc for post-sprint work. Sprint pipeline is complete; this folder is **isolated**.

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Scaffold (`paths`, `load_manifest`, docs) | **done** |
| 2 | `prep_volume.py` — normalize + mask + SITK | **done** |
| 2b | `qc_mask_overlay.py` | **done** |
| 3 | `extract_radiomics.py` | **done** (PyRadiomics default + fastrad backend) |
| 4 | `run_all_radiomics.py`, `compare_longitudinal.py` | **done** |
| 5 | `PRANEETH_HANDOFF.md`, CSV to Praneeth | pending |
| 6 | `stretch/tests/` | **done** (prep + dual-backend extraction) |

## Isolation

- All code under `philip-chandan/stretch/`
- Read-only: raw extracts + manifest
- Write-only: `data/processed/radiomics-philip-chandan/`, `data/qc/radiomics-philip-chandan/`
- Do **not** import from `vinesh/` — mirror algorithms only

## Vinesh reference (read-only)

| File | Use |
|------|-----|
| `vinesh/prepare_pde_input.py` | **Not** used as radiomics input (64³ crop) — PDE prep only |

## ROI masks (no Otsu)

- **Baseline:** TCIA `.les` via [`load_les_mask.py`](load_les_mask.py) or aligned-bbox export ([`../validation/run_aligned_bbox_workflow.py`](../validation/run_aligned_bbox_workflow.py))
- **Follow-up:** aligned-bbox or nnU-Net ([`../segmentation/PLAN.md`](../segmentation/PLAN.md)) — no TCIA `.les`
- Global Otsu **retired** (see [`../validation/VALIDATION.md`](../validation/VALIDATION.md))

## Dual backend (Phase 3)

- **Canonical CSV:** PyRadiomics via [`radiomics_params.yaml`](radiomics_params.yaml) (`normalize: false`; prep already scales to [0, 1], `binWidth: 0.05`).
- **Optional fast path / parity:** `fastrad` (`--backend fastrad`); same feature classes, `device=auto`.
- Do not double-normalize or resample in both prep and extractor.

## Next steps

1. Wire aligned-bbox / `.les` masks in `prep_volume.py` (Otsu retired)
2. Run `qc_mask_overlay.py` on all four slugs; confirm mask covers tumor, not whole breast
3. Run `run_all_radiomics.py` → `features_all.csv` (basal first; LumA follow-up after mask)
4. Run `compare_longitudinal.py` → `features_longitudinal.csv` for Praneeth (join on `tcga_id`)
5. Write `PRANEETH_HANDOFF.md` and share CSV

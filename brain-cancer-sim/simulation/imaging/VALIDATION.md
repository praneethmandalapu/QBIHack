# Validation & QC

Interactive 3D inspection with napari — see [`view_volume_napari.py`](view_volume_napari.py) (run instructions at top of that file).

Brain datasets (UCSF, MU-Glioma-Post, MSSEG2) ship expert segmentations; use those as ground truth instead of Otsu.

Quick start with no data:

```bash
cd brain-cancer-sim
source .venv/bin/activate
python simulation/imaging/view_volume_napari.py --demo
```

When raw extracts exist:

```bash
python simulation/imaging/view_volume_napari.py --list
python simulation/imaging/view_volume_napari.py --slug <slug>
```

Direct NIfTI (before export pipeline):

```bash
python simulation/imaging/view_volume_napari.py \
  --mr data/raw/.../T1.nii.gz \
  --mask data/raw/.../seg.nii.gz
```

Breast reference (TCIA `.les` validation): `breast-cancer-sim/simulation-vinesh-philip-chandan/philip-chandan/validation/`

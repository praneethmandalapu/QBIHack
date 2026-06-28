# Validation & QC

Interactive 3D inspection with napari — see [`view_volume_napari.py`](view_volume_napari.py).

Clinical-style QC dock (right side):

- **Window / Level** sliders (brain-masked defaults; ignores black padding outside head)
- **Ctrl + drag** on canvas for click-drag WW/WL
- **Plane** dropdown — axial / coronal / sagittal on the main canvas
- **CLAHE** — local contrast for QC only (not for diagnosis)
- **Orthogonal MPR grid** — linked 1×3 axial / coronal / sagittal panels
- **FLAIR / T2 / T1** loaded automatically for UCSF when raw NIfTI exists (toggle in layer list)
- **Hide overlay** — toggle expert segmentation

Brain datasets (UCSF, MU-Glioma-Post, MSSEG2) ship expert segmentations; use those as ground truth instead of Otsu.

Quick start with no data:

```bash
cd brain-cancer-sim
source .venv/bin/activate
python simulation-vinesh-philip-chandan/philip-chandan/view_volume_napari.py --demo
```

When raw extracts exist:

```bash
python simulation-vinesh-philip-chandan/philip-chandan/view_volume_napari.py --list
python simulation-vinesh-philip-chandan/philip-chandan/view_volume_napari.py --slug glioma_ucsf_100002_baseline
```

Direct NIfTI (before export pipeline):

```bash
python simulation-vinesh-philip-chandan/philip-chandan/view_volume_napari.py \
  --mr data/raw/ucsf_alptdg/100002/100002_time1_t1ce.nii.gz \
  --mask data/raw/ucsf_alptdg/100002/100002_time1_seg.nii.gz
```

Breast reference (TCIA `.les` validation): `breast-cancer-sim/simulation-vinesh-philip-chandan/philip-chandan/validation/`

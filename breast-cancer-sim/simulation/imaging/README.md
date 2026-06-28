# Brain imaging napari viewer (demo mode)

No dataset required — run from `breast-cancer-sim` with the shared venv:

```bash
cd breast-cancer-sim
.venv/bin/python simulation/imaging/view_volume_napari.py --demo
```

Reads real data from sibling `../brain-cancer-sim/data/` when exported.

Breast `.les` validation viewer (has local TCIA data):

```bash
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/validation/view_les_napari.py --list

# Expert .les + cuboid_enhancement predicted mask (run segment.py first)
.venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/validation/view_les_napari.py \
  --slug luminal_a_TCGA-AR-A1AX_baseline --cuboid-enhancement
```

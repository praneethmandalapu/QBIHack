"""Interactive 3D viewer: brain MR + expert segmentation overlay (napari).

Run from breast-cancer-sim (shared .venv — no brain dataset required for demo):

    cd breast-cancer-sim
    source .venv/bin/activate
    pip install -r requirements.txt   # napari[pyqt6]

    # Synthetic glioma demo (default when no data on disk)
    python simulation/imaging/view_volume_napari.py
    python simulation/imaging/view_volume_napari.py --demo

    # After brain data is exported to ../brain-cancer-sim/data/...
    python simulation/imaging/view_volume_napari.py --list
    python simulation/imaging/view_volume_napari.py --slug glioma_ucsf_P001_baseline

    # Direct NIfTI pair
    python simulation/imaging/view_volume_napari.py \\
        --mr ../brain-cancer-sim/data/raw/patient001/T1.nii.gz \\
        --mask ../brain-cancer-sim/data/raw/patient001/seg.nii.gz

Run (Windows):
    cd breast-cancer-sim
    .venv\\Scripts\\Activate.ps1
    .venv\\Scripts\\python.exe simulation\\imaging\\view_volume_napari.py --demo

Requires napari[pyqt6]. First launch may take ~10s while Qt initializes.
Canonical copy also lives in ../brain-cancer-sim/simulation/imaging/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

IMAGING_DIR = Path(__file__).resolve().parent
BREAST_ROOT = IMAGING_DIR.parents[1]
BRAIN_ROOT = BREAST_ROOT.parent / "brain-cancer-sim"


def _brain_repo() -> Path:
    if BRAIN_ROOT.is_dir():
        return BRAIN_ROOT
    return BREAST_ROOT


def _repo_path(relative: str) -> Path:
    return _brain_repo() / relative


def _ensure_brain_solver_path() -> None:
    solver = _brain_repo() / "simulation" / "solver"
    sim = _brain_repo() / "simulation"
    for path in (solver, sim):
        if path.is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def dummy_volume(shape=(48, 48, 48), radius: float = 10.0, seed: int = 0) -> np.ndarray:
    """Synthetic tumor blob — used when ../brain-cancer-sim solver is unavailable."""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.indices(shape)
    cz, cy, cx = (s / 2 for s in shape)
    r2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    blob = np.exp(-r2 / (2.0 * radius**2))
    blob = blob + 0.02 * rng.standard_normal(shape)
    return np.clip(blob, 0.0, 1.0).astype(np.float32)


def get_dummy_volume():
    _ensure_brain_solver_path()
    try:
        from tumor_pde_solver import dummy_volume as solver_dummy  # type: ignore

        return solver_dummy(shape=(48, 48, 48), radius=10.0)
    except ImportError:
        return dummy_volume()


def normalize_for_display(volume: np.ndarray) -> np.ndarray:
    flat = volume.astype(np.float32, copy=False).ravel()
    lo, hi = np.percentile(flat, (1.0, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((volume.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def load_handoff_contract() -> dict[str, Any]:
    path = _brain_repo() / "simulation" / "handoff_contract.json"
    if not path.exists():
        return {
            "raw_extract": {"output_dir": "data/processed/raw-extract-imaging"},
            "pde_input": {"output_dir": "data/processed/pde-input-solver"},
        }
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def raw_extract_spec() -> dict[str, Any]:
    return dict(load_handoff_contract()["raw_extract"])


def pde_input_spec() -> dict[str, Any]:
    return dict(load_handoff_contract()["pde_input"])


def load_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {data.shape} from {path}")
    volume = np.transpose(data, (2, 1, 0)).astype(np.float32)
    zooms = img.header.get_zooms()[:3]
    spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    return volume, spacing


def load_raw_extract(slug: str) -> tuple[np.ndarray, dict[str, Any], Path]:
    spec = raw_extract_spec()
    out_dir = _repo_path(spec["output_dir"])
    npy_path = out_dir / f"{slug}.npy"
    json_path = out_dir / f"{slug}.json"
    if not npy_path.exists() or not json_path.exists():
        raise FileNotFoundError(f"Missing raw extract for {slug!r} under {out_dir}")
    with json_path.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    return np.load(npy_path).astype(np.float32), meta, json_path


def resolve_segmentation_path(meta: dict[str, Any], slug: str) -> Path | None:
    if meta.get("segmentation_path"):
        path = Path(meta["segmentation_path"])
        if not path.is_absolute():
            path = _brain_repo() / path
        if path.exists():
            return path
    seg_dir = _repo_path("data/processed/segmentations")
    for candidate in (
        seg_dir / f"{slug}_mask.npy",
        seg_dir / f"{slug}.npy",
        seg_dir / f"{slug}_mask.nii.gz",
        seg_dir / f"{slug}.nii.gz",
    ):
        if candidate.exists():
            return candidate
    return None


def load_segmentation(path: Path, target_shape: tuple[int, ...]) -> np.ndarray:
    if path.suffix == ".npy" or path.name.endswith(".npy"):
        mask = np.load(path)
    else:
        mask, _ = load_nifti(path)
    if mask.shape != target_shape:
        raise ValueError(f"Segmentation shape {mask.shape} != MR shape {target_shape}")
    return (mask > 0).astype(np.uint8)


def list_slugs() -> list[str]:
    spec = raw_extract_spec()
    out_dir = _repo_path(spec["output_dir"])
    if not out_dir.exists():
        return []
    slugs: list[str] = []
    for json_path in sorted(out_dir.glob("*.json")):
        slug = json_path.stem
        meta = json.loads(json_path.read_text())
        if resolve_segmentation_path(meta, slug) is not None:
            slugs.append(slug)
    return slugs


def view_demo() -> None:
    import napari

    volume = get_dummy_volume()
    display_mr = normalize_for_display(volume)
    mask = (volume >= 0.35).astype(np.uint8)
    scale = (1.0, 1.0, 1.0)

    viewer = napari.Viewer(title="brain tumor — demo (no dataset yet)")
    viewer.add_image(display_mr, name="MR (synthetic)", scale=scale, colormap="gray",
                     contrast_limits=(0.0, 1.0))
    viewer.add_labels(mask, name="tumor (synthetic)", scale=scale, opacity=0.55)
    print("Demo mode: synthetic MR + mask. No brain dataset required.")
    print("When UCSF/MU-Glioma data arrives, export to brain-cancer-sim/data/ and use --slug.")
    napari.run()


def view_nifti(mr_path: Path, mask_path: Path | None, *, show_pde: bool = False) -> None:
    import napari

    volume, spacing = load_nifti(mr_path)
    display_mr = normalize_for_display(volume)
    scale = tuple(float(s) for s in spacing)

    viewer = napari.Viewer(title=f"{mr_path.name} — napari")
    viewer.add_image(display_mr, name="MR", scale=scale, colormap="gray",
                     contrast_limits=(0.0, 1.0))
    mask_arr: np.ndarray | None = None
    if mask_path:
        mask_arr, _ = load_nifti(mask_path)
        if mask_arr.shape != volume.shape:
            raise ValueError(f"Mask shape {mask_arr.shape} != MR {volume.shape}")
        viewer.add_labels((mask_arr > 0).astype(np.uint8), name=f"seg ({mask_path.name})",
                          scale=scale, opacity=0.55)
    if show_pde and mask_arr is not None:
        pde = normalize_for_display(volume * (mask_arr > 0))
        viewer.add_image(pde, name="PDE preview", scale=scale, opacity=0.4, colormap="magma")
    print(f"MR: {mr_path}  shape={volume.shape}  spacing_mm={spacing}")
    napari.run()


def view_slug(slug: str, *, show_pde: bool = False) -> None:
    import napari

    volume, meta, json_path = load_raw_extract(slug)
    spacing = meta.get("spacing_mm", [1.0, 1.0, 1.0])
    scale = tuple(float(s) for s in spacing)
    display_mr = normalize_for_display(volume)

    seg_path = resolve_segmentation_path(meta, slug)
    if seg_path is None:
        raise FileNotFoundError(f"No segmentation for {slug!r} (see {json_path})")
    labels = load_segmentation(seg_path, volume.shape)

    viewer = napari.Viewer(title=f"{slug} — brain MR + seg")
    viewer.add_image(display_mr, name="MR (normalized)", scale=scale, colormap="gray",
                     contrast_limits=(0.0, 1.0))
    viewer.add_labels(labels, name=f"segmentation ({seg_path.name})",
                      scale=scale, opacity=0.55)

    if show_pde:
        pde_path = _repo_path(pde_input_spec()["output_dir"]) / f"{slug}.npy"
        if pde_path.exists():
            pde = np.load(pde_path).astype(np.float32)
            viewer.add_image(pde, name="PDE input", scale=scale, opacity=0.45,
                             colormap="magma", contrast_limits=(0.0, 1.0))

    print(f"{slug}  shape={volume.shape}  seg_voxels={int(labels.sum()):,}")
    napari.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Brain MR + segmentation napari viewer.")
    parser.add_argument("--slug")
    parser.add_argument("--mr", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--pde-input", action="store_true")
    parser.add_argument("--demo", action="store_true", help="Synthetic demo (also the default with no data)")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        slugs = list_slugs()
        for slug in slugs:
            print(slug)
        if not slugs:
            print("(none yet — use --demo or pull brain-cancer-sim data when ready)")
        return

    if args.demo or (not args.slug and not args.mr):
        view_demo()
        return

    if args.mr:
        view_nifti(args.mr, args.mask, show_pde=args.pde_input)
        return

    slug = args.slug
    if not slug:
        available = list_slugs()
        if len(available) == 1:
            slug = available[0]
        else:
            parser.error("Provide --slug or use --demo. Available: " + ", ".join(available))

    view_slug(slug, show_pde=args.pde_input)


if __name__ == "__main__":
    main()

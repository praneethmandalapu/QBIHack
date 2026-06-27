"""Interactive 3D viewer: brain MR + expert segmentation overlay (napari).

Run (macOS/Linux):
    cd brain-cancer-sim
    python3.11 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt   # includes napari[pyqt6]

    # Synthetic demo (default when no data on disk — no args needed)
    python simulation-vinesh-philip-chandan/philip-chandan/view_volume_napari.py
    python simulation-vinesh-philip-chandan/philip-chandan/view_volume_napari.py --demo

Run (Windows):
    cd brain-cancer-sim
    py -3 -m venv .venv
    .venv\\Scripts\\Activate.ps1
    pip install -r requirements.txt
    .venv\\Scripts\\python.exe simulation-vinesh-philip-chandan\\philip-chandan\\view_volume_napari.py --demo

Requires napari[pyqt6]. First launch may take ~10s while Qt initializes.
Arrays are (Z, Y, X); napari scale is (dz, dy, dx) mm from sidecar metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

IMAGING_DIR = Path(__file__).resolve().parent
SIM_ROOT = IMAGING_DIR.parent
REPO_ROOT = SIM_ROOT.parent

sys.path.insert(0, str(SIM_ROOT))
sys.path.insert(0, str(SIM_ROOT / "vinesh"))

from handoff_contract import load_handoff_contract, pde_input_spec, raw_extract_spec  # noqa: E402
from tumor_pde_solver import dummy_volume  # noqa: E402


def _repo_path(relative: str) -> Path:
    return REPO_ROOT / relative


def normalize_for_display(volume: np.ndarray) -> np.ndarray:
    """Percentile clip to [0, 1] for napari contrast."""
    flat = volume.astype(np.float32, copy=False).ravel()
    lo, hi = np.percentile(flat, (1.0, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((volume.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return out.astype(np.float32)


def load_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    # NIfTI is typically (X, Y, Z) or (X, Y, Z, 1) — reorder to (Z, Y, X)
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
        raise FileNotFoundError(
            f"Missing raw extract for {slug!r}. Expected:\n"
            f"  {npy_path}\n  {json_path}"
        )
    with json_path.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    volume = np.load(npy_path).astype(np.float32)
    return volume, meta, json_path


def resolve_segmentation_path(meta: dict[str, Any], slug: str) -> Path | None:
    if meta.get("segmentation_path"):
        path = Path(meta["segmentation_path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
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
        raise ValueError(
            f"Segmentation shape {mask.shape} != MR shape {target_shape} ({path})"
        )
    labels = (mask > 0).astype(np.uint8)
    return labels


def list_slugs() -> list[str]:
    spec = raw_extract_spec()
    out_dir = _repo_path(spec["output_dir"])
    if not out_dir.exists():
        return []
    slugs: list[str] = []
    for json_path in sorted(out_dir.glob("*.json")):
        slug = json_path.stem
        seg = resolve_segmentation_path(json.loads(json_path.read_text()), slug)
        if seg is not None:
            slugs.append(slug)
    return slugs


def view_demo() -> None:
    import napari

    volume = dummy_volume(shape=(48, 48, 48), radius=10.0)
    display_mr = normalize_for_display(volume)
    mask = (volume >= 0.35).astype(np.uint8)
    scale = (1.0, 1.0, 1.0)

    viewer = napari.Viewer(title="brain-cancer-sim — demo")
    viewer.add_image(display_mr, name="MR (synthetic)", scale=scale, colormap="gray",
                     contrast_limits=(0.0, 1.0))
    viewer.add_labels(mask, name="tumor (synthetic)", scale=scale, opacity=0.55)
    print("Demo mode: synthetic MR + threshold mask. Replace with --slug when data exists.")
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
    if show_pde:
        weight = (mask_arr > 0) if mask_arr is not None else 1.0
        pde = normalize_for_display(volume * weight)
        viewer.add_image(pde, name="PDE preview", scale=scale, opacity=0.4, colormap="magma")
    print(f"MR: {mr_path}\n  shape={volume.shape} spacing_mm={spacing}")
    if mask_path:
        print(f"mask: {mask_path}")
    napari.run()


def view_slug(slug: str, *, show_pde: bool = False) -> None:
    import napari

    volume, meta, json_path = load_raw_extract(slug)
    spacing = meta.get("spacing_mm", [1.0, 1.0, 1.0])
    scale = tuple(float(s) for s in spacing)
    display_mr = normalize_for_display(volume)

    seg_path = resolve_segmentation_path(meta, slug)
    if seg_path is None:
        raise FileNotFoundError(
            f"No segmentation for {slug!r}. Set segmentation_path in {json_path} "
            f"or place mask under data/processed/segmentations/"
        )
    labels = load_segmentation(seg_path, volume.shape)

    viewer = napari.Viewer(title=f"{slug} — brain MR + seg")
    viewer.add_image(display_mr, name="MR (normalized)", scale=scale, colormap="gray",
                     contrast_limits=(0.0, 1.0))
    viewer.add_labels(labels, name=f"segmentation ({seg_path.name})",
                      scale=scale, opacity=0.55)

    if show_pde:
        pde_spec = pde_input_spec()
        pde_path = _repo_path(pde_spec["output_dir"]) / f"{slug}.npy"
        if pde_path.exists():
            pde = np.load(pde_path).astype(np.float32)
            viewer.add_image(pde, name="PDE input", scale=scale, opacity=0.45,
                             colormap="magma", contrast_limits=(0.0, 1.0))
        else:
            print(f"  (no PDE input at {pde_path})")

    print(
        f"{slug}\n"
        f"  dataset:  {meta.get('dataset', '?')}\n"
        f"  patient:  {meta.get('patient_id', '?')}\n"
        f"  MR:       shape={volume.shape} spacing_mm={spacing}\n"
        f"  seg:      {seg_path} voxels={int(labels.sum()):,}"
    )
    napari.run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View brain MR with expert segmentation overlay in napari.",
    )
    parser.add_argument("--slug", help="Raw extract slug under data/processed/raw-extract-philip-chandan/")
    parser.add_argument("--mr", type=Path, help="MR NIfTI path (.nii / .nii.gz)")
    parser.add_argument("--mask", type=Path, help="Segmentation NIfTI path")
    parser.add_argument("--pde-input", action="store_true", help="Overlay PDE-ready volume if present")
    parser.add_argument("--demo", action="store_true", help="Synthetic MR + mask (default when no data)")
    parser.add_argument("--list", action="store_true", help="List slugs with paired segmentations")
    args = parser.parse_args()

    _ = load_handoff_contract()

    if args.list:
        for slug in list_slugs():
            print(slug)
        if not list_slugs():
            print("(none yet — use --demo or export to data/processed/raw-extract-philip-chandan/)")
        return

    if args.demo:
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
        elif available:
            parser.error("Provide --slug. Available: " + ", ".join(available))
        else:
            print("No brain data on disk — opening synthetic demo.")
            view_demo()
            return

    view_slug(slug, show_pde=args.pde_input)


if __name__ == "__main__":
    main()

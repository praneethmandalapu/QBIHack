"""Fill necrotic core inside an aligned-bbox tumor mask (.npy).

Rim-enhancing lesions (bright ring, dark center on post-contrast DCE) often yield a
donut-shaped threshold mask. This post-process turns the ring into a solid lesion
volume by filling enclosed background voxels.

**Is .npy level OK?** Yes — for our pipeline this is the right place. The mask is
already a derived binary segmentation in full ``(Z, Y, X)`` stack coordinates, not
raw DICOM. Hole-filling is standard morphological post-processing on label maps.
Raw MR intensities are unchanged; only the handoff mask used for PDE / radiomics
ROI updates. Commit JSON sidecar updates; keep ``.npy`` local per repo rules.

Fill modes:

``2d`` (default)
    ``binary_fill_holes`` **per slice** inside the ``.les`` cuboid. Best for
    in-plane necrotic cores surrounded by an enhancing rim.

``3d``
    Single 3D fill inside the cuboid. Only fills cavities fully enclosed in 3D
    (often adds fewer voxels when the core opens through slice direction).

Example::

    cd breast-cancer-sim
    .venv/bin/python simulation-vinesh-philip-chandan/philip-chandan/validation/fill_necrotic_core.py \\
      --slug basal_TCGA-AR-A1AQ_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np

VALIDATION_DIR = Path(__file__).resolve().parent
PHILIP_CHANDAN_DIR = VALIDATION_DIR.parent
REPO_ROOT = PHILIP_CHANDAN_DIR.parents[1]
SEGMENTATION_OUT_DIR = REPO_ROOT / "data" / "processed" / "segmentation-philip-chandan"
METHOD_ID = "aligned_bbox_tumor"

sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(VALIDATION_DIR))

FillMode = Literal["2d", "3d"]


def les_bounds_from_meta(meta: dict[str, Any]) -> tuple[slice, slice, slice]:
    return (
        slice(int(meta["z_start"]), int(meta["z_end"]) + 1),
        slice(int(meta["y_start"]), int(meta["y_end"]) + 1),
        slice(int(meta["x_start"]), int(meta["x_end"]) + 1),
    )


def fill_necrotic_core_in_mask(
    mask: np.ndarray,
    *,
    z_slice: slice,
    y_slice: slice,
    x_slice: slice,
    mode: FillMode = "2d",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return filled mask copy + stats (voxels added inside cuboid)."""
    from scipy.ndimage import binary_fill_holes

    if mask.ndim != 3:
        raise ValueError(f"Expected (Z, Y, X) mask, got shape {mask.shape}")

    out = mask.astype(np.uint8, copy=True)
    region = out[z_slice, y_slice, x_slice].astype(bool)
    before = int(region.sum())

    if before == 0:
        return out, {
            "mode": mode,
            "voxels_before": 0,
            "voxels_after": 0,
            "core_voxels_filled": 0,
        }

    if mode == "3d":
        filled = binary_fill_holes(region)
    elif mode == "2d":
        filled = np.zeros_like(region, dtype=bool)
        for index in range(region.shape[0]):
            slab = region[index]
            if slab.any():
                filled[index] = binary_fill_holes(slab)
    else:
        raise ValueError(f"Unknown fill mode: {mode!r}")

    filled_u8 = filled.astype(np.uint8)
    out[z_slice, y_slice, x_slice] = filled_u8
    after = int(filled_u8.sum())
    return out, {
        "mode": mode,
        "voxels_before": before,
        "voxels_after": after,
        "core_voxels_filled": after - before,
    }


def mask_paths(slug: str) -> tuple[Path, Path]:
    base = SEGMENTATION_OUT_DIR / f"{slug}_{METHOD_ID}_mask"
    return base.with_suffix(".npy"), base.with_suffix(".json")


def load_mask_artifact(slug: str) -> tuple[np.ndarray, dict[str, Any], Path, Path]:
    npy_path, json_path = mask_paths(slug)
    if not npy_path.is_file():
        raise FileNotFoundError(f"Mask not found: {npy_path}")
    mask = np.load(npy_path)
    meta: dict[str, Any] = {}
    if json_path.is_file():
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    return mask, meta, npy_path, json_path


def apply_fill_to_slug(
    slug: str,
    *,
    mode: FillMode = "2d",
    dry_run: bool = False,
    output_npy: Path | None = None,
) -> dict[str, Any]:
    mask, meta, npy_path, json_path = load_mask_artifact(slug)
    required = ("z_start", "z_end", "y_start", "y_end", "x_start", "x_end")
    missing = [key for key in required if key not in meta]
    if missing:
        raise ValueError(f"Mask sidecar missing .les bounds {missing}: {json_path}")

    z_sl, y_sl, x_sl = les_bounds_from_meta(meta)
    filled, fill_stats = fill_necrotic_core_in_mask(
        mask,
        z_slice=z_sl,
        y_slice=y_sl,
        x_slice=x_sl,
        mode=mode,
    )

    report = {
        "slug": slug,
        "mask_npy": str(npy_path),
        "full_volume_voxels_before": int(mask.sum()),
        "full_volume_voxels_after": int(filled.sum()),
        **fill_stats,
    }

    if dry_run:
        return report

    out_npy = output_npy or npy_path
    np.save(out_npy, filled)

    meta = dict(meta)
    meta["mask_voxels"] = int(filled[z_sl, y_sl, x_sl].sum())
    meta["mask_npy"] = str(out_npy)
    meta["necrotic_core_fill"] = {
        "method": "binary_fill_holes",
        "mode": mode,
        "cuboid_voxels_before": fill_stats["voxels_before"],
        "cuboid_voxels_after": fill_stats["voxels_after"],
        "core_voxels_filled": fill_stats["core_voxels_filled"],
        "export_source": "fill_necrotic_core.py",
    }
    json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    report["mask_json"] = str(json_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill necrotic core (2D/3D hole fill) in aligned-bbox tumor mask .npy",
    )
    parser.add_argument("--slug", required=True, help="Baseline slug with aligned_bbox_tumor mask")
    parser.add_argument(
        "--mode",
        choices=("2d", "3d"),
        default="2d",
        help="2d=per-slice fill in .les cuboid (default); 3d=single 3D fill in cuboid",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print stats only; do not write")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output .npy (default: overwrite existing mask path)",
    )
    args = parser.parse_args()

    report = apply_fill_to_slug(
        args.slug,
        mode=args.mode,
        dry_run=args.dry_run,
        output_npy=args.output,
    )
    print(
        f"{report['slug']}: cuboid {report['voxels_before']:,} → {report['voxels_after']:,} "
        f"(+{report['core_voxels_filled']:,} core voxels, mode={report['mode']})"
    )
    if not args.dry_run:
        print(f"  wrote {report['mask_npy']}")
        print(f"  updated {report['mask_json']}")


if __name__ == "__main__":
    main()

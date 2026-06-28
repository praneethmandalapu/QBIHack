"""Publish napari aligned-bbox masks to brain-parity segmentations/{slug}_mask.nii.gz."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PHILIP_CHANDAN_DIR = Path(__file__).resolve().parent
SPIKE_ROOT = PHILIP_CHANDAN_DIR.parent
REPO_ROOT = SPIKE_ROOT.parent

sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(SPIKE_ROOT))

from spike_paths import (  # noqa: E402
    resolve_raw_extract_metadata,
    segmentation_mask_path,
)

SEGMENTATION_OUT_DIR = REPO_ROOT / "data" / "processed" / "segmentation-philip-chandan"
ALIGNED_BBOX_METHOD = "aligned_bbox_tumor"


def aligned_bbox_mask_npy(slug: str) -> Path:
    return SEGMENTATION_OUT_DIR / f"{slug}_{ALIGNED_BBOX_METHOD}_mask.npy"


def write_mask_nifti(mask_zyx: np.ndarray, path: Path, spacing_mm: list[float]) -> None:
    """Write (Z,Y,X) binary mask as NIfTI compatible with prepare_pde_input loader."""
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.transpose(np.asarray(mask_zyx > 0, dtype=np.uint8), (2, 1, 0))
    sx, sy, sz = (float(spacing_mm[2]), float(spacing_mm[1]), float(spacing_mm[0]))
    affine = np.array(
        [
            [sx, 0.0, 0.0, 0.0],
            [0.0, sy, 0.0, 0.0],
            [0.0, 0.0, sz, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    nib.save(nib.Nifti1Image(data, affine), str(path))


def _load_spacing_for_slug(slug: str) -> list[float]:
    sidecar = aligned_bbox_mask_npy(slug).with_suffix(".json")
    if sidecar.is_file():
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        spacing = meta.get("spacing_mm")
        if spacing and len(spacing) == 3:
            return [float(v) for v in spacing]
    raw_json = resolve_raw_extract_metadata(slug)
    if raw_json.is_file():
        meta = json.loads(raw_json.read_text(encoding="utf-8"))
        return [float(v) for v in meta["spacing_mm"]]
    raise FileNotFoundError(f"No spacing metadata for slug {slug!r}")


def link_segmentation_in_raw_extract(slug: str, seg_rel: str) -> Path:
    """Set segmentation_path on the raw extract sidecar for this slug."""
    json_path = resolve_raw_extract_metadata(slug)
    if not json_path.is_file():
        raise FileNotFoundError(f"Missing raw extract JSON for {slug}: {json_path}")
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    meta["segmentation_path"] = seg_rel
    json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return json_path


def publish_expert_mask_from_volume(
    slug: str,
    mask_zyx: np.ndarray,
    *,
    spacing_mm: list[float] | None = None,
) -> Path:
    """Write segmentations/{slug}_mask.nii.gz and link raw extract JSON."""
    spacing = spacing_mm or _load_spacing_for_slug(slug)
    out_path = segmentation_mask_path(slug)
    write_mask_nifti(mask_zyx, out_path, spacing)
    link_segmentation_in_raw_extract(slug, str(out_path.relative_to(REPO_ROOT)))
    return out_path


def publish_expert_mask(slug: str) -> Path:
    """Publish existing aligned_bbox .npy mask to segmentations/ + raw JSON link."""
    npy_path = aligned_bbox_mask_npy(slug)
    if not npy_path.is_file():
        raise FileNotFoundError(
            f"Missing aligned bbox mask for {slug}. Expected {npy_path}. "
            "Export from napari first."
        )
    mask = np.load(npy_path)
    return publish_expert_mask_from_volume(slug, mask)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Publish aligned_bbox tumor mask to segmentations/{slug}_mask.nii.gz",
    )
    parser.add_argument("--slug", required=True, help="Cohort slug (baseline with napari mask)")
    args = parser.parse_args(argv)
    out = publish_expert_mask(args.slug)
    print(f"Wrote {out}")
    print(f"Linked segmentation_path in {resolve_raw_extract_metadata(args.slug)}")


if __name__ == "__main__":
    main()

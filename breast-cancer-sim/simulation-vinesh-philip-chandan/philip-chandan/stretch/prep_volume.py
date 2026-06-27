"""Prepare full-resolution raw MR + tumor mask for PyRadiomics (stretch-owned).

Default ROI is TCIA radiologist ``*.les`` masks (see ``load_les_mask.py``).
Otsu + largest connected component remains available via ``mask_source="otsu"``
for validation comparisons only — not for production radiomics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label
from skimage.filters import threshold_otsu

STRETCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STRETCH_DIR))

from load_les_mask import find_les_files, load_les_mask  # noqa: E402
from load_manifest import find_volume  # noqa: E402
from paths import (  # noqa: E402
    ensure_radiomics_dirs,
    radiomics_mask_npy,
    raw_extract_json,
    raw_extract_npy,
)

MASK_SOURCES = ("les", "otsu")


def load_raw_extract(slug: str) -> tuple[np.ndarray, dict[str, Any]]:
    npy_path = raw_extract_npy(slug)
    json_path = raw_extract_json(slug)
    if not npy_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"Missing raw extract for {slug}. Expected {npy_path} and {json_path}"
        )
    volume = np.load(npy_path)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    return volume.astype(np.float32), metadata


def normalize_volume(
    volume: np.ndarray,
    *,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """Clip to percentile range and scale to [0, 1] at full resolution."""
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    lo, hi = np.percentile(finite, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.float32)
    clipped = np.clip(volume, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def tumor_mask_largest_component(norm: np.ndarray) -> np.ndarray:
    """Binary ROI mask: Otsu on nonzero voxels, keep largest connected component."""
    nonzero = norm[norm > 0]
    if nonzero.size == 0 or float(nonzero.max()) <= float(nonzero.min()):
        return np.zeros_like(norm, dtype=np.uint8)
    thresh = threshold_otsu(nonzero)
    candidate = norm > thresh
    labels, n = label(candidate)
    if n == 0:
        return np.zeros_like(norm, dtype=np.uint8)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    largest = int(sizes.argmax())
    return (labels == largest).astype(np.uint8)


def crop_to_mask_bbox(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    margin: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop image and mask to mask bounding box plus margin voxels."""
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return image, mask
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    slices = tuple(
        slice(max(int(lo) - margin, 0), min(int(hi) + margin + 1, size))
        for lo, hi, size in zip(mins, maxs, image.shape)
    )
    return image[slices], mask[slices]


def resolve_les_path(tcga_id: str, *, lesions_dir: Path | None = None) -> Path | None:
    """Return the first local ``*.les`` file for a TCGA patient, or None."""
    les_files = find_les_files(tcga_id, lesions_dir)
    return les_files[0] if les_files else None


def load_les_mask_for_slug(
    slug: str,
    volume_shape: tuple[int, ...],
    *,
    lesions_dir: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load radiologist ``*.les`` ROI embedded in ``(Z, Y, X)`` volume shape."""
    entry = find_volume(slug=slug)
    tcga_id = entry["tcga_id"]
    les_path = resolve_les_path(tcga_id, lesions_dir=lesions_dir)
    if les_path is None:
        raise FileNotFoundError(
            f"No radiologist .les mask for {tcga_id} ({slug}). "
            "TCIA Radiogenomics expert annotations cover baseline DCE only — "
            "follow-up timepoints have no .les file."
        )
    mask, les_meta = load_les_mask(les_path, volume_shape)
    if not mask.any():
        raise ValueError(f".les mask is empty after embed: {les_path.name}")
    return mask, {
        **les_meta,
        "mask_source": "les",
        "les_file": les_path.name,
    }


def numpy_to_sitk(
    image: np.ndarray,
    mask: np.ndarray,
    spacing_mm: list[float],
) -> tuple[sitk.Image, sitk.Image]:
    """Build aligned SimpleITK images from (Z, Y, X) arrays."""
    sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
    sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
    spacing = tuple(float(s) for s in spacing_mm)
    sitk_image.SetSpacing(spacing)
    sitk_mask.SetSpacing(spacing)
    return sitk_image, sitk_mask


def prep_for_radiomics(
    slug: str,
    *,
    crop: bool = True,
    margin: int = 10,
    save_mask: bool = True,
) -> tuple[sitk.Image, sitk.Image, dict[str, Any]]:
    """Load raw extract, normalize, segment, optionally crop; return SITK pair + metadata."""
    volume, sidecar = load_raw_extract(slug)
    manifest_entry = find_volume(slug=slug)
    spacing_mm = list(sidecar["spacing_mm"])

    norm = normalize_volume(volume)
    mask = tumor_mask_largest_component(norm)
    if crop:
        norm, mask = crop_to_mask_bbox(norm, mask, margin=margin)

    sitk_image, sitk_mask = numpy_to_sitk(norm, mask, spacing_mm)

    mask_path = None
    if save_mask:
        ensure_radiomics_dirs()
        mask_path = radiomics_mask_npy(slug)
        np.save(mask_path, mask)

    metadata = {
        "slug": slug,
        "tcga_id": manifest_entry.get("tcga_id", sidecar.get("tcga_id")),
        "subtype": manifest_entry.get("subtype", sidecar.get("subtype")),
        "timepoint": manifest_entry.get("timepoint"),
        "study_date": manifest_entry.get("study_date", sidecar.get("study_date")),
        "source_raw_npy": str(raw_extract_npy(slug)),
        "shape": list(norm.shape),
        "spacing_mm": spacing_mm,
        "mask_voxels": int(mask.sum()),
        "mask_fraction": float(mask.mean()),
        "cropped": crop,
        "mask_npy": str(mask_path) if mask_path else None,
    }
    return sitk_image, sitk_mask, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prep raw MR + tumor mask for PyRadiomics.")
    parser.add_argument(
        "--slug",
        default="luminal_a_TCGA-AR-A1AX_baseline",
        help="Manifest slug (default: spike baseline)",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Keep full FOV (slower PyRadiomics on large follow-ups)",
    )
    args = parser.parse_args()

    _, _, meta = prep_for_radiomics(args.slug, crop=not args.no_crop)
    print(f"Prepared {args.slug}")
    print(f"  shape={meta['shape']} spacing_mm={meta['spacing_mm']}")
    print(f"  mask_voxels={meta['mask_voxels']} fraction={meta['mask_fraction']:.4f}")
    if meta["mask_npy"]:
        print(f"  wrote {meta['mask_npy']}")


if __name__ == "__main__":
    main()

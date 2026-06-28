"""Embed TCIA radiologist .les masks as segmentation reference outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SEGMENTATION_DIR = Path(__file__).resolve().parent
STRETCH_DIR = SEGMENTATION_DIR.parent / "stretch"
PHILIP_CHANDAN_DIR = SEGMENTATION_DIR.parent

sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(STRETCH_DIR))
sys.path.insert(0, str(SEGMENTATION_DIR))

from load_les_mask import find_les_files, load_les_mask  # noqa: E402
from load_manifest import find_volume  # noqa: E402
from validate_segmentation import load_annotation_volume  # noqa: E402

from seg_paths import ensure_segmentation_dirs, mask_metadata_json, mask_npy  # noqa: E402


def write_les_reference(slug: str, *, lesions_dir: Path | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Embed .les ground truth for a baseline slug and write mask + sidecar JSON."""
    entry = find_volume(slug=slug)
    tcga_id = entry["tcga_id"]
    study_date = entry["study_date"]

    les_files = find_les_files(tcga_id, lesions_dir)
    if not les_files:
        raise FileNotFoundError(f"No .les file for {tcga_id} ({slug})")

    les_path = les_files[0]
    volume, spacing_mm, series_description, dce_index, volume_source = load_annotation_volume(
        tcga_id=tcga_id,
        study_date=study_date,
        les_path=les_path,
        slug=slug,
    )
    mask, les_meta = load_les_mask(les_path, volume.shape)
    if not mask.any():
        raise ValueError(f"Empty .les mask after embed: {les_path.name}")

    ensure_segmentation_dirs()
    mask_path = mask_npy(slug, "les")
    meta_path = mask_metadata_json(slug, "les")
    np.save(mask_path, mask)

    metadata: dict[str, Any] = {
        "slug": slug,
        "method": "les",
        "role": "reference",
        "tcga_id": tcga_id,
        "subtype": entry.get("subtype"),
        "timepoint": entry.get("timepoint"),
        "study_date": study_date,
        "les_file": les_path.name,
        "dce_index": dce_index,
        "annotated_series": series_description,
        "volume_source": volume_source,
        "shape_zyx": list(volume.shape),
        "spacing_mm": spacing_mm,
        "mask_voxels": int(mask.sum()),
        "mask_npy": str(mask_path),
        **les_meta,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return mask, metadata

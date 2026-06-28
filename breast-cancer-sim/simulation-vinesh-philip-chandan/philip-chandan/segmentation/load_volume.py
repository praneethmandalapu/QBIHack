"""Load aligned MR volumes for segmentation (raw extract + manifest metadata)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from seg_paths import raw_extract_json, raw_extract_npy


def load_slug_volume(slug: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (Z, Y, X) float32 volume and sidecar metadata for a manifest slug."""
    npy_path = raw_extract_npy(slug)
    json_path = raw_extract_json(slug)
    if not npy_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"Missing raw extract for {slug}. Expected {npy_path} and {json_path}"
        )
    volume = np.load(npy_path).astype(np.float32)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    metadata["slug"] = slug
    metadata["shape_zyx"] = list(volume.shape)
    return volume, metadata

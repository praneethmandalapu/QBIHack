"""Load Philip-Chandan raw extract and prepare PDE-ready input (Vinesh-owned)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

VINESH_DIR = Path(__file__).resolve().parent
SPIKE_ROOT = VINESH_DIR.parent
sys.path.insert(0, str(SPIKE_ROOT))

from spike_paths import (  # noqa: E402
    SPIKE_PATIENT,
    ensure_spike_dirs,
    pde_input_metadata,
    pde_input_npy,
    raw_extract_metadata,
    raw_extract_npy,
)

DEFAULT_MAX_SHAPE = 64
DEFAULT_TARGET_SPACING_MM = 1.0


def load_raw_extract(slug: str | None = None) -> tuple[np.ndarray, dict]:
    name = slug or SPIKE_PATIENT["slug"]
    npy_path = raw_extract_npy(name)
    json_path = raw_extract_metadata(name)
    if not npy_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"Missing raw extract. Ask Philip-Chandan to run export_raw_extract.py "
            f"(expected {npy_path} and {json_path})"
        )
    volume = np.load(npy_path)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    return volume, metadata


def prepare_pde_input(
    volume: np.ndarray,
    spacing_mm: list[float],
    *,
    max_shape: int = DEFAULT_MAX_SHAPE,
    target_spacing_mm: float = DEFAULT_TARGET_SPACING_MM,
) -> tuple[np.ndarray, list[float]]:
    """Resample, crop, and normalize for solve_growth. Implement for the spike."""
    raise NotImplementedError(
        "Vinesh: resample with scipy.ndimage.zoom using spacing_mm, "
        f"crop/downsample to <= {max_shape}^3, normalize to [0, 1], "
        f"target spacing {target_spacing_mm} mm"
    )


def save_pde_input(
    volume: np.ndarray,
    spacing_mm: list[float],
    raw_metadata: dict,
    *,
    slug: str | None = None,
) -> tuple[Path, Path]:
    ensure_spike_dirs()
    name = slug or SPIKE_PATIENT["slug"]
    npy_path = pde_input_npy(name)
    json_path = pde_input_metadata(name)

    np.save(npy_path, volume.astype(np.float32))
    metadata = {
        "slug": name,
        "source_raw_extract": str(raw_extract_npy(name).relative_to(SPIKE_ROOT.parent)),
        "shape": list(volume.shape),
        "dtype": "float32",
        "spacing_mm": spacing_mm,
        "value_semantics": {
            "0": "background/healthy",
            ">0": "initial tumor burden",
        },
        "upstream": {
            "tcga_id": raw_metadata.get("tcga_id"),
            "study_date": raw_metadata.get("study_date"),
        },
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return npy_path, json_path


def main() -> None:
    raw_volume, raw_metadata = load_raw_extract()
    spacing_mm = raw_metadata["spacing_mm"]
    pde_volume, pde_spacing = prepare_pde_input(raw_volume, spacing_mm)
    npy_path, json_path = save_pde_input(pde_volume, pde_spacing, raw_metadata)
    print(f"Wrote {npy_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()

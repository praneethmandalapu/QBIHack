"""Load Philip-Chandan raw extract and prepare PDE-ready input (Vinesh-owned)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

VINESH_DIR = Path(__file__).resolve().parent
SPIKE_ROOT = VINESH_DIR.parent
sys.path.insert(0, str(SPIKE_ROOT))

from handoff_contract import (  # noqa: E402
    contract_version,
    max_shape,
    pde_input_spec,
    target_spacing_mm,
)
from spike_paths import (  # noqa: E402
    SPIKE_PATIENT,
    ensure_spike_dirs,
    pde_input_metadata,
    pde_input_npy,
    raw_extract_metadata,
    raw_extract_npy,
)


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
    max_shape_xyz: tuple[int, int, int] | None = None,
    target_spacing: list[float] | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Resample, crop, and normalize for solve_growth. Implement for the spike."""
    pde_spec = pde_input_spec()
    shape_limit = max_shape_xyz or max_shape()
    spacing_target = target_spacing or target_spacing_mm()
    value_range = pde_spec["value_range"]
    raise NotImplementedError(
        "Vinesh: resample with scipy.ndimage.zoom using spacing_mm, "
        f"crop/downsample to {shape_limit}, normalize to {value_range}, "
        f"target spacing {spacing_target} mm, segmentation={pde_spec['segmentation']['method']}"
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

    pde_spec = pde_input_spec()
    np.save(npy_path, volume.astype(np.float32))
    metadata = {
        "contract_version": contract_version(),
        "slug": name,
        "source_raw_extract": str(raw_extract_npy(name).relative_to(SPIKE_ROOT.parent)),
        "shape": list(volume.shape),
        "dtype": pde_spec["dtype"],
        "axis_order": pde_spec["axis_order"],
        "spacing_mm": spacing_mm,
        "value_range": pde_spec["value_range"],
        "background_value": pde_spec["background_value"],
        "tumor_burden_rule": pde_spec["tumor_burden_rule"],
        "value_semantics": {
            str(pde_spec["background_value"]): "background/healthy",
            ">0": "initial tumor burden",
        },
        "upstream": {
            "tcga_id": raw_metadata.get("tcga_id"),
            "study_date": raw_metadata.get("study_date"),
            "raw_contract_version": raw_metadata.get("contract_version"),
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

"""Export raw DICOM extract for the Option B Vinesh handoff."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PHILIP_CHANDAN_DIR = Path(__file__).resolve().parent
SPIKE_ROOT = PHILIP_CHANDAN_DIR.parent
sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(SPIKE_ROOT))

from handoff_contract import contract_version, raw_extract_spec  # noqa: E402
from spike_paths import (  # noqa: E402
    SPIKE_PATIENT,
    ensure_spike_dirs,
    raw_extract_metadata,
    raw_extract_npy,
)
from tcia_extractor import (  # noqa: E402
    extract_volume_with_spacing_for_timepoint,
    resolve_study_dir,
    validate_series,
)


def export_raw_extract(
    tcga_id: str,
    subtype: str,
    study_date: str,
    *,
    slug: str | None = None,
) -> tuple[Path, Path]:
    """Write raw float32 volume + JSON sidecar for Vinesh."""
    ensure_spike_dirs()

    dicom_dir = resolve_study_dir(tcga_id, subtype, study_date)
    report = validate_series(dicom_dir)
    if not report["ok"]:
        raise ValueError(f"{dicom_dir}: {'; '.join(report['errors'])}")

    volume, spacing_mm = extract_volume_with_spacing_for_timepoint(
        tcga_id,
        subtype,
        study_date,
    )

    output_slug = slug or SPIKE_PATIENT["slug"]
    npy_path = raw_extract_npy(output_slug)
    json_path = raw_extract_metadata(output_slug)

    tmp_npy = npy_path.with_suffix(".tmp.npy")
    np.save(tmp_npy, volume)
    tmp_npy.replace(npy_path)

    raw_spec = raw_extract_spec()
    metadata = {
        "contract_version": contract_version(),
        "slug": output_slug,
        "tcga_id": tcga_id,
        "subtype": subtype,
        "study_date": study_date,
        "source_dicom_dir": str(dicom_dir.relative_to(SPIKE_ROOT.parent)),
        "shape": list(volume.shape),
        "dtype": str(volume.dtype),
        "spacing_mm": spacing_mm,
        "axis_order": raw_spec["axis_order"],
        "normalize": raw_spec["normalize"],
        "value_semantics": raw_spec["value_semantics"],
        "handoff": "Option B — Vinesh resamples in prepare_pde_input.py",
        "validate_series": {
            "n_slices": report["n_slices"],
            "ok": report["ok"],
        },
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return npy_path, json_path


def main() -> None:
    patient = SPIKE_PATIENT
    npy_path, json_path = export_raw_extract(
        patient["tcga_id"],
        patient["subtype"],
        patient["study_date"],
        slug=patient["slug"],
    )
    print(f"Wrote {npy_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()

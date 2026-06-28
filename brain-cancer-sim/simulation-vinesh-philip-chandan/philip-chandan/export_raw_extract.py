"""Export raw NIfTI extract + JSON sidecar for the Vinesh handoff."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

PHILIP_CHANDAN_DIR = Path(__file__).resolve().parent
SIM_ROOT = PHILIP_CHANDAN_DIR.parent
REPO_ROOT = SIM_ROOT.parent

sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(SIM_ROOT))

from handoff_contract import contract_version, raw_extract_spec, spike_patient  # noqa: E402
from nifti_extractor import (  # noqa: E402
    extract_volume_with_spacing,
    load_expert_mask,
    validate_nifti_pair,
)
from spike_paths import (  # noqa: E402
    ensure_spike_dirs,
    raw_extract_metadata,
    raw_extract_npy,
    segmentation_mask_path,
)


def export_raw_extract(
    mr_path: Path,
    seg_path: Path,
    *,
    slug: str,
    patient_id: str,
    dataset: str,
    disease: str,
    timepoint: str,
    study_date: str,
    copy_mask: bool = True,
) -> tuple[Path, Path]:
    """Write raw float32 MR volume + contract JSON; optionally copy expert mask."""
    ensure_spike_dirs()

    report = validate_nifti_pair(mr_path, seg_path)
    if not report["ok"]:
        raise ValueError(f"{mr_path}: {'; '.join(report['errors'])}")

    volume, spacing_mm = extract_volume_with_spacing(mr_path)
    _ = load_expert_mask(seg_path, volume.shape)

    npy_path = raw_extract_npy(slug, patient_id=patient_id, timepoint=timepoint)
    json_path = raw_extract_metadata(slug, patient_id=patient_id, timepoint=timepoint)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_npy = npy_path.with_suffix(".tmp.npy")
    np.save(tmp_npy, volume)
    tmp_npy.replace(npy_path)

    if copy_mask:
        mask_out = segmentation_mask_path(slug)
        shutil.copy2(seg_path, mask_out)
        seg_ref = str(mask_out.relative_to(REPO_ROOT))
    else:
        seg_ref = str(seg_path.relative_to(REPO_ROOT))

    raw_spec = raw_extract_spec()
    metadata: dict[str, Any] = {
        "contract_version": contract_version(),
        "slug": slug,
        "patient_id": patient_id,
        "dataset": dataset,
        "disease": disease,
        "timepoint": timepoint,
        "study_date": study_date,
        "source_path": str(mr_path.relative_to(REPO_ROOT)),
        "shape": list(volume.shape),
        "dtype": str(volume.dtype),
        "spacing_mm": list(spacing_mm),
        "axis_order": raw_spec["axis_order"],
        "normalize": raw_spec["normalize"],
        "value_semantics": raw_spec["value_semantics"],
        "segmentation_path": seg_ref,
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return npy_path, json_path


def export_spike_patient() -> tuple[Path, Path]:
    """Export the locked spike patient from handoff_contract.json + cohort paths."""
    from nifti_extractor import resolve_ucsf_paths

    patient = spike_patient()
    patient_id = patient["patient_id"]
    timepoint = patient.get("timepoint", "baseline")
    patient_dir = REPO_ROOT / "data" / "raw" / "ucsf_alptdg" / patient_id
    mr_path, seg_path = resolve_ucsf_paths(patient_dir, timepoint)

    return export_raw_extract(
        mr_path,
        seg_path,
        slug=patient["slug"],
        patient_id=patient_id,
        dataset=patient["dataset"],
        disease=patient["disease"],
        timepoint=timepoint,
        study_date="time1" if timepoint == "baseline" else "time2",
    )


def main() -> None:
    npy_path, json_path = export_spike_patient()
    print(f"Wrote {npy_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()

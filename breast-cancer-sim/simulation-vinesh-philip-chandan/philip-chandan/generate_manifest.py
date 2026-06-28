"""Build manifest.json from raw-extract sidecars and cohort.json."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

PHILIP_CHANDAN_DIR = Path(__file__).resolve().parent
SPIKE_ROOT = PHILIP_CHANDAN_DIR.parent
REPO_ROOT = SPIKE_ROOT.parent
MANIFEST_PATH = REPO_ROOT / "data/processed/raw-extract-philip-chandan/manifest.json"

sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(SPIKE_ROOT))

from export_all_raw import build_slug  # noqa: E402
from handoff_contract import contract_version, default_grid_size  # noqa: E402
from spike_paths import (  # noqa: E402
    RAW_EXTRACT_PHILIP_CHANDAN,
    longitudinal_slice_plot_path,
    raw_extract_metadata_legacy,
    raw_extract_npy_legacy,
    resolve_pde_input_metadata,
    resolve_pde_input_npy,
    resolve_raw_extract_metadata,
    resolve_raw_extract_npy,
    slice_plot_path,
    slug_to_tcga_timepoint,
)
from tcia_extractor import load_cohort  # noqa: E402

MANIFEST_VERSION = "1.2.0"


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def _discover_raw_sidecars() -> list[tuple[Path, dict[str, Any]]]:
    entries: list[tuple[Path, dict[str, Any]]] = []
    seen_slugs: set[str] = set()

    for patient_dir in sorted(RAW_EXTRACT_PHILIP_CHANDAN.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name.startswith("."):
            continue
        for json_path in sorted(patient_dir.glob("*.json")):
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            slug = str(meta.get("slug") or "")
            if not slug:
                continue
            seen_slugs.add(slug)
            entries.append((json_path, meta))

    for json_path in sorted(RAW_EXTRACT_PHILIP_CHANDAN.glob("*.json")):
        if json_path.name in ("manifest.json",):
            continue
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        slug = str(meta.get("slug") or json_path.stem)
        if slug in seen_slugs:
            continue
        if raw_extract_npy_legacy(slug).is_file():
            seen_slugs.add(slug)
            entries.append((json_path, meta))

    return sorted(entries, key=lambda item: item[1].get("slug", ""))


def _volume_entry(json_path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    slug = str(meta["slug"])
    tcga_id = str(meta.get("tcga_id") or slug_to_tcga_timepoint(slug)[0])
    timepoint = str(meta.get("timepoint") or slug_to_tcga_timepoint(slug)[1])
    raw_npy = resolve_raw_extract_npy(slug)
    raw_json = resolve_raw_extract_metadata(slug)
    pde_npy = resolve_pde_input_npy(slug)
    pde_json = resolve_pde_input_metadata(slug)
    pde_meta: dict[str, Any] = {}
    if pde_json.is_file():
        pde_meta = json.loads(pde_json.read_text(encoding="utf-8"))

    return {
        "slug": slug,
        "subtype": meta.get("subtype"),
        "tcga_id": tcga_id,
        "timepoint": timepoint,
        "study_date": meta.get("study_date"),
        "raw_npy": _rel(raw_npy),
        "raw_json": _rel(raw_json),
        "pde_npy": _rel(pde_npy) if pde_npy.is_file() else None,
        "pde_json": _rel(pde_json) if pde_json.is_file() else None,
        "shape": meta.get("shape"),
        "spacing_mm": meta.get("spacing_mm"),
        "pde_shape": pde_meta.get("shape"),
        "pde_spacing_mm": pde_meta.get("spacing_mm"),
        "grid_size": pde_meta.get("grid_size", default_grid_size()),
        "segmentation_path": meta.get("segmentation_path"),
        "qc_plot": _rel(slice_plot_path(slug)),
    }


def _patient_entries(volumes: list[dict[str, Any]], cohort: dict[str, Any]) -> list[dict[str, Any]]:
    by_tcga: dict[str, dict[str, Any]] = {}
    for vol in volumes:
        tcga_id = str(vol["tcga_id"])
        by_tcga.setdefault(tcga_id, {"tcga_id": tcga_id, "subtype": vol.get("subtype")})
        if vol["timepoint"] == "baseline":
            by_tcga[tcga_id]["baseline_slug"] = vol["slug"]
            by_tcga[tcga_id]["baseline_study_date"] = vol.get("study_date")
        elif vol["timepoint"] == "followup":
            by_tcga[tcga_id]["followup_slug"] = vol["slug"]
            by_tcga[tcga_id]["followup_study_date"] = vol.get("study_date")

    patients: list[dict[str, Any]] = []
    for patient in cohort.get("primary", []):
        tcga_id = str(patient["tcga_id"])
        row = by_tcga.get(tcga_id, {"tcga_id": tcga_id, "subtype": patient.get("subtype")})
        timepoints = patient.get("imaging", {}).get("timepoints", [])
        baseline_tp = next((tp for tp in timepoints if tp.get("label") == "baseline"), None)
        followup_tp = next((tp for tp in timepoints if tp.get("label") == "followup"), None)
        if baseline_tp and followup_tp:
            row["interval_days"] = int(followup_tp["relative_day"]) - int(baseline_tp["relative_day"])
        if "baseline_slug" not in row and baseline_tp:
            row["baseline_slug"] = build_slug(patient, baseline_tp)
        if "followup_slug" not in row and followup_tp:
            row["followup_slug"] = build_slug(patient, followup_tp)
        long_qc = longitudinal_slice_plot_path(tcga_id)
        if long_qc.is_file():
            row["longitudinal_qc_plot"] = _rel(long_qc)
        patients.append(row)
    return patients


def build_manifest() -> dict[str, Any]:
    cohort = load_cohort()
    sidecars = _discover_raw_sidecars()
    volumes = [_volume_entry(json_path, meta) for json_path, meta in sidecars]
    return {
        "version": MANIFEST_VERSION,
        "contract_version": contract_version(),
        "generated_at": date.today().isoformat(),
        "description": "Philip-Chandan raw extract index with nested patient volume paths",
        "patients": _patient_entries(volumes, cohort),
        "volumes": volumes,
    }


def write_manifest(path: Path = MANIFEST_PATH) -> Path:
    manifest = build_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    path = write_manifest()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    print(f"Wrote {path}")
    print(f"  patients: {len(manifest.get('patients', []))}")
    print(f"  volumes: {len(manifest.get('volumes', []))}")


if __name__ == "__main__":
    main()

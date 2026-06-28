"""Export imaging-cohort genomic risk scores for brain visualization handoff.

Reads philip-chandan cohort.json, looks up scores via neuropulse, writes
patients.csv next to this script. Safe to re-run after model or cohort updates.

    cd brain-cancer-sim
    ../breast-cancer-sim/.venv/bin/python visualization-jasim/risk/export_risk.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BRAIN = HERE.parents[1]
COHORT = (
    BRAIN
    / "simulation-vinesh-philip-chandan"
    / "philip-chandan"
    / "cohort"
    / "cohort.json"
)
OUT = HERE / "patients.csv"

sys.path.insert(0, str(BRAIN))
from neuropulse import get_patient, growth_multiplier  # noqa: E402


def _roles(cohort: dict) -> dict[str, str]:
    roles: dict[str, str] = {}
    for entry in cohort.get("primary", []):
        roles[str(entry["patient_id"])] = "primary"
    for entry in cohort.get("patients", []):
        pid = str(entry["patient_id"])
        if pid in roles:
            continue
        if entry.get("backup"):
            roles[pid] = "backup"
        else:
            roles[pid] = "imaging_cohort"
    return roles


def main() -> int:
    cohort = json.loads(COHORT.read_text())
    roles = _roles(cohort)
    seen: set[str] = set()
    rows: list[dict] = []

    for source in (cohort.get("primary", []), cohort.get("patients", [])):
        for entry in source:
            pid = str(entry["patient_id"])
            if pid in seen:
                continue
            seen.add(pid)
            scored = get_patient(int(pid))
            growth = entry.get("measured_growth") or {}
            wt_growth_pct = growth.get("wt_growth_pct")
            actually_grew = ""
            if wt_growth_pct is not None:
                actually_grew = str(float(wt_growth_pct) > 0)
            rows.append(
                {
                    "patient_id": pid,
                    "idh": scored["idh"] or "",
                    "grade": scored["grade"] if scored["grade"] is not None else "",
                    "mgmt": scored["mgmt"] or "",
                    "diagnosis": scored["diagnosis"] or "",
                    "risk": round(scored["risk"], 4),
                    "growth_multiplier": round(growth_multiplier(int(pid)), 4),
                    "wt_growth_pct": wt_growth_pct if wt_growth_pct is not None else "",
                    "actually_grew": actually_grew,
                    "cohort_role": roles.get(pid, "imaging_cohort"),
                }
            )

    rows.sort(key=lambda r: int(r["patient_id"]))
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "patient_id",
                "idh",
                "grade",
                "mgmt",
                "diagnosis",
                "risk",
                "growth_multiplier",
                "wt_growth_pct",
                "actually_grew",
                "cohort_role",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT} ({len(rows)} patients)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

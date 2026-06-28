"""Export imaging-cohort genomic risk scores for breast visualization handoff.

Reads philip-chandan cohort.json, looks up scores via oncopulse, writes
patients.csv next to this script. Safe to re-run after model or cohort updates.

    cd breast-cancer-sim
    .venv/bin/python visualization-jasim/risk/export_risk.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BREAST = HERE.parents[1]
REPO = BREAST.parent
COHORT = (
    BREAST
    / "simulation-vinesh-philip-chandan"
    / "philip-chandan"
    / "cohort"
    / "cohort.json"
)
OUT = HERE / "patients.csv"

sys.path.insert(0, str(BREAST))
from oncopulse import get_patient, growth_multiplier  # noqa: E402


def _roles(cohort: dict) -> dict[str, str]:
    roles: dict[str, str] = {}
    for entry in cohort.get("primary", []):
        roles[entry["tcga_id"]] = "primary"
    for subtype, backups in cohort.get("backups", {}).items():
        for entry in backups:
            bc = entry["tcga_id"]
            if bc not in roles:
                roles[bc] = f"backup_{subtype.replace(' ', '_').lower()}"
    return roles


def main() -> int:
    cohort = json.loads(COHORT.read_text())
    roles = _roles(cohort)
    rows: list[dict] = []

    for entry in cohort.get("primary", []):
        bc = entry["tcga_id"]
        scored = get_patient(bc)
        rows.append(
            {
                "patient_id": bc,
                "subtype": entry["subtype"],
                "pam50": scored["pam50"],
                "risk": round(scored["risk"], 5),
                "growth_multiplier": round(growth_multiplier(bc), 5),
                "cohort_role": roles.get(bc, "primary"),
            }
        )

    rows.sort(key=lambda r: r["patient_id"])
    with OUT.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "patient_id",
                "subtype",
                "pam50",
                "risk",
                "growth_multiplier",
                "cohort_role",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT} ({len(rows)} patients)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Download Option B spike imaging data (gitignored — not pushed to GitHub).

Share this script with Vinesh or anyone cloning the repo without ``data/``.

Spike case (from ``handoff_contract.json``):

| Field | Value |
|-------|-------|
| TCGA ID | ``TCGA-AR-A1AX`` |
| Subtype | Luminal A |
| Baseline study | ``2002-09-12`` |
| DICOM output | ``data/raw/tcia/luminal_a/TCGA-AR-A1AX/2002-09-12/`` |
| Raw handoff slug | ``luminal_a_TCGA-AR-A1AX_baseline`` |

**Vinesh (Option B):** you normally only need the raw extract, not DICOM::

    data/processed/raw-extract-philip-chandan/TCGA-AR-A1AX/baseline.npy
    data/processed/raw-extract-philip-chandan/TCGA-AR-A1AX/baseline.json

Ask Philip-Chandan to share those two files (Drive/Slack), or run this script with
``--export-raw`` after downloading DICOM locally.

Setup once (macOS/Linux)::

    cd breast-cancer-sim
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

Setup once (Windows / Vinesh)::

    cd breast-cancer-sim
    .\\simulation-vinesh-philip-chandan\\download_spike_data.ps1

Download baseline DICOM (~350 slices, ~50 MB)::

    python simulation-vinesh-philip-chandan/download_spike_data.py

Also build raw extract for ``prepare_pde_input.py``::

    python simulation-vinesh-philip-chandan/download_spike_data.py --export-raw

Windows one-liner (download + export)::

    .\\simulation-vinesh-philip-chandan\\download_spike_data.ps1 -ExportRaw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SPIKE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SPIKE_ROOT.parent
PHILIP_CHANDAN_DIR = SPIKE_ROOT / "philip-chandan"

sys.path.insert(0, str(SPIKE_ROOT))
sys.path.insert(0, str(PHILIP_CHANDAN_DIR))

from handoff_contract import spike_patient  # noqa: E402
from download_tcia import download_patient_longitudinal  # noqa: E402
from spike_paths import ensure_spike_dirs, raw_extract_metadata, raw_extract_npy  # noqa: E402

# Follow-up exists on TCIA but is not required for the spike baseline.
SPIKE_FOLLOWUP_STUDY_DATE = "2003-09-24"


def download_spike_dicom(*, include_followup: bool = False) -> list[dict]:
    """Download MR DICOM for the spike patient from TCIA NBIA."""
    patient = spike_patient()
    study_dates = [patient["study_date"]]
    if include_followup:
        study_dates.append(SPIKE_FOLLOWUP_STUDY_DATE)

    return download_patient_longitudinal(
        patient["tcga_id"],
        patient["subtype"],
        prefer_contrast=True,
        study_dates=study_dates,
    )


def export_spike_raw_extract() -> tuple[Path, Path]:
    """Build raw .npy + .json handoff from local DICOM (Philip-Chandan export)."""
    from export_raw_extract import export_raw_extract

    patient = spike_patient()
    return export_raw_extract(
        patient["tcga_id"],
        patient["subtype"],
        patient["study_date"],
        slug=patient["slug"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download TCIA DICOM for the Option B spike case (TCGA-AR-A1AX baseline).",
    )
    parser.add_argument(
        "--include-followup",
        action="store_true",
        help=f"Also download follow-up study {SPIKE_FOLLOWUP_STUDY_DATE} (~200 MB extra).",
    )
    parser.add_argument(
        "--export-raw",
        action="store_true",
        help="After download, run export_raw_extract.py for Vinesh handoff (.npy + .json).",
    )
    parser.add_argument(
        "--export-raw-only",
        action="store_true",
        help="Skip download; only export raw extract (DICOM already on disk).",
    )
    args = parser.parse_args()

    patient = spike_patient()
    ensure_spike_dirs()

    print("Spike case:")
    print(f"  tcga_id:    {patient['tcga_id']}")
    print(f"  subtype:    {patient['subtype']}")
    print(f"  study_date: {patient['study_date']}")
    print(f"  slug:       {patient['slug']}")
    print()

    if not args.export_raw_only:
        print("Downloading from TCIA NBIA (public TCGA-BRCA MR)...")
        results = download_spike_dicom(include_followup=args.include_followup)
        for result in results:
            print(
                f"OK {result['study_date']}: {result['series_description']} "
                f"({result['image_count']} slices) → {result['dicom_dir']}"
            )
        print()

    if args.export_raw or args.export_raw_only:
        print("Exporting raw extract for Vinesh...")
        npy_path, json_path = export_spike_raw_extract()
        print(f"Wrote {npy_path}")
        print(f"Wrote {json_path}")
        print()
        print("Vinesh next step:")
        print("  python simulation-vinesh-philip-chandan/vinesh/prepare_pde_input.py")
        return

    print("Raw handoff paths (after --export-raw):")
    print(f"  {raw_extract_npy()}")
    print(f"  {raw_extract_metadata()}")


if __name__ == "__main__":
    main()

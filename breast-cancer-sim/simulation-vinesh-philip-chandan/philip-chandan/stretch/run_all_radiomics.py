"""Batch radiomics extraction over manifest slugs → features_all.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

STRETCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STRETCH_DIR))

from extract_radiomics import BACKENDS, extract_for_slug  # noqa: E402
from load_manifest import load_volumes  # noqa: E402
from paths import DEFAULT_PARAMS_PATH, ensure_radiomics_dirs, features_all_csv  # noqa: E402

METADATA_COLUMNS = ("slug", "tcga_id", "subtype", "timepoint", "study_date", "backend")


def features_to_row(
    features: dict[str, float],
    meta: dict[str, Any],
    *,
    backend: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "slug": meta["slug"],
        "tcga_id": meta.get("tcga_id"),
        "subtype": meta.get("subtype"),
        "timepoint": meta.get("timepoint"),
        "study_date": meta.get("study_date"),
        "backend": backend,
    }
    row.update(features)
    return row


def run_all(
    *,
    backend: str = "pyradiomics",
    crop: bool = True,
    params_path: Path | None = None,
    bin_width: float = 0.05,
    device: str = "auto",
    slugs: list[str] | None = None,
) -> Path:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; choose from {BACKENDS}")

    ensure_radiomics_dirs()
    volumes = load_volumes()
    if slugs is not None:
        slug_set = set(slugs)
        volumes = [entry for entry in volumes if entry["slug"] in slug_set]

    rows: list[dict[str, Any]] = []
    for entry in volumes:
        slug = entry["slug"]
        features, meta = extract_for_slug(
            slug,
            backend=backend,
            crop=crop,
            params_path=params_path,
            bin_width=bin_width,
            device=device,
        )
        rows.append(features_to_row(features, meta, backend=backend))
        print(f"  {slug}: {len(features)} features")

    out_path = features_all_csv()
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract radiomics for all manifest slugs.")
    parser.add_argument("--backend", choices=BACKENDS, default="pyradiomics")
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--bin-width", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--slug", action="append", dest="slugs", help="Limit to slug(s)")
    args = parser.parse_args()

    print(f"Running radiomics batch ({args.backend})...")
    out_path = run_all(
        backend=args.backend,
        crop=not args.no_crop,
        params_path=args.params,
        bin_width=args.bin_width,
        device=args.device,
        slugs=args.slugs,
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

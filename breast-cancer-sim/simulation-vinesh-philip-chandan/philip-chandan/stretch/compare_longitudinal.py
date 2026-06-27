"""Compute baseline → follow-up feature deltas from features_all.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

STRETCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STRETCH_DIR))

from paths import ensure_radiomics_dirs, features_all_csv, features_longitudinal_csv  # noqa: E402

METADATA_COLUMNS = {"slug", "tcga_id", "subtype", "timepoint", "study_date", "backend"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in METADATA_COLUMNS]


def compare_longitudinal(
    input_csv: Path | None = None,
    output_csv: Path | None = None,
) -> Path:
    in_path = input_csv or features_all_csv()
    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing {in_path}. Run run_all_radiomics.py first."
        )

    df = pd.read_csv(in_path)
    if df.empty:
        raise ValueError(f"No rows in {in_path}")

    feature_cols = feature_columns(df)
    rows: list[dict] = []

    for tcga_id, group in df.groupby("tcga_id", sort=False):
        baseline = group[group["timepoint"] == "baseline"]
        followup = group[group["timepoint"] == "followup"]
        if baseline.empty or followup.empty:
            continue
        if len(baseline) > 1 or len(followup) > 1:
            raise ValueError(f"Expected one baseline and one followup row for {tcga_id}")

        base_row = baseline.iloc[0]
        follow_row = followup.iloc[0]
        out: dict = {
            "tcga_id": tcga_id,
            "subtype": base_row.get("subtype"),
            "baseline_slug": base_row["slug"],
            "followup_slug": follow_row["slug"],
            "baseline_study_date": base_row.get("study_date"),
            "followup_study_date": follow_row.get("study_date"),
            "backend": base_row.get("backend"),
        }
        for col in feature_cols:
            base_val = float(base_row[col])
            follow_val = float(follow_row[col])
            delta = follow_val - base_val
            out[f"{col}_baseline"] = base_val
            out[f"{col}_followup"] = follow_val
            out[f"{col}_delta"] = delta
            if base_val != 0:
                out[f"{col}_pct_change"] = 100.0 * delta / base_val
            else:
                out[f"{col}_pct_change"] = float("nan")
        rows.append(out)

    if not rows:
        raise ValueError("No longitudinal pairs found (need baseline + followup per tcga_id)")

    ensure_radiomics_dirs()
    out_path = output_csv or features_longitudinal_csv()
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs follow-up radiomics features per patient."
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    out_path = compare_longitudinal(args.input, args.output)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

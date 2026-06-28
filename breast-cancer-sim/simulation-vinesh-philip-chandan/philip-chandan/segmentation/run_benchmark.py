"""Run segmentation benchmark: .les reference + evaluate on-disk method masks."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

SEGMENTATION_DIR = Path(__file__).resolve().parent
STRETCH_DIR = SEGMENTATION_DIR.parent / "stretch"
PHILIP_CHANDAN_DIR = SEGMENTATION_DIR.parent

sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(STRETCH_DIR))

from load_les_mask import find_les_files  # noqa: E402
from load_manifest import find_volume, load_volumes  # noqa: E402
from prep_volume import normalize_volume  # noqa: E402
from validate_segmentation import load_annotation_volume  # noqa: E402

sys.path.insert(0, str(SEGMENTATION_DIR))

from evaluate import compare_to_reference, load_mask  # noqa: E402
from ground_truth import write_les_reference  # noqa: E402
from qc_overlay import save_mid_z_overlay  # noqa: E402
from seg_paths import (  # noqa: E402
    BENCHMARK_METHODS,
    REFERENCE_METHOD,
    comparison_metrics_csv,
    ensure_segmentation_dirs,
    mask_npy,
)

CSV_FIELDS = [
    "slug",
    "tcga_id",
    "subtype",
    "timepoint",
    "method",
    "reference_method",
    "dice",
    "reference_volume_mm3",
    "predicted_volume_mm3",
    "relative_volume_error",
    "reference_voxels",
    "predicted_voxels",
    "area_fraction_pred_over_ref",
    "les_file",
    "annotated_series",
    "volume_source",
    "reference_overlay_png",
    "predicted_overlay_png",
]


def _baseline_slugs_with_les() -> list[str]:
    slugs: list[str] = []
    for entry in load_volumes():
        if entry.get("timepoint") != "baseline":
            continue
        if find_les_files(entry["tcga_id"]):
            slugs.append(entry["slug"])
    return slugs


def _load_display_volume(slug: str, ref_meta: dict[str, Any]) -> Any:
    """MR volume aligned to the .les annotation series."""
    entry = find_volume(slug=slug)
    les_path = Path(ref_meta["path"])
    volume, _, _, _, _ = load_annotation_volume(
        tcga_id=entry["tcga_id"],
        study_date=entry["study_date"],
        les_path=les_path,
        slug=slug,
    )
    return volume


def benchmark_slug(slug: str, *, lesions_dir: Path | None = None) -> list[dict[str, Any]]:
    """Write .les reference and evaluate any on-disk method masks for one baseline slug."""
    ref_mask, ref_meta = write_les_reference(slug, lesions_dir=lesions_dir)
    spacing_mm = list(ref_meta["spacing_mm"])
    volume = _load_display_volume(slug, ref_meta)
    norm = normalize_volume(volume)

    ref_overlay = save_mid_z_overlay(
        slug,
        REFERENCE_METHOD,
        norm,
        ref_mask,
        title=(
            f"{ref_meta['tcga_id']} baseline | .les S{ref_meta['dce_index']} "
            f"({ref_meta['mask_voxels']:,} vox)"
        ),
        fill_rgba=(0.0, 1.0, 0.0, 0.45),
        contour_color="#00FF00",
    )

    rows: list[dict[str, Any]] = []
    for method in BENCHMARK_METHODS:
        pred_path = mask_npy(slug, method)
        if not pred_path.exists():
            continue

        pred_mask = load_mask(pred_path)
        if pred_mask.shape != ref_mask.shape:
            raise ValueError(
                f"{method} mask shape {pred_mask.shape} != reference {ref_mask.shape} for {slug}"
            )

        metrics = compare_to_reference(pred_mask, ref_mask, spacing_mm)
        pred_overlay = save_mid_z_overlay(
            slug,
            method,
            norm,
            pred_mask,
            title=(
                f"{ref_meta['tcga_id']} baseline | {method} "
                f"({metrics['predicted_voxels']:,} vox, Dice={metrics['dice']:.3f})"
            ),
            fill_rgba=(1.0, 0.5, 0.0, 0.35),
            contour_color="#FF8800",
        )

        rows.append(
            {
                "slug": slug,
                "tcga_id": ref_meta["tcga_id"],
                "subtype": ref_meta.get("subtype"),
                "timepoint": ref_meta.get("timepoint"),
                "method": method,
                "reference_method": REFERENCE_METHOD,
                "les_file": ref_meta["les_file"],
                "annotated_series": ref_meta["annotated_series"],
                "volume_source": ref_meta["volume_source"],
                "reference_overlay_png": str(ref_overlay),
                "predicted_overlay_png": str(pred_overlay),
                **metrics,
            }
        )

    return rows


def write_comparison_csv(rows: list[dict[str, Any]], path: Path | None = None) -> Path:
    ensure_segmentation_dirs()
    out_path = path or comparison_metrics_csv()
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_FIELDS})
    return out_path


def run_benchmark(
    *,
    slugs: list[str] | None = None,
    lesions_dir: Path | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    """Benchmark baseline slugs; returns (csv_path, metric rows)."""
    target_slugs = slugs or _baseline_slugs_with_les()
    if not target_slugs:
        raise FileNotFoundError("No baseline slugs with local .les files found.")

    all_rows: list[dict[str, Any]] = []
    for slug in target_slugs:
        all_rows.extend(benchmark_slug(slug, lesions_dir=lesions_dir))

    csv_path = write_comparison_csv(all_rows)
    return csv_path, all_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segmentation benchmark: .les reference + evaluate method masks on disk.",
    )
    parser.add_argument("--slug", help="One baseline manifest slug")
    parser.add_argument(
        "--all-primary",
        action="store_true",
        help="All baseline primaries with .les (default when no --slug)",
    )
    args = parser.parse_args()

    if args.slug:
        slugs = [args.slug]
    elif args.all_primary or not args.slug:
        slugs = _baseline_slugs_with_les()
    else:
        parser.error("Provide --slug or --all-primary")

    csv_path, rows = run_benchmark(slugs=slugs)
    print(f"Wrote reference masks + {csv_path}")
    for slug in slugs:
        ref_path = mask_npy(slug, REFERENCE_METHOD)
        print(f"  {slug}: reference → {ref_path}")

    if rows:
        for row in rows:
            print(
                f"  {row['slug']} [{row['method']}]: Dice={row['dice']:.3f} "
                f"area={row['area_fraction_pred_over_ref']:.1f}x"
            )
    else:
        print(
            "  No method masks evaluated yet. Add predictions under "
            f"data/processed/segmentation-philip-chandan/{{slug}}_{{method}}_mask.npy "
            f"for methods: {', '.join(BENCHMARK_METHODS)}"
        )


if __name__ == "__main__":
    main()

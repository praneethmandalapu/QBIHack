"""Plot P1–P4 cuboid intensity histograms for a baseline slug with .les."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VALIDATION_DIR = Path(__file__).resolve().parent
PHILIP_CHANDAN_DIR = VALIDATION_DIR.parent
STRETCH_DIR = PHILIP_CHANDAN_DIR / "stretch"
QC_DIR = PHILIP_CHANDAN_DIR.parents[1] / "data" / "qc" / "segmentation-philip-chandan"

sys.path.insert(0, str(VALIDATION_DIR))
sys.path.insert(0, str(STRETCH_DIR))
sys.path.insert(0, str(PHILIP_CHANDAN_DIR))

from dce_phases import (  # noqa: E402
    resolve_dce_dicom_dir_for_study,
    resolve_phase_ranges,
    split_dce_phases,
)
from les_cuboid_brightness import plot_phase_cuboid_histograms  # noqa: E402
from load_les_mask import find_les_files, load_les_mask  # noqa: E402
from load_manifest import find_volume, load_volumes  # noqa: E402
from prep_volume import load_raw_extract  # noqa: E402
from validate_segmentation import load_annotation_volume  # noqa: E402
from view_les_napari import slugs_with_les  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Histogram of normalized intensities inside .les cuboid for P1–P4.",
    )
    parser.add_argument("--slug", help="Baseline manifest slug")
    parser.add_argument("--bins", type=int, default=50, help="Histogram bin count")
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG path (default: data/qc/segmentation-philip-chandan/{slug}_cuboid_hist_p1-p4.png)",
    )
    parser.add_argument("--no-show", action="store_true", help="Save PNG only, do not open window")
    parser.add_argument("--list", action="store_true", help="List slugs with .les")
    args = parser.parse_args()

    if args.list:
        for slug in slugs_with_les():
            print(slug)
        return

    slug = args.slug
    if not slug:
        available = slugs_with_les()
        if len(available) == 1:
            slug = available[0]
        elif available:
            parser.error("Provide --slug. Baselines with .les: " + ", ".join(available))
        else:
            parser.error("No baseline slugs with .les found.")

    entry = find_volume(slug=slug)
    tcga_id = entry["tcga_id"]
    study_date = entry["study_date"]
    les_files = find_les_files(tcga_id)
    if not les_files:
        raise FileNotFoundError(f"No .les file for {tcga_id}")
    les_path = les_files[0]

    volume, spacing_mm, _series_desc, dce_index, _source = load_annotation_volume(
        tcga_id=tcga_id,
        study_date=study_date,
        les_path=les_path,
        slug=slug,
    )
    expert_mask, les_meta = load_les_mask(les_path, volume.shape)

    source_dicom_dir: Path | None = None
    try:
        _, sidecar = load_raw_extract(slug)
        rel = sidecar.get("source_dicom_dir")
        if rel:
            candidate = PHILIP_CHANDAN_DIR.parents[1] / rel
            if candidate.is_dir():
                source_dicom_dir = candidate
    except FileNotFoundError:
        pass

    dicom_dir = resolve_dce_dicom_dir_for_study(
        tcga_id=tcga_id,
        study_date=study_date,
        dce_index=dce_index,
        source_dicom_dir=source_dicom_dir,
    )
    phases = resolve_phase_ranges(volume_shape=volume.shape, dicom_dir=dicom_dir)
    phase_volumes = split_dce_phases(volume, phases)

    output_path = args.output or (QC_DIR / f"{slug}_cuboid_hist_p1-p4.png")
    saved = plot_phase_cuboid_histograms(
        phase_volumes,
        phases,
        les_meta,
        expert_mask,
        slug=slug,
        bins=args.bins,
        output_path=output_path,
        show=not args.no_show,
    )
    if saved:
        print(f"Saved {saved}")


if __name__ == "__main__":
    main()

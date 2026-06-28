"""Align P1 z-band slabs, plot bbox bright-fraction curves, optional tumor mask."""

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

from aligned_bbox_tumor import write_aligned_bbox_tumor_mask  # noqa: E402
from cuboid_phase_registration import (  # noqa: E402
    align_phase_z_bands_to_p1,
    attach_les_overlays_on_z_band,
    format_alignment_metrics,
)
from dce_phases import (  # noqa: E402
    resolve_dce_dicom_dir_for_study,
    resolve_phase_ranges,
    select_phases,
    split_dce_phases,
)
from les_cuboid_brightness import (  # noqa: E402
    ALIGNED_BBOX_REGISTRATION_PHASES,
    compute_aligned_bbox_connected_table,
    format_brightness_table,
    plot_aligned_bbox_bright_fraction_grid,
    plot_aligned_bbox_bright_fraction_vs_threshold,
)
from load_les_mask import find_les_files, load_les_mask  # noqa: E402
from load_manifest import find_volume  # noqa: E402
from prep_volume import load_raw_extract  # noqa: E402
from validate_segmentation import load_annotation_volume  # noqa: E402
from view_les_napari import slugs_with_les  # noqa: E402


def run_workflow(
    slug: str,
    *,
    registration_iterations: int = 200,
    threshold_step: float = 0.05,
    gap_voxels: int = 0,
    output_path: Path | None = None,
    plot_layout: str = "both",
    show_plot: bool = False,
    write_mask: bool = True,
) -> None:
    entry = find_volume(slug=slug)
    tcga_id = entry["tcga_id"]
    study_date = entry["study_date"]
    les_files = find_les_files(tcga_id)
    if not les_files:
        raise FileNotFoundError(f"No .les file for {tcga_id}")
    les_path = les_files[0]

    volume, spacing_mm, series_desc, dce_index, volume_source = load_annotation_volume(
        tcga_id=tcga_id,
        study_date=study_date,
        les_path=les_path,
        slug=slug,
    )
    scale = tuple(float(s) for s in spacing_mm)
    _, les_meta = load_les_mask(les_path, volume.shape)

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
    phases, phase_volumes = select_phases(
        phases,
        phase_volumes,
        ALIGNED_BBOX_REGISTRATION_PHASES,
    )

    result = align_phase_z_bands_to_p1(
        phase_volumes,
        phases,
        les_meta,
        scale,
        number_of_iterations=registration_iterations,
    )
    attach_les_overlays_on_z_band(
        result,
        les_path=les_path,
        full_volume_shape=volume.shape,
        reference_phase=phases[0],
    )

    rows = compute_aligned_bbox_connected_table(
        result.slabs_aligned,
        phases,
        les_meta,
        result.expert_slab,
        threshold_step=threshold_step,
        gap_voxels=gap_voxels,
    )

    print(
        f"{slug}\n"
        f"  series: {series_desc!r} (S{dce_index})\n"
        f"  P1 .les local z-band: {result.z_band_local[0]}–{result.z_band_local[1]}\n"
        f"  phases: P{', P'.join(str(p.index) for p in phases)} (register P2–P3 → P1)\n"
        f"  alignment ROI: full Y/X z-band; metric ROI: tight .les bbox\n"
        f"  spacing_mm: {spacing_mm}\n"
    )
    print(format_alignment_metrics(result.metrics))
    print()
    print("Post-alignment center-connected bbox fraction vs threshold (P2–P3):")
    print(format_brightness_table(rows))

    if plot_layout in ("overlay", "both"):
        png_path = output_path or (QC_DIR / f"{slug}_aligned_bbox_bright_vs_threshold.png")
        saved = plot_aligned_bbox_bright_fraction_vs_threshold(
            result.slabs_aligned,
            phases,
            les_meta,
            result.expert_slab,
            slug=slug,
            threshold_step=threshold_step,
            gap_voxels=gap_voxels,
            output_path=png_path,
            show=show_plot and plot_layout == "overlay",
        )
        if saved:
            print(f"\nSaved overlay plot: {saved}")

    if plot_layout in ("grid", "both"):
        grid_path = QC_DIR / f"{slug}_aligned_bbox_bright_vs_threshold_grid.png"
        saved_grid = plot_aligned_bbox_bright_fraction_grid(
            result.slabs_aligned,
            phases,
            les_meta,
            result.expert_slab,
            slug=slug,
            threshold_step=threshold_step,
            gap_voxels=gap_voxels,
            output_path=grid_path,
            show=show_plot and plot_layout == "grid",
        )
        if saved_grid:
            print(f"Saved grid plot: {saved_grid}")

    if write_mask:
        mask, meta = write_aligned_bbox_tumor_mask(
            slug,
            result,
            phases,
            les_meta,
            volume.shape,
            gap_voxels=gap_voxels,
            extra_metadata={
                "tcga_id": tcga_id,
                "study_date": study_date,
                "volume_source": volume_source,
                "spacing_mm": spacing_mm,
            },
        )
        print(
            f"\nTumor mask: P{meta['selected_phase']} @ threshold={meta['threshold']:.3f} "
            f"(gap={meta.get('gap_voxels', 0)}) "
            f"→ {meta['mask_voxels']:,} voxels in bbox "
            f"({int(mask.sum()):,} in full volume)\n"
            f"  {meta['mask_npy']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rigid-align P2–P4 to P1 on .les z-band (full Y/X), sweep bbox bright fraction, "
            "optionally write aligned_bbox_tumor mask."
        ),
    )
    parser.add_argument("--slug", help="Baseline manifest slug")
    parser.add_argument("--registration-iterations", type=int, default=200)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument(
        "--gap-voxels",
        type=int,
        default=0,
        help="Morphological closing radius before center CC (0=strict, 1–10 bridges gaps)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Overlay PNG path (default: .../{slug}_aligned_bbox_bright_vs_threshold.png)",
    )
    parser.add_argument(
        "--plot-layout",
        choices=("overlay", "grid", "both"),
        default="both",
        help="Single combined curve plot, 2×2 per-phase grid, or both (default: both)",
    )
    parser.add_argument("--show", action="store_true", help="Open matplotlib window")
    parser.add_argument("--no-mask", action="store_true", help="Plot/table only; skip mask write")
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

    run_workflow(
        slug,
        registration_iterations=args.registration_iterations,
        threshold_step=args.threshold_step,
        gap_voxels=args.gap_voxels,
        output_path=args.output,
        plot_layout=args.plot_layout,
        show_plot=args.show,
        write_mask=not args.no_mask,
    )


if __name__ == "__main__":
    main()

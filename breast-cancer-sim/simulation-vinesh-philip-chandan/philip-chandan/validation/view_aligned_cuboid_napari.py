"""Napari viewer: P1 .les z-band (full Y/X) with P2–P4 rigidly aligned + metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

VALIDATION_DIR = Path(__file__).resolve().parent
PHILIP_CHANDAN_DIR = VALIDATION_DIR.parent
STRETCH_DIR = PHILIP_CHANDAN_DIR / "stretch"

sys.path.insert(0, str(VALIDATION_DIR))
sys.path.insert(0, str(STRETCH_DIR))
sys.path.insert(0, str(PHILIP_CHANDAN_DIR))

import napari  # noqa: E402
import numpy as np  # noqa: E402

from aligned_bbox_threshold_dock import add_bbox_threshold_dock  # noqa: E402
from clinical_layout import link_viewers  # noqa: E402
from cuboid_phase_registration import (  # noqa: E402
    align_phase_z_bands_to_p1,
    attach_les_overlays_on_z_band,
    display_slab_for_napari,
    format_alignment_metrics,
)
from dce_phases import (  # noqa: E402
    resolve_dce_dicom_dir_for_study,
    resolve_phase_ranges,
    split_dce_phases,
)
from les_cuboid_brightness import (  # noqa: E402
    plot_aligned_bbox_bright_fraction_grid,
    plot_aligned_bbox_bright_fraction_vs_threshold,
)
from load_les_mask import find_les_files, load_les_mask  # noqa: E402
from load_manifest import find_volume  # noqa: E402
from prep_volume import load_raw_extract  # noqa: E402
from validate_segmentation import load_annotation_volume  # noqa: E402
from view_les_napari import slugs_with_les  # noqa: E402

QC_DIR = PHILIP_CHANDAN_DIR.parents[1] / "data" / "qc" / "segmentation-philip-chandan"


def save_aligned_bbox_plot_pngs(
    slug: str,
    *,
    slabs_aligned: dict[int, Any],
    phases: list,
    les_meta: dict[str, Any],
    expert_slab: Any,
    threshold_step: float = 0.05,
) -> list[Path]:
    """Write overlay + 2×2 grid bright-fraction PNGs (same paths as run_aligned_bbox_workflow)."""
    saved: list[Path] = []
    overlay_path = QC_DIR / f"{slug}_aligned_bbox_bright_vs_threshold.png"
    if plot_aligned_bbox_bright_fraction_vs_threshold(
        slabs_aligned,
        phases,
        les_meta,
        expert_slab,
        slug=slug,
        threshold_step=threshold_step,
        output_path=overlay_path,
        show=False,
    ):
        saved.append(overlay_path)
    grid_path = QC_DIR / f"{slug}_aligned_bbox_bright_vs_threshold_grid.png"
    if plot_aligned_bbox_bright_fraction_grid(
        slabs_aligned,
        phases,
        les_meta,
        expert_slab,
        slug=slug,
        threshold_step=threshold_step,
        output_path=grid_path,
        show=False,
    ):
        saved.append(grid_path)
    return saved


def setup_aligned_slab_grid(
    viewer: napari.Viewer,
    *,
    phases: list,
    slabs_aligned: dict[int, Any],
    slabs_raw: dict[int, Any],
    expert_slab: Any,
    boundary_slab: Any,
    spacing_mm: tuple[float, float, float],
    z_band_local: tuple[int, int],
    show_raw: bool = False,
) -> tuple[list[Any], dict[int, Any], dict[int, Any]]:
    from napari._qt.qt_viewer import QtViewer
    from napari.components.viewer_model import ViewerModel
    from qtpy.QtWidgets import QGridLayout, QLabel, QWidget

    scale = spacing_mm
    phase_viewers: list[Any] = []
    threshold_layers: dict[int, Any] = {}
    boundary_layers: dict[int, Any] = {}

    grid = QGridLayout()
    grid.setSpacing(4)
    z0, z1 = z_band_local
    slab_shape = next(iter(slabs_aligned.values())).shape

    for column, phase in enumerate(phases):
        volume = slabs_raw[phase.index] if show_raw else slabs_aligned[phase.index]
        title = f"P{phase.index} z{z0}-{z1}" + (" raw" if show_raw else " → P1")
        phase_model = ViewerModel(title=title)
        phase_qt = QtViewer(phase_model)
        phase_model.add_image(
            display_slab_for_napari(volume),
            name=f"z-band P{phase.index}",
            scale=scale,
            colormap="gray",
            contrast_limits=(0.0, 1.0),
        )
        # Overlays on P1 only — threshold/bbox on P2–P4 block the aligned tissue view.
        if phase.index == 1:
            if expert_slab.any():
                phase_model.add_labels(expert_slab, name=".les expert", scale=scale, opacity=0.55)
            boundary_layer = phase_model.add_labels(
                boundary_slab if boundary_slab.any() else np.zeros(slab_shape, dtype=np.uint8),
                name="bbox shell",
                scale=scale,
                opacity=0.85,
            )
            boundary_layers[phase.index] = boundary_layer
            threshold_layer = phase_model.add_labels(
                np.zeros(slab_shape, dtype=np.uint8),
                name="threshold mask",
                scale=scale,
                opacity=0.5,
            )
            threshold_layers[phase.index] = threshold_layer
        phase_viewers.append(phase_model)

        header = QLabel(title)
        if phase.acquisition_time:
            header.setText(f"{title}\n{phase.acquisition_time}")
        grid.addWidget(header, 0, column)
        grid.addWidget(phase_qt, 1, column)

    link_viewers(*phase_viewers)

    grid_widget = QWidget()
    grid_widget.setLayout(grid)
    viewer.window._qt_window.setCentralWidget(grid_widget)
    return phase_viewers, threshold_layers, boundary_layers


def view_aligned_cuboids(
    slug: str,
    *,
    show_raw: bool = False,
    registration_iterations: int = 200,
    save_plots: bool = True,
    threshold_step: float = 0.05,
) -> None:
    entry = find_volume(slug=slug)
    tcga_id = entry["tcga_id"]
    study_date = entry["study_date"]
    les_files = find_les_files(tcga_id)
    if not les_files:
        raise FileNotFoundError(f"No .les file for {tcga_id}")
    les_path = les_files[0]

    volume, spacing_mm, series_desc, dce_index, _source = load_annotation_volume(
        tcga_id=tcga_id,
        study_date=study_date,
        les_path=les_path,
        slug=slug,
    )
    scale = tuple(float(s) for s in spacing_mm)

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

    _, les_meta = load_les_mask(les_path, volume.shape)
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

    slab_shape = next(iter(result.slabs_aligned.values())).shape
    print(
        f"{slug}\n"
        f"  series: {series_desc!r} (S{dce_index})\n"
        f"  P1 .les local z-band: {result.z_band_local[0]}–{result.z_band_local[1]}\n"
        f"  slab shape (Z,Y,X): {slab_shape} — full in-plane Y/X\n"
        f"  spacing_mm: {spacing_mm}\n"
        f"  mode: {'raw slabs' if show_raw else 'aligned to P1 slab grid'}\n"
    )
    print(format_alignment_metrics(result.metrics))

    if save_plots and not show_raw:
        plot_paths = save_aligned_bbox_plot_pngs(
            slug,
            slabs_aligned=result.slabs_aligned,
            phases=phases,
            les_meta=les_meta,
            expert_slab=result.expert_slab,
            threshold_step=threshold_step,
        )
        for path in plot_paths:
            print(f"  saved plot: {path}")

    viewer = napari.Viewer(title=f"{slug} — aligned P1 z-band slabs")
    _phase_viewers, threshold_layers, boundary_layers = setup_aligned_slab_grid(
        viewer,
        phases=phases,
        slabs_aligned=result.slabs_aligned,
        slabs_raw=result.slabs_raw,
        expert_slab=result.expert_slab,
        boundary_slab=result.boundary_slab,
        spacing_mm=scale,
        z_band_local=result.z_band_local,
        show_raw=show_raw,
    )
    if not show_raw:
        add_bbox_threshold_dock(
            viewer,
            phases=phases,
            slabs_aligned=result.slabs_aligned,
            expert_slab=result.expert_slab,
            les_meta=les_meta,
            phase_viewers=_phase_viewers,
            threshold_layers=threshold_layers,
            boundary_layers=boundary_layers,
            initial_phase=2,
        )
    napari.run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View P1 .les z-band (full Y/X) with P2–P4 registered + metrics.",
    )
    parser.add_argument("--slug", help="Baseline manifest slug")
    parser.add_argument("--raw", action="store_true", help="Show unregistered slabs")
    parser.add_argument("--registration-iterations", type=int, default=200)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip regenerating bright-fraction PNGs on launch",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
        help="Threshold step for saved PNG curves (default 0.05)",
    )
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

    view_aligned_cuboids(
        slug,
        show_raw=args.raw,
        registration_iterations=args.registration_iterations,
        save_plots=not args.no_plots,
        threshold_step=args.threshold_step,
    )


if __name__ == "__main__":
    main()

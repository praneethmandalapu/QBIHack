"""Napari viewer: P1 .les z-band (full Y/X) with P2–P4 rigidly aligned + metrics."""

from __future__ import annotations

import argparse
import math
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

from clinical_layout import link_viewers  # noqa: E402
from cuboid_phase_registration import (  # noqa: E402
    CuboidAlignmentMetrics,
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
from load_les_mask import find_les_files, load_les_mask  # noqa: E402
from load_manifest import find_volume  # noqa: E402
from prep_volume import load_raw_extract  # noqa: E402
from validate_segmentation import load_annotation_volume  # noqa: E402
from view_les_napari import slugs_with_les  # noqa: E402


def add_alignment_metrics_dock(viewer: napari.Viewer, metrics: list[CuboidAlignmentMetrics]) -> None:
    from qtpy.QtWidgets import (
        QHeaderView,
        QLabel,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.addWidget(
        QLabel(
            "Rigid registration on P1 .les z-band slabs (full breast Y/X). "
            "Each P2–P4 volume is aligned to the P1 slab grid."
        )
    )

    table = QTableWidget(len(metrics), 8)
    table.setHorizontalHeaderLabels(
        ["Phase", "|T| mm", "|R| deg", "NCC pre", "NCC post", "MSE pre", "MSE post", "MI final"]
    )
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    for row_index, row in enumerate(metrics):
        trans = math.sqrt(sum(v * v for v in row.translation_mm))
        values = [
            f"P{row.moving_phase}",
            f"{trans:.3f}",
            f"{row.rotation_magnitude_deg:.3f}",
            f"{row.ncc_before:.4f}",
            f"{row.ncc_after:.4f}",
            f"{row.mse_before:.2f}",
            f"{row.mse_after:.2f}",
            f"{row.optimizer_metric_value:.4f}",
        ]
        for col_index, text in enumerate(values):
            table.setItem(row_index, col_index, QTableWidgetItem(text))

    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    layout.addWidget(table)
    viewer.window.add_dock_widget(widget, area="bottom", name="Alignment metrics")


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
) -> None:
    from napari._qt.qt_viewer import QtViewer
    from napari.components.viewer_model import ViewerModel
    from qtpy.QtWidgets import QGridLayout, QLabel, QWidget

    scale = spacing_mm
    phase_viewers: list[Any] = []

    grid = QGridLayout()
    grid.setSpacing(4)
    z0, z1 = z_band_local

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
        if expert_slab.any():
            phase_model.add_labels(expert_slab, name=".les expert", scale=scale, opacity=0.55)
        if boundary_slab.any():
            phase_model.add_labels(boundary_slab, name=".les bbox", scale=scale, opacity=0.85)
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


def view_aligned_cuboids(
    slug: str,
    *,
    show_raw: bool = False,
    registration_iterations: int = 200,
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

    viewer = napari.Viewer(title=f"{slug} — aligned P1 z-band slabs")
    setup_aligned_slab_grid(
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
    add_alignment_metrics_dock(viewer, result.metrics)
    napari.run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View P1 .les z-band (full Y/X) with P2–P4 registered + metrics.",
    )
    parser.add_argument("--slug", help="Baseline manifest slug")
    parser.add_argument("--raw", action="store_true", help="Show unregistered slabs")
    parser.add_argument("--registration-iterations", type=int, default=200)
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
    )


if __name__ == "__main__":
    main()

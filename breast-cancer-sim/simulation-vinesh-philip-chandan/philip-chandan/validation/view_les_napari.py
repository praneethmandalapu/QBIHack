"""Interactive 3D viewer: baseline MR + TCIA radiologist .les overlay (napari).

Clinical-style breast DCE QC: side-by-side temporal phases, optional MIP row,
pre-contrast subtraction, and expert-mask toggle.

Run (macOS/Linux):
    cd breast-cancer-sim
    source .venv/bin/activate
    pip install -r requirements.txt   # includes napari[pyqt6]

    python simulation-vinesh-philip-chandan/philip-chandan/validation/view_les_napari.py --list
    python simulation-vinesh-philip-chandan/philip-chandan/validation/view_les_napari.py \\
        --slug luminal_a_TCGA-AR-A1AX_baseline --phases-only

    python simulation-vinesh-philip-chandan/philip-chandan/validation/view_les_napari.py \\
        --slug luminal_a_TCGA-AR-A1AX_baseline --wide --collapse-controls

Optional: add --mip for the bottom maximum-intensity-projection row beneath P1–P4.
Phases-only mode shows .les expert voxels + cuboid bounding box and a docked table of
bright-voxel fraction inside the bbox for each normalized threshold (default step 0.05).

Run (Windows):
    cd breast-cancer-sim
    .venv\\Scripts\\python.exe simulation-vinesh-philip-chandan\\philip-chandan\\validation\\view_les_napari.py --list

Requires napari[pyqt6]. First launch may take ~10s while Qt initializes.
See validation/VALIDATION.md for dataset paths and validation context.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

VALIDATION_DIR = Path(__file__).resolve().parent
PHILIP_CHANDAN_DIR = VALIDATION_DIR.parent
STRETCH_DIR = PHILIP_CHANDAN_DIR / "stretch"

sys.path.insert(0, str(VALIDATION_DIR))
sys.path.insert(0, str(STRETCH_DIR))
sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
SEGMENTATION_DIR = PHILIP_CHANDAN_DIR / "segmentation"
sys.path.insert(0, str(SEGMENTATION_DIR))

import napari  # noqa: E402

from clinical_layout import (  # noqa: E402
    setup_clinical_hanging_protocol,
    setup_phases_only_view,
)
from dce_phases import (  # noqa: E402
    DcePhase,
    compute_subtraction,
    lesion_z_in_phase,
    load_precontrast_volume,
    mask_for_phase,
    resample_volume,
    resolve_dce_dicom_dir_for_study,
    resolve_phase_ranges,
    split_dce_phases,
)
from les_cuboid_brightness import (  # noqa: E402
    CuboidBrightnessRow,
    compute_cuboid_brightness_table,
    format_brightness_table,
)
from load_les_mask import (  # noqa: E402
    find_les_files,
    load_les_cuboid_boundary,
    load_les_mask,
)
from load_manifest import find_volume, load_volumes  # noqa: E402
from prep_volume import load_raw_extract, normalize_volume, tumor_mask_largest_component  # noqa: E402
from validate_segmentation import load_annotation_volume  # noqa: E402


@dataclass
class BreastDisplayState:
    phases: list[DcePhase]
    phase_volumes: list[np.ndarray]
    subtraction_volumes: list[np.ndarray]
    spacing_mm: tuple[float, float, float]
    active_phase: int = 1
    show_subtraction: bool = False
    show_precontrast: bool = False
    show_mip_row: bool = False
    precontrast_display: np.ndarray | None = None
    mr_layer: Any | None = None
    subtraction_layer: Any | None = None
    precontrast_layer: Any | None = None
    expert_layer: Any | None = None
    boundary_layer: Any | None = None
    prediction_layer: Any | None = None
    expert_mask_full: np.ndarray | None = None
    boundary_mask_full: np.ndarray | None = None
    prediction_mask_full: np.ndarray | None = None
    cuboid_boundary: bool = False
    hanging: dict[str, Any] = field(default_factory=dict)


def slugs_with_les() -> list[str]:
    """Baseline manifest slugs whose TCGA ID has a local .les file."""
    slugs: list[str] = []
    for entry in load_volumes():
        if entry.get("timepoint") != "baseline":
            continue
        if find_les_files(entry["tcga_id"]):
            slugs.append(entry["slug"])
    return slugs


def add_collapsible_dock(
    viewer: napari.Viewer,
    widget: Any,
    *,
    name: str,
    collapsed: bool = False,
) -> None:
    """Right-side dock with a button to collapse controls and free canvas space."""
    from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

    outer = QWidget()
    layout = QVBoxLayout(outer)
    layout.setContentsMargins(4, 4, 4, 4)
    toggle = QPushButton("Show controls ▼" if collapsed else "Hide controls ▲")

    inner = QWidget()
    inner_layout = QVBoxLayout(inner)
    inner_layout.setContentsMargins(0, 0, 0, 0)
    native = widget.native if hasattr(widget, "native") else widget
    inner_layout.addWidget(native)
    inner.setVisible(not collapsed)

    def on_toggle() -> None:
        show = not inner.isVisible()
        inner.setVisible(show)
        toggle.setText("Hide controls ▲" if show else "Show controls ▼")

    toggle.clicked.connect(on_toggle)
    layout.addWidget(toggle)
    layout.addWidget(inner)
    viewer.window.add_dock_widget(outer, area="right", name=name)


def _configure_viewer_chrome(viewer: napari.Viewer, *, wide: bool) -> None:
    """Hide napari layer list/controls so MR images use more horizontal space."""
    if not wide:
        return
    qt_viewer = viewer.window.qt_viewer
    for attr in ("dockLayerList", "dockLayerControls", "layerListDock", "layerControlsDock"):
        dock = getattr(qt_viewer, attr, None)
        if dock is not None:
            dock.setVisible(False)


def add_overlay_toggle_button(
    viewer: napari.Viewer,
    button_label: str,
    *overlay_layers: Any,
) -> None:
    """Dock button to show/hide overlay label layers."""
    if not overlay_layers:
        return

    from qtpy.QtWidgets import QPushButton

    hide_text = f"Hide {button_label}"
    show_text = f"Show {button_label}"
    button = QPushButton(hide_text)

    def toggle() -> None:
        visible = not overlay_layers[0].visible
        for layer in overlay_layers:
            layer.visible = visible
        button.setText(hide_text if visible else show_text)

    button.clicked.connect(toggle)
    viewer.window.add_dock_widget(button, area="right", name=button_label)


def add_brightness_fraction_dock(
    viewer: napari.Viewer,
    rows: list[CuboidBrightnessRow],
    *,
    threshold_step: float,
) -> None:
    """Docked table: bright fraction inside .les cuboid for each phase × threshold."""
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
            "Bright fraction = voxels ≥ threshold inside .les cuboid "
            f"(1–99% normalized per phase; step={threshold_step:.2f}). "
            "Les frac = expert .les voxels / cuboid voxels."
        )
    )

    table = QTableWidget(len(rows), 7)
    table.setHorizontalHeaderLabels(
        [
            "Phase",
            "Threshold",
            "Bright frac",
            "Bright vox",
            "Cuboid vox",
            "Les frac",
            "Les vox",
        ]
    )
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    for row_index, row in enumerate(rows):
        values = [
            f"P{row.phase_index}",
            f"{row.threshold:.2f}",
            f"{row.bright_fraction:.3f}",
            str(row.bright_voxels),
            str(row.cuboid_voxels),
            f"{row.les_fraction:.3f}",
            str(row.les_voxels),
        ]
        for col_index, text in enumerate(values):
            table.setItem(row_index, col_index, QTableWidgetItem(text))

    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    layout.addWidget(table)
    viewer.window.add_dock_widget(widget, area="bottom", name="Cuboid brightness")


def _phase_label(phase: DcePhase) -> str:
    acq = phase.acquisition_time
    suffix = f" acq={acq}" if acq else ""
    return f"Phase {phase.index} (z {phase.z_start}-{phase.z_end - 1}{suffix})"


def _build_subtraction_volumes(
    phase_volumes: list[np.ndarray],
    spacing_mm: list[float],
    precontrast: tuple[np.ndarray, list[float]] | None,
) -> list[np.ndarray]:
    if precontrast is None:
        return []
    pre_volume, pre_spacing = precontrast
    target_shape = phase_volumes[0].shape
    pre_resampled = resample_volume(pre_volume, pre_spacing, target_shape, spacing_mm)
    return [compute_subtraction(phase, pre_resampled) for phase in phase_volumes]


def _update_boundary_layer(state: BreastDisplayState) -> None:
    if state.boundary_layer is None or state.boundary_mask_full is None:
        return
    phase = state.phases[state.active_phase - 1]
    state.boundary_layer.data = mask_for_phase(state.boundary_mask_full, phase)


def _update_expert_layer(state: BreastDisplayState) -> None:
    if state.expert_layer is None or state.expert_mask_full is None:
        return
    phase = state.phases[state.active_phase - 1]
    state.expert_layer.data = mask_for_phase(state.expert_mask_full, phase)


def _update_prediction_layer(state: BreastDisplayState) -> None:
    if state.prediction_layer is None or state.prediction_mask_full is None:
        return
    phase = state.phases[state.active_phase - 1]
    state.prediction_layer.data = mask_for_phase(state.prediction_mask_full, phase)


def _refresh_display_layers(state: BreastDisplayState) -> None:
    phase_index = state.active_phase - 1
    phase_volume = state.phase_volumes[phase_index]
    if state.mr_layer is not None:
        state.mr_layer.data = normalize_volume(phase_volume)
        state.mr_layer.name = f"DCE {_phase_label(state.phases[phase_index])}"
        state.mr_layer.visible = not state.show_subtraction

    if state.subtraction_layer is not None and state.subtraction_volumes:
        state.subtraction_layer.data = normalize_volume(state.subtraction_volumes[phase_index])
        state.subtraction_layer.visible = state.show_subtraction

    if state.precontrast_layer is not None and state.precontrast_display is not None:
        state.precontrast_layer.visible = state.show_precontrast

    _update_expert_layer(state)
    _update_boundary_layer(state)
    _update_prediction_layer(state)


def _set_hanging_visibility(state: BreastDisplayState) -> None:
    for layer in state.hanging.get("mip_layers", []):
        layer.visible = state.show_mip_row
    for widget in state.hanging.get("mip_qt_widgets", []):
        widget.setVisible(state.show_mip_row)
    for header in state.hanging.get("mip_headers", []):
        header.setVisible(state.show_mip_row)


def _jump_to_lesion(viewer: napari.Viewer, state: BreastDisplayState) -> None:
    if state.expert_mask_full is None:
        return
    phase = state.phases[state.active_phase - 1]
    lesion_z = lesion_z_in_phase(state.expert_mask_full, phase)
    if lesion_z is None:
        return
    point = list(viewer.dims.point)
    point[0] = float(lesion_z)
    viewer.dims.point = tuple(point)
    for phase_viewer in state.hanging.get("phase_viewers", []):
        phase_viewer.dims.point = tuple(point)


def _set_hanging_panel_visible(state: BreastDisplayState, visible: bool) -> None:
    splitter = state.hanging.get("splitter")
    hanging_wrap = state.hanging.get("hanging_wrap")
    if splitter is None or hanging_wrap is None:
        return
    hanging_wrap.setVisible(visible)
    if visible:
        splitter.setSizes([640, 960])
    else:
        splitter.setSizes([1_000_000, 0])


def add_breast_display_controls(
    viewer: napari.Viewer,
    state: BreastDisplayState,
    *,
    collapse_controls: bool = False,
) -> None:
    from magicgui.widgets import Checkbox, ComboBox, Container, PushButton

    phase_choices = {f"Phase {phase.index}": phase.index for phase in state.phases}
    phase_box = ComboBox(label="DCE phase", choices=list(phase_choices.keys()))
    phase_box.value = f"Phase {state.active_phase}"

    subtraction_box = Checkbox(value=state.show_subtraction, text="Subtraction (phase − pre-contrast S1)")
    subtraction_box.enabled = bool(state.subtraction_volumes)
    precontrast_box = Checkbox(value=False, text="Pre-contrast S1 (Ax T1, resampled)")
    precontrast_box.enabled = state.precontrast_display is not None
    mip_box = Checkbox(value=state.show_mip_row, text="MIP row (bottom of hanging protocol)")
    jump_button = PushButton(text="Jump to expert lesion")
    hanging_box = Checkbox(
        value=True,
        text="Hanging protocol panel (4 phases on right)",
    )
    hanging_box.enabled = bool(state.hanging)

    def on_phase_change(_event: Any = None) -> None:
        state.active_phase = int(phase_choices[str(phase_box.value)])
        _refresh_display_layers(state)

    def on_subtraction_change(_event: Any = None) -> None:
        state.show_subtraction = bool(subtraction_box.value)
        _refresh_display_layers(state)

    def on_precontrast_change(_event: Any = None) -> None:
        state.show_precontrast = bool(precontrast_box.value)
        if state.precontrast_layer is not None:
            state.precontrast_layer.visible = state.show_precontrast

    def on_mip_change(_event: Any = None) -> None:
        state.show_mip_row = bool(mip_box.value)
        _set_hanging_visibility(state)

    def on_hanging_change(_event: Any = None) -> None:
        _set_hanging_panel_visible(state, bool(hanging_box.value))

    def on_jump(_event: Any = None) -> None:
        _jump_to_lesion(viewer, state)

    phase_box.changed.connect(on_phase_change)
    subtraction_box.changed.connect(on_subtraction_change)
    precontrast_box.changed.connect(on_precontrast_change)
    mip_box.changed.connect(on_mip_change)
    if state.hanging:
        hanging_box.changed.connect(on_hanging_change)
    jump_button.clicked.connect(on_jump)

    widgets = [
        phase_box,
        subtraction_box,
        precontrast_box,
        mip_box,
        jump_button,
    ]
    if state.hanging:
        widgets.insert(0, hanging_box)

    controls = Container(widgets=widgets)
    add_collapsible_dock(
        viewer,
        controls,
        name="DCE controls",
        collapsed=collapse_controls,
    )


def view_slug(
    slug: str,
    *,
    show_otsu: bool = False,
    show_cuboid_enhancement: bool = False,
    cuboid_boundary: bool = False,
    skip_precontrast: bool = False,
    no_hanging: bool = False,
    phases_only: bool = False,
    show_mip_row: bool = False,
    threshold_step: float = 0.05,
    wide: bool = False,
    collapse_controls: bool = False,
) -> None:
    entry = find_volume(slug=slug)
    tcga_id = entry["tcga_id"]
    study_date = entry["study_date"]

    les_files = find_les_files(tcga_id)
    if not les_files:
        raise FileNotFoundError(
            f"No .les file for {tcga_id}. Expected under "
            "data/raw/tcia-radiogenomics/lesions/TCGA_Segmented_Lesions_UofC/"
        )
    les_path = les_files[0]

    volume, spacing_mm, series_desc, dce_index, source = load_annotation_volume(
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

    precontrast: tuple[np.ndarray, list[float], str] | None = None
    if not skip_precontrast:
        precontrast = load_precontrast_volume(tcga_id=tcga_id, study_date=study_date)

    precontrast_display: np.ndarray | None = None
    precontrast_label = ""
    if precontrast is not None:
        pre_volume, pre_spacing, pre_desc = precontrast
        precontrast_display = resample_volume(
            pre_volume,
            pre_spacing,
            phase_volumes[0].shape,
            list(spacing_mm),
        )
        precontrast_label = pre_desc

    subtraction_volumes = _build_subtraction_volumes(
        phase_volumes,
        list(spacing_mm),
        (precontrast[0], precontrast[1]) if precontrast is not None else None,
    )

    expert_mask, meta = load_les_mask(les_path, volume.shape)
    boundary_mask, boundary_meta = load_les_cuboid_boundary(les_path, volume.shape)

    expert_name = f".les expert ({meta['lesion_voxels']:,} vox)"
    boundary_name = (
        f".les cuboid bbox "
        f"(y[{boundary_meta['y_start']},{boundary_meta['y_end']}] "
        f"x[{boundary_meta['x_start']},{boundary_meta['x_end']}] "
        f"z[{boundary_meta['z_start']},{boundary_meta['z_end']}])"
    )

    if cuboid_boundary:
        overlay_mask = boundary_mask
        overlay_name = boundary_name
        overlay_detail = f"boundary={boundary_meta['boundary_voxels']:,} vox"
    else:
        overlay_mask = expert_mask
        overlay_name = expert_name
        overlay_detail = f"lesion={meta['lesion_voxels']:,} vox"

    active_phase = 1
    for phase in phases:
        if overlay_mask[phase.z_start : phase.z_end].any():
            active_phase = phase.index
            break

    lesion_z = lesion_z_in_phase(overlay_mask, phases[active_phase - 1])
    phase_summary = ", ".join(_phase_label(phase) for phase in phases)

    print(
        f"{slug}\n"
        f"  series: {series_desc!r} (S{dce_index})\n"
        f"  .les:   {les_path.name} ({overlay_detail})\n"
        f"  MR:     shape={volume.shape} source={source} spacing_mm={spacing_mm}\n"
        f"  phases: {len(phases)} temporal phases — {phase_summary}\n"
        f"  pre:    {precontrast_label or '(unavailable — will download S1 on first run)'}\n"
        f"  layout: "
        + (
            "P1–P4 only"
            if phases_only
            else ("detail + hanging protocol" if not no_hanging else "detail only")
        )
        + (", MIP row on" if show_mip_row and (phases_only or not no_hanging) else "")
        + "\n"
        f"  lesion: phase {active_phase}"
        + (f", z={lesion_z} within phase" if lesion_z is not None else "")
    )

    show_subtraction = bool(subtraction_volumes)

    brightness_rows = compute_cuboid_brightness_table(
        phase_volumes,
        phases,
        meta,
        expert_mask,
        threshold_step=threshold_step,
    )
    print("\nCuboid bright fraction sweep:\n" + format_brightness_table(brightness_rows))

    if phases_only:
        viewer = napari.Viewer(title=f"{slug} — P1–P4")
        _configure_viewer_chrome(viewer, wide=wide)
        hanging = setup_phases_only_view(
            viewer,
            phase_volumes=phase_volumes,
            phases=phases,
            subtraction_volumes=[],
            spacing_mm=scale,
            expert_mask_full=expert_mask,
            expert_layer_name=expert_name,
            boundary_mask_full=boundary_mask,
            boundary_layer_name=boundary_name,
            show_mip_row=show_mip_row,
        )
        if hanging.get("expert_layers"):
            add_overlay_toggle_button(viewer, "Expert .les", *hanging["expert_layers"])
        if hanging.get("boundary_layers"):
            add_overlay_toggle_button(viewer, "Cuboid bbox", *hanging["boundary_layers"])
        add_brightness_fraction_dock(viewer, brightness_rows, threshold_step=threshold_step)
        if lesion_z is not None:
            point = [0.0, 0.0, 0.0]
            point[0] = float(lesion_z)
            for phase_viewer in hanging.get("phase_viewers", []):
                phase_viewer.dims.point = tuple(point)
        napari.run()
        return

    state = BreastDisplayState(
        phases=phases,
        phase_volumes=phase_volumes,
        subtraction_volumes=subtraction_volumes,
        spacing_mm=scale,
        active_phase=active_phase,
        show_subtraction=show_subtraction,
        show_mip_row=show_mip_row,
        precontrast_display=precontrast_display,
        expert_mask_full=overlay_mask,
        boundary_mask_full=boundary_mask if not cuboid_boundary else None,
        cuboid_boundary=cuboid_boundary,
    )

    viewer = napari.Viewer(title=f"{slug} — breast DCE hanging protocol")
    _configure_viewer_chrome(viewer, wide=wide)
    active_volume = phase_volumes[active_phase - 1]
    state.mr_layer = viewer.add_image(
        normalize_volume(active_volume),
        name=f"DCE {_phase_label(phases[active_phase - 1])}",
        scale=scale,
        colormap="gray",
        contrast_limits=(0.0, 1.0),
        visible=not show_subtraction,
    )

    if subtraction_volumes:
        state.subtraction_layer = viewer.add_image(
            normalize_volume(subtraction_volumes[active_phase - 1]),
            name="Subtraction (phase − pre S1)",
            scale=scale,
            colormap="gray",
            contrast_limits=(0.0, 1.0),
            blending="additive",
            visible=show_subtraction,
        )

    if precontrast_display is not None:
        state.precontrast_layer = viewer.add_image(
            normalize_volume(precontrast_display),
            name=f"Pre-contrast S1 ({precontrast_label})",
            scale=scale,
            colormap="gray",
            contrast_limits=(0.0, 1.0),
            opacity=0.45,
            visible=False,
        )

    phase_mask = mask_for_phase(overlay_mask, phases[active_phase - 1])
    state.expert_layer = viewer.add_labels(
        phase_mask,
        name=overlay_name,
        scale=scale,
        opacity=0.85 if cuboid_boundary else 0.55,
    )

    if not cuboid_boundary:
        phase_boundary = mask_for_phase(boundary_mask, phases[active_phase - 1])
        state.boundary_layer = viewer.add_labels(
            phase_boundary,
            name=boundary_name,
            scale=scale,
            opacity=0.85,
        )

    if show_otsu:
        detail_volume = (
            subtraction_volumes[active_phase - 1]
            if show_subtraction
            else active_volume
        )
        viewer.add_labels(
            tumor_mask_largest_component(normalize_volume(detail_volume)),
            name="Otsu (phase only)",
            scale=scale,
            opacity=0.35,
        )

    if show_cuboid_enhancement:
        from seg_paths import mask_npy  # noqa: WPS433

        pred_path = mask_npy(slug, "cuboid_enhancement")
        if pred_path.exists():
            state.prediction_mask_full = np.load(pred_path).astype(np.uint8)
            phase_pred = mask_for_phase(state.prediction_mask_full, phases[active_phase - 1])
            state.prediction_layer = viewer.add_labels(
                phase_pred,
                name="cuboid_enhancement",
                scale=scale,
                opacity=0.4,
            )
            print(f"  overlay: {pred_path} ({int(state.prediction_mask_full.sum()):,} vox)")
        else:
            print(f"  cuboid_enhancement mask not found: {pred_path}")

    if not no_hanging:
        state.hanging = setup_clinical_hanging_protocol(
            viewer,
            phase_volumes=phase_volumes,
            phases=phases,
            subtraction_volumes=subtraction_volumes if show_subtraction else [],
            spacing_mm=scale,
            expert_mask_full=expert_mask if not cuboid_boundary else None,
            expert_layer_name=expert_name,
            boundary_mask_full=boundary_mask,
            boundary_layer_name=boundary_name,
            show_mip_row=state.show_mip_row,
        )

    if state.expert_layer is not None:
        add_overlay_toggle_button(
            viewer,
            "Expert .les",
            state.expert_layer,
            *state.hanging.get("expert_layers", []),
        )
    if state.boundary_layer is not None or state.hanging.get("boundary_layers"):
        boundary_layers = [state.boundary_layer] if state.boundary_layer is not None else []
        add_overlay_toggle_button(
            viewer,
            "Cuboid bbox",
            *boundary_layers,
            *state.hanging.get("boundary_layers", []),
        )
    if show_cuboid_enhancement and state.prediction_layer is not None:
        add_overlay_toggle_button(viewer, "cuboid_enhancement", state.prediction_layer)

    add_brightness_fraction_dock(viewer, brightness_rows, threshold_step=threshold_step)

    add_breast_display_controls(viewer, state, collapse_controls=collapse_controls)
    _jump_to_lesion(viewer, state)
    napari.run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View baseline MR with TCIA radiologist .les overlay in napari.",
    )
    parser.add_argument(
        "--slug",
        help="Manifest slug (baseline only; e.g. luminal_a_TCGA-AR-A1AX_baseline)",
    )
    parser.add_argument(
        "--cuboid",
        action="store_true",
        help="Show .les annotation bounding cuboid shell instead of filled lesion mask",
    )
    parser.add_argument(
        "--otsu",
        action="store_true",
        help="Also overlay Otsu + largest connected component on the active phase",
    )
    parser.add_argument(
        "--cuboid-enhancement",
        action="store_true",
        help="Overlay cuboid_enhancement predicted mask if on disk",
    )
    parser.add_argument(
        "--no-precontrast",
        action="store_true",
        help="Skip downloading/loading pre-contrast S1 (no subtraction layer)",
    )
    parser.add_argument(
        "--no-hanging",
        action="store_true",
        help="Disable side-by-side hanging protocol pane (detail viewer only)",
    )
    parser.add_argument(
        "--phases-only",
        action="store_true",
        help="Full-window P1–P4 grid only (no left detail viewer)",
    )
    parser.add_argument(
        "--mip",
        action="store_true",
        help="Show MIP row under the 4 phase viewers in the hanging protocol panel",
    )
    parser.add_argument(
        "--wide",
        action="store_true",
        help="Hide napari layer list/controls on the left for a wider MR canvas",
    )
    parser.add_argument(
        "--collapse-controls",
        action="store_true",
        help="Start with the DCE controls dock collapsed (click Show controls to expand)",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
        help="Normalized threshold step for cuboid bright-fraction table (default 0.05)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List baseline slugs that have a local .les file",
    )
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
            parser.error(
                "Provide --slug. Baselines with .les: "
                + ", ".join(available)
            )
        else:
            parser.error(
                "No baseline slugs with .les found. Download TCIA masks — see validation/VALIDATION.md"
            )

    view_slug(
        slug,
        show_otsu=args.otsu,
        show_cuboid_enhancement=args.cuboid_enhancement,
        cuboid_boundary=args.cuboid,
        skip_precontrast=args.no_precontrast,
        no_hanging=args.no_hanging,
        phases_only=args.phases_only,
        show_mip_row=args.mip,
        threshold_step=args.threshold_step,
        wide=args.wide,
        collapse_controls=args.collapse_controls,
    )


if __name__ == "__main__":
    main()

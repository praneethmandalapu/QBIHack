"""Interactive bbox threshold slider dock for aligned cuboid napari viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from dce_phases import DcePhase
from les_cuboid_brightness import (
    POSTCONTRAST_ANALYSIS_PHASES,
    bbox_boundary_slab,
    bbox_yx_slices,
    center_connected_mask_in_slab,
    connected_fraction_curve_for_slab,
    elbow_threshold,
    expand_les_meta_yx,
    les_fraction_in_bbox_slab,
)


@dataclass
class BboxThresholdState:
    phase_index: int = 2
    threshold: float = 0.5
    margin_yx: int = 0
    gap_voxels: int = 0
    threshold_step: float = 0.01
    show_postcontrast_bright: bool = False


def add_bbox_threshold_dock(
    viewer: Any,
    *,
    phases: list[DcePhase],
    slabs_aligned: dict[int, np.ndarray],
    expert_slab: np.ndarray,
    les_meta: dict[str, Any],
    threshold_layers: dict[int, Any],
    boundary_layers: dict[int, Any],
    postcontrast_bright_layers: dict[int, Any] | None = None,
    initial_phase: int = 2,
    show_postcontrast_bright: bool = False,
    export_mask: Callable[[float, int, int], str] | None = None,
) -> None:
    """Right dock: P2/P3 threshold slider, center-connected mask, optional export."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    analysis_phases = [p for p in phases if p.index in POSTCONTRAST_ANALYSIS_PHASES]
    if not analysis_phases:
        analysis_phases = [p for p in phases if p.index > 1] or phases[:1]

    slab_shape = next(iter(slabs_aligned.values())).shape
    _, y_size, x_size = slab_shape
    bright_layers = postcontrast_bright_layers or {}

    state = BboxThresholdState(
        phase_index=initial_phase,
        show_postcontrast_bright=show_postcontrast_bright,
    )

    def expanded_meta() -> dict[str, Any]:
        return expand_les_meta_yx(
            les_meta,
            state.margin_yx,
            y_size=y_size,
            x_size=x_size,
        )

    def curve_for_phase(phase_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return connected_fraction_curve_for_slab(
            slabs_aligned[phase_index],
            expanded_meta(),
            threshold_step=state.threshold_step,
            gap_voxels=state.gap_voxels,
        )

    def update_boundary_layers() -> None:
        meta = expanded_meta()
        y_sl, x_sl = bbox_yx_slices(meta)
        boundary = bbox_boundary_slab(slab_shape, y_sl, x_sl)
        for layer in boundary_layers.values():
            layer.data = boundary

    def update_threshold_layers() -> None:
        meta = expanded_meta()
        if 1 in threshold_layers:
            threshold_layers[1].data = center_connected_mask_in_slab(
                slabs_aligned[state.phase_index],
                meta,
                state.threshold,
                gap_voxels=state.gap_voxels,
            )
        if state.show_postcontrast_bright:
            for phase_index, layer in bright_layers.items():
                layer.data = center_connected_mask_in_slab(
                    slabs_aligned[phase_index],
                    meta,
                    state.threshold,
                    gap_voxels=state.gap_voxels,
                )

    def refresh_labels(
        thresholds: np.ndarray,
        fractions: np.ndarray,
        values: np.ndarray,
    ) -> None:
        meta = expanded_meta()
        norm_bbox_voxels = values.size
        connected = center_connected_mask_in_slab(
            slabs_aligned[state.phase_index],
            meta,
            state.threshold,
            gap_voxels=state.gap_voxels,
        )
        y_sl, x_sl = bbox_yx_slices(meta)
        connected_voxels = int(connected[:, y_sl, x_sl].sum())
        elbow_t, elbow_dist = elbow_threshold(thresholds, fractions)
        les_frac, les_vox, bbox_vox = les_fraction_in_bbox_slab(expert_slab, meta)

        fraction_label.setText(
            f"Center-connected @ threshold: {connected_voxels / norm_bbox_voxels * 100:.1f}% "
            f"({connected_voxels:,} / {norm_bbox_voxels:,} bbox voxels)"
        )
        knee_label.setText(
            f"Elbow (connected curve): {elbow_t:.2f}  (perp. dist {elbow_dist:.3f})"
        )
        les_label.setText(
            f".les fill in bbox: {les_frac * 100:.1f}% ({les_vox:,} / {bbox_vox:,}) — reference only"
        )

        slider.blockSignals(True)
        slider.setValue(int(round(state.threshold * 1000)))
        slider.blockSignals(False)
        spin.blockSignals(True)
        spin.setValue(state.threshold)
        spin.blockSignals(False)

    def apply_update(*, recompute_curve: bool = False) -> None:
        thresholds, fractions, values = curve_for_phase(state.phase_index)
        if recompute_curve:
            cached["thresholds"] = thresholds
            cached["fractions"] = fractions
            cached["values"] = values
        update_boundary_layers()
        update_threshold_layers()
        refresh_labels(thresholds, fractions, values)

    def on_slider(value: int) -> None:
        state.threshold = value / 1000.0
        spin.blockSignals(True)
        spin.setValue(state.threshold)
        spin.blockSignals(False)
        update_threshold_layers()
        refresh_labels(cached["thresholds"], cached["fractions"], cached["values"])

    def on_spin(value: float) -> None:
        state.threshold = float(value)
        slider.blockSignals(True)
        slider.setValue(int(round(state.threshold * 1000)))
        slider.blockSignals(False)
        update_threshold_layers()
        refresh_labels(cached["thresholds"], cached["fractions"], cached["values"])

    def on_phase_changed(index: int) -> None:
        state.phase_index = analysis_phases[index].index
        apply_update(recompute_curve=True)

    def on_margin_changed(value: int) -> None:
        state.margin_yx = int(value)
        apply_update(recompute_curve=True)

    def on_gap_changed(value: int) -> None:
        state.gap_voxels = int(value)
        apply_update(recompute_curve=True)

    def set_postcontrast_bright_visible(visible: bool) -> None:
        state.show_postcontrast_bright = visible
        for layer in bright_layers.values():
            layer.visible = visible
        if visible:
            update_threshold_layers()

    def on_postcontrast_bright_toggled(checked: bool) -> None:
        set_postcontrast_bright_visible(checked)

    def jump_to_elbow() -> None:
        thresholds = cached["thresholds"]
        fractions = cached["fractions"]
        elbow_t, _ = elbow_threshold(thresholds, fractions)
        state.threshold = elbow_t
        apply_update(recompute_curve=False)

    def on_export() -> None:
        if export_mask is None:
            export_status.setText("Export not configured for this viewer.")
            return
        try:
            path = export_mask(state.threshold, state.phase_index, state.gap_voxels)
            export_status.setText(f"Saved: {path}")
        except Exception as exc:  # noqa: BLE001 — show in dock
            export_status.setText(f"Export failed: {exc}")

    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.addWidget(
        QLabel(
            "P2/P3 analysis: center-connected region from bbox center (gap=0 is strict 26-NN). "
            "P1 shows the selected phase mask; optional red overlay on P2–P3. "
            "Tools → Segmentation (nsbatwm) available for manual threshold on a single layer."
        )
    )

    form = QFormLayout()
    phase_combo = QComboBox()
    for phase in analysis_phases:
        label = f"P{phase.index}"
        if phase.acquisition_time:
            label += f" ({phase.acquisition_time})"
        phase_combo.addItem(label)
    default_index = next(
        (index for index, phase in enumerate(analysis_phases) if phase.index == initial_phase),
        0,
    )
    phase_combo.setCurrentIndex(default_index)
    phase_combo.currentIndexChanged.connect(on_phase_changed)
    form.addRow("Threshold phase", phase_combo)

    margin_spin = QSpinBox()
    margin_spin.setRange(0, 40)
    margin_spin.setValue(0)
    margin_spin.setToolTip("Expand .les Y/X bounds symmetrically (voxels)")
    margin_spin.valueChanged.connect(on_margin_changed)
    form.addRow("Bbox margin (Y/X)", margin_spin)

    gap_spin = QSpinBox()
    gap_spin.setRange(0, 10)
    gap_spin.setValue(0)
    gap_spin.setToolTip(
        "Stub: morphological closing radius before CC (0=strict singly connected, 1–10 bridges gaps)"
    )
    gap_spin.valueChanged.connect(on_gap_changed)
    form.addRow("Connectivity gap", gap_spin)

    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, 1000)
    slider.setValue(int(round(state.threshold * 1000)))
    slider.valueChanged.connect(on_slider)

    spin = QDoubleSpinBox()
    spin.setRange(0.0, 1.0)
    spin.setSingleStep(0.01)
    spin.setDecimals(3)
    spin.setValue(state.threshold)
    spin.valueChanged.connect(on_spin)

    slider_row = QHBoxLayout()
    slider_row.addWidget(QLabel("0"))
    slider_row.addWidget(slider, stretch=1)
    slider_row.addWidget(QLabel("1"))
    slider_row.addWidget(spin)
    form.addRow("Threshold", slider_row)

    postcontrast_check = QCheckBox("Red center-connected on P2–P3")
    postcontrast_check.setChecked(show_postcontrast_bright)
    postcontrast_check.setToolTip("Each P2/P3 panel shows its own center-connected mask in red.")
    postcontrast_check.setEnabled(bool(bright_layers))
    postcontrast_check.toggled.connect(on_postcontrast_bright_toggled)
    form.addRow("Post-contrast", postcontrast_check)

    layout.addLayout(form)

    elbow_button = QPushButton("Jump to elbow (connected curve)")
    elbow_button.clicked.connect(jump_to_elbow)
    layout.addWidget(elbow_button)

    export_button = QPushButton("Export mask → .npy")
    export_button.setEnabled(export_mask is not None)
    export_button.clicked.connect(on_export)
    layout.addWidget(export_button)

    fraction_label = QLabel("")
    knee_label = QLabel("")
    les_label = QLabel("")
    export_status = QLabel("")
    for label in (fraction_label, knee_label, les_label, export_status):
        label.setWordWrap(True)
        layout.addWidget(label)

    layout.addStretch(1)
    viewer.window.add_dock_widget(widget, area="right", name="Bbox threshold")

    cached: dict[str, Any] = {}
    state.phase_index = analysis_phases[default_index].index
    thresholds, fractions, _values = curve_for_phase(state.phase_index)
    elbow_t, _ = elbow_threshold(thresholds, fractions)
    state.threshold = elbow_t
    cached["thresholds"] = thresholds
    cached["fractions"] = fractions
    cached["values"] = _values
    apply_update(recompute_curve=False)

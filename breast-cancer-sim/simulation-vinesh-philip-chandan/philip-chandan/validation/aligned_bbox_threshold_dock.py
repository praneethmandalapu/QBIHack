"""Interactive bbox threshold slider dock for aligned cuboid napari viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dce_phases import DcePhase
from les_cuboid_brightness import (
    bbox_boundary_slab,
    bbox_yx_slices,
    bright_fraction_at_threshold,
    expand_les_meta_yx,
    fraction_curve_for_slab,
    les_fraction_in_bbox_slab,
    steepest_dropout_threshold,
    threshold_mask_in_slab,
)


@dataclass
class BboxThresholdState:
    phase_index: int = 2
    threshold: float = 0.5
    margin_yx: int = 0
    threshold_step: float = 0.01


def add_bbox_threshold_dock(
    viewer: Any,
    *,
    phases: list[DcePhase],
    slabs_aligned: dict[int, np.ndarray],
    expert_slab: np.ndarray,
    les_meta: dict[str, Any],
    phase_viewers: list[Any],
    threshold_layers: dict[int, Any],
    boundary_layers: dict[int, Any],
    initial_phase: int = 2,
) -> None:
    """Right dock: phase picker, threshold slider, bbox margin, knee hint."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
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

    slab_shape = next(iter(slabs_aligned.values())).shape
    _, y_size, x_size = slab_shape

    state = BboxThresholdState(phase_index=initial_phase)

    def expanded_meta() -> dict[str, Any]:
        return expand_les_meta_yx(
            les_meta,
            state.margin_yx,
            y_size=y_size,
            x_size=x_size,
        )

    def curve_for_phase(phase_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return fraction_curve_for_slab(
            slabs_aligned[phase_index],
            expanded_meta(),
            threshold_step=state.threshold_step,
        )

    def update_boundary_layers() -> None:
        meta = expanded_meta()
        y_sl, x_sl = bbox_yx_slices(meta)
        boundary = bbox_boundary_slab(slab_shape, y_sl, x_sl)
        for layer in boundary_layers.values():
            layer.data = boundary

    def update_threshold_layers() -> None:
        meta = expanded_meta()
        slab = slabs_aligned[state.phase_index]
        mask = threshold_mask_in_slab(slab, meta, state.threshold)
        for layer in threshold_layers.values():
            layer.data = mask

    def refresh_labels(
        thresholds: np.ndarray,
        fractions: np.ndarray,
        values: np.ndarray,
    ) -> None:
        frac = bright_fraction_at_threshold(values, state.threshold)
        knee_t, knee_slope = steepest_dropout_threshold(thresholds, fractions)
        les_frac, les_vox, bbox_vox = les_fraction_in_bbox_slab(expert_slab, expanded_meta())

        fraction_label.setText(
            f"Bright fraction @ threshold: {frac * 100:.1f}% "
            f"({int(frac * values.size):,} / {values.size:,} bbox voxels)"
        )
        knee_label.setText(
            f"Steepest drop (knee): {knee_t:.2f}  "
            f"(slope {knee_slope * 100:.1f} %/unit)"
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
        thresholds = cached["thresholds"]
        fractions = cached["fractions"]
        values = cached["values"]
        update_threshold_layers()
        refresh_labels(thresholds, fractions, values)

    def on_spin(value: float) -> None:
        state.threshold = float(value)
        slider.blockSignals(True)
        slider.setValue(int(round(state.threshold * 1000)))
        slider.blockSignals(False)
        thresholds = cached["thresholds"]
        fractions = cached["fractions"]
        values = cached["values"]
        update_threshold_layers()
        refresh_labels(thresholds, fractions, values)

    def on_phase_changed(index: int) -> None:
        state.phase_index = phases[index].index
        apply_update(recompute_curve=True)

    def on_margin_changed(value: int) -> None:
        state.margin_yx = int(value)
        apply_update(recompute_curve=True)

    def jump_to_knee() -> None:
        thresholds = cached["thresholds"]
        fractions = cached["fractions"]
        knee_t, _ = steepest_dropout_threshold(thresholds, fractions)
        state.threshold = knee_t
        apply_update(recompute_curve=False)

    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.addWidget(
        QLabel(
            "Threshold voxels inside the bbox on P1 (post alignment). "
            "P2–P4 panels show aligned tissue only. "
            "Use the slider to find a sharp drop on the fraction curve."
        )
    )

    form = QFormLayout()
    phase_combo = QComboBox()
    for phase in phases:
        label = f"P{phase.index}"
        if phase.acquisition_time:
            label += f" ({phase.acquisition_time})"
        phase_combo.addItem(label)
    default_index = next(
        (index for index, phase in enumerate(phases) if phase.index == initial_phase),
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

    layout.addLayout(form)

    knee_button = QPushButton("Jump to steepest drop (knee)")
    knee_button.clicked.connect(jump_to_knee)
    layout.addWidget(knee_button)

    fraction_label = QLabel("")
    knee_label = QLabel("")
    les_label = QLabel("")
    for label in (fraction_label, knee_label, les_label):
        label.setWordWrap(True)
        layout.addWidget(label)

    layout.addStretch(1)
    viewer.window.add_dock_widget(widget, area="right", name="Bbox threshold")

    cached: dict[str, Any] = {}
    state.phase_index = phases[default_index].index
    thresholds, fractions, _values = curve_for_phase(state.phase_index)
    knee_t, _ = steepest_dropout_threshold(thresholds, fractions)
    state.threshold = knee_t
    cached["thresholds"] = thresholds
    cached["fractions"] = fractions
    cached["values"] = _values
    apply_update(recompute_curve=False)

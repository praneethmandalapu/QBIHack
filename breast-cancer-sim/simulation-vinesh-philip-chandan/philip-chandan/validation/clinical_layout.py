"""Clinical hanging-protocol layout for breast DCE napari QC."""

from __future__ import annotations

from typing import Any

import numpy as np

from dce_phases import DcePhase, mask_for_phase, mip_as_volume
from prep_volume import normalize_volume


def link_viewer_dims(left: Any, right: Any) -> None:
    """Keep slice position and viewing plane in sync across two napari viewers."""
    lock = {"active": False}

    def sync(src: Any, dst: Any):
        def _cb(_event: Any = None) -> None:
            if lock["active"]:
                return
            lock["active"] = True
            try:
                dst.dims.point = src.dims.point
                if tuple(dst.dims.order) != tuple(src.dims.order):
                    dst.dims.order = src.dims.order
            finally:
                lock["active"] = False

        return _cb

    left.dims.events.point.connect(sync(left, right))
    right.dims.events.point.connect(sync(right, left))
    left.dims.events.order.connect(sync(left, right))
    right.dims.events.order.connect(sync(right, left))


def link_viewers(*viewers: Any) -> None:
    """Link dimensions across many viewers using the first as reference."""
    if len(viewers) < 2:
        return
    reference = viewers[0]
    for viewer in viewers[1:]:
        link_viewer_dims(reference, viewer)


def _display_volume(volume: np.ndarray) -> np.ndarray:
    return normalize_volume(volume)


def _mip_scale(scale: tuple[float, float, float]) -> tuple[float, float, float]:
    _dz, dy, dx = scale
    return (1.0, dy, dx)


def _build_phase_grid(
    *,
    phase_volumes: list[np.ndarray],
    phases: list[DcePhase],
    subtraction_volumes: list[np.ndarray],
    spacing_mm: tuple[float, float, float],
    expert_mask_full: np.ndarray | None,
    expert_layer_name: str,
    prediction_mask_full: np.ndarray | None,
    show_mip_row: bool,
) -> tuple[Any, dict[str, Any]]:
    """Build a 4-column grid of linked phase viewers (optional MIP row beneath)."""
    from napari._qt.qt_viewer import QtViewer
    from napari.components.viewer_model import ViewerModel
    from qtpy.QtWidgets import QGridLayout, QLabel, QWidget

    display_volumes = subtraction_volumes or phase_volumes
    n_phases = len(display_volumes)
    volume_kind = "Subtraction" if subtraction_volumes else "DCE"

    phase_viewers: list[Any] = []
    mip_viewers: list[Any] = []
    phase_layers: list[Any] = []
    mip_layers: list[Any] = []
    mip_headers: list[Any] = []
    mip_qt_widgets: list[Any] = []
    expert_layers: list[Any] = []
    prediction_layers: list[Any] = []

    grid = QGridLayout()
    grid.setSpacing(4)

    for column, phase in enumerate(phases[:n_phases]):
        volume = display_volumes[column]
        phase_model = ViewerModel(title=f"P{phase.index}")
        phase_qt = QtViewer(phase_model)
        phase_layer = phase_model.add_image(
            _display_volume(volume),
            name=f"{volume_kind} P{phase.index}",
            scale=spacing_mm,
            colormap="gray",
            contrast_limits=(0.0, 1.0),
        )
        phase_viewers.append(phase_model)
        phase_layers.append(phase_layer)

        header = QLabel(f"P{phase.index}")
        if phase.acquisition_time:
            header.setText(f"P{phase.index} ({phase.acquisition_time})")
        grid.addWidget(header, 0, column)
        grid.addWidget(phase_qt, 1, column)

        if expert_mask_full is not None:
            phase_expert = mask_for_phase(expert_mask_full, phase)
            if phase_expert.any():
                expert_layer = phase_model.add_labels(
                    phase_expert,
                    name=expert_layer_name,
                    scale=spacing_mm,
                    opacity=0.55,
                )
                expert_layers.append(expert_layer)

        if prediction_mask_full is not None:
            phase_pred = mask_for_phase(prediction_mask_full, phase)
            if phase_pred.any():
                pred_layer = phase_model.add_labels(
                    phase_pred,
                    name="cuboid_enhancement",
                    scale=spacing_mm,
                    opacity=0.4,
                )
                prediction_layers.append(pred_layer)

        mip_model = ViewerModel(title=f"MIP P{phase.index}")
        mip_qt = QtViewer(mip_model)
        mip_scale = _mip_scale(spacing_mm)
        mip_layer = mip_model.add_image(
            _display_volume(mip_as_volume(volume)),
            name=f"MIP P{phase.index}",
            scale=mip_scale,
            colormap="gray",
            contrast_limits=(0.0, 1.0),
            visible=show_mip_row,
        )
        mip_viewers.append(mip_model)
        mip_layers.append(mip_layer)

        mip_header = QLabel(f"P{phase.index} MIP")
        mip_header.setVisible(show_mip_row)
        mip_headers.append(mip_header)
        mip_qt_widgets.append(mip_qt)
        grid.addWidget(mip_header, 2, column)
        grid.addWidget(mip_qt, 3, column)
        mip_qt.setVisible(show_mip_row)

    link_viewers(*phase_viewers)
    if mip_viewers:
        link_viewers(*mip_viewers)

    grid_widget = QWidget()
    grid_widget.setLayout(grid)

    return grid_widget, {
        "phase_viewers": phase_viewers,
        "mip_viewers": mip_viewers,
        "phase_layers": phase_layers,
        "mip_layers": mip_layers,
        "mip_headers": mip_headers,
        "mip_qt_widgets": mip_qt_widgets,
        "expert_layers": expert_layers,
        "prediction_layers": prediction_layers,
    }


def setup_phases_only_view(
    main_viewer: Any,
    *,
    phase_volumes: list[np.ndarray],
    phases: list[DcePhase],
    subtraction_volumes: list[np.ndarray],
    spacing_mm: tuple[float, float, float],
    expert_mask_full: np.ndarray | None,
    expert_layer_name: str,
    prediction_mask_full: np.ndarray | None = None,
    show_mip_row: bool = False,
) -> dict[str, Any]:
    """Full-window P1–P4 grid (no detail pane)."""
    from qtpy.QtWidgets import QVBoxLayout, QWidget

    grid_widget, hanging = _build_phase_grid(
        phase_volumes=phase_volumes,
        phases=phases,
        subtraction_volumes=subtraction_volumes,
        spacing_mm=spacing_mm,
        expert_mask_full=expert_mask_full,
        expert_layer_name=expert_layer_name,
        prediction_mask_full=prediction_mask_full,
        show_mip_row=show_mip_row,
    )

    wrap = QWidget()
    layout = QVBoxLayout(wrap)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(grid_widget, stretch=1)
    main_viewer.window._qt_window.setCentralWidget(wrap)
    return hanging


def setup_clinical_hanging_protocol(
    main_viewer: Any,
    *,
    phase_volumes: list[np.ndarray],
    phases: list[DcePhase],
    subtraction_volumes: list[np.ndarray],
    spacing_mm: tuple[float, float, float],
    expert_mask_full: np.ndarray | None,
    expert_layer_name: str,
    prediction_mask_full: np.ndarray | None = None,
    show_mip_row: bool = False,
) -> dict[str, Any]:
    """Split the window: detail viewer left, 4-up phases + optional MIP row right."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QSplitter, QVBoxLayout, QWidget

    grid_widget, hanging = _build_phase_grid(
        phase_volumes=phase_volumes,
        phases=phases,
        subtraction_volumes=subtraction_volumes,
        spacing_mm=spacing_mm,
        expert_mask_full=expert_mask_full,
        expert_layer_name=expert_layer_name,
        prediction_mask_full=prediction_mask_full,
        show_mip_row=show_mip_row,
    )

    phase_viewers = hanging["phase_viewers"]
    if phase_viewers and main_viewer is not None:
        link_viewer_dims(main_viewer, phase_viewers[0])

    qt_main = main_viewer.window._qt_window._qt_viewer
    detail_wrap = QWidget()
    detail_layout = QVBoxLayout(detail_wrap)
    detail_layout.setContentsMargins(0, 0, 0, 0)
    detail_layout.setSpacing(2)
    detail_layout.addWidget(qt_main, stretch=1)

    hanging_wrap = QWidget()
    hanging_layout = QVBoxLayout(hanging_wrap)
    hanging_layout.setContentsMargins(0, 0, 0, 0)
    hanging_layout.setSpacing(2)
    hanging_layout.addWidget(grid_widget, stretch=1)

    splitter = QSplitter(Qt.Horizontal)
    splitter.addWidget(detail_wrap)
    splitter.addWidget(hanging_wrap)
    splitter.setStretchFactor(0, 2)
    splitter.setStretchFactor(1, 3)
    splitter.setSizes([640, 960])
    main_viewer.window._qt_window.setCentralWidget(splitter)

    hanging["splitter"] = splitter
    hanging["hanging_wrap"] = hanging_wrap
    return hanging

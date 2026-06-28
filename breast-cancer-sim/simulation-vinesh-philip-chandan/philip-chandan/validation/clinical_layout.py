"""Clinical hanging-protocol layout for breast DCE napari QC."""

from __future__ import annotations

from typing import Any

import numpy as np

from dce_phases import DcePhase, detect_cad_markers, expert_centroid_zyx, mip_as_volume
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


def setup_clinical_hanging_protocol(
    main_viewer: Any,
    *,
    phase_volumes: list[np.ndarray],
    phases: list[DcePhase],
    subtraction_volumes: list[np.ndarray],
    spacing_mm: tuple[float, float, float],
    expert_mask_full: np.ndarray | None,
    expert_layer_name: str,
    show_mip_row: bool = True,
    show_cad: bool = True,
) -> dict[str, Any]:
    """Split the window: detail viewer left, 4-up phases + MIP row right."""
    from napari._qt.qt_viewer import QtViewer
    from napari.components.viewer_model import ViewerModel
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QGridLayout, QLabel, QSplitter, QVBoxLayout, QWidget

    display_volumes = subtraction_volumes or phase_volumes
    n_phases = len(display_volumes)
    volume_kind = "Subtraction" if subtraction_volumes else "DCE"

    phase_viewers: list[Any] = []
    mip_viewers: list[Any] = []
    phase_layers: list[Any] = []
    mip_layers: list[Any] = []
    cad_layers: list[Any] = []
    expert_layers: list[Any] = []

    grid = QGridLayout()
    grid.setSpacing(4)

    for column, phase in enumerate(phases[:n_phases]):
        volume = display_volumes[column]
        cad_coords = np.empty((0, 3), dtype=np.float32)
        phase_model = ViewerModel(title=f"{volume_kind} phase {phase.index}")
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

        header = QLabel(f"P{phase.index} — {volume_kind.lower()}")
        if phase.acquisition_time:
            header.setText(f"P{phase.index} — {volume_kind.lower()} ({phase.acquisition_time})")
        grid.addWidget(header, 0, column)
        grid.addWidget(phase_qt, 1, column)

        if expert_mask_full is not None:
            from dce_phases import mask_for_phase

            phase_expert = mask_for_phase(expert_mask_full, phase)
            if phase_expert.any():
                expert_layer = phase_model.add_labels(
                    phase_expert,
                    name=expert_layer_name,
                    scale=spacing_mm,
                    opacity=0.55,
                )
                expert_layers.append(expert_layer)

        if show_cad:
            cad_coords = detect_cad_markers(volume)
            expert_centroid = (
                expert_centroid_zyx(expert_mask_full, phase)
                if expert_mask_full is not None
                else None
            )
            if expert_centroid is not None:
                cad_coords = (
                    np.vstack([cad_coords, np.array([expert_centroid], dtype=np.float32)])
                    if cad_coords.size
                    else np.array([expert_centroid], dtype=np.float32)
                )
            if cad_coords.size:
                cad_layer = phase_model.add_points(
                    cad_coords,
                    name="CAD markers",
                    size=10,
                    face_color="yellow",
                    border_color="black",
                    symbol="disc",
                )
                if expert_centroid is not None:
                    cad_layer.face_color = [
                        "lime" if np.allclose(row, expert_centroid, atol=0.5) else "yellow"
                        for row in cad_coords
                    ]
                cad_layers.append(cad_layer)

        if show_mip_row:
            mip_model = ViewerModel(title=f"MIP phase {phase.index}")
            mip_qt = QtViewer(mip_model)
            mip_scale = _mip_scale(spacing_mm)
            mip_layer = mip_model.add_image(
                _display_volume(mip_as_volume(volume)),
                name=f"MIP P{phase.index}",
                scale=mip_scale,
                colormap="gray",
                contrast_limits=(0.0, 1.0),
            )
            mip_viewers.append(mip_model)
            mip_layers.append(mip_layer)

            mip_header = QLabel(f"P{phase.index} — MIP")
            grid.addWidget(mip_header, 2, column)
            grid.addWidget(mip_qt, 3, column)

            if show_cad and cad_coords.size:
                mip_points = cad_coords.copy()
                mip_points[:, 0] = 0.0
                mip_model.add_points(
                    mip_points,
                    name="CAD MIP",
                    size=10,
                    face_color="yellow",
                    border_color="black",
                    symbol="disc",
                )

    link_viewers(*phase_viewers)
    if mip_viewers:
        link_viewers(*mip_viewers)
    if phase_viewers and main_viewer is not None:
        link_viewer_dims(main_viewer, phase_viewers[0])

    grid_widget = QWidget()
    grid_widget.setLayout(grid)

    qt_main = main_viewer.window._qt_window._qt_viewer
    detail_wrap = QWidget()
    detail_layout = QVBoxLayout(detail_wrap)
    detail_layout.setContentsMargins(0, 0, 0, 0)
    detail_layout.setSpacing(2)
    detail_layout.addWidget(QLabel("Detail — phase picker + overlays"))
    detail_layout.addWidget(qt_main, stretch=1)

    hanging_wrap = QWidget()
    hanging_layout = QVBoxLayout(hanging_wrap)
    hanging_layout.setContentsMargins(0, 0, 0, 0)
    hanging_layout.setSpacing(2)
    hanging_layout.addWidget(QLabel("Hanging protocol — linked phases + MIPs + CAD"))
    hanging_layout.addWidget(grid_widget, stretch=1)

    splitter = QSplitter(Qt.Horizontal)
    splitter.addWidget(detail_wrap)
    splitter.addWidget(hanging_wrap)
    splitter.setStretchFactor(0, 2)
    splitter.setStretchFactor(1, 3)
    splitter.setSizes([640, 960])
    main_viewer.window._qt_window.setCentralWidget(splitter)

    return {
        "phase_viewers": phase_viewers,
        "mip_viewers": mip_viewers,
        "phase_layers": phase_layers,
        "mip_layers": mip_layers,
        "cad_layers": cad_layers,
        "expert_layers": expert_layers,
    }

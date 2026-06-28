"""Interactive 3D viewer: brain MR + expert segmentation overlay (napari).

Clinical-style QC controls: brain-masked window/level, optional CLAHE, FLAIR/T2
when available, linked dual-pane layout (annotation + sequence explorer), and
optional orthogonal MPR grid on the annotation pane.

Run (macOS/Linux):
    cd brain-cancer-sim
    python3.11 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt   # includes napari[pyqt6]

    python simulation-vinesh-philip-chandan/philip-chandan/view_volume_napari.py
    python simulation-vinesh-philip-chandan/philip-chandan/view_volume_napari.py --slug glioma_ucsf_100002_baseline

Run (Windows):
    cd brain-cancer-sim
    py -3 -m venv .venv
    .venv\\Scripts\\Activate.ps1
    pip install -r requirements.txt
    .venv\\Scripts\\python.exe simulation-vinesh-philip-chandan\\philip-chandan\\view_volume_napari.py --demo

Requires napari[pyqt6]. First launch may take ~10s while Qt initializes.
Arrays are (Z, Y, X); napari scale is (dz, dy, dx) mm from sidecar metadata.

CLAHE and WW/WL are for QC only — not for diagnostic reading.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

IMAGING_DIR = Path(__file__).resolve().parent
SIM_ROOT = IMAGING_DIR.parent
REPO_ROOT = SIM_ROOT.parent

sys.path.insert(0, str(IMAGING_DIR))
sys.path.insert(0, str(SIM_ROOT))
sys.path.insert(0, str(SIM_ROOT / "vinesh"))

from handoff_contract import (  # noqa: E402
    default_grid_size,
    grid_size_options,
    load_handoff_contract,
    pde_input_spec,
    raw_extract_spec,
)
from spike_paths import pde_input_npy, pde_input_npy_legacy  # noqa: E402
from nifti_extractor import resolve_ucsf_supplementary_paths  # noqa: E402
from tumor_pde_solver import dummy_volume  # noqa: E402

PLANE_ORDERS = {
    "axial": (0, 1, 2),
    "coronal": (1, 0, 2),
    "sagittal": (2, 0, 1),
}
MPR_PLANE_TRANSPOSE = {
    "axial": (0, 1, 2),
    "coronal": (1, 0, 2),
    "sagittal": (2, 0, 1),
}
MPR_SCALE_PERMUTE = {
    "axial": (0, 1, 2),
    "coronal": (1, 0, 2),
    "sagittal": (2, 0, 1),
}


@dataclass
class SeriesVolume:
    name: str
    volume: np.ndarray
    scale: tuple[float, float, float]
    path: Path | None = None


@dataclass
class DisplayState:
    level: float
    width: float
    clahe: bool = False
    series: dict[str, SeriesVolume] = field(default_factory=dict)
    brain_masks: dict[str, np.ndarray] = field(default_factory=dict)
    mr_layers: dict[str, Any] = field(default_factory=dict)
    mpr_layers: dict[str, Any] = field(default_factory=dict)
    seg_layer: Any | None = None
    explorer: Any | None = None
    explorer_layers: dict[str, Any] = field(default_factory=dict)
    primary: str = "t1ce"
    orthogonal_grid: bool = False
    _syncing_dims: bool = False


def _repo_path(relative: str) -> Path:
    return REPO_ROOT / relative


def brain_tissue_mask(volume: np.ndarray) -> np.ndarray:
    """Mask intracranial voxels; ignore black padding outside the head."""
    positive = volume > 0
    if not positive.any():
        return positive
    threshold = max(float(np.percentile(volume[positive], 0.5)), 1.0)
    return volume > threshold


def brain_window_level(
    volume: np.ndarray,
    mask: np.ndarray,
    *,
    lo_percentile: float = 2.0,
    hi_percentile: float = 98.0,
) -> tuple[float, float]:
    """Default level/width from brain-masked percentiles."""
    if not mask.any():
        values = volume.ravel()
    else:
        values = volume[mask]
    lo = float(np.percentile(values, lo_percentile))
    hi = float(np.percentile(values, hi_percentile))
    if hi <= lo:
        hi = lo + 1.0
    return (lo + hi) / 2.0, hi - lo


def contrast_limits_from_wl(level: float, width: float) -> tuple[float, float]:
    half = max(width / 2.0, 0.5)
    return level - half, level + half


def wl_from_contrast_limits(lo: float, hi: float) -> tuple[float, float]:
    return (lo + hi) / 2.0, max(hi - lo, 1.0)


def apply_clahe_slice(slice_2d: np.ndarray, mask_2d: np.ndarray) -> np.ndarray:
    from skimage import exposure

    if not mask_2d.any():
        return slice_2d
    out = slice_2d.astype(np.float32, copy=True)
    brain_vals = out[mask_2d]
    lo, hi = np.percentile(brain_vals, (1.0, 99.0))
    if hi <= lo:
        return out
    norm = np.clip((out - lo) / (hi - lo), 0.0, 1.0)
    enhanced = exposure.equalize_adapthist(norm, clip_limit=0.02)
    return (enhanced * (hi - lo) + lo).astype(np.float32)


def apply_clahe(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Slice-wise CLAHE inside the brain mask (QC only)."""
    out = volume.astype(np.float32, copy=True)
    for z in range(volume.shape[0]):
        if mask[z].any():
            out[z] = apply_clahe_slice(volume[z], mask[z])
    return out


def render_mr_volume(state: DisplayState, name: str) -> np.ndarray:
    series = state.series[name]
    mask = state.brain_masks[name]
    volume = apply_clahe(series.volume, mask) if state.clahe else series.volume
    return volume.astype(np.float32, copy=False)


def update_mr_contrast(state: DisplayState) -> None:
    limits = contrast_limits_from_wl(state.level, state.width)
    for layer in state.mr_layers.values():
        layer.contrast_limits = limits
    for layer in state.explorer_layers.values():
        layer.contrast_limits = limits


def refresh_mr_layers(state: DisplayState) -> None:
    limits = contrast_limits_from_wl(state.level, state.width)
    for name, layer in state.mr_layers.items():
        layer.data = render_mr_volume(state, name)
        layer.contrast_limits = limits
    for name, layer in state.explorer_layers.items():
        layer.data = render_mr_volume(state, name)
        layer.contrast_limits = limits
    if state.orthogonal_grid:
        refresh_mpr_layers(state)


def permute_scale(
    scale: tuple[float, float, float],
    order: tuple[int, int, int],
) -> tuple[float, float, float]:
    return tuple(float(scale[i]) for i in order)


def transpose_volume(volume: np.ndarray, order: tuple[int, int, int]) -> np.ndarray:
    return np.transpose(volume, order)


def load_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {data.shape} from {path}")
    volume = np.transpose(data, (2, 1, 0)).astype(np.float32)
    zooms = img.header.get_zooms()[:3]
    spacing = (float(zooms[2]), float(zooms[1]), float(zooms[0]))
    return volume, spacing


def load_raw_extract(slug: str) -> tuple[np.ndarray, dict[str, Any], Path]:
    spec = raw_extract_spec()
    out_dir = _repo_path(spec["output_dir"])
    npy_path = out_dir / f"{slug}.npy"
    json_path = out_dir / f"{slug}.json"
    if not npy_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"Missing raw extract for {slug!r}. Expected:\n"
            f"  {npy_path}\n  {json_path}"
        )
    with json_path.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    volume = np.load(npy_path).astype(np.float32)
    return volume, meta, json_path


def resolve_segmentation_path(meta: dict[str, Any], slug: str) -> Path | None:
    if meta.get("segmentation_path"):
        path = Path(meta["segmentation_path"])
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.exists():
            return path
    seg_dir = _repo_path("data/processed/segmentations")
    for candidate in (
        seg_dir / f"{slug}_mask.npy",
        seg_dir / f"{slug}.npy",
        seg_dir / f"{slug}_mask.nii.gz",
        seg_dir / f"{slug}.nii.gz",
    ):
        if candidate.exists():
            return candidate
    return None


def load_segmentation(path: Path, target_shape: tuple[int, ...]) -> np.ndarray:
    if path.suffix == ".npy" or path.name.endswith(".npy"):
        mask = np.load(path)
    else:
        mask, _ = load_nifti(path)
    if mask.shape != target_shape:
        raise ValueError(
            f"Segmentation shape {mask.shape} != MR shape {target_shape} ({path})"
        )
    return (mask > 0).astype(np.uint8)


def discover_ucsf_series(meta: dict[str, Any]) -> dict[str, Path]:
    patient_id = meta.get("patient_id")
    if not patient_id:
        return {}
    patient_dir = REPO_ROOT / "data" / "raw" / "ucsf_alptdg" / str(patient_id)
    if not patient_dir.is_dir():
        return {}
    timepoint = str(meta.get("timepoint", "baseline"))
    return resolve_ucsf_supplementary_paths(patient_dir, timepoint)


def discover_series_next_to_mr(mr_path: Path) -> dict[str, Path]:
    if mr_path.parent.name.isdigit():
        return resolve_ucsf_supplementary_paths(mr_path.parent, "baseline")
    return {}


def build_series_catalog(
    primary: SeriesVolume,
    extra_paths: dict[str, Path],
) -> dict[str, SeriesVolume]:
    catalog: dict[str, SeriesVolume] = {primary.name.lower(): primary}
    for name, path in extra_paths.items():
        key = name.lower()
        if key in catalog:
            continue
        volume, spacing = load_nifti(path)
        catalog[key] = SeriesVolume(name=name.upper(), volume=volume, scale=spacing, path=path)
    return catalog


def link_viewer_dims(left: Any, right: Any) -> None:
    """Keep slice position and viewing plane in sync across two viewers."""
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


def setup_dual_view_panes(viewer: Any, state: DisplayState) -> None:
    """Split the window: annotation (T1CE + seg) left, all sequences right."""
    from napari._qt.qt_viewer import QtViewer
    from napari.components.viewer_model import ViewerModel
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QLabel, QSplitter, QVBoxLayout, QWidget

    explorer = ViewerModel(title="Sequences")
    qt_explorer = QtViewer(explorer)
    limits = contrast_limits_from_wl(state.level, state.width)
    for name, series in state.series.items():
        layer = explorer.add_image(
            render_mr_volume(state, name),
            name=series.name,
            scale=series.scale,
            colormap="gray",
            contrast_limits=limits,
            blending="additive",
            visible=(name == state.primary),
        )
        state.explorer_layers[name] = layer
    state.explorer = explorer
    explorer.dims.order = viewer.dims.order
    explorer.dims.point = viewer.dims.point
    link_viewer_dims(viewer, explorer)

    qt_annotation = viewer.window._qt_window._qt_viewer

    annotation_wrap = QWidget()
    annotation_layout = QVBoxLayout(annotation_wrap)
    annotation_layout.setContentsMargins(0, 0, 0, 0)
    annotation_layout.setSpacing(2)
    annotation_layout.addWidget(QLabel("Annotation on T1CE"))
    annotation_layout.addWidget(qt_annotation, stretch=1)

    explorer_wrap = QWidget()
    explorer_layout = QVBoxLayout(explorer_wrap)
    explorer_layout.setContentsMargins(0, 0, 0, 0)
    explorer_layout.setSpacing(2)
    explorer_layout.addWidget(QLabel("Sequences — toggle layers in dock →"))
    explorer_layout.addWidget(qt_explorer, stretch=1)

    splitter = QSplitter(Qt.Horizontal)
    splitter.addWidget(annotation_wrap)
    splitter.addWidget(explorer_wrap)
    splitter.setStretchFactor(0, 1)
    splitter.setStretchFactor(1, 1)
    splitter.setSizes([700, 700])
    viewer.window._qt_window.setCentralWidget(splitter)

    viewer.window.add_dock_widget(
        qt_explorer.dockLayerList.widget(),
        area="right",
        name="Sequences",
    )


def add_overlay_toggle_button(viewer: Any, *overlay_layers: Any) -> None:
    if not overlay_layers:
        return

    from qtpy.QtWidgets import QPushButton

    button = QPushButton("Hide overlay")

    def toggle() -> None:
        visible = not overlay_layers[0].visible
        for layer in overlay_layers:
            layer.visible = visible
        button.setText("Hide overlay" if visible else "Show overlay")

    button.clicked.connect(toggle)
    viewer.window.add_dock_widget(button, area="right", name="Overlay")


def point_for_plane(point_zyx: tuple[float, ...], plane: str) -> tuple[float, ...]:
  z, y, x = (float(point_zyx[i]) for i in range(3))
  if plane == "axial":
      return (z, y, x)
  if plane == "coronal":
      return (y, z, x)
  return (x, z, y)


def zyx_from_plane_point(point: tuple[float, ...], plane: str) -> tuple[float, float, float]:
    if plane == "axial":
        return float(point[0]), float(point[1]), float(point[2])
    if plane == "coronal":
        return float(point[1]), float(point[0]), float(point[2])
    return float(point[1]), float(point[2]), float(point[0])


def refresh_mpr_layers(state: DisplayState) -> None:
    primary = state.primary
    if primary not in state.series:
        return
    display = render_mr_volume(state, primary)
    scale = state.series[primary].scale
    for plane, layer in state.mpr_layers.items():
        order = MPR_PLANE_TRANSPOSE[plane]
        layer.data = transpose_volume(display, order)
        layer.scale = permute_scale(scale, MPR_SCALE_PERMUTE[plane])


def set_orthogonal_grid(viewer: Any, state: DisplayState, enabled: bool) -> None:
    state.orthogonal_grid = enabled
    if enabled:
        primary = state.primary
        if primary not in state.series:
            return
        if not state.mpr_layers:
            display = render_mr_volume(state, primary)
            scale = state.series[primary].scale
            limits = contrast_limits_from_wl(state.level, state.width)
            for plane in ("sagittal", "coronal", "axial"):
                order = MPR_PLANE_TRANSPOSE[plane]
                layer = viewer.add_image(
                    transpose_volume(display, order),
                    name=f"{state.series[primary].name} {plane}",
                    scale=permute_scale(scale, MPR_SCALE_PERMUTE[plane]),
                    colormap="gray",
                    contrast_limits=limits,
                    blending="additive",
                )
                viewer.layers.move(len(viewer.layers) - 1, 0)
                state.mpr_layers[plane] = layer
        else:
            refresh_mpr_layers(state)
        viewer.grid.enabled = True
        viewer.grid.shape = (1, 3)
        for layer in state.mr_layers.values():
            layer.visible = False
        if state.seg_layer is not None:
            state.seg_layer.visible = False
        for layer in state.mpr_layers.values():
            layer.visible = True
    else:
        viewer.grid.enabled = False
        for layer in state.mpr_layers.values():
            layer.visible = False
        primary_layer = state.mr_layers.get(state.primary)
        if primary_layer is not None:
            primary_layer.visible = True
        if state.seg_layer is not None:
            state.seg_layer.visible = True


def setup_orthogonal_dim_linking(viewer: Any, state: DisplayState) -> None:
    """Best-effort sync of MPR crosshair positions across grid canvases."""
    if len(state.mpr_layers) < 3:
        return

    def on_point(event: Any) -> None:
        if state._syncing_dims or not state.orthogonal_grid:
            return
        order = tuple(int(i) for i in viewer.dims.order)
        plane = {0: "axial", 1: "coronal", 2: "sagittal"}.get(order[0], "axial")
        zyx = zyx_from_plane_point(tuple(float(v) for v in event.source.point), plane)
        state._syncing_dims = True
        try:
            for target_plane, layer in state.mpr_layers.items():
                if target_plane == plane:
                    continue
                idx = viewer.layers.index(layer)
                sub_viewer = viewer.window._qt_window._sub_viewers.get(idx)  # noqa: SLF001
                if sub_viewer is None:
                    continue
                sub_viewer.viewer.dims.point = point_for_plane(zyx, target_plane)
        finally:
            state._syncing_dims = False

    viewer.dims.events.point.connect(on_point)


def add_display_controls(viewer: Any, state: DisplayState) -> None:
    from magicgui.widgets import Checkbox, ComboBox, Container, FloatSlider, Label

    lo, hi = contrast_limits_from_wl(state.level, state.width)
    level_slider = FloatSlider(
        name="Level",
        value=state.level,
        min=lo - state.width * 2,
        max=hi + state.width * 2,
        step=max(state.width / 200.0, 1.0),
    )
    width_slider = FloatSlider(
        name="Width",
        value=state.width,
        min=1.0,
        max=max(state.width * 4.0, 10.0),
        step=max(state.width / 200.0, 1.0),
    )
    clahe_box = Checkbox(value=False, text="CLAHE (QC only)")
    ortho_box = Checkbox(value=False, text="Orthogonal MPR grid")
    plane_box = ComboBox(label="Plane", choices=list(PLANE_ORDERS), value="axial")

    def on_wl_change(_event: Any = None) -> None:
        state.level = float(level_slider.value)
        state.width = float(width_slider.value)
        update_mr_contrast(state)
        if state.orthogonal_grid:
            limits = contrast_limits_from_wl(state.level, state.width)
            for layer in state.mpr_layers.values():
                layer.contrast_limits = limits

    def on_clahe_change(_event: Any = None) -> None:
        state.clahe = bool(clahe_box.value)
        refresh_mr_layers(state)

    def on_ortho_change(_event: Any = None) -> None:
        set_orthogonal_grid(viewer, state, bool(ortho_box.value))
        if ortho_box.value:
            setup_orthogonal_dim_linking(viewer, state)

    def on_plane_change(_event: Any = None) -> None:
        if state.orthogonal_grid:
            return
        order = PLANE_ORDERS[str(plane_box.value)]
        viewer.dims.order = order
        if state.explorer is not None:
            state.explorer.dims.order = order

    level_slider.changed.connect(on_wl_change)
    width_slider.changed.connect(on_wl_change)
    clahe_box.changed.connect(on_clahe_change)
    ortho_box.changed.connect(on_ortho_change)
    plane_box.changed.connect(on_plane_change)

    @viewer.mouse_drag_callbacks.append
    def window_level_drag(viewer: Any, event: Any) -> None:
        if event.type != "mouse_move":
            return
        modifiers = getattr(event, "modifiers", None) or set()
        if "Control" not in modifiers:
            return
        native = getattr(event, "native", None)
        if native is None:
            return
        pos = native.pos()
        last = native.last().pos()
        dx = float(pos.x() - last.x())
        dy = float(pos.y() - last.y())
        state.width = max(1.0, state.width + dx * (state.width / 200.0))
        state.level = state.level - dy * (state.width / 200.0)
        level_slider.value = state.level
        width_slider.value = state.width
        update_mr_contrast(state)

    controls = Container(
        widgets=[
            Label(label="Ctrl + drag on canvas adjusts window/level."),
            Label(
                label="Left: T1CE + segmentation. Right: toggle FLAIR/T2 in Sequences dock."
            ),
            Label(label="CLAHE is QC only."),
            level_slider,
            width_slider,
            plane_box,
            clahe_box,
            ortho_box,
        ],
    )
    viewer.window.add_dock_widget(controls, area="right", name="Window / Level")


def launch_clinical_viewer(
    viewer: Any,
    *,
    series_catalog: dict[str, SeriesVolume],
    labels: np.ndarray | None,
    seg_path: Path | None,
    primary: str = "t1ce",
    scale: tuple[float, float, float] | None = None,
) -> DisplayState:
    if primary not in series_catalog and series_catalog:
        primary = next(iter(series_catalog))
    primary_series = series_catalog[primary]
    if scale is not None:
        primary_series = SeriesVolume(
            name=primary_series.name,
            volume=primary_series.volume,
            scale=scale,
            path=primary_series.path,
        )
        series_catalog[primary] = primary_series

    brain_masks = {
        name: brain_tissue_mask(series.volume) for name, series in series_catalog.items()
    }
    level, width = brain_window_level(primary_series.volume, brain_masks[primary])
    state = DisplayState(level=level, width=width, series=series_catalog, primary=primary)
    state.brain_masks = brain_masks
    limits = contrast_limits_from_wl(level, width)

    primary_layer = viewer.add_image(
        render_mr_volume(state, primary),
        name=primary_series.name,
        scale=primary_series.scale,
        colormap="gray",
        contrast_limits=limits,
        blending="additive",
        visible=True,
    )
    state.mr_layers[primary] = primary_layer

    if labels is not None:
        seg_name = f"segmentation ({seg_path.name})" if seg_path else "segmentation"
        state.seg_layer = viewer.add_labels(
            labels,
            name=seg_name,
            scale=primary_series.scale,
            opacity=0.55,
        )
        add_overlay_toggle_button(viewer, state.seg_layer)

    viewer.dims.order = PLANE_ORDERS["axial"]
    setup_dual_view_panes(viewer, state)
    add_display_controls(viewer, state)
    return state


def list_slugs() -> list[str]:
    spec = raw_extract_spec()
    out_dir = _repo_path(spec["output_dir"])
    if not out_dir.exists():
        return []
    slugs: list[str] = []
    for json_path in sorted(out_dir.glob("*.json")):
        slug = json_path.stem
        seg = resolve_segmentation_path(json.loads(json_path.read_text()), slug)
        if seg is not None:
            slugs.append(slug)
    return slugs


def view_demo() -> None:
    import napari

    volume = dummy_volume(shape=(48, 48, 48), radius=10.0)
    mask = (volume >= 0.35).astype(np.uint8)
    primary = SeriesVolume(name="T1ce", volume=volume, scale=(1.0, 1.0, 1.0))
    viewer = napari.Viewer(title="brain-cancer-sim — demo")
    launch_clinical_viewer(
        viewer,
        series_catalog={"t1ce": primary},
        labels=mask,
        seg_path=None,
        primary="t1ce",
    )
    print("Demo mode: synthetic MR + threshold mask. Replace with --slug when data exists.")
    napari.run()


def view_nifti(
    mr_path: Path,
    mask_path: Path | None,
    *,
    show_pde: bool = False,
) -> None:
    import napari

    volume, spacing = load_nifti(mr_path)
    primary = SeriesVolume(name="T1ce", volume=volume, scale=spacing, path=mr_path)
    extra_paths = discover_series_next_to_mr(mr_path)
    catalog = build_series_catalog(primary, extra_paths)
    labels = None
    if mask_path:
        mask_arr, _ = load_nifti(mask_path)
        if mask_arr.shape != volume.shape:
            raise ValueError(f"Mask shape {mask_arr.shape} != MR {volume.shape}")
        labels = (mask_arr > 0).astype(np.uint8)

    viewer = napari.Viewer(title=f"{mr_path.name} — napari")
    launch_clinical_viewer(
        viewer,
        series_catalog=catalog,
        labels=labels,
        seg_path=mask_path,
        primary="t1ce",
    )
    if show_pde and labels is not None:
        weight = labels > 0
        pde = volume * weight
        viewer.add_image(
            pde,
            name="PDE preview",
            scale=spacing,
            opacity=0.4,
            colormap="magma",
            contrast_limits=contrast_limits_from_wl(*brain_window_level(pde, weight)),
        )
    loaded = ", ".join(sorted(catalog))
    print(f"MR: {mr_path}\n  shape={volume.shape} spacing_mm={spacing}\n  series: {loaded}")
    if mask_path:
        print(f"mask: {mask_path}")
    napari.run()


def view_slug(slug: str, *, show_pde: bool = False, pde_grid_size: int | None = None) -> None:
    import napari

    volume, meta, json_path = load_raw_extract(slug)
    spacing = tuple(float(s) for s in meta.get("spacing_mm", [1.0, 1.0, 1.0]))
    primary = SeriesVolume(
        name="T1ce",
        volume=volume,
        scale=spacing,
        path=Path(meta.get("source_path", "")),
    )
    catalog = build_series_catalog(primary, discover_ucsf_series(meta))

    seg_path = resolve_segmentation_path(meta, slug)
    labels = None
    if seg_path is not None:
        labels = load_segmentation(seg_path, volume.shape)

    viewer = napari.Viewer(title=f"{slug} — brain MR + seg")
    launch_clinical_viewer(
        viewer,
        series_catalog=catalog,
        labels=labels,
        seg_path=seg_path,
        primary="t1ce",
        scale=spacing,
    )

    if show_pde:
        grid_size = pde_grid_size or default_grid_size()
        pde_path = pde_input_npy(slug, grid_size=grid_size)
        if not pde_path.exists():
            pde_path = pde_input_npy_legacy(slug)
        if pde_path.exists():
            pde = np.load(pde_path).astype(np.float32)
            viewer.add_image(
                pde,
                name=f"PDE input g{grid_size}",
                scale=spacing,
                opacity=0.45,
                colormap="magma",
                contrast_limits=(0.0, 1.0),
            )
        else:
            print(f"  (no PDE input at {pde_path})")

    loaded = ", ".join(name.upper() for name in sorted(catalog))
    print(
        f"{slug}\n"
        f"  dataset:  {meta.get('dataset', '?')}\n"
        f"  patient:  {meta.get('patient_id', '?')}\n"
        f"  MR:       shape={volume.shape} spacing_mm={list(spacing)}\n"
        f"  series:   {loaded}\n"
        f"  seg:      {seg_path} voxels={int(labels.sum()) if labels is not None else 0:,}"
    )
    napari.run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="View brain MR with expert segmentation overlay in napari.",
    )
    parser.add_argument("--slug", help="Raw extract slug under data/processed/raw-extract-philip-chandan/")
    parser.add_argument("--mr", type=Path, help="MR NIfTI path (.nii / .nii.gz)")
    parser.add_argument("--mask", type=Path, help="Segmentation NIfTI path")
    parser.add_argument("--pde-input", action="store_true", help="Overlay PDE-ready volume if present")
    parser.add_argument(
        "--grid-size",
        type=int,
        choices=grid_size_options(),
        default=None,
        help="PDE crop grid (default: contract default_grid_size)",
    )
    parser.add_argument("--demo", action="store_true", help="Synthetic MR + mask (default when no data)")
    parser.add_argument("--list", action="store_true", help="List slugs with paired segmentations")
    args = parser.parse_args()

    _ = load_handoff_contract()

    if args.list:
        slugs = list_slugs()
        for slug in slugs:
            print(slug)
        if not slugs:
            print("(none yet — use --demo or export to data/processed/raw-extract-philip-chandan/)")
        return

    if args.demo:
        view_demo()
        return

    if args.mr:
        view_nifti(args.mr, args.mask, show_pde=args.pde_input)
        return

    slug = args.slug
    if not slug:
        available = list_slugs()
        if len(available) == 1:
            slug = available[0]
        elif available:
            parser.error("Provide --slug. Available: " + ", ".join(available))
        else:
            print("No brain data on disk — opening synthetic demo.")
            view_demo()
            return

    view_slug(slug, show_pde=args.pde_input, pde_grid_size=args.grid_size)


if __name__ == "__main__":
    main()

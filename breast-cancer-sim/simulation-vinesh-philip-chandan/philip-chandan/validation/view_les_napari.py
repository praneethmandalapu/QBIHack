"""Interactive 3D viewer: baseline MR + TCIA radiologist .les overlay (napari).

Clinical-style breast DCE QC: side-by-side temporal phases, MIP row, CAD-style
enhancement markers, optional pre-contrast subtraction, and expert-mask toggle.

Run (macOS/Linux):
    cd breast-cancer-sim
    source .venv/bin/activate
    pip install -r requirements.txt   # includes napari[pyqt6]

    python simulation-vinesh-philip-chandan/philip-chandan/validation/view_les_napari.py --list
    python simulation-vinesh-philip-chandan/philip-chandan/validation/view_les_napari.py \\
        --slug luminal_a_TCGA-AR-A1AX_baseline

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

import napari  # noqa: E402

from clinical_layout import setup_clinical_hanging_protocol  # noqa: E402
from dce_phases import (  # noqa: E402
    DcePhase,
    compute_subtraction,
    detect_cad_markers,
    expert_centroid_zyx,
    lesion_z_in_phase,
    load_precontrast_volume,
    mask_for_phase,
    resample_volume,
    resolve_dce_dicom_dir_for_study,
    resolve_phase_ranges,
    split_dce_phases,
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
    show_mip_row: bool = True
    show_cad: bool = True
    precontrast_display: np.ndarray | None = None
    mr_layer: Any | None = None
    subtraction_layer: Any | None = None
    precontrast_layer: Any | None = None
    expert_layer: Any | None = None
    cad_layer: Any | None = None
    expert_mask_full: np.ndarray | None = None
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


def add_overlay_toggle_button(viewer: napari.Viewer, *overlay_layers: Any) -> None:
    """Dock button to show/hide expert (and optional) overlay layers."""
    if not overlay_layers:
        return

    from qtpy.QtWidgets import QPushButton

    button = QPushButton("Hide expert mask")

    def toggle() -> None:
        visible = not overlay_layers[0].visible
        for layer in overlay_layers:
            layer.visible = visible
        button.setText("Hide expert mask" if visible else "Show expert mask")

    button.clicked.connect(toggle)
    viewer.window.add_dock_widget(button, area="right", name="Expert mask")


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


def _update_expert_layer(state: BreastDisplayState) -> None:
    if state.expert_layer is None or state.expert_mask_full is None:
        return
    phase = state.phases[state.active_phase - 1]
    state.expert_layer.data = mask_for_phase(state.expert_mask_full, phase)


def _update_cad_layer(state: BreastDisplayState) -> None:
    if state.cad_layer is None:
        return
    phase = state.phases[state.active_phase - 1]
    volume = (
        state.subtraction_volumes[state.active_phase - 1]
        if state.show_subtraction and state.subtraction_volumes
        else state.phase_volumes[state.active_phase - 1]
    )
    coords = detect_cad_markers(volume)
    expert_centroid = (
        expert_centroid_zyx(state.expert_mask_full, phase)
        if state.expert_mask_full is not None
        else None
    )
    if expert_centroid is not None:
        coords = (
            np.vstack([coords, np.array([expert_centroid], dtype=np.float32)])
            if coords.size
            else np.array([expert_centroid], dtype=np.float32)
        )
    state.cad_layer.data = coords
    if expert_centroid is not None and coords.size:
        state.cad_layer.face_color = [
            "lime" if np.allclose(row, expert_centroid, atol=0.5) else "yellow"
            for row in coords
        ]


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
    if state.show_cad:
        _update_cad_layer(state)


def _set_hanging_visibility(state: BreastDisplayState) -> None:
    for layer in state.hanging.get("mip_layers", []):
        layer.visible = state.show_mip_row
    for layer in state.hanging.get("cad_layers", []):
        layer.visible = state.show_cad


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


def add_breast_display_controls(viewer: napari.Viewer, state: BreastDisplayState) -> None:
    from magicgui.widgets import Checkbox, ComboBox, Container, PushButton

    phase_choices = {f"Phase {phase.index}": phase.index for phase in state.phases}
    phase_box = ComboBox(label="DCE phase", choices=list(phase_choices.keys()))
    phase_box.value = f"Phase {state.active_phase}"

    subtraction_box = Checkbox(value=state.show_subtraction, text="Subtraction (phase − pre-contrast S1)")
    subtraction_box.enabled = bool(state.subtraction_volumes)
    precontrast_box = Checkbox(value=False, text="Pre-contrast S1 (Ax T1, resampled)")
    precontrast_box.enabled = state.precontrast_display is not None
    mip_box = Checkbox(value=state.show_mip_row, text="MIP row (hanging protocol)")
    cad_box = Checkbox(value=state.show_cad, text="CAD markers (enhancement peaks)")
    jump_button = PushButton(text="Jump to expert lesion")

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

    def on_cad_change(_event: Any = None) -> None:
        state.show_cad = bool(cad_box.value)
        if state.cad_layer is not None:
            state.cad_layer.visible = state.show_cad
        _set_hanging_visibility(state)

    def on_jump(_event: Any = None) -> None:
        _jump_to_lesion(viewer, state)

    phase_box.changed.connect(on_phase_change)
    subtraction_box.changed.connect(on_subtraction_change)
    precontrast_box.changed.connect(on_precontrast_change)
    mip_box.changed.connect(on_mip_change)
    cad_box.changed.connect(on_cad_change)
    jump_button.clicked.connect(on_jump)

    controls = Container(
        widgets=[
            phase_box,
            subtraction_box,
            precontrast_box,
            mip_box,
            cad_box,
            jump_button,
        ],
    )
    viewer.window.add_dock_widget(controls, area="right", name="DCE controls")


def view_slug(
    slug: str,
    *,
    show_otsu: bool = False,
    cuboid_boundary: bool = False,
    skip_precontrast: bool = False,
    no_hanging: bool = False,
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

    if cuboid_boundary:
        overlay_mask, meta = load_les_cuboid_boundary(les_path, volume.shape)
        overlay_name = (
            f".les cuboid shell "
            f"(y[{meta['y_start']},{meta['y_end']}] "
            f"x[{meta['x_start']},{meta['x_end']}] "
            f"z[{meta['z_start']},{meta['z_end']}])"
        )
        overlay_detail = f"boundary={meta['boundary_voxels']:,} vox"
    else:
        overlay_mask, meta = load_les_mask(les_path, volume.shape)
        overlay_name = f".les expert ({meta['lesion_voxels']:,} vox)"
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
        f"  layout: {'detail + hanging protocol' if not no_hanging else 'detail only'}\n"
        f"  lesion: phase {active_phase}"
        + (f", z={lesion_z} within phase" if lesion_z is not None else "")
    )

    show_subtraction = bool(subtraction_volumes)
    state = BreastDisplayState(
        phases=phases,
        phase_volumes=phase_volumes,
        subtraction_volumes=subtraction_volumes,
        spacing_mm=scale,
        active_phase=active_phase,
        show_subtraction=show_subtraction,
        precontrast_display=precontrast_display,
        expert_mask_full=overlay_mask,
        cuboid_boundary=cuboid_boundary,
    )

    viewer = napari.Viewer(title=f"{slug} — breast DCE hanging protocol")
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

    if state.show_cad:
        detail_volume = (
            subtraction_volumes[active_phase - 1]
            if show_subtraction
            else active_volume
        )
        cad_coords = detect_cad_markers(detail_volume)
        expert_centroid = expert_centroid_zyx(overlay_mask, phases[active_phase - 1])
        if expert_centroid is not None:
            cad_coords = (
                np.vstack([cad_coords, np.array([expert_centroid], dtype=np.float32)])
                if cad_coords.size
                else np.array([expert_centroid], dtype=np.float32)
            )
        if cad_coords.size:
            state.cad_layer = viewer.add_points(
                cad_coords,
                name="CAD markers (detail)",
                size=12,
                face_color="yellow",
                border_color="black",
                symbol="disc",
            )
            if expert_centroid is not None:
                state.cad_layer.face_color = [
                    "lime" if np.allclose(row, expert_centroid, atol=0.5) else "yellow"
                    for row in cad_coords
                ]

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

    if not no_hanging:
        state.hanging = setup_clinical_hanging_protocol(
            viewer,
            phase_volumes=phase_volumes,
            phases=phases,
            subtraction_volumes=subtraction_volumes if show_subtraction else [],
            spacing_mm=scale,
            expert_mask_full=overlay_mask,
            expert_layer_name=overlay_name,
            show_mip_row=state.show_mip_row,
            show_cad=state.show_cad,
        )

    add_overlay_toggle_button(
        viewer,
        state.expert_layer,
        *state.hanging.get("expert_layers", []),
    )

    add_breast_display_controls(viewer, state)
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
        cuboid_boundary=args.cuboid,
        skip_precontrast=args.no_precontrast,
        no_hanging=args.no_hanging,
    )


if __name__ == "__main__":
    main()

"""Rigid registration of P1 z-band slabs (full Y/X) from P2–P4 onto P1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import SimpleITK as sitk

from dce_phases import DcePhase, mask_for_phase
from les_cuboid_brightness import (
    extract_phase_z_band_full_xy,
    phase_z_band_slice,
)
from load_les_mask import load_les_cuboid_boundary, load_les_mask
from prep_volume import normalize_volume


@dataclass(frozen=True)
class CuboidAlignmentMetrics:
    moving_phase: int
    translation_mm: tuple[float, float, float]
    rotation_deg: tuple[float, float, float]
    rotation_magnitude_deg: float
    ncc_before: float
    ncc_after: float
    mse_before: float
    mse_after: float
    optimizer_iterations: int
    optimizer_metric_value: float


@dataclass
class ZBandAlignmentResult:
    reference_phase_index: int
    z_band_local: tuple[int, int]
    slabs_raw: dict[int, np.ndarray]
    slabs_aligned: dict[int, np.ndarray]
    expert_slab: np.ndarray
    boundary_slab: np.ndarray
    metrics: list[CuboidAlignmentMetrics]
    spacing_mm: tuple[float, float, float]


def _sitk_spacing_xyz(spacing_zyx: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
    dz, dy, dx = (float(s) for s in spacing_zyx)
    return (dx, dy, dz)


def _array_to_sitk(volume: np.ndarray, spacing_zyx: tuple[float, float, float]) -> sitk.Image:
    image = sitk.GetImageFromArray(volume.astype(np.float32))
    image.SetSpacing(_sitk_spacing_xyz(spacing_zyx))
    return image


def normalized_cross_correlation(fixed: np.ndarray, moving: np.ndarray) -> float:
    a = fixed.astype(np.float64).ravel()
    b = moving.astype(np.float64).ravel()
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def mean_squared_error(fixed: np.ndarray, moving: np.ndarray) -> float:
    if fixed.shape != moving.shape or fixed.size == 0:
        return float("nan")
    diff = fixed.astype(np.float64) - moving.astype(np.float64)
    return float(np.mean(diff * diff))


def _euler_magnitude_deg(transform: sitk.Euler3DTransform) -> tuple[tuple[float, float, float], float]:
    rx, ry, rz = transform.GetAngleX(), transform.GetAngleY(), transform.GetAngleZ()
    deg = (math.degrees(rx), math.degrees(ry), math.degrees(rz))
    magnitude = math.degrees(math.sqrt(rx * rx + ry * ry + rz * rz))
    return deg, magnitude


def register_rigid_slab(
    fixed: np.ndarray,
    moving: np.ndarray,
    spacing_zyx: tuple[float, float, float],
    *,
    moving_phase: int,
    number_of_iterations: int = 200,
) -> tuple[np.ndarray, CuboidAlignmentMetrics]:
    """Rigidly register ``moving`` z-band slab onto ``fixed`` (P1 grid, full Y/X)."""
    fixed_img = _array_to_sitk(fixed, spacing_zyx)
    moving_img = _array_to_sitk(moving, spacing_zyx)

    ncc_before = normalized_cross_correlation(fixed, moving)
    mse_before = mean_squared_error(fixed, moving)

    if moving_phase == 1 or np.allclose(fixed, moving):
        metrics = CuboidAlignmentMetrics(
            moving_phase=moving_phase,
            translation_mm=(0.0, 0.0, 0.0),
            rotation_deg=(0.0, 0.0, 0.0),
            rotation_magnitude_deg=0.0,
            ncc_before=ncc_before,
            ncc_after=ncc_before,
            mse_before=mse_before,
            mse_after=mse_before,
            optimizer_iterations=0,
            optimizer_metric_value=0.0,
        )
        return moving.copy(), metrics

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.25)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=number_of_iterations,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-8,
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    initial = sitk.CenteredTransformInitializer(
        fixed_img,
        moving_img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration.SetInitialTransform(initial, inPlace=False)
    final_transform = registration.Execute(fixed_img, moving_img)

    aligned_img = sitk.Resample(
        moving_img,
        fixed_img,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_img.GetPixelID(),
    )
    aligned = sitk.GetArrayFromImage(aligned_img)

    euler = sitk.Euler3DTransform()
    if isinstance(final_transform, sitk.CompositeTransform):
        for index in range(final_transform.GetNumberOfTransforms()):
            component = final_transform.GetNthTransform(index)
            if isinstance(component, sitk.Euler3DTransform):
                euler = component
                break
    elif isinstance(final_transform, sitk.Euler3DTransform):
        euler = final_transform

    rotation_deg, rotation_mag = _euler_magnitude_deg(euler)
    translation_mm = tuple(float(v) for v in euler.GetTranslation())

    ncc_after = normalized_cross_correlation(fixed, aligned)
    mse_after = mean_squared_error(fixed, aligned)

    metrics = CuboidAlignmentMetrics(
        moving_phase=moving_phase,
        translation_mm=translation_mm,
        rotation_deg=rotation_deg,
        rotation_magnitude_deg=rotation_mag,
        ncc_before=ncc_before,
        ncc_after=ncc_after,
        mse_before=mse_before,
        mse_after=mse_after,
        optimizer_iterations=number_of_iterations,
        optimizer_metric_value=float(registration.GetMetricValue()),
    )
    return aligned.astype(np.float32), metrics


def align_phase_z_bands_to_p1(
    phase_volumes: list[np.ndarray],
    phases: list[DcePhase],
    les_meta: dict[str, Any],
    spacing_mm: tuple[float, float, float] | list[float],
    *,
    number_of_iterations: int = 200,
) -> ZBandAlignmentResult:
    """Extract P1 .les z-band with full Y/X per phase; register P2–P4 slabs → P1."""
    reference = phases[0]
    spacing = tuple(float(s) for s in spacing_mm)
    z_sl = phase_z_band_slice(les_meta, reference, reference_phase=reference)
    z_band_local = (z_sl.start or 0, (z_sl.stop or 1) - 1)

    slabs_raw: dict[int, np.ndarray] = {}
    slabs_aligned: dict[int, np.ndarray] = {}
    metrics: list[CuboidAlignmentMetrics] = []

    for phase, volume in zip(phases, phase_volumes, strict=True):
        slabs_raw[phase.index] = extract_phase_z_band_full_xy(
            volume,
            les_meta,
            phase,
            reference_phase=reference,
        )

    fixed = slabs_raw[reference.index]
    slabs_aligned[reference.index] = fixed.copy()
    metrics.append(
        CuboidAlignmentMetrics(
            moving_phase=reference.index,
            translation_mm=(0.0, 0.0, 0.0),
            rotation_deg=(0.0, 0.0, 0.0),
            rotation_magnitude_deg=0.0,
            ncc_before=1.0,
            ncc_after=1.0,
            mse_before=0.0,
            mse_after=0.0,
            optimizer_iterations=0,
            optimizer_metric_value=0.0,
        )
    )

    for phase in phases[1:]:
        aligned, phase_metrics = register_rigid_slab(
            fixed,
            slabs_raw[phase.index],
            spacing,
            moving_phase=phase.index,
            number_of_iterations=number_of_iterations,
        )
        slabs_aligned[phase.index] = aligned
        metrics.append(phase_metrics)

    return ZBandAlignmentResult(
        reference_phase_index=reference.index,
        z_band_local=z_band_local,
        slabs_raw=slabs_raw,
        slabs_aligned=slabs_aligned,
        expert_slab=np.zeros(fixed.shape, dtype=np.uint8),
        boundary_slab=np.zeros(fixed.shape, dtype=np.uint8),
        metrics=metrics,
        spacing_mm=spacing,
    )


def attach_les_overlays_on_z_band(
    result: ZBandAlignmentResult,
    *,
    les_path: Any,
    full_volume_shape: tuple[int, int, int],
    reference_phase: DcePhase,
) -> ZBandAlignmentResult:
    """Expert + bbox overlays on the P1 z-band slab (full Y/X)."""
    expert_full, _les_meta = load_les_mask(les_path, full_volume_shape)
    boundary_full, _ = load_les_cuboid_boundary(les_path, full_volume_shape)

    z_sl = slice(result.z_band_local[0], result.z_band_local[1] + 1)
    expert_phase = mask_for_phase(expert_full, reference_phase)
    boundary_phase = mask_for_phase(boundary_full, reference_phase)

    result.expert_slab = expert_phase[z_sl, :, :].astype(np.uint8)
    result.boundary_slab = boundary_phase[z_sl, :, :].astype(np.uint8)
    return result


def format_alignment_metrics(metrics: list[CuboidAlignmentMetrics]) -> str:
    header = (
        f"{'Phase':>5}  {'|T| mm':>7}  {'|R| deg':>7}  "
        f"{'NCC pre':>8}  {'NCC post':>9}  {'MSE pre':>9}  {'MSE post':>9}  {'MI opt':>8}"
    )
    lines = [header, "-" * len(header)]
    for row in metrics:
        trans = math.sqrt(sum(v * v for v in row.translation_mm))
        lines.append(
            f"P{row.moving_phase:>4}  {trans:7.3f}  {row.rotation_magnitude_deg:7.3f}  "
            f"{row.ncc_before:8.4f}  {row.ncc_after:9.4f}  "
            f"{row.mse_before:9.2f}  {row.mse_after:9.2f}  "
            f"{row.optimizer_metric_value:8.4f}"
        )
        if row.moving_phase > 1:
            tx, ty, tz = row.translation_mm
            rx, ry, rz = row.rotation_deg
            lines.append(
                f"       translation (x,y,z) mm = ({tx:+.3f}, {ty:+.3f}, {tz:+.3f})  "
                f"rotation (x,y,z) deg = ({rx:+.3f}, {ry:+.3f}, {rz:+.3f})"
            )
    return "\n".join(lines)


def display_slab_for_napari(slab: np.ndarray) -> np.ndarray:
    return normalize_volume(slab)


# Back-compat aliases
CuboidAlignmentResult = ZBandAlignmentResult
align_phase_cuboids_to_p1 = align_phase_z_bands_to_p1
attach_les_overlays_in_cuboid_space = attach_les_overlays_on_z_band
display_cuboid_for_napari = display_slab_for_napari

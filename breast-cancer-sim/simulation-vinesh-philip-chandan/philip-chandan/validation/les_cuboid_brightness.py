"""Bright-voxel fraction inside .les cuboid for threshold sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dce_phases import DcePhase, mask_for_phase
from prep_volume import normalize_volume


@dataclass(frozen=True)
class CuboidBrightnessRow:
    phase_index: int
    threshold: float
    bright_fraction: float
    bright_voxels: int
    cuboid_voxels: int
    les_fraction: float
    les_voxels: int


def default_thresholds(*, step: float = 0.05) -> np.ndarray:
    """Normalized intensity cutoffs in (0, 1] for cuboid ROI sweeps."""
    if step <= 0 or step > 1:
        raise ValueError(f"step must be in (0, 1], got {step}")
    values = np.arange(step, 1.0 + step / 2, step, dtype=np.float64)
    return np.clip(values, 0.0, 1.0)


def _normalize_roi(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    clipped = np.clip(values, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def cuboid_slices_from_meta(les_meta: dict[str, Any]) -> tuple[slice, slice, slice]:
    """Return (Z, Y, X) slices for the .les cuboid interior."""
    return (
        slice(int(les_meta["z_start"]), int(les_meta["z_end"]) + 1),
        slice(int(les_meta["y_start"]), int(les_meta["y_end"]) + 1),
        slice(int(les_meta["x_start"]), int(les_meta["x_end"]) + 1),
    )


def les_local_z_slice(les_meta: dict[str, Any], reference_phase: DcePhase) -> slice:
    """Phase-local z band from .les global indices, using ``reference_phase`` as P1 anchor."""
    z0 = int(les_meta["z_start"]) - reference_phase.z_start
    z1 = int(les_meta["z_end"]) - reference_phase.z_start
    return slice(z0, z1 + 1)


def phase_cuboid_slices(
    les_meta: dict[str, Any],
    phase: DcePhase,
    *,
    reference_phase: DcePhase,
    phase_local: bool = True,
) -> tuple[slice, slice, slice]:
    """Cuboid Y/X plus P1-anchored local z band applied to every phase volume."""
    _z_sl, y_sl, x_sl = cuboid_slices_from_meta(les_meta)
    if phase_local:
        z_sl = les_local_z_slice(les_meta, reference_phase)
        depth = phase.z_end - phase.z_start
        z0 = max(0, min(z_sl.start or 0, depth - 1))
        z1 = max(0, min((z_sl.stop or 1) - 1, depth - 1))
        if z0 > z1:
            z1 = z0
        z_sl = slice(z0, z1 + 1)
    else:
        z_sl = slice(int(les_meta["z_start"]), int(les_meta["z_end"]) + 1)
    return z_sl, y_sl, x_sl


def extract_phase_cuboid(
    volume: np.ndarray,
    les_meta: dict[str, Any],
    phase: DcePhase,
    *,
    reference_phase: DcePhase,
) -> np.ndarray:
    """Crop the .les cuboid from one phase volume (P1-anchored local z)."""
    z_sl, y_sl, x_sl = phase_cuboid_slices(
        les_meta,
        phase,
        reference_phase=reference_phase,
        phase_local=True,
    )
    return np.ascontiguousarray(volume[z_sl, y_sl, x_sl].astype(np.float32))


def phase_z_band_slice(
    les_meta: dict[str, Any],
    phase: DcePhase,
    *,
    reference_phase: DcePhase,
) -> slice:
    """P1-anchored local z band, clamped to ``phase`` depth."""
    z_sl = les_local_z_slice(les_meta, reference_phase)
    depth = phase.z_end - phase.z_start
    z0 = max(0, min(z_sl.start or 0, depth - 1))
    z1 = max(0, min((z_sl.stop or 1) - 1, depth - 1))
    if z0 > z1:
        z1 = z0
    return slice(z0, z1 + 1)


def extract_phase_z_band_full_xy(
    volume: np.ndarray,
    les_meta: dict[str, Any],
    phase: DcePhase,
    *,
    reference_phase: DcePhase,
) -> np.ndarray:
    """P1 .les z-band through full in-plane (Y, X) extent — for cross-phase alignment."""
    z_sl = phase_z_band_slice(les_meta, phase, reference_phase=reference_phase)
    return np.ascontiguousarray(volume[z_sl, :, :].astype(np.float32))


def values_in_phase_cuboid(
    volume: np.ndarray,
    les_meta: dict[str, Any],
    phase: DcePhase,
    *,
    reference_phase: DcePhase,
    normalize: bool = True,
    phase_local: bool = True,
) -> tuple[np.ndarray, int]:
    """Flattened cuboid voxels for one phase; second value is cuboid voxel count."""
    z_sl, y_sl, x_sl = phase_cuboid_slices(
        les_meta,
        phase,
        reference_phase=reference_phase,
        phase_local=phase_local,
    )
    roi = volume[z_sl, y_sl, x_sl].astype(np.float32, copy=False)
    flat = roi.ravel()
    if normalize and flat.size:
        flat = _normalize_roi(flat)
    return flat, int(flat.size)


def bright_fraction_sweep(
    values: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """For each threshold, fraction and count of voxels >= threshold."""
    if values.size == 0:
        zeros = np.zeros(len(thresholds), dtype=np.float64)
        return zeros, zeros.astype(np.int64)

    n = values.size
    fractions = np.array([(values >= t).sum() / n for t in thresholds], dtype=np.float64)
    counts = np.array([(values >= t).sum() for t in thresholds], dtype=np.int64)
    return fractions, counts


def les_fraction_in_phase_cuboid(
    expert_mask: np.ndarray,
    les_meta: dict[str, Any],
    phase: DcePhase,
    *,
    reference_phase: DcePhase,
    phase_local: bool = True,
) -> tuple[float, int, int]:
    """Expert .les fill fraction inside the phase-local cuboid."""
    z_sl, y_sl, x_sl = phase_cuboid_slices(
        les_meta,
        phase,
        reference_phase=reference_phase,
        phase_local=phase_local,
    )
    expert = expert_mask[z_sl, y_sl, x_sl].astype(bool)
    cuboid_voxels = int(expert.size)
    les_voxels = int(expert.sum())
    if cuboid_voxels == 0:
        return 0.0, 0, 0
    return les_voxels / cuboid_voxels, les_voxels, cuboid_voxels


def compute_cuboid_brightness_table(
    phase_volumes: list[np.ndarray],
    phases: list[DcePhase],
    les_meta: dict[str, Any],
    expert_mask: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
    threshold_step: float = 0.05,
    normalize: bool = True,
) -> list[CuboidBrightnessRow]:
    """Bright fraction inside .les cuboid for each phase × threshold."""
    cutoffs = thresholds if thresholds is not None else default_thresholds(step=threshold_step)
    rows: list[CuboidBrightnessRow] = []
    reference_phase = phases[0]

    for phase, volume in zip(phases, phase_volumes, strict=True):
        values, cuboid_voxels = values_in_phase_cuboid(
            volume,
            les_meta,
            phase,
            reference_phase=reference_phase,
            normalize=normalize,
        )
        les_frac, les_voxels, cuboid_from_expert = les_fraction_in_phase_cuboid(
            expert_mask,
            les_meta,
            phase,
            reference_phase=reference_phase,
        )
        if cuboid_voxels == 0:
            cuboid_voxels = cuboid_from_expert

        fractions, counts = bright_fraction_sweep(values, cutoffs)
        for threshold, bright_fraction, bright_voxels in zip(
            cutoffs,
            fractions,
            counts,
            strict=True,
        ):
            rows.append(
                CuboidBrightnessRow(
                    phase_index=phase.index,
                    threshold=float(threshold),
                    bright_fraction=float(bright_fraction),
                    bright_voxels=int(bright_voxels),
                    cuboid_voxels=int(cuboid_voxels),
                    les_fraction=float(les_frac),
                    les_voxels=int(les_voxels),
                )
            )
    return rows


def format_brightness_table(rows: list[CuboidBrightnessRow]) -> str:
    """Human-readable table for terminal logging."""
    if not rows:
        return "(no cuboid voxels in any phase)"

    header = (
        f"{'Phase':>5}  {'Thresh':>6}  {'Bright':>7}  {'BrightVox':>9}  "
        f"{'CuboidVox':>9}  {'LesFrac':>7}  {'LesVox':>6}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"P{row.phase_index:>4}  {row.threshold:6.2f}  {row.bright_fraction:7.3f}  "
            f"{row.bright_voxels:9d}  {row.cuboid_voxels:9d}  "
            f"{row.les_fraction:7.3f}  {row.les_voxels:6d}"
        )
    return "\n".join(lines)


def expert_mask_for_phase_grid(expert_mask: np.ndarray, phase: DcePhase) -> np.ndarray:
    """Phase-local .les voxels for napari overlay."""
    return mask_for_phase(expert_mask, phase)


def phase_cuboid_histograms(
    phase_volumes: list[np.ndarray],
    phases: list[DcePhase],
    les_meta: dict[str, Any],
    *,
    normalize: bool = True,
    bins: int = 50,
) -> dict[int, np.ndarray]:
    """Normalized cuboid voxel values per phase index."""
    out: dict[int, np.ndarray] = {}
    reference_phase = phases[0]
    for phase, volume in zip(phases, phase_volumes, strict=True):
        values, _ = values_in_phase_cuboid(
            volume,
            les_meta,
            phase,
            reference_phase=reference_phase,
            normalize=normalize,
        )
        out[phase.index] = values
    return out


def plot_phase_cuboid_histograms(
    phase_volumes: list[np.ndarray],
    phases: list[DcePhase],
    les_meta: dict[str, Any],
    expert_mask: np.ndarray,
    *,
    slug: str = "",
    bins: int = 50,
    normalize: bool = True,
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """2×2 histogram grid for P1–P4 intensities inside the .les cuboid."""
    import matplotlib.pyplot as plt

    histograms = phase_cuboid_histograms(
        phase_volumes,
        phases,
        les_meta,
        normalize=normalize,
        bins=bins,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    title = f"{slug} — cuboid intensity histograms (1–99% norm per phase)" if slug else "Cuboid intensity histograms"
    fig.suptitle(title, fontsize=12)

    for phase, ax in zip(phases, axes.ravel(), strict=True):
        values = histograms.get(phase.index, np.array([], dtype=np.float32))
        les_frac, les_voxels, cuboid_voxels = les_fraction_in_phase_cuboid(
            expert_mask,
            les_meta,
            phase,
            reference_phase=phases[0],
        )
        label = f"P{phase.index} (n={values.size:,})"
        if phase.acquisition_time:
            label += f"\n{phase.acquisition_time}"
        if values.size:
            ax.hist(values, bins=bins, range=(0.0, 1.0), color="#4C72B0", alpha=0.85, edgecolor="white")
        ax.axvline(les_frac, color="#C44E52", linestyle="--", linewidth=1.5, label=f".les fill {les_frac:.2%}")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Normalized intensity")
        ax.set_ylabel("Voxels")
        ax.legend(fontsize=8, loc="upper right")
        ax.text(
            0.02,
            0.98,
            f"les={les_voxels:,} / cuboid={cuboid_voxels:,}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )

    fig.tight_layout()

    saved: Path | None = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        saved = output_path

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved


def bright_fraction_curves_by_phase(
    rows: list[CuboidBrightnessRow],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Map phase index -> (thresholds, bright_fractions)."""
    by_phase: dict[int, list[CuboidBrightnessRow]] = {}
    for row in rows:
        by_phase.setdefault(row.phase_index, []).append(row)

    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for phase_index, phase_rows in sorted(by_phase.items()):
        ordered = sorted(phase_rows, key=lambda row: row.threshold)
        thresholds = np.array([row.threshold for row in ordered], dtype=np.float64)
        fractions = np.array([row.bright_fraction for row in ordered], dtype=np.float64)
        curves[phase_index] = (thresholds, fractions)
    return curves


def plot_cuboid_bright_fraction_vs_threshold(
    phase_volumes: list[np.ndarray],
    phases: list[DcePhase],
    les_meta: dict[str, Any],
    expert_mask: np.ndarray,
    *,
    slug: str = "",
    threshold_step: float = 0.05,
    normalize: bool = True,
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Plot % cuboid voxels ≥ threshold vs threshold for P1–P4."""
    import matplotlib.pyplot as plt

    rows = compute_cuboid_brightness_table(
        phase_volumes,
        phases,
        les_meta,
        expert_mask,
        threshold_step=threshold_step,
        normalize=normalize,
    )
    curves = bright_fraction_curves_by_phase(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    title = (
        f"{slug} — % cuboid bright vs threshold (1–99% norm per phase)"
        if slug
        else "% cuboid bright vs threshold"
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Normalized intensity threshold")
    ax.set_ylabel("% cuboid voxels ≥ threshold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 100.0)
    ax.grid(True, alpha=0.3)

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    for phase, color in zip(phases, colors, strict=True):
        thresholds, fractions = curves.get(phase.index, (np.array([]), np.array([])))
        if thresholds.size == 0:
            continue
        pct = fractions * 100.0
        label = f"P{phase.index}"
        if phase.acquisition_time:
            label += f" ({phase.acquisition_time})"
        ax.plot(thresholds, pct, marker="o", markersize=3, linewidth=1.8, color=color, label=label)

        les_frac, _, _ = les_fraction_in_phase_cuboid(
            expert_mask,
            les_meta,
            phase,
            reference_phase=phases[0],
        )
        if les_frac > 0:
            ax.axhline(
                les_frac * 100.0,
                color=color,
                linestyle=":",
                linewidth=1.0,
                alpha=0.7,
            )

    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    saved: Path | None = None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        saved = output_path

    if show:
        plt.show()
    else:
        plt.close(fig)

    return saved

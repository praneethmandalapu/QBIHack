"""Load Philip-Chandan raw extract + expert mask → PDE-ready input (Vinesh-owned)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import center_of_mass, zoom

VINESH_DIR = Path(__file__).resolve().parent
SPIKE_ROOT = VINESH_DIR.parent
sys.path.insert(0, str(SPIKE_ROOT))
sys.path.insert(0, str(VINESH_DIR))

from mask_seeding import seed_from_mask  # noqa: E402

# Core density assigned to the densest part of the lesion when seeding from the
# expert mask. Kept just below carrying capacity so the logistic term stays
# active (growth comes from the invading margin, not from filling headroom).
DENSITY_PEAK = 0.9

from handoff_contract import (  # noqa: E402
    contract_version,
    default_grid_size,
    grid_size_options,
    max_shape_for_grid,
    pde_input_spec,
    target_spacing_mm,
)
from spike_paths import (  # noqa: E402
    REPO_ROOT,
    SPIKE_PATIENT,
    ensure_spike_dirs,
    pde_input_metadata,
    pde_input_npy,
    pde_input_npy_legacy,
    raw_extract_npy,
    resolve_raw_extract_metadata,
    resolve_raw_extract_npy,
    segmentation_mask_path,
)


def load_raw_extract(slug: str | None = None) -> tuple[np.ndarray, dict]:
    name = slug or SPIKE_PATIENT["slug"]
    npy_path = resolve_raw_extract_npy(name)
    json_path = resolve_raw_extract_metadata(name)
    if not npy_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f"Missing raw extract. Ask Philip-Chandan to run export_raw_extract.py "
            f"(expected {npy_path} and {json_path})"
        )
    volume = np.load(npy_path)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    return volume, metadata


def _load_mask_nifti(mask_path: Path, mr_shape: tuple[int, ...]) -> np.ndarray:
    """Load expert mask as (Z, Y, X) float32 with values 0/1."""
    import nibabel as nib

    img = nib.load(str(mask_path))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {data.shape} from {mask_path}")
    mask = np.transpose(data, (2, 1, 0))
    if mask.shape != mr_shape:
        raise ValueError(f"Mask shape {mask.shape} != MR shape {mr_shape} ({mask_path})")
    return (mask > 0).astype(np.float32)


def load_expert_mask(metadata: dict, mr_shape: tuple[int, ...]) -> tuple[np.ndarray, Path]:
    """Resolve segmentation_path from raw metadata (or spike default) and load mask."""
    seg_rel = metadata.get("segmentation_path")
    if seg_rel:
        mask_path = REPO_ROOT / seg_rel
    else:
        mask_path = segmentation_mask_path(metadata.get("slug"))
    if not mask_path.is_file():
        raise FileNotFoundError(
            f"Missing expert mask at {mask_path}. "
            "Re-run export_raw_extract.py with copy_mask=True."
        )
    return _load_mask_nifti(mask_path, mr_shape), mask_path


def _crop_or_pad_centered(
    volume: np.ndarray,
    target_shape: tuple[int, int, int],
    center: tuple[float, ...],
    pad_value: float,
) -> np.ndarray:
    """Return a `target_shape` box of `volume` centered on `center` (voxel coords)."""
    out = np.full(target_shape, pad_value, dtype=volume.dtype)
    for_slices_src: list[slice] = []
    for_slices_dst: list[slice] = []
    for axis, (size, tgt, c) in enumerate(zip(volume.shape, target_shape, center)):
        start = int(round(c - tgt / 2))
        end = start + tgt
        src_start = max(start, 0)
        src_end = min(end, size)
        dst_start = src_start - start
        dst_end = dst_start + (src_end - src_start)
        for_slices_src.append(slice(src_start, src_end))
        for_slices_dst.append(slice(dst_start, dst_end))
    out[tuple(for_slices_dst)] = volume[tuple(for_slices_src)]
    return out


def prepare_pde_stages(
    volume: np.ndarray,
    spacing_mm: list[float],
    expert_mask: np.ndarray,
    *,
    grid_size: int | None = None,
    max_shape_xyz: tuple[int, int, int] | None = None,
    target_spacing: list[float] | None = None,
) -> dict[str, np.ndarray | list[float] | float | int]:
    """Run the full PDE prep pipeline and return intermediate arrays for QC."""
    pde_spec = pde_input_spec()
    size = grid_size or default_grid_size()
    shape_limit = tuple(max_shape_xyz or max_shape_for_grid(size))
    spacing_target = target_spacing or target_spacing_mm()
    vmin_out, vmax_out = (float(v) for v in pde_spec["value_range"])
    background = float(pde_spec["background_value"])

    vol = np.asarray(volume, dtype=np.float32)
    mask = np.asarray(expert_mask, dtype=np.float32)
    if mask.shape != vol.shape:
        raise ValueError(f"expert_mask shape {mask.shape} != volume shape {vol.shape}")

    zoom_factors = [s / t for s, t in zip(spacing_mm, spacing_target)]
    resampled = zoom(vol, zoom_factors, order=1)
    resampled_mask = zoom(mask, zoom_factors, order=0) > 0.5

    rmin, rmax = float(resampled.min()), float(resampled.max())
    if rmax > rmin:
        norm = (resampled - rmin) / (rmax - rmin)
    else:
        norm = np.zeros_like(resampled)
    norm = norm * (vmax_out - vmin_out) + vmin_out

    # Seed the PDE density from the expert mask's GEOMETRY (dense core + low-density
    # infiltrative rim), not from MR intensity. MRI does not measure cell density;
    # keeping normalized intensity put the whole tumor at a flat ~0.30 plateau, so
    # the logistic term grew each voxel in place (densified) instead of advancing
    # the invasive front — the unrealistic growth Philip flagged. seed_from_mask is
    # the standard glioma reaction-diffusion initialization (see mask_seeding.py).
    seed = seed_from_mask(resampled_mask.astype(np.float32),
                          profile="ramp", peak=DENSITY_PEAK)
    segmented = np.where(resampled_mask, seed, background).astype(np.float32)

    if resampled_mask.any():
        center = center_of_mass(resampled_mask.astype(np.float32))
    else:
        center = tuple(s / 2 for s in segmented.shape)
    pde_volume = _crop_or_pad_centered(segmented, shape_limit, center, background)

    return {
        "grid_size": size,
        "resampled": resampled.astype(np.float32),
        "resampled_mask": resampled_mask,
        "normalized": norm.astype(np.float32),
        "segmented": segmented,
        "pde_volume": pde_volume.astype(np.float32),
        "spacing_mm": list(spacing_target),
        "background_value": background,
    }


def prepare_pde_input(
    volume: np.ndarray,
    spacing_mm: list[float],
    expert_mask: np.ndarray,
    *,
    grid_size: int | None = None,
    max_shape_xyz: tuple[int, int, int] | None = None,
    target_spacing: list[float] | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Resample, normalize, and crop raw MR + expert mask into solve_growth input."""
    stages = prepare_pde_stages(
        volume,
        spacing_mm,
        expert_mask,
        grid_size=grid_size,
        max_shape_xyz=max_shape_xyz,
        target_spacing=target_spacing,
    )
    return stages["pde_volume"], stages["spacing_mm"]  # type: ignore[return-value]


def save_pde_input(
    volume: np.ndarray,
    spacing_mm: list[float],
    raw_metadata: dict,
    *,
    segmentation_path: Path,
    slug: str | None = None,
    grid_size: int | None = None,
    write_legacy_compat: bool = True,
) -> tuple[Path, Path]:
    ensure_spike_dirs()
    name = slug or SPIKE_PATIENT["slug"]
    size = grid_size or default_grid_size()
    npy_path = pde_input_npy(name, grid_size=size)
    json_path = pde_input_metadata(name, grid_size=size)
    npy_path.parent.mkdir(parents=True, exist_ok=True)

    pde_spec = pde_input_spec()
    np.save(npy_path, volume.astype(np.float32))
    metadata = {
        "contract_version": contract_version(),
        "slug": name,
        "grid_size": size,
        "source_raw_extract": str(raw_extract_npy(name).relative_to(REPO_ROOT)),
        "source_segmentation": str(segmentation_path.relative_to(REPO_ROOT)),
        "shape": list(volume.shape),
        "dtype": pde_spec["dtype"],
        "axis_order": pde_spec["axis_order"],
        "spacing_mm": spacing_mm,
        "value_range": pde_spec["value_range"],
        "background_value": pde_spec["background_value"],
        "tumor_burden_rule": pde_spec["tumor_burden_rule"],
        "segmentation": pde_spec["segmentation"],
        "value_semantics": {
            str(pde_spec["background_value"]): "background/healthy",
            ">0": "modeled initial tumor cell density, seeded from expert-mask "
                  "geometry (dense core + infiltrative rim); NOT MR intensity",
        },
        "upstream": {
            "patient_id": raw_metadata.get("patient_id"),
            "dataset": raw_metadata.get("dataset"),
            "timepoint": raw_metadata.get("timepoint"),
            "study_date": raw_metadata.get("study_date"),
            "raw_contract_version": raw_metadata.get("contract_version"),
        },
    }
    json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if write_legacy_compat and size == default_grid_size():
        legacy = pde_input_npy_legacy(name)
        shutil.copy2(npy_path, legacy)
        legacy_json = legacy.with_suffix(".json")
        shutil.copy2(json_path, legacy_json)

    return npy_path, json_path


def run_prepare_for_slug(
    slug: str | None = None,
    *,
    grid_size: int | None = None,
) -> tuple[Path, Path]:
    """Load raw extract + mask, prepare PDE input, and write outputs for one grid size."""
    raw_volume, raw_metadata = load_raw_extract(slug)
    expert_mask, mask_path = load_expert_mask(raw_metadata, raw_volume.shape)
    size = grid_size or default_grid_size()
    pde_volume, pde_spacing = prepare_pde_input(
        raw_volume,
        raw_metadata["spacing_mm"],
        expert_mask,
        grid_size=size,
    )
    return save_pde_input(
        pde_volume,
        pde_spacing,
        raw_metadata,
        segmentation_path=mask_path,
        slug=raw_metadata.get("slug") or slug,
        grid_size=size,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare PDE input from raw extract + expert mask.")
    parser.add_argument("--slug", default=None, help="Cohort slug (default: spike patient)")
    parser.add_argument(
        "--grid-size",
        type=int,
        choices=grid_size_options(),
        default=None,
        help=f"Crop cube edge length (options: {list(grid_size_options())})",
    )
    parser.add_argument(
        "--all-grids",
        action="store_true",
        help="Run for every grid_size in handoff_contract.json",
    )
    args = parser.parse_args(argv)

    sizes = list(grid_size_options()) if args.all_grids else [args.grid_size or default_grid_size()]
    for size in sizes:
        npy_path, json_path = run_prepare_for_slug(args.slug, grid_size=size)
        print(f"g{size}: wrote {npy_path}")
        print(f"g{size}: wrote {json_path}")


if __name__ == "__main__":
    main()

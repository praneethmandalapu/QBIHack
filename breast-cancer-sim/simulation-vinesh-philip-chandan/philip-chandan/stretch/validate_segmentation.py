"""Compare Otsu tumor masks to TCIA radiologist .les annotations."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

STRETCH_DIR = Path(__file__).resolve().parent
PHILIP_CHANDAN_DIR = STRETCH_DIR.parent
SPIKE_ROOT = PHILIP_CHANDAN_DIR.parent
sys.path.insert(0, str(STRETCH_DIR))
sys.path.insert(0, str(PHILIP_CHANDAN_DIR))

from download_tcia import (  # noqa: E402
    download_series_to_dir,
    group_series_by_study,
    list_mr_series,
)
from load_les_mask import find_les_files, load_les_mask, parse_les_filename  # noqa: E402
from load_manifest import find_volume, load_volumes  # noqa: E402
from paths import (  # noqa: E402
    ensure_validation_dirs,
    validation_metrics_csv,
    validation_qc_les_overlay,
    validation_qc_otsu_overlay,
    VALIDATION_DICOM_DIR,
)
from prep_volume import (  # noqa: E402
    load_raw_extract,
    normalize_volume,
    tumor_mask_largest_component,
)
from tcia_extractor import extract_volume_with_spacing  # noqa: E402

_DCE_TOKENS = ("VIBRANT", "BRAVA", "AX T1", "AXIAL T1", "DYN", "DCE")


def _is_dce_series(description: str) -> bool:
    upper = description.strip().upper()
    if any(token in upper for token in ("SAG", "SCOUT", "CAL", "(", "CAD", "IDEAL", "MIP", "B=800")):
        return False
    return any(token in upper for token in _DCE_TOKENS)


def _dce_sort_key(entry: dict[str, Any]) -> tuple[int, str]:
    """Order DCE series like TCIA/NBIA: ax t1, VIBRANT, BRAVA."""
    upper = str(entry.get("SeriesDescription", "")).upper()
    if "AX T1" in upper or "AXIAL T1" in upper:
        return (0, upper)
    if "VIBRANT" in upper:
        return (1, upper)
    if "BRAVA" in upper:
        return (2, upper)
    return (9, upper)


def pick_dce_series(series_list: list[dict[str, Any]], dce_index: int) -> dict[str, Any]:
    """Pick the Nth DCE-MRI sequence (1-based index) for a study."""
    numbered = [entry for entry in series_list if entry.get("SeriesNumber") not in (None, "")]
    if numbered:
        ordered = sorted(numbered, key=lambda entry: int(entry.get("SeriesNumber") or 0))
        dce_series = [
            entry
            for entry in ordered
            if _is_dce_series(str(entry.get("SeriesDescription", "")))
        ]
    else:
        dce_series = sorted(
            [
                entry
                for entry in series_list
                if _is_dce_series(str(entry.get("SeriesDescription", "")))
            ],
            key=_dce_sort_key,
        )
    if not dce_series:
        raise ValueError("No DCE-like MR series found for study")
    if dce_index < 1 or dce_index > len(dce_series):
        raise ValueError(
            f"DCE index S{dce_index} out of range; study has {len(dce_series)} DCE series"
        )
    return dce_series[dce_index - 1]


def _validation_series_dir(tcga_id: str, study_date: str, series_uid: str) -> Path:
    return VALIDATION_DICOM_DIR / tcga_id / study_date / series_uid


def load_annotation_volume(
    *,
    tcga_id: str,
    study_date: str,
    les_path: Path,
    slug: str | None = None,
) -> tuple[np.ndarray, list[float], str, int, str]:
    """Load the MR volume that matches the annotated DCE sequence for a .les file."""
    _, dce_index, _ = parse_les_filename(les_path.name)
    series_list = list_mr_series(tcga_id)
    grouped = group_series_by_study(series_list)
    if study_date not in grouped:
        raise FileNotFoundError(f"No MR study on {study_date} for {tcga_id}")
    target_series = pick_dce_series(grouped[study_date], dce_index)
    series_description = str(target_series.get("SeriesDescription", ""))
    image_count = int(target_series.get("ImageCount") or 0)

    if slug:
        volume, sidecar = load_raw_extract(slug)
        if volume.shape[0] == image_count:
            return (
                volume,
                list(sidecar["spacing_mm"]),
                series_description,
                dce_index,
                "raw_extract",
            )

    cache_dir = _validation_series_dir(
        tcga_id,
        study_date,
        str(target_series["SeriesInstanceUID"]),
    )
    if not any(cache_dir.rglob("*")):
        download_series_to_dir(target_series, cache_dir)
    volume, spacing = extract_volume_with_spacing(cache_dir)
    return volume, spacing, series_description, dce_index, "validation_dicom"


def dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    denom = int(a.sum() + b.sum())
    if denom == 0:
        return 1.0
    intersection = int(np.logical_and(a, b).sum())
    return 2.0 * intersection / denom


def volume_mm3(mask: np.ndarray, spacing_mm: list[float]) -> float:
    dz, dy, dx = (float(s) for s in spacing_mm)
    return float(mask.astype(bool).sum()) * dz * dy * dx


def compare_masks(
    *,
    expert_mask: np.ndarray,
    otsu_mask: np.ndarray,
    spacing_mm: list[float],
) -> dict[str, float]:
    expert_vol = volume_mm3(expert_mask, spacing_mm)
    otsu_vol = volume_mm3(otsu_mask, spacing_mm)
    if expert_vol <= 0:
        rel_volume_error = float("nan")
    else:
        rel_volume_error = (otsu_vol - expert_vol) / expert_vol
    return {
        "dice": dice_coefficient(expert_mask, otsu_mask),
        "expert_volume_mm3": expert_vol,
        "otsu_volume_mm3": otsu_vol,
        "relative_volume_error": rel_volume_error,
        "expert_voxels": int(expert_mask.sum()),
        "otsu_voxels": int(otsu_mask.sum()),
        "area_fraction_otsu_over_les": (
            float(otsu_mask.sum()) / float(expert_mask.sum())
            if expert_mask.sum() > 0
            else float("nan")
        ),
    }


def _pick_expert_z_index(expert_mask: np.ndarray, norm: np.ndarray) -> int:
    """Slice where the radiologist .les mask is most visible."""
    if expert_mask.any():
        return int(expert_mask.sum(axis=(1, 2)).argmax())
    return norm.shape[0] // 2


def _pick_otsu_z_index(otsu_mask: np.ndarray, norm: np.ndarray) -> int:
    """Slice where the Otsu mask is most visible."""
    if otsu_mask.any():
        return int(otsu_mask.sum(axis=(1, 2)).argmax())
    return norm.shape[0] // 2


def _format_area_fraction(otsu_voxels: int, expert_voxels: int) -> str:
    if expert_voxels <= 0:
        return "area n/a"
    return f"area={otsu_voxels / expert_voxels:.1f}x"


def _save_mask_overlay(
    path: Path,
    norm: np.ndarray,
    mask: np.ndarray,
    z_idx: int,
    *,
    title: str,
    fill_rgba: tuple[float, float, float, float],
    contour_color: str,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib import colors

    fig, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(norm[z_idx], cmap="gray", vmin=0, vmax=1)
    slice_mask = mask[z_idx]
    if slice_mask.any():
        overlay = colors.to_rgba(contour_color, alpha=fill_rgba[3])
        tint = np.zeros((*slice_mask.shape, 4), dtype=np.float32)
        tint[slice_mask > 0] = overlay
        axis.imshow(tint)
        axis.contour(slice_mask, levels=[0.5], colors=[contour_color], linewidths=2.0)
    axis.set_title(title, fontsize=10)
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def save_validation_overlays(
    slug: str,
    norm: np.ndarray,
    expert_mask: np.ndarray,
    otsu_mask: np.ndarray,
    *,
    tcga_id: str,
    timepoint: str,
    dce_index: int,
    dice: float,
    expert_voxels: int,
    otsu_voxels: int,
) -> dict[str, Path]:
    """Save separate mid-Z QC PNGs for radiologist .les and Otsu masks."""
    ensure_validation_dirs()
    expert_z = _pick_expert_z_index(expert_mask, norm)
    otsu_z = _pick_otsu_z_index(otsu_mask, norm)
    area_text = _format_area_fraction(otsu_voxels, expert_voxels)

    les_path = _save_mask_overlay(
        validation_qc_les_overlay(slug),
        norm,
        expert_mask,
        expert_z,
        title=(
            f"{tcga_id} {timepoint} z={expert_z} | .les S{dce_index} "
            f"({expert_voxels:,} vox, {area_text})"
        ),
        fill_rgba=(0.0, 1.0, 0.0, 0.45),
        contour_color="#00FF00",
    )
    otsu_path = _save_mask_overlay(
        validation_qc_otsu_overlay(slug),
        norm,
        otsu_mask,
        otsu_z,
        title=(
            f"{tcga_id} {timepoint} z={otsu_z} | Otsu "
            f"({otsu_voxels:,} vox, Dice={dice:.3f}, {area_text})"
        ),
        fill_rgba=(1.0, 0.0, 1.0, 0.25),
        contour_color="#FF00FF",
    )
    return {"les": les_path, "otsu": otsu_path}


def validate_slug(slug: str, *, lesions_dir: Path | None = None) -> dict[str, Any]:
    """Run .les vs Otsu comparison for one manifest slug (baseline only)."""
    entry = find_volume(slug=slug)
    tcga_id = entry["tcga_id"]
    study_date = entry["study_date"]
    timepoint = entry.get("timepoint", "")

    les_files = find_les_files(tcga_id, lesions_dir)
    if not les_files:
        raise FileNotFoundError(f"No .les file found for {tcga_id}")

    # Radiogenomics annotations are single-timepoint; prefer the only file or S*-1.les.
    les_path = les_files[0]
    volume, spacing_mm, series_description, dce_index, volume_source = load_annotation_volume(
        tcga_id=tcga_id,
        study_date=study_date,
        les_path=les_path,
        slug=slug,
    )
    expert_mask, les_meta = load_les_mask(les_path, volume.shape)
    norm = normalize_volume(volume)
    otsu_mask = tumor_mask_largest_component(norm)
    metrics = compare_masks(
        expert_mask=expert_mask,
        otsu_mask=otsu_mask,
        spacing_mm=spacing_mm,
    )

    overlay_paths = save_validation_overlays(
        slug,
        norm,
        expert_mask,
        otsu_mask,
        tcga_id=tcga_id,
        timepoint=timepoint,
        dce_index=dce_index,
        dice=metrics["dice"],
        expert_voxels=metrics["expert_voxels"],
        otsu_voxels=metrics["otsu_voxels"],
    )

    return {
        "slug": slug,
        "tcga_id": tcga_id,
        "subtype": entry.get("subtype"),
        "timepoint": timepoint,
        "study_date": study_date,
        "les_file": les_path.name,
        "dce_index": dce_index,
        "annotated_series": series_description,
        "volume_source": volume_source,
        "volume_shape_zyx": list(volume.shape),
        "spacing_mm": spacing_mm,
        **metrics,
        "les_overlay_png": str(overlay_paths["les"]),
        "otsu_overlay_png": str(overlay_paths["otsu"]),
        "les_metadata": les_meta,
    }


def validate_primaries(*, lesions_dir: Path | None = None) -> list[dict[str, Any]]:
    """Validate baseline slugs for rev2 primaries that have .les annotations."""
    results: list[dict[str, Any]] = []
    for entry in load_volumes():
        if entry.get("timepoint") != "baseline":
            continue
        tcga_id = entry["tcga_id"]
        if not find_les_files(tcga_id, lesions_dir):
            continue
        results.append(validate_slug(entry["slug"], lesions_dir=lesions_dir))
    return results


def write_metrics_csv(rows: list[dict[str, Any]], path: Path | None = None) -> Path:
    ensure_validation_dirs()
    out_path = path or validation_metrics_csv()
    fieldnames = [
        "slug",
        "tcga_id",
        "subtype",
        "study_date",
        "les_file",
        "dce_index",
        "annotated_series",
        "volume_source",
        "dice",
        "expert_volume_mm3",
        "otsu_volume_mm3",
        "relative_volume_error",
        "expert_voxels",
        "otsu_voxels",
        "area_fraction_otsu_over_les",
        "les_overlay_png",
        "otsu_overlay_png",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return out_path


def ensure_validation_artifacts() -> tuple[Path | None, list[dict[str, Any]]]:
    """Generate validation metrics + QC overlays if lesions are present."""
    if not find_les_files("TCGA-AR-A1AX") and not find_les_files("TCGA-AR-A1AQ"):
        return None, []
    rows = validate_primaries()
    csv_path = write_metrics_csv(rows)
    return csv_path, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Otsu masks against TCIA .les masks.")
    parser.add_argument("--slug", help="Validate one manifest slug (baseline)")
    parser.add_argument(
        "--all-primary",
        action="store_true",
        help="Validate all baseline primaries with .les files",
    )
    args = parser.parse_args()

    if args.slug:
        rows = [validate_slug(args.slug)]
    elif args.all_primary:
        rows = validate_primaries()
    else:
        parser.error("Provide --slug or --all-primary")

    csv_path = write_metrics_csv(rows)
    print(f"Wrote {csv_path}")
    for row in rows:
        print(
            f"  {row['slug']}: Dice={row['dice']:.3f} "
            f"vol_err={row['relative_volume_error']:.1%} "
            f"series={row['annotated_series']!r}"
        )
        print(f"    .les overlay: {row['les_overlay_png']}")
        print(f"    Otsu overlay: {row['otsu_overlay_png']}")


if __name__ == "__main__":
    main()

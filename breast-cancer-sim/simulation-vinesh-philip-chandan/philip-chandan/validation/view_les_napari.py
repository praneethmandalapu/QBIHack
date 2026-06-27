"""Interactive 3D viewer: baseline MR + TCIA radiologist .les overlay (napari).

Run (macOS/Linux):
    cd breast-cancer-sim
    source .venv/bin/activate
    pip install -r requirements.txt   # includes napari[pyqt6]

    # List baseline slugs that have a local .les file
    python simulation-vinesh-philip-chandan/philip-chandan/validation/view_les_napari.py --list

    # Luminal A baseline (VIBRANT S2)
    python simulation-vinesh-philip-chandan/philip-chandan/validation/view_les_napari.py \\
        --slug luminal_a_TCGA-AR-A1AX_baseline

    # Basal-like baseline + Otsu comparison overlay
    python simulation-vinesh-philip-chandan/philip-chandan/validation/view_les_napari.py \\
        --slug basal_TCGA-AR-A1AQ_baseline --otsu

    # Cuboid annotation shell (see MR inside the box)
    python simulation-vinesh-philip-chandan/philip-chandan/validation/view_les_napari.py \\
        --slug luminal_a_TCGA-AR-A1AX_baseline --cuboid

Run (Windows):
    cd breast-cancer-sim
    .venv\\Scripts\\Activate.ps1
    .venv\\Scripts\\python.exe simulation-vinesh-philip-chandan\\philip-chandan\\validation\\view_les_napari.py --list

Requires napari[pyqt6]. First launch may take ~10s while Qt initializes.
See validation/VALIDATION.md for dataset paths and validation context.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

VALIDATION_DIR = Path(__file__).resolve().parent
PHILIP_CHANDAN_DIR = VALIDATION_DIR.parent
STRETCH_DIR = PHILIP_CHANDAN_DIR / "stretch"

sys.path.insert(0, str(STRETCH_DIR))
sys.path.insert(0, str(PHILIP_CHANDAN_DIR))

import napari  # noqa: E402

from load_les_mask import (  # noqa: E402
    find_les_files,
    load_les_cuboid_boundary,
    load_les_mask,
)
from load_manifest import find_volume, load_volumes  # noqa: E402
from prep_volume import normalize_volume, tumor_mask_largest_component  # noqa: E402
from validate_segmentation import load_annotation_volume  # noqa: E402


def slugs_with_les() -> list[str]:
    """Baseline manifest slugs whose TCGA ID has a local .les file."""
    slugs: list[str] = []
    for entry in load_volumes():
        if entry.get("timepoint") != "baseline":
            continue
        if find_les_files(entry["tcga_id"]):
            slugs.append(entry["slug"])
    return slugs


def view_slug(
    slug: str,
    *,
    show_otsu: bool = False,
    cuboid_boundary: bool = False,
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
    display_mr = normalize_volume(volume)
    scale = tuple(float(s) for s in spacing_mm)

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
        overlay_name = f".les ({meta['lesion_voxels']:,} vox)"
        overlay_detail = f"lesion={meta['lesion_voxels']:,} vox"

    print(
        f"{slug}\n"
        f"  series: {series_desc!r} (S{dce_index})\n"
        f"  .les:   {les_path.name} ({overlay_detail})\n"
        f"  MR:     shape={volume.shape} source={source} spacing_mm={spacing_mm}\n"
        f"  overlay: {'cuboid boundary' if cuboid_boundary else 'filled lesion'}"
    )

    viewer = napari.Viewer(title=f"{slug} — .les overlay")
    viewer.add_image(
        display_mr,
        name="MR (normalized)",
        scale=scale,
        colormap="gray",
        contrast_limits=(0.0, 1.0),
    )
    viewer.add_labels(
        overlay_mask,
        name=overlay_name,
        scale=scale,
        opacity=0.85 if cuboid_boundary else 0.55,
    )
    if show_otsu:
        viewer.add_labels(
            tumor_mask_largest_component(display_mr),
            name="Otsu",
            scale=scale,
            opacity=0.35,
        )
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
        help="Also overlay Otsu + largest connected component",
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

    view_slug(slug, show_otsu=args.otsu, cuboid_boundary=args.cuboid)


if __name__ == "__main__":
    main()

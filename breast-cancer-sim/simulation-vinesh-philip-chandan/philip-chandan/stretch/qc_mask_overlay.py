"""Mid-Z overlay of tumor mask on normalized MR (stretch QC)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import SimpleITK as sitk

STRETCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STRETCH_DIR))

from paths import ensure_radiomics_dirs, radiomics_qc_overlay  # noqa: E402
from prep_volume import prep_for_radiomics  # noqa: E402


def save_mask_overlay(slug: str, *, crop: bool = True) -> Path:
    ensure_radiomics_dirs()
    sitk_image, sitk_mask, meta = prep_for_radiomics(slug, crop=crop, save_mask=True)
    image = sitk.GetArrayFromImage(sitk_image)
    mask = sitk.GetArrayFromImage(sitk_mask)

    z_idx = int(mask.sum(axis=(1, 2)).argmax()) if mask.any() else image.shape[0] // 2
    out_path = radiomics_qc_overlay(slug)

    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(image[z_idx], cmap="gray")
    if mask.any():
        axis.contour(mask[z_idx], levels=[0.5], colors="lime", linewidths=1.0)
    axis.set_title(
        f"{meta['tcga_id']} {meta.get('timepoint', '')} z={z_idx} "
        f"(mask vox={meta['mask_voxels']})"
    )
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Save mask overlay QC PNG for a slug.")
    parser.add_argument("--slug", default="luminal_a_TCGA-AR-A1AX_baseline")
    parser.add_argument("--no-crop", action="store_true")
    args = parser.parse_args()
    out_path = save_mask_overlay(args.slug, crop=not args.no_crop)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

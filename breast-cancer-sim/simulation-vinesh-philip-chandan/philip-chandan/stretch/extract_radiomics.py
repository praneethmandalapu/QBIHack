"""Extract radiomics features from prepped SITK image/mask pairs (dual backend)."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

STRETCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STRETCH_DIR))

from paths import DEFAULT_PARAMS_PATH  # noqa: E402
from prep_volume import prep_for_radiomics  # noqa: E402

BACKENDS = ("pyradiomics", "fastrad")
DEFAULT_BIN_WIDTH = 0.05
PARITY_TOLERANCE = 1e-3

# Pairs known to align on pre-normalized [0, 1] volumes with binWidth 0.05.
# Entropy is excluded: fastrad anchors bins at image minimum, PyRadiomics differs.
PARITY_FEATURE_PAIRS: tuple[tuple[str, str], ...] = (
    ("original_firstorder_Mean", "firstorder:mean"),
    ("original_firstorder_Variance", "firstorder:variance"),
    ("original_shape_VoxelVolume", "shape:voxel_volume"),
    ("original_glcm_Contrast", "glcm:contrast"),
    ("original_glcm_Correlation", "glcm:correlation"),
)


def _is_feature_key(key: str) -> bool:
    return key.startswith("original_")


def _to_float(value: Any) -> float | None:
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray) and value.size == 1:
        return float(value.item())
    return None


def _sitk_to_fastrad(
    sitk_image: sitk.Image,
    sitk_mask: sitk.Image,
) -> tuple[Any, Any]:
    import torch
    from fastrad import Mask, MedicalImage

    spacing = tuple(float(s) for s in sitk_image.GetSpacing())
    image_arr = np.ascontiguousarray(sitk.GetArrayFromImage(sitk_image), dtype=np.float32)
    mask_arr = np.ascontiguousarray(sitk.GetArrayFromImage(sitk_mask), dtype=np.float32)
    image = MedicalImage(
        torch.tensor(image_arr, dtype=torch.float32),
        spacing,
    )
    mask = Mask(
        torch.tensor(mask_arr, dtype=torch.float32),
        spacing,
    )
    return image, mask


def extract_features_pyradiomics(
    sitk_image: sitk.Image,
    sitk_mask: sitk.Image,
    *,
    params_path: Path | None = None,
) -> dict[str, float]:
    from radiomics import featureextractor

    params = str(params_path or DEFAULT_PARAMS_PATH)
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = extractor.execute(sitk_image, sitk_mask)
    return {
        key: scalar
        for key, value in raw.items()
        if _is_feature_key(key) and (scalar := _to_float(value)) is not None
    }


def extract_features_fastrad(
    sitk_image: sitk.Image,
    sitk_mask: sitk.Image,
    *,
    bin_width: float = DEFAULT_BIN_WIDTH,
    device: str = "auto",
) -> dict[str, float]:
    from fastrad import FeatureExtractor, FeatureSettings

    image, mask = _sitk_to_fastrad(sitk_image, sitk_mask)
    settings = FeatureSettings(
        feature_classes=["firstorder", "shape", "glcm"],
        bin_width=bin_width,
        device=device,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = FeatureExtractor(settings).extract(image, mask)
    return {key: float(value) for key, value in raw.items()}


def extract_features(
    sitk_image: sitk.Image,
    sitk_mask: sitk.Image,
    *,
    backend: str = "pyradiomics",
    params_path: Path | None = None,
    bin_width: float = DEFAULT_BIN_WIDTH,
    device: str = "auto",
) -> dict[str, float]:
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}; choose from {BACKENDS}")
    if backend == "pyradiomics":
        return extract_features_pyradiomics(sitk_image, sitk_mask, params_path=params_path)
    return extract_features_fastrad(
        sitk_image,
        sitk_mask,
        bin_width=bin_width,
        device=device,
    )


def extract_for_slug(
    slug: str,
    *,
    backend: str = "pyradiomics",
    crop: bool = True,
    params_path: Path | None = None,
    bin_width: float = DEFAULT_BIN_WIDTH,
    device: str = "auto",
) -> tuple[dict[str, float], dict[str, Any]]:
    sitk_image, sitk_mask, meta = prep_for_radiomics(slug, crop=crop, save_mask=True)
    features = extract_features(
        sitk_image,
        sitk_mask,
        backend=backend,
        params_path=params_path,
        bin_width=bin_width,
        device=device,
    )
    return features, meta


def compare_parity(
    sitk_image: sitk.Image,
    sitk_mask: sitk.Image,
    *,
    params_path: Path | None = None,
    bin_width: float = DEFAULT_BIN_WIDTH,
    tolerance: float = PARITY_TOLERANCE,
) -> list[tuple[str, float, float, float]]:
    """Return (pyradiomics_key, pyradiomics, fastrad, abs_diff) for mapped pairs."""
    py_feats = extract_features_pyradiomics(sitk_image, sitk_mask, params_path=params_path)
    fr_feats = extract_features_fastrad(sitk_image, sitk_mask, bin_width=bin_width, device="cpu")
    rows: list[tuple[str, float, float, float]] = []
    for py_key, fr_key in PARITY_FEATURE_PAIRS:
        if py_key not in py_feats or fr_key not in fr_feats:
            continue
        py_val = py_feats[py_key]
        fr_val = fr_feats[fr_key]
        rows.append((py_key, py_val, fr_val, abs(py_val - fr_val)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract radiomics features for one slug.")
    parser.add_argument("--slug", default="luminal_a_TCGA-AR-A1AX_baseline")
    parser.add_argument("--backend", choices=BACKENDS, default="pyradiomics")
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH)
    parser.add_argument("--bin-width", type=float, default=DEFAULT_BIN_WIDTH)
    parser.add_argument("--device", default="auto", help="fastrad only: cpu, cuda, or auto")
    args = parser.parse_args()

    features, meta = extract_for_slug(
        args.slug,
        backend=args.backend,
        crop=not args.no_crop,
        params_path=args.params,
        bin_width=args.bin_width,
        device=args.device,
    )
    print(f"Extracted {len(features)} features for {args.slug} via {args.backend}")
    print(f"  tcga_id={meta.get('tcga_id')} timepoint={meta.get('timepoint')}")
    for key in sorted(features)[:8]:
        print(f"  {key}: {features[key]}")
    if len(features) > 8:
        print(f"  ... ({len(features) - 8} more)")


if __name__ == "__main__":
    main()

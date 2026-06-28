"""On-disk paths for automated breast tumor segmentation (Philip-Chandan)."""

from __future__ import annotations

from pathlib import Path

SEGMENTATION_DIR = Path(__file__).resolve().parent
PHILIP_CHANDAN_DIR = SEGMENTATION_DIR.parent
SPIKE_ROOT = PHILIP_CHANDAN_DIR.parent
REPO_ROOT = SPIKE_ROOT.parent

RAW_EXTRACT_DIR = REPO_ROOT / "data" / "processed" / "raw-extract-philip-chandan"
MANIFEST_PATH = RAW_EXTRACT_DIR / "manifest.json"

# Segmentation outputs (gitignored via data/)
SEGMENTATION_OUT_DIR = REPO_ROOT / "data" / "processed" / "segmentation-philip-chandan"
QC_SEGMENTATION_DIR = REPO_ROOT / "data" / "qc" / "segmentation-philip-chandan"

# Ground truth + validation (shared with stretch/)
LESIONS_DIR = REPO_ROOT / "data" / "raw" / "tcia-radiogenomics" / "lesions"
LESIONS_EXTRACT_DIR = LESIONS_DIR / "TCGA_Segmented_Lesions_UofC"
VALIDATION_DIR = REPO_ROOT / "data" / "processed" / "validation-philip-chandan"

# Optional: downloaded nnU-Net weights (local only, not committed)
MODELS_DIR = REPO_ROOT / "data" / "models" / "segmentation-philip-chandan"
MAMA_MIA_NNUNET_DIR = MODELS_DIR / "mama-mia-nnunet"

REFERENCE_METHOD = "les"
# Automated methods evaluated by run_benchmark.py when {slug}_{method}_mask.npy exists.
BENCHMARK_METHODS = ("nnunet", "medsam")
# Legacy heuristic — not benchmarked here (see stretch/validate_segmentation.py).
MASK_SOURCES = (REFERENCE_METHOD, *BENCHMARK_METHODS, "otsu")
SEGMENTATION_METHODS = MASK_SOURCES  # alias


def ensure_segmentation_dirs() -> None:
    """Create gitignored output folders for segmentation artifacts."""
    for path in (SEGMENTATION_OUT_DIR, QC_SEGMENTATION_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def raw_extract_npy(slug: str) -> Path:
    return RAW_EXTRACT_DIR / f"{slug}.npy"


def raw_extract_json(slug: str) -> Path:
    return RAW_EXTRACT_DIR / f"{slug}.json"


def mask_npy(slug: str, method: str) -> Path:
    """Per-slug binary tumor mask for a segmentation method."""
    if method not in MASK_SOURCES:
        raise ValueError(f"Unknown method {method!r}; choose from {MASK_SOURCES}")
    return SEGMENTATION_OUT_DIR / f"{slug}_{method}_mask.npy"


def mask_metadata_json(slug: str, method: str) -> Path:
    return SEGMENTATION_OUT_DIR / f"{slug}_{method}_mask.json"


def comparison_metrics_csv() -> Path:
    return SEGMENTATION_OUT_DIR / "segmentation_comparison.csv"


def qc_overlay_png(slug: str, method: str) -> Path:
    return QC_SEGMENTATION_DIR / f"{slug}_{method}_overlay_mid-z.png"

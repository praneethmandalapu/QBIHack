"""On-disk paths for the Philip-Chandan PyRadiomics stretch (isolated from sprint handoff)."""

from __future__ import annotations

from pathlib import Path

STRETCH_DIR = Path(__file__).resolve().parent
PHILIP_CHANDAN_DIR = STRETCH_DIR.parent
SPIKE_ROOT = PHILIP_CHANDAN_DIR.parent
REPO_ROOT = SPIKE_ROOT.parent

RAW_EXTRACT_DIR = REPO_ROOT / "data" / "processed" / "raw-extract-philip-chandan"
MANIFEST_PATH = RAW_EXTRACT_DIR / "manifest.json"

RADIOMICS_DIR = REPO_ROOT / "data" / "processed" / "radiomics-philip-chandan"
QC_RADIOMICS_DIR = REPO_ROOT / "data" / "qc" / "radiomics-philip-chandan"

LESIONS_DIR = REPO_ROOT / "data" / "raw" / "tcia-radiogenomics" / "lesions"
LESIONS_EXTRACT_DIR = LESIONS_DIR / "TCGA_Segmented_Lesions_UofC"
VALIDATION_DICOM_DIR = REPO_ROOT / "data" / "raw" / "tcia-radiogenomics" / "validation-dicom"
VALIDATION_DIR = REPO_ROOT / "data" / "processed" / "validation-philip-chandan"
QC_VALIDATION_DIR = REPO_ROOT / "data" / "qc" / "validation-philip-chandan"


def ensure_radiomics_dirs() -> None:
    """Create gitignored output folders for stretch artifacts."""
    for path in (RADIOMICS_DIR, QC_RADIOMICS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def ensure_validation_dirs() -> None:
    """Create gitignored folders for segmentation validation artifacts."""
    for path in (VALIDATION_DIR, QC_VALIDATION_DIR, VALIDATION_DICOM_DIR):
        path.mkdir(parents=True, exist_ok=True)


def raw_extract_npy(slug: str) -> Path:
    return RAW_EXTRACT_DIR / f"{slug}.npy"


def raw_extract_json(slug: str) -> Path:
    return RAW_EXTRACT_DIR / f"{slug}.json"


def radiomics_mask_npy(slug: str) -> Path:
    return RADIOMICS_DIR / f"{slug}_mask.npy"


def radiomics_qc_overlay(slug: str) -> Path:
    return QC_RADIOMICS_DIR / f"{slug}_mask_overlay_mid-z.png"


DEFAULT_PARAMS_PATH = STRETCH_DIR / "radiomics_params.yaml"


def features_all_csv() -> Path:
    return RADIOMICS_DIR / "features_all.csv"


def features_longitudinal_csv() -> Path:
    return RADIOMICS_DIR / "features_longitudinal.csv"


def validation_metrics_csv() -> Path:
    return VALIDATION_DIR / "validation_metrics.csv"


def validation_qc_les_overlay(slug: str) -> Path:
    return QC_VALIDATION_DIR / f"{slug}_les_mid-z.png"


def validation_qc_otsu_overlay(slug: str) -> Path:
    return QC_VALIDATION_DIR / f"{slug}_otsu_mid-z.png"

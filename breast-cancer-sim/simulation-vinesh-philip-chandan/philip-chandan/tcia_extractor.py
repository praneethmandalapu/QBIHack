"""Extract 3D tumor volume from TCIA DICOM series."""

from pathlib import Path

RAW_DICOM_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def extract_volume(dicom_dir: Path) -> "np.ndarray":
    """Load DICOM stack and return a 3D numpy volume."""
    raise NotImplementedError

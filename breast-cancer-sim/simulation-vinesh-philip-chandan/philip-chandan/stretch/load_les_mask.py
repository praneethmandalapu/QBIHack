"""Parse TCIA TCGA-Breast-Radiogenomics radiologist lesion masks (*.les)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from paths import LESIONS_EXTRACT_DIR

# TCGA barcode + optional -Sn lesion index (TCIA Radiogenomics naming).
_LES_NAME_RE = re.compile(
    r"^(?P<patient>TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})(?:-S(?P<dce>\d+))?(?:-(?P<lesion>\d+))?\.les$",
    re.IGNORECASE,
)


def parse_les_filename(name: str) -> tuple[str, int, int]:
    """Return patient ID, 1-based DCE sequence index, and lesion index."""
    match = _LES_NAME_RE.match(name)
    if not match:
        raise ValueError(f"Unrecognized .les filename: {name}")
    patient = match.group("patient")
    dce_index = int(match.group("dce") or 1)
    lesion_index = int(match.group("lesion") or 1)
    return patient, dce_index, lesion_index


def read_les_cuboid(path: Path | str) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a .les file and return cuboid mask with shape (Y, X, Z) plus metadata."""
    path = Path(path)
    payload = path.read_bytes()
    if len(payload) < 12:
        raise ValueError(f".les file too small: {path}")

    header = np.frombuffer(payload[:12], dtype="<u2")
    y_start, x_start, z_start, y_end, x_end, z_end = (int(v) for v in header)
    y_size = y_end - y_start + 1
    x_size = x_end - x_start + 1
    z_size = z_end - z_start + 1
    expected_voxels = y_size * x_size * z_size
    voxels = np.frombuffer(payload[12:], dtype=np.int8)
    if voxels.size != expected_voxels:
        raise ValueError(
            f"{path.name}: expected {expected_voxels} voxels, got {voxels.size} "
            f"for bounds y[{y_start},{y_end}] x[{x_start},{x_end}] z[{z_start},{z_end}]"
        )

    cuboid_yxz = voxels.reshape(y_size, x_size, z_size)
    metadata = {
        "path": str(path),
        "y_start": y_start,
        "y_end": y_end,
        "x_start": x_start,
        "x_end": x_end,
        "z_start": z_start,
        "z_end": z_end,
        "cuboid_shape_yxz": [y_size, x_size, z_size],
        "lesion_voxels": int((cuboid_yxz > 0).sum()),
    }
    return (cuboid_yxz > 0).astype(np.uint8), metadata


def embed_les_mask(
    cuboid_yxz: np.ndarray,
    metadata: dict[str, Any],
    volume_shape: tuple[int, ...],
) -> np.ndarray:
    """Embed a (Y, X, Z) cuboid mask into a dense (Z, Y, X) volume."""
    if len(volume_shape) != 3:
        raise ValueError(f"volume_shape must be (Z, Y, X), got {volume_shape}")

    z_size, y_size, x_size = volume_shape
    y0, y1 = metadata["y_start"], metadata["y_end"]
    x0, x1 = metadata["x_start"], metadata["x_end"]
    z0, z1 = metadata["z_start"], metadata["z_end"]

    if y0 < 0 or x0 < 0 or z0 < 0 or y1 >= y_size or x1 >= x_size or z1 >= z_size:
        raise ValueError(
            f"Lesion bounds y[{y0},{y1}] x[{x0},{x1}] z[{z0},{z1}] "
            f"outside volume shape (Z,Y,X)={volume_shape}"
        )

    cuboid_zyx = np.transpose(cuboid_yxz, (2, 0, 1))
    mask = np.zeros(volume_shape, dtype=np.uint8)
    mask[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] = cuboid_zyx
    return mask


def cuboid_boundary_mask(
    metadata: dict[str, Any],
    volume_shape: tuple[int, ...],
) -> np.ndarray:
    """Return a (Z, Y, X) mask that is 1 on the .les annotation cuboid shell only."""
    if len(volume_shape) != 3:
        raise ValueError(f"volume_shape must be (Z, Y, X), got {volume_shape}")

    z_size, y_size, x_size = volume_shape
    y0, y1 = metadata["y_start"], metadata["y_end"]
    x0, x1 = metadata["x_start"], metadata["x_end"]
    z0, z1 = metadata["z_start"], metadata["z_end"]

    if y0 < 0 or x0 < 0 or z0 < 0 or y1 >= y_size or x1 >= x_size or z1 >= z_size:
        raise ValueError(
            f"Lesion bounds y[{y0},{y1}] x[{x0},{x1}] z[{z0},{z1}] "
            f"outside volume shape (Z,Y,X)={volume_shape}"
        )

    mask = np.zeros(volume_shape, dtype=np.uint8)
    region = mask[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]
    if region.size == 0:
        return mask

    shell = np.zeros_like(region)
    shell[0, :, :] = 1
    shell[-1, :, :] = 1
    shell[:, 0, :] = 1
    shell[:, -1, :] = 1
    shell[:, :, 0] = 1
    shell[:, :, -1] = 1
    region[:] = shell
    return mask


def cuboid_filled_mask(
    metadata: dict[str, Any],
    volume_shape: tuple[int, ...],
) -> np.ndarray:
    """Return a (Z, Y, X) mask that is 1 throughout the .les annotation cuboid."""
    if len(volume_shape) != 3:
        raise ValueError(f"volume_shape must be (Z, Y, X), got {volume_shape}")

    z_size, y_size, x_size = volume_shape
    y0, y1 = metadata["y_start"], metadata["y_end"]
    x0, x1 = metadata["x_start"], metadata["x_end"]
    z0, z1 = metadata["z_start"], metadata["z_end"]

    if y0 < 0 or x0 < 0 or z0 < 0 or y1 >= y_size or x1 >= x_size or z1 >= z_size:
        raise ValueError(
            f"Lesion bounds y[{y0},{y1}] x[{x0},{x1}] z[{z0},{z1}] "
            f"outside volume shape (Z,Y,X)={volume_shape}"
        )

    mask = np.zeros(volume_shape, dtype=np.uint8)
    mask[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1] = 1
    return mask


def load_les_cuboid_boundary(
    path: Path | str,
    volume_shape: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Parse .les header bounds and return a dense cuboid-shell mask (Z, Y, X)."""
    _, metadata = read_les_cuboid(path)
    patient, dce_index, lesion_index = parse_les_filename(Path(path).name)
    mask = cuboid_boundary_mask(metadata, volume_shape)
    metadata.update(
        {
            "patient_id": patient,
            "dce_index": dce_index,
            "lesion_index": lesion_index,
            "mask_shape_zyx": list(volume_shape),
            "boundary_voxels": int(mask.sum()),
        }
    )
    return mask, metadata


def load_les_mask(
    path: Path | str,
    volume_shape: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Parse .les and return a dense binary mask with shape (Z, Y, X)."""
    cuboid, metadata = read_les_cuboid(path)
    patient, dce_index, lesion_index = parse_les_filename(Path(path).name)
    mask = embed_les_mask(cuboid, metadata, volume_shape)
    metadata.update(
        {
            "patient_id": patient,
            "dce_index": dce_index,
            "lesion_index": lesion_index,
            "mask_shape_zyx": list(volume_shape),
            "mask_voxels": int(mask.sum()),
        }
    )
    return mask, metadata


def find_les_files(tcga_id: str, lesions_dir: Path | None = None) -> list[Path]:
    """Return all .les files for a TCGA patient ID."""
    root = lesions_dir or LESIONS_EXTRACT_DIR
    if not root.exists():
        return []
    return sorted(root.glob(f"{tcga_id}*.les"))


def write_synthetic_les(
    path: Path,
    *,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
    z_start: int,
    z_end: int,
    cuboid_yxz: np.ndarray,
) -> None:
    """Write a minimal .les file (for unit tests)."""
    header = np.array(
        [y_start, x_start, z_start, y_end, x_end, z_end],
        dtype="<u2",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header.tobytes() + cuboid_yxz.astype(np.int8).ravel().tobytes())

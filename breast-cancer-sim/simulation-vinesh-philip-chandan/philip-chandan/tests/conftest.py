"""Shared pytest fixtures for tcia_extractor tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def _write_synthetic_slice(
    output_dir: Path,
    *,
    instance_number: int,
    z_position: float,
    pixel_value: int,
    rows: int = 32,
    columns: int = 32,
    series_uid: str,
) -> Path:
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = FileDataset(
        str(output_dir / f"slice_{instance_number:03d}.dcm"),
        {},
        file_meta=file_meta,
        preamble=b"\0" * 128,
    )
    dataset.is_little_endian = True
    dataset.is_implicit_VR = False

    dataset.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    dataset.SOPInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = series_uid
    dataset.StudyInstanceUID = generate_uid()
    dataset.Modality = "MR"
    dataset.InstanceNumber = instance_number
    dataset.Rows = rows
    dataset.Columns = columns
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.SliceThickness = 2.0
    dataset.SpacingBetweenSlices = 2.0
    dataset.PixelSpacing = [1.0, 1.0]
    dataset.ImagePositionPatient = [0.0, 0.0, z_position]
    dataset.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    dataset.StudyDate = "20260101"
    dataset.ContentDate = "20260101"
    dataset.StudyTime = "120000"
    dataset.ContentTime = "120000"
    dataset.PatientName = "SYNTHETIC^PATIENT"
    dataset.PatientID = "SYNTH-001"
    dataset.SeriesDescription = "Synthetic MR series"
    dataset.StudyDescription = "Synthetic study"
    dataset.Manufacturer = "pytest"
    dataset.InstitutionName = "QBIHack"
    dataset.SeriesNumber = 1
    dataset.AcquisitionNumber = 1
    dataset.InstanceCreationDate = datetime.now().strftime("%Y%m%d")
    dataset.InstanceCreationTime = datetime.now().strftime("%H%M%S")

    pixel_array = np.full((rows, columns), pixel_value, dtype=np.uint16)
    dataset.PixelData = pixel_array.tobytes()

    output_path = Path(dataset.filename)
    dataset.save_as(output_path)
    return output_path


@pytest.fixture
def synthetic_dicom_dir(tmp_path: Path) -> Path:
    """Create a deterministic 5-slice synthetic MR DICOM series."""
    series_uid = generate_uid()
    for index in range(1, 6):
        _write_synthetic_slice(
            tmp_path,
            instance_number=index,
            z_position=float((index - 1) * 2),
            pixel_value=index * 10,
            series_uid=series_uid,
        )
    return tmp_path

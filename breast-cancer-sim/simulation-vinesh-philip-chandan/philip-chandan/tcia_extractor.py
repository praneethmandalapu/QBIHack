"""Philip + Chandan TCIA/DICOM extraction lane.

This module gives the team three Day 1 handoffs:
1. Print the locked TCGA primary pair and backup roster.
2. Query TCIA/NBIA for image series by patient ID.
3. Convert downloaded DICOM slices into clean 3D NumPy volumes.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DICOM_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_COHORT_PATH = PROJECT_ROOT / "cohort.json"
DEFAULT_COLLECTION = "TCGA-Breast-Radiogenomics"
DEFAULT_FALLBACK_COLLECTION = "TCGA-BRCA"
TCIA_BASE_URL = os.environ.get(
    "TCIA_BASE_URL",
    "https://services.cancerimagingarchive.net/nbia-api/services/v1",
)


@dataclass(frozen=True)
class SeriesRecord:
    collection: str
    patient_id: str
    study_uid: str
    series_uid: str
    modality: str
    image_count: int | None = None
    body_part: str | None = None
    series_description: str | None = None

    @classmethod
    def from_api(cls, row: dict[str, Any], collection: str) -> "SeriesRecord":
        image_count = row.get("ImageCount")
        try:
            parsed_count = int(image_count) if image_count is not None else None
        except (TypeError, ValueError):
            parsed_count = None

        return cls(
            collection=row.get("Collection", collection),
            patient_id=row.get("PatientID", ""),
            study_uid=row.get("StudyInstanceUID", ""),
            series_uid=row.get("SeriesInstanceUID", ""),
            modality=row.get("Modality", ""),
            image_count=parsed_count,
            body_part=row.get("BodyPartExamined"),
            series_description=row.get("SeriesDescription"),
        )


class TciaClient:
    def __init__(self, base_url: str = TCIA_BASE_URL, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get_json(self, endpoint: str, params: dict[str, str]) -> list[dict[str, Any]]:
        import requests

        response = requests.get(
            f"{self.base_url}/{endpoint.lstrip('/')}",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list from TCIA, got {type(payload).__name__}")
        return payload

    def get_series(
        self,
        collection: str,
        patient_id: str | None = None,
        modality: str | None = None,
    ) -> list[SeriesRecord]:
        params = {"Collection": collection}
        if patient_id:
            params["PatientID"] = patient_id
        if modality:
            params["Modality"] = modality

        rows = self._get_json("getSeries", params=params)
        return [SeriesRecord.from_api(row, collection=collection) for row in rows]

    def download_series_zip(self, series_uid: str, output_dir: Path) -> Path:
        import requests

        output_dir.mkdir(parents=True, exist_ok=True)
        zip_path = output_dir / f"{series_uid.replace('.', '_')}.zip"
        with requests.get(
            f"{self.base_url}/getImage",
            params={"SeriesInstanceUID": series_uid},
            stream=True,
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            with zip_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        return zip_path


def load_cohort(path: Path = DEFAULT_COHORT_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cohort_patients(cohort: dict[str, Any], include_backups: bool = False) -> list[dict[str, str]]:
    patients = [
        {
            "patient_id": entry["patient_id"],
            "pam50_subtype": entry["pam50_subtype"],
            "cohort_group": "primary_pair",
            "demo_role": entry.get("demo_role", ""),
        }
        for entry in cohort.get("primary_pair", [])
    ]

    if include_backups:
        for subtype, patient_ids in cohort.get("backup_roster", {}).items():
            for patient_id in patient_ids:
                patients.append(
                    {
                        "patient_id": patient_id,
                        "pam50_subtype": subtype,
                        "cohort_group": "backup_roster",
                        "demo_role": "backup case",
                    }
                )
    return patients


def pick_best_series(records: list[SeriesRecord]) -> SeriesRecord:
    if not records:
        raise ValueError("No matching TCIA series found")

    modality_priority = {"MR": 0, "CT": 1, "PT": 2, "MG": 3, "CR": 4}

    def sort_key(record: SeriesRecord) -> tuple[int, int]:
        priority = modality_priority.get(record.modality.upper(), 99)
        return (priority, -(record.image_count or 0))

    return sorted(records, key=sort_key)[0]


def series_for_patient(
    client: TciaClient,
    patient_id: str,
    collection: str,
    modality: str | None = "MR",
    fallback_collection: str | None = DEFAULT_FALLBACK_COLLECTION,
) -> tuple[str, list[SeriesRecord]]:
    records = client.get_series(collection=collection, patient_id=patient_id, modality=modality)
    if records or not fallback_collection:
        return collection, records

    fallback_records = client.get_series(
        collection=fallback_collection,
        patient_id=patient_id,
        modality=modality,
    )
    return fallback_collection, fallback_records


def extract_zip_safely(zip_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_output = output_dir.resolve()

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            target = (output_dir / member.filename).resolve()
            if resolved_output not in target.parents and target != resolved_output:
                raise ValueError(f"Unsafe path in zip archive: {member.filename}")
            archive.extract(member, output_dir)
    return output_dir


def iter_dicom_files(dicom_dir: Path) -> Iterable[Path]:
    for path in dicom_dir.rglob("*"):
        if path.is_file():
            yield path


def _slice_sort_key(dataset: Any, fallback: str) -> tuple[int, float | str]:
    image_position = getattr(dataset, "ImagePositionPatient", None)
    if image_position and len(image_position) >= 3:
        return (0, float(image_position[2]))

    slice_location = getattr(dataset, "SliceLocation", None)
    if slice_location is not None:
        return (1, float(slice_location))

    instance_number = getattr(dataset, "InstanceNumber", None)
    if instance_number is not None:
        return (2, float(instance_number))

    return (3, fallback)


def _pixel_spacing(dataset: Any) -> dict[str, Any]:
    spacing = getattr(dataset, "PixelSpacing", None)
    thickness = getattr(dataset, "SliceThickness", None)
    pixel_spacing = [float(value) for value in spacing] if spacing else None
    slice_thickness = float(thickness) if thickness is not None else None
    return {
        "pixel_spacing": pixel_spacing,
        "slice_thickness": slice_thickness,
        "voxel_spacing_mm": [slice_thickness, *pixel_spacing] if pixel_spacing and slice_thickness else None,
        "voxel_axis_order": "z,y,x",
    }


def extract_volume(dicom_dir: Path) -> np.ndarray:
    """Load one local DICOM series and return a `(slices, height, width)` volume."""

    volume, _metadata = load_dicom_series(dicom_dir)
    return volume


def load_dicom_series(dicom_dir: Path) -> tuple[np.ndarray, dict[str, Any]]:
    import pydicom

    slices: list[tuple[tuple[int, float | str], np.ndarray, Any]] = []
    for path in iter_dicom_files(dicom_dir):
        try:
            dataset = pydicom.dcmread(path, force=True)
            pixel_array = dataset.pixel_array.astype(np.float32)
        except Exception:
            continue

        if pixel_array.ndim != 2:
            continue

        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        calibrated = pixel_array * slope + intercept
        slices.append((_slice_sort_key(dataset, path.name), calibrated, dataset))

    if not slices:
        raise ValueError(f"No readable 2D DICOM image slices found in {dicom_dir}")

    slices.sort(key=lambda item: item[0])
    volume = np.stack([item[1] for item in slices]).astype(np.float32)
    first_dataset = slices[0][2]
    metadata = {
        "source_dir": str(dicom_dir),
        "shape": list(volume.shape),
        "dtype": str(volume.dtype),
        "patient_id": getattr(first_dataset, "PatientID", None),
        "modality": getattr(first_dataset, "Modality", None),
        "study_instance_uid": getattr(first_dataset, "StudyInstanceUID", None),
        "series_instance_uid": getattr(first_dataset, "SeriesInstanceUID", None),
        "series_description": getattr(first_dataset, "SeriesDescription", None),
        "slice_count": len(slices),
        **_pixel_spacing(first_dataset),
    }
    return volume, metadata


def save_volume(volume: np.ndarray, output: Path, metadata: dict[str, Any] | None = None) -> Path:
    if output.suffix != ".npy":
        output = output.with_suffix(".npy")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, volume.astype(np.float32))

    if metadata is not None:
        output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def create_dummy_tumor_volume(
    shape: tuple[int, int, int] = (64, 64, 64),
    radius: float = 15.0,
    noise: float = 0.02,
    seed: int = 7,
) -> np.ndarray:
    z_axis, y_axis, x_axis = np.indices(shape)
    center = np.array([(size - 1) / 2 for size in shape], dtype=np.float32)
    distance = np.sqrt(
        (z_axis - center[0]) ** 2
        + (y_axis - center[1]) ** 2
        + (x_axis - center[2]) ** 2
    )
    tumor = np.clip(1.0 - (distance / radius), 0.0, 1.0)
    tumor[distance < radius * 0.35] *= 0.35

    rng = np.random.default_rng(seed)
    if noise > 0:
        tumor = tumor + rng.normal(0.0, noise, size=shape)
    return np.clip(tumor, 0.0, 1.0).astype(np.float32)


def command_cohort_summary(args: argparse.Namespace) -> None:
    cohort = load_cohort(Path(args.cohort))
    print(f"Cohort: {cohort['cohort_name']}")
    print(f"Imaging collection: {cohort['imaging_collection']}")
    print(f"Fallback collection: {cohort.get('fallback_imaging_collection', '')}")
    print(f"Genomics project: {cohort['genomics_project']}")
    print("\nPrimary pair:")
    for patient in cohort_patients(cohort):
        print(f"- {patient['patient_id']}\t{patient['pam50_subtype']}\t{patient['demo_role']}")

    print("\nBackup roster:")
    for patient in cohort_patients(cohort, include_backups=True):
        if patient["cohort_group"] == "backup_roster":
            print(f"- {patient['patient_id']}\t{patient['pam50_subtype']}")


def command_list_cohort(args: argparse.Namespace) -> None:
    cohort = load_cohort(Path(args.cohort))
    client = TciaClient(base_url=args.base_url)
    collection = args.collection or cohort.get("imaging_collection", DEFAULT_COLLECTION)
    fallback = args.fallback_collection or cohort.get("fallback_imaging_collection")

    for patient in cohort_patients(cohort, include_backups=args.include_backups):
        used_collection, records = series_for_patient(
            client,
            patient_id=patient["patient_id"],
            collection=collection,
            modality=args.modality,
            fallback_collection=fallback,
        )
        print(
            f"\n{patient['patient_id']} | {patient['pam50_subtype']} | "
            f"{patient['cohort_group']} | collection={used_collection} | series={len(records)}"
        )
        for record in records[: args.limit_per_patient]:
            print(
                "\t".join(
                    [
                        record.modality,
                        str(record.image_count or ""),
                        record.series_uid,
                        record.series_description or "",
                    ]
                )
            )


def command_download_cohort(args: argparse.Namespace) -> None:
    cohort = load_cohort(Path(args.cohort))
    client = TciaClient(base_url=args.base_url)
    collection = args.collection or cohort.get("imaging_collection", DEFAULT_COLLECTION)
    fallback = args.fallback_collection or cohort.get("fallback_imaging_collection")

    for patient in cohort_patients(cohort, include_backups=args.include_backups):
        used_collection, records = series_for_patient(
            client,
            patient_id=patient["patient_id"],
            collection=collection,
            modality=args.modality,
            fallback_collection=fallback,
        )
        if not records:
            print(f"Skipping {patient['patient_id']}: no matching series found")
            continue

        selected = pick_best_series(records)
        output_dir = Path(args.output_dir) / patient["patient_id"]
        raw_dir = RAW_DICOM_DIR / "tcia" / patient["patient_id"] / selected.series_uid.replace(".", "_")
        print(
            f"Downloading {patient['patient_id']} ({patient['pam50_subtype']}) "
            f"from {used_collection}: {selected.series_uid}"
        )
        zip_path = client.download_series_zip(selected.series_uid, raw_dir)
        dicom_dir = extract_zip_safely(zip_path, raw_dir / "dicom")
        volume, metadata = load_dicom_series(dicom_dir)
        output = save_volume(volume, output_dir / f"{patient['patient_id']}.npy", metadata=metadata)
        print(f"Saved volume: {output}")
        print(f"Shape: {volume.shape}")


def command_convert_local(args: argparse.Namespace) -> None:
    volume, metadata = load_dicom_series(Path(args.dicom_dir))
    output = save_volume(volume, Path(args.output), metadata=metadata)
    print(f"Saved volume: {output}")
    print(f"Shape: {volume.shape}")


def command_dummy(args: argparse.Namespace) -> None:
    shape = tuple(int(value) for value in args.shape.split(","))
    if len(shape) != 3:
        raise ValueError("--shape must have exactly three comma-separated integers")

    volume = create_dummy_tumor_volume(
        shape=shape,
        radius=args.radius,
        noise=args.noise,
        seed=args.seed,
    )
    output = save_volume(
        volume,
        Path(args.output),
        metadata={
            "source": "dummy",
            "shape": list(volume.shape),
            "dtype": str(volume.dtype),
            "density_range": [0.0, 1.0],
            "voxel_spacing_mm": [1.0, 1.0, 1.0],
            "voxel_axis_order": "z,y,x",
            "radius": args.radius,
            "noise": args.noise,
            "seed": args.seed,
        },
    )
    print(f"Saved dummy volume: {output}")
    print(f"Shape: {volume.shape}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TCIA DICOM extraction helper")
    parser.add_argument("--base-url", default=TCIA_BASE_URL)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cohort_parser = subparsers.add_parser("cohort-summary", help="Print locked sprint cohort")
    cohort_parser.add_argument("--cohort", default=str(DEFAULT_COHORT_PATH))
    cohort_parser.set_defaults(func=command_cohort_summary)

    list_parser = subparsers.add_parser("list-cohort", help="List TCIA series for cohort patients")
    list_parser.add_argument("--cohort", default=str(DEFAULT_COHORT_PATH))
    list_parser.add_argument("--collection")
    list_parser.add_argument("--fallback-collection")
    list_parser.add_argument("--modality", default="MR")
    list_parser.add_argument("--limit-per-patient", type=int, default=10)
    list_parser.add_argument("--include-backups", action="store_true")
    list_parser.set_defaults(func=command_list_cohort)

    download_parser = subparsers.add_parser("download-cohort", help="Download best series for cohort patients")
    download_parser.add_argument("--cohort", default=str(DEFAULT_COHORT_PATH))
    download_parser.add_argument("--collection")
    download_parser.add_argument("--fallback-collection")
    download_parser.add_argument("--modality", default="MR")
    download_parser.add_argument("--output-dir", default=str(PROCESSED_DIR / "cohort_volumes"))
    download_parser.add_argument("--include-backups", action="store_true")
    download_parser.set_defaults(func=command_download_cohort)

    convert_parser = subparsers.add_parser("convert-local", help="Convert local DICOM folder to .npy")
    convert_parser.add_argument("--dicom-dir", required=True)
    convert_parser.add_argument("--output", default=str(PROCESSED_DIR / "tumor_volume.npy"))
    convert_parser.set_defaults(func=command_convert_local)

    dummy_parser = subparsers.add_parser("dummy", help="Write a dummy 3D tumor volume")
    dummy_parser.add_argument("--output", default=str(PROCESSED_DIR / "dummy_tumor_volume.npy"))
    dummy_parser.add_argument("--shape", default="64,64,64")
    dummy_parser.add_argument("--radius", type=float, default=15.0)
    dummy_parser.add_argument("--noise", type=float, default=0.02)
    dummy_parser.add_argument("--seed", type=int, default=7)
    dummy_parser.set_defaults(func=command_dummy)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""Candidate brain MRI datasets from repo-root DATASETS.md."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AccessMode = Literal["tcia_nbia", "tcia_nifti", "ucsf_portal", "local_only"]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    disease: str
    access: AccessMode
    longitudinal: bool
    segmentation: str
    format: str
    raw_dir: str
    tcia_collection: str | None = None
    portal_url: str | None = None
    download_notes: str = ""
    growth_model_score: int = 0


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "ucsf_longitudinal_glioma": DatasetSpec(
        key="ucsf_longitudinal_glioma",
        label="UCSF Longitudinal Glioma",
        disease="Glioma",
        access="ucsf_portal",
        longitudinal=True,
        segmentation="expert",
        format="nifti",
        raw_dir="data/raw/ucsf_alptdg",
        portal_url="https://imagingdatasets.ucsf.edu/dataset/2",
        download_notes=(
            "UCSF-ALPTDG: register on the UCSF imaging portal, accept the DUA, "
            "then download NIfTI + expert masks into data/raw/ucsf_alptdg/<patient_id>/."
        ),
        growth_model_score=5,
    ),
    "mu_glioma_post": DatasetSpec(
        key="mu_glioma_post",
        label="MU-Glioma-Post",
        disease="Glioma",
        access="tcia_nifti",
        longitudinal=True,
        segmentation="expert_validated",
        format="nifti",
        raw_dir="data/raw/mu_glioma_post",
        tcia_collection="MU-Glioma-Post",
        portal_url="https://www.cancerimagingarchive.net/collection/mu-glioma-post/",
        download_notes=(
            "Bulk NIfTI download via TCIA web portal or NBIA Data Retriever "
            "(collection MU-Glioma-Post). Expect patient/timepoint subfolders with "
            "t1n, t1c, t2w, t2f sequences and segmentation labels."
        ),
        growth_model_score=5,
    ),
    "yale_brain_mets": DatasetSpec(
        key="yale_brain_mets",
        label="Yale Brain Mets",
        disease="Metastases",
        access="local_only",
        longitudinal=True,
        segmentation="variable",
        format="nifti",
        raw_dir="data/raw/yale_brain_mets",
        download_notes="Confirm access path before cohort lock; store under data/raw/yale_brain_mets/.",
        growth_model_score=4,
    ),
    "lumiere": DatasetSpec(
        key="lumiere",
        label="LUMIERE",
        disease="GBM",
        access="local_only",
        longitudinal=True,
        segmentation="auto_plus_rano",
        format="nifti",
        raw_dir="data/raw/lumiere",
        portal_url="https://springernature.figshare.com/collections/The_LUMIERE_Dataset_Longitudinal_Glioblastoma_MRI_with_Expert_RANO_Evaluation/5904905",
        download_notes=(
            "Figshare (non-commercial): 91 GBM patients, longitudinal MRI, automated "
            "segmentations (DeepBraTumIA / HD-GLIO-AUTO), expert RANO ratings, MGMT/IDH1 "
            "subset. Download into data/raw/lumiere/<patient_id>/."
        ),
        growth_model_score=5,
    ),
    "ms_longitudinal": DatasetSpec(
        key="ms_longitudinal",
        label="MS Longitudinal",
        disease="MS",
        access="local_only",
        longitudinal=True,
        segmentation="expert",
        format="nifti",
        raw_dir="data/raw/ms_longitudinal",
        download_notes="Stretch dataset for lesion dynamics; out of glioma v1 scope.",
        growth_model_score=5,
    ),
    "msseg2": DatasetSpec(
        key="msseg2",
        label="MSSEG2",
        disease="MS",
        access="local_only",
        longitudinal=True,
        segmentation="expert",
        format="nifti",
        raw_dir="data/raw/msseg2",
        download_notes="Stretch dataset; lesion segmentation challenge data.",
        growth_model_score=5,
    ),
    "upenn_gbm": DatasetSpec(
        key="upenn_gbm",
        label="UPENN-GBM",
        disease="GBM",
        access="tcia_nbia",
        longitudinal=False,
        segmentation="none",
        format="dicom",
        raw_dir="data/raw/upenn_gbm",
        tcia_collection="UPENN-GBM",
        portal_url="https://www.cancerimagingarchive.net/collection/upenn-gbm/",
        download_notes=(
            "NBIA REST collection UPENN-GBM (DICOM). Useful for API smoke tests; "
            "not longitudinal and lacks expert segmentations for growth v1."
        ),
        growth_model_score=2,
    ),
}

PREFERRED_DATASET_KEYS = ("ucsf_longitudinal_glioma", "mu_glioma_post", "lumiere")


def get_dataset(key: str) -> DatasetSpec:
    if key not in DATASET_REGISTRY:
        known = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset {key!r}. Known keys: {known}")
    return DATASET_REGISTRY[key]


def iter_datasets(*, preferred_only: bool = False) -> list[DatasetSpec]:
    if preferred_only:
        return [DATASET_REGISTRY[key] for key in PREFERRED_DATASET_KEYS]
    return list(DATASET_REGISTRY.values())

"""Cohort definitions and dataset discovery helpers for brain-cancer-sim."""

from __future__ import annotations

from pathlib import Path

COHORT_DIR = Path(__file__).resolve().parent
COHORT_PATH = COHORT_DIR / "cohort.json"
COHORT_DISCOVERY_UCSF_PATH = COHORT_DIR / "cohort_discovery_ucsf.json"
REPO_ROOT = COHORT_DIR.parents[2]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
UCSF_MASTER_CSV = PROCESSED_DATA_DIR / "ucsf_longitudinal_master.csv"

"""Cohort definitions and dataset discovery helpers for brain-cancer-sim."""

from __future__ import annotations

from pathlib import Path

COHORT_DIR = Path(__file__).resolve().parent
COHORT_PATH = COHORT_DIR / "cohort.json"
REPO_ROOT = COHORT_DIR.parents[2]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"

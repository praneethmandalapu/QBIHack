"""NeuroPulse shared loader -- the brain (glioma) contract surface for the team.

Mirror of breast-cancer-sim/oncopulse.py, keyed by UCSF SubjectID:

    from neuropulse import get_patient, growth_multiplier

    p = get_patient(100118)
    p["risk"]    -> float in [0, 1]   (prognostic mortality risk; relative rank)
    p["idh"]     -> "WT" | "mut" | None
    p["grade"]   -> 2.0 | 3.0 | 4.0 | None
    growth_multiplier(100118)  -> float for Vinesh's PDE solver (rho * multiplier)

Scores are precomputed (data/processed/brain_patient_features.csv, out-of-fold so
they are honest), so get_patient / growth_multiplier are pure CSV lookups -- no
xgboost / libomp needed. Only predict() loads the model for a brand-new patient.
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_PROC = _HERE / "data" / "processed"
_MODEL = _HERE / "models-praneeth" / "saved" / "brain_model.pkl"
GROWTH_LO, GROWTH_HI = 0.8, 1.8   # same convention as breast oncopulse


@lru_cache(maxsize=1)
def _table() -> pd.DataFrame:
    return pd.read_csv(_PROC / "brain_patient_features.csv").set_index("subjectid")


@lru_cache(maxsize=1)
def _bundle():
    with open(_MODEL, "rb") as fh:
        return pickle.load(fh)   # {"model":..., "features":[...]}


def get_patient(subject_id) -> dict:
    """Look up a precomputed UCSF glioma patient. Raises KeyError if unknown."""
    sid = int(subject_id)
    t = _table()
    if sid not in t.index:
        raise KeyError(f"{sid} not in brain_patient_features.csv")
    r = t.loc[sid]
    return {
        "subject_id": sid,
        "risk": float(r["risk"]),
        "growth_multiplier": float(r["growth_multiplier"]),
        "idh": None if pd.isna(r["idh"]) else str(r["idh"]),
        "grade": None if pd.isna(r["grade"]) else float(r["grade"]),
        "mgmt": None if pd.isna(r["mgmt"]) else str(r["mgmt"]),
        "diagnosis": None if pd.isna(r["who_2021_diagnosis"]) else str(r["who_2021_diagnosis"]),
    }


def growth_multiplier(subject_id, lo: float = GROWTH_LO, hi: float = GROWTH_HI) -> float:
    """Map a patient's prognostic risk to Vinesh's solver risk_multiplier
    (lo + (hi-lo)*risk). Same shape/convention as breast oncopulse."""
    risk = get_patient(subject_id)["risk"]
    return float(lo + (hi - lo) * risk)


def predict(features: dict) -> float:
    """Score a brand-new patient (loads brain_model.pkl -> needs xgboost+libomp).
    `features` keys are a subset of the trained feature columns; missing -> NaN."""
    b = _bundle()
    row = pd.DataFrame([{c: features.get(c, np.nan) for c in b["features"]}])
    return float(b["model"].predict_proba(row)[0, 1])


def list_patients() -> list[int]:
    return _table().index.tolist()

"""OncoPulse shared loader -- the contract surface for the whole team.

Person 1 (Praneeth) owns this. Everyone else imports from it instead of touching
model internals:

    from oncopulse import get_patient, GENE_LIST

    p = get_patient("TCGA-BH-A0BR")
    p["risk"]    -> float in [0, 1]   (disease-specific-survival risk)
    p["expr"]    -> np.ndarray(30)    (z-scored driver genes, GENE_LIST order)
    p["pam50"]   -> str               (LumA / LumB / Basal / Her2 / Normal)

The TCGA barcode is the team join key: Philip renders it, you score it, Vinesh
simulates it. Scores are precomputed (data/processed/tcga_patient_features.csv),
so this is a lookup -- no live z-scoring.
"""

from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_PROC = _HERE / "data" / "processed"
_MODEL = _HERE / "models-praneeth" / "saved" / "model.pkl"


@lru_cache(maxsize=1)
def _gene_list() -> list[str]:
    return json.loads((_PROC / "gene_list.json").read_text())["genes"]


@lru_cache(maxsize=1)
def _table() -> pd.DataFrame:
    return pd.read_csv(_PROC / "tcga_patient_features.csv", index_col="barcode")


@lru_cache(maxsize=1)
def _model():
    with open(_MODEL, "rb") as fh:
        return pickle.load(fh)


def __getattr__(name: str):
    # Lazy module attribute so `from oncopulse import GENE_LIST` works even
    # before the model is trained (resolved on first access, not import).
    if name == "GENE_LIST":
        return _gene_list()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_patient(barcode: str) -> dict:
    """Look up a precomputed TCGA patient. Raises KeyError if not scored."""
    bc = barcode[:12]
    table = _table()
    if bc not in table.index:
        raise KeyError(f"{bc} not in tcga_patient_features.csv -- pick a backup barcode")
    row = table.loc[bc]
    genes = _gene_list()
    return {
        "barcode": bc,
        "risk": float(row["risk"]),
        "pam50": str(row["pam50"]),
        "expr": row[genes].to_numpy(dtype="float64"),
        "genes": genes,
    }


def growth_multiplier(barcode: str, lo: float = 0.8, hi: float = 1.8) -> float:
    """Map a patient's genomic risk to Vinesh's solver `risk_multiplier`.

    tumor_pde_solver scales the growth rate rho by `risk_multiplier`
    (rho_eff = rho * risk_multiplier). This converts risk in [0, 1] linearly to
    [lo, hi]: lo + (hi - lo) * risk. Defaults (0.8, 1.8) sit in the 0.7/1.6
    ballpark used in vinesh/test_solver.py. The lo/hi anchors are the knobs to
    reconcile against vinesh/calibrate.py (which fits the multiplier from the two
    imaging timepoints) -- genomics sets the per-patient ratio, calibration sets
    the absolute scale.
    """
    risk = get_patient(barcode)["risk"]
    return float(lo + (hi - lo) * risk)


def score_expression(expr) -> float:
    """Escape hatch: score a raw z-scored vector/dict if a barcode is missing.
    `expr` = {gene: zscore} or sequence in GENE_LIST order."""
    genes = _gene_list()
    if isinstance(expr, dict):
        vec = [[expr.get(g, 0.0) for g in genes]]
    else:
        vec = [list(expr)]
    return float(_model().predict_proba(np.asarray(vec))[0, 1])


def gene_correlation() -> pd.DataFrame:
    """30x30 gene-gene correlation matrix for Vinesh's ODE coupling."""
    return pd.read_csv(_PROC / "gene_correlation_matrix.csv", index_col=0)


def list_barcodes() -> list[str]:
    return _table().index.tolist()

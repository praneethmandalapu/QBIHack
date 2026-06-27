"""SHAP explanations for the trained OncoPulse risk model.

Two entry points:

  build_global()  - run once after training. Computes SHAP over the full pooled
                    cohort and writes the artifacts Vihari's EXPLAIN tab reads:
                        saved/shap_importance.csv   gene, mean_abs_shap (ranked)
                        saved/shap_values.pkl       {genes, base_value, values, X}

  explain(model_path, features) - per-patient explanation for the app. `features`
                    is a dict {gene: zscore} or a 1-row DataFrame in gene order.
                    Returns {risk, base_value, contributions:{gene: shap}}.

Run:
    python generate_shap.py            # builds global artifacts
"""

from __future__ import annotations

import _macos_omp_fix  # noqa: F401  (libomp; before xgboost/shap)

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SAVED = Path(__file__).resolve().parent / "saved"


def _load_model_and_genes():
    with open(SAVED / "model.pkl", "rb") as fh:
        model = pickle.load(fh)
    genes = json.loads((PROC / "gene_list.json").read_text())["genes"]
    return model, genes


def _load_pooled(genes: list[str]) -> pd.DataFrame:
    csv = PROC / "train_matrix.csv"
    pq = PROC / "train_matrix.parquet"
    df = pd.read_parquet(pq) if pq.exists() else pd.read_csv(csv, index_col="sample_id")
    return df[genes]


def build_global() -> None:
    model, genes = _load_model_and_genes()
    X = _load_pooled(genes)
    print(f"Computing SHAP over {X.shape[0]} samples x {len(genes)} genes ...", flush=True)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    base = explainer.expected_value
    base = float(np.ravel(base)[0])

    mean_abs = np.abs(sv).mean(axis=0)
    importance = (pd.DataFrame({"gene": genes, "mean_abs_shap": mean_abs})
                  .sort_values("mean_abs_shap", ascending=False)
                  .reset_index(drop=True))
    importance.to_csv(SAVED / "shap_importance.csv", index=False)

    with open(SAVED / "shap_values.pkl", "wb") as fh:
        pickle.dump({"genes": genes, "base_value": base,
                     "values": sv, "X": X.to_numpy(), "index": X.index.tolist()}, fh)

    print("Top 10 by mean|SHAP|:", flush=True)
    print(importance.head(10).to_string(index=False), flush=True)
    print(f"Saved shap_importance.csv + shap_values.pkl (base={base:.3f})", flush=True)


def explain(model_path: Path, features) -> dict:
    """Per-patient SHAP. `features` = {gene: zscore} or 1-row DataFrame."""
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)
    genes = json.loads((PROC / "gene_list.json").read_text())["genes"]

    if isinstance(features, dict):
        row = pd.DataFrame([[features.get(g, 0.0) for g in genes]], columns=genes)
    else:
        row = features[genes]

    explainer = shap.TreeExplainer(model)
    sv = np.ravel(explainer.shap_values(row))
    base = float(np.ravel(explainer.expected_value)[0])
    risk = float(model.predict_proba(row)[0, 1])
    return {
        "risk": risk,
        "base_value": base,
        "contributions": dict(sorted(
            {g: float(v) for g, v in zip(genes, sv)}.items(),
            key=lambda kv: abs(kv[1]), reverse=True)),
    }


if __name__ == "__main__":
    build_global()

"""Train the OncoPulse XGBoost disease-specific-survival risk model.

Two stages, both with live progress to stdout (and a log file):

  Stage A - Feature selection
    Fit an XGBoost on the full concordant candidate pool, rank features by gain
    importance, keep the top N_FINAL_GENES driver genes -> gene_list.json.

  Stage B - Hyperparameter search (this is the long part; ~1-2h is fine)
    Randomized search over a wide XGBoost grid, each config scored by repeated
    stratified k-fold CV with early stopping on a held-out fold. The best config
    is refit on all data -> model.pkl, plus a held-out cross-cohort
    generalization check (train METABRIC -> test TCGA and vice versa).

Run:
    python train_xgboost.py                 # full search (default N_TRIALS)
    N_TRIALS=40 python train_xgboost.py     # quicker
"""

from __future__ import annotations

import _macos_omp_fix  # noqa: F401  (sets up libomp; must precede xgboost)

import json
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SAVED = Path(__file__).resolve().parent / "saved"
LOG = SAVED / "training_log.txt"

N_FINAL_GENES = 30
N_TRIALS = int(os.environ.get("N_TRIALS", "120"))
N_SPLITS = int(os.environ.get("N_SPLITS", "5"))
N_REPEATS = int(os.environ.get("N_REPEATS", "3"))
EARLY_STOP = 50
MAX_BOOST_ROUNDS = 2000
SEED = 17


# ----------------------------------------------------------------------------
# logging
# ----------------------------------------------------------------------------
def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_matrix() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    pq, csv = PROC / "train_matrix.parquet", PROC / "train_matrix.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
    else:
        df = pd.read_csv(csv, index_col="sample_id")
    genes = [c for c in df.columns if c not in ("label", "cohort")]
    X = df[genes]
    y = df["label"].to_numpy().astype(int)
    cohort = df["cohort"].to_numpy()
    return X, y, cohort, genes


# ----------------------------------------------------------------------------
# CV scoring
# ----------------------------------------------------------------------------
def sample_params(rng: np.random.Generator) -> dict:
    return {
        "max_depth": int(rng.integers(2, 7)),
        "learning_rate": float(10 ** rng.uniform(-2.3, -0.7)),   # ~0.005 - 0.2
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "min_child_weight": float(rng.uniform(1, 10)),
        "gamma": float(10 ** rng.uniform(-3, 0.3)),
        "reg_lambda": float(10 ** rng.uniform(-2, 1.3)),
        "reg_alpha": float(10 ** rng.uniform(-3, 1.0)),
    }


def cv_score(X: pd.DataFrame, y: np.ndarray, params: dict) -> tuple[float, float, float]:
    """Repeated stratified CV. Returns (mean_auc, std_auc, mean_best_rounds)."""
    pos_w = float((y == 0).sum()) / max((y == 1).sum(), 1)
    aucs, rounds = [], []
    for rep in range(N_REPEATS):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + rep)
        for tr, va in skf.split(X, y):
            dtr = xgb.DMatrix(X.iloc[tr], label=y[tr])
            dva = xgb.DMatrix(X.iloc[va], label=y[va])
            booster = xgb.train(
                {**params, "objective": "binary:logistic", "eval_metric": "auc",
                 "scale_pos_weight": pos_w, "tree_method": "hist", "seed": SEED},
                dtr, num_boost_round=MAX_BOOST_ROUNDS,
                evals=[(dva, "val")], early_stopping_rounds=EARLY_STOP,
                verbose_eval=False,
            )
            p = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
            aucs.append(roc_auc_score(y[va], p))
            rounds.append(booster.best_iteration + 1)
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(rounds))


# ----------------------------------------------------------------------------
# stages
# ----------------------------------------------------------------------------
def select_genes(X: pd.DataFrame, y: np.ndarray) -> list[str]:
    log(f"Stage A: selecting top {N_FINAL_GENES} drivers from {X.shape[1]} candidates")
    pos_w = float((y == 0).sum()) / max((y == 1).sum(), 1)
    dall = xgb.DMatrix(X, label=y)
    booster = xgb.train(
        {"objective": "binary:logistic", "eval_metric": "auc", "max_depth": 4,
         "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.6,
         "scale_pos_weight": pos_w, "tree_method": "hist", "seed": SEED},
        dall, num_boost_round=400, verbose_eval=False,
    )
    gain = booster.get_score(importance_type="gain")
    ranked = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)
    top = [g for g, _ in ranked[:N_FINAL_GENES]]
    log(f"Stage A: top drivers -> {', '.join(top)}")
    return top


def cross_cohort_check(X: pd.DataFrame, y: np.ndarray, cohort: np.ndarray,
                       params: dict, rounds: int) -> dict:
    """Honest generalization test: train on one cohort, test on the other."""
    out = {}
    for tr_name, te_name in [("METABRIC", "TCGA"), ("TCGA", "METABRIC")]:
        tr, te = cohort == tr_name, cohort == te_name
        pos_w = float((y[tr] == 0).sum()) / max((y[tr] == 1).sum(), 1)
        booster = xgb.train(
            {**params, "objective": "binary:logistic", "eval_metric": "auc",
             "scale_pos_weight": pos_w, "tree_method": "hist", "seed": SEED},
            xgb.DMatrix(X[tr], label=y[tr]), num_boost_round=rounds, verbose_eval=False,
        )
        p = booster.predict(xgb.DMatrix(X[te]))
        out[f"train_{tr_name}_test_{te_name}"] = {
            "auc": round(roc_auc_score(y[te], p), 4),
            "ap": round(average_precision_score(y[te], p), 4),
        }
    return out


def search(X: pd.DataFrame, y: np.ndarray) -> tuple[dict, dict]:
    log(f"Stage B: randomized CV search, {N_TRIALS} trials x "
        f"{N_REPEATS}x{N_SPLITS}-fold (~{N_TRIALS * N_REPEATS * N_SPLITS} fits)")
    rng = np.random.default_rng(SEED)
    best = {"auc": -1.0}
    t0 = time.time()
    for i in range(1, N_TRIALS + 1):
        params = sample_params(rng)
        auc, std, rounds = cv_score(X, y, params)
        elapsed = time.time() - t0
        eta = elapsed / i * (N_TRIALS - i)
        flag = ""
        if auc > best["auc"]:
            best = {"auc": auc, "std": std, "rounds": int(round(rounds)), "params": params}
            flag = "  <-- new best"
        log(f"  trial {i:3d}/{N_TRIALS}  AUC={auc:.4f}+-{std:.3f}  "
            f"depth={params['max_depth']} lr={params['learning_rate']:.3f} "
            f"rounds={rounds:.0f}  [{elapsed/60:.1f}m elapsed, ETA {eta/60:.1f}m]{flag}")
    log(f"Stage B done. Best CV AUC={best['auc']:.4f}+-{best['std']:.3f}")
    return best["params"], best


def train() -> Path:
    SAVED.mkdir(parents=True, exist_ok=True)
    LOG.write_text("")  # reset
    log("=== OncoPulse XGBoost training ===")

    X, y, cohort, candidates = load_matrix()
    log(f"Loaded matrix: {X.shape[0]} samples, {len(candidates)} candidate genes, "
        f"{int(y.sum())} positive ({y.mean()*100:.1f}%)")
    for name in ("METABRIC", "TCGA"):
        m = cohort == name
        log(f"  {name}: {m.sum()} samples, {int(y[m].sum())} pos")

    genes = select_genes(X, y)
    Xf = X[genes]

    best_params, best = search(Xf, y)

    # cross-cohort generalization
    log("Cross-cohort generalization check ...")
    cc = cross_cohort_check(Xf, y, cohort, best_params, best["rounds"])
    for k, v in cc.items():
        log(f"  {k}: AUC={v['auc']} AP={v['ap']}")

    # refit final model on all data
    log("Refitting final model on all samples ...")
    pos_w = float((y == 0).sum()) / max((y == 1).sum(), 1)
    final = xgb.XGBClassifier(
        **best_params, n_estimators=best["rounds"], objective="binary:logistic",
        eval_metric="auc", scale_pos_weight=pos_w, tree_method="hist", random_state=SEED,
    )
    final.fit(Xf, y)

    model_path = SAVED / "model.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump(final, fh)

    with open(PROC / "gene_list.json", "w") as fh:
        json.dump({
            "genes": genes,
            "n_genes": len(genes),
            "input": "per-gene z-scores in this order; risk = predict_proba[:,1]",
            "label": "disease-specific survival (1=died of disease)",
        }, fh, indent=2)

    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "cv_auc": round(best["auc"], 4),
        "cv_auc_std": round(best["std"], 4),
        "best_rounds": best["rounds"],
        "best_params": {k: round(v, 5) if isinstance(v, float) else v
                        for k, v in best_params.items()},
        "cross_cohort": cc,
        "genes": genes,
        "n_samples": int(X.shape[0]),
        "n_trials": N_TRIALS,
    }
    with open(SAVED / "metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    log(f"Saved model -> {model_path}")
    log(f"Final CV AUC {best['auc']:.4f} | cross-cohort {cc}")
    log("=== done ===")
    return model_path


if __name__ == "__main__":
    train()

"""Train the brain (glioma) risk-score model on the UCSF-LPTDG cohort.

Predicts MORTALITY RISK (death recorded during follow-up) from BASELINE features
only (molecular markers, grade, baseline tumour volumes, treatment status). The
predicted probability is the per-patient prognostic risk score, mapped to a
growth_multiplier for the PDE — the brain analog of the breast oncopulse model.
Higher risk = more aggressive biology = faster simulated growth (IDH-WT GBM high,
IDH-mut low), which matches the observed t1->t2 growth direction.

Why mortality, not 2-month growth: whole-tumour growth is edema-dominated and
swings with steroids/RT between scans (CV AUC ~0.54, noise). Mortality is a real
prognostic endpoint that the molecular markers genuinely predict (CV AUC ~0.79).

Leakage guards: only baseline-available features are used. All t2 / change /
growth / future-event / outcome columns are excluded from X.

Outputs:
  saved/brain_model.pkl              fitted XGBClassifier
  saved/brain_metrics.json           CV AUC, params, feature list
  saved/brain_shap_importance.csv    mean|SHAP| per feature
  ../data/processed/brain_patient_features.csv   per-patient risk + growth_multiplier

Run:  python train_brain_risk.py
"""
from __future__ import annotations

import _macos_omp_fix  # noqa: F401  (libomp shim; before xgboost)

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]            # brain-cancer-sim/
PROC = ROOT / "data" / "processed"
SAVED = Path(__file__).resolve().parent / "saved"
SEED = 17

NUMERIC = [
    "patient_age", "grade", "mgmt_methylation_index", "number_of_surgeries",
    "days_from_1st_surgery_dx_to_1st_scan",
    "ncr_volume_label1_t1", "snfh_volume_label2_t1", "et_volume_label3_t1",
    "rc_volume_label4_t1", "wt_volume_label1_plus_2_plus_3_t1",
    "tc_volume_label_1_plus_2_t1",
]
CATEGORICAL = [
    "patient_sex", "who_2021_diagnosis", "idh", "mgmt", "1p19q", "atrx",
    "eor_at_1st_sx", "on_tx_at_1st_scan_c_chemo_rt_n_none",
]
GROWTH_LO, GROWTH_HI = 0.8, 1.8   # same convention as breast oncopulse


def load() -> pd.DataFrame:
    m = pd.read_csv(PROC / "ucsf_longitudinal_master.csv")
    il = pd.read_csv(PROC / "ucsf_imaging_long_clean.csv")
    base = il[il["timepoint"] == 1][["subjectid", "patient_age", "patient_sex"]]
    return m.merge(base, on="subjectid", how="left")


def build_X(df: pd.DataFrame) -> pd.DataFrame:
    num = df[NUMERIC].apply(pd.to_numeric, errors="coerce")
    cat = pd.get_dummies(df[CATEGORICAL].astype("object"), dummy_na=False, dtype=float)
    X = pd.concat([num, cat], axis=1)
    X.columns = [str(c) for c in X.columns]
    return X


def build_target(df: pd.DataFrame) -> np.ndarray:
    # mortality risk: a death date was recorded during follow-up
    return df["days_from_death_to_1st_scan"].notna().astype(int).to_numpy()


def main() -> int:
    SAVED.mkdir(parents=True, exist_ok=True)
    df = load()
    X = build_X(df)
    y = build_target(df)
    print(f"samples={len(y)}  features={X.shape[1]}  deceased={int(y.sum())} ({y.mean()*100:.0f}%)")

    params = dict(objective="binary:logistic", eval_metric="auc", max_depth=3,
                  learning_rate=0.05, subsample=0.9, colsample_bytree=0.8,
                  n_estimators=300, reg_lambda=2.0, tree_method="hist", random_state=SEED)
    clf = xgb.XGBClassifier(**params)

    # average out-of-fold AUC over a few 5-fold partitions for stability
    aucs, oofs = [], []
    for s in range(5):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + s)
        p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, p)); oofs.append(p)
    oof = np.mean(oofs, axis=0)
    auc = float(np.mean(aucs))
    print(f"out-of-fold CV AUC = {auc:.3f} +/- {np.std(aucs):.3f}  "
          f"(mortality, baseline features only)")

    clf.fit(X, y)
    with open(SAVED / "brain_model.pkl", "wb") as fh:
        pickle.dump({"model": clf, "features": list(X.columns)}, fh)

    # SHAP
    expl = shap.TreeExplainer(clf)
    sv = expl.shap_values(X)
    imp = (pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(0)})
           .sort_values("mean_abs_shap", ascending=False))
    imp.to_csv(SAVED / "brain_shap_importance.csv", index=False)
    print("top risk drivers:", ", ".join(imp["feature"].head(6)))

    # per-patient scored table (OOF risk = honest, not overfit)
    out = df[["subjectid", "idh", "grade", "mgmt", "who_2021_diagnosis",
              "wt_volume_label1_plus_2_plus_3_t1", "wt_grew"]].copy()
    out = out.rename(columns={"wt_volume_label1_plus_2_plus_3_t1": "wt_volume_t1",
                              "wt_grew": "actually_grew"})
    out["risk"] = np.round(oof, 4)
    out["growth_multiplier"] = np.round(GROWTH_LO + (GROWTH_HI - GROWTH_LO) * oof, 4)
    out = out.sort_values("risk", ascending=False)
    out.to_csv(PROC / "brain_patient_features.csv", index=False)

    json.dump({"cv_auc_oof": round(auc, 4), "n": int(len(y)),
               "n_deceased": int(y.sum()),
               "n_features": int(X.shape[1]),
               "target": "mortality (death recorded during follow-up)",
               "risk_meaning": "predicted mortality probability; relative prognostic rank",
               "growth_multiplier": f"{GROWTH_LO} + {GROWTH_HI-GROWTH_LO}*risk",
               "params": params},
              open(SAVED / "brain_metrics.json", "w"), indent=2)
    print(f"wrote brain_model.pkl, brain_patient_features.csv ({len(out)} patients)")

    # transparent molecular sanity check
    print("\nmean risk by IDH x grade (sanity):")
    print(out.assign(idh=out.idh.fillna("NA"))
          .groupby(["idh"])["risk"].mean().round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

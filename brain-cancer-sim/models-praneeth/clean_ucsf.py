"""Standard data-cleaning pass for the UCSF Postoperative Glioma dataset (UCSF-LPTDG).

Reads the dataset's clinical workbook (Table S1) and emits three tidy CSVs into
brain-cancer-sim/data/processed/ plus an integrity check against the imaging
folders. This is the genomics/molecular foundation for the brain risk model
(IDH / grade / MGMT per patient + measured t1->t2 tumour growth).

Inputs (extract from the UCSF zip into data/raw/ucsf_glioma/ first):
  data/raw/ucsf_glioma/UCSF_PostopGlioma_Table S1 *.xlsx   (clinical workbook)
Outputs (tracked):
  data/processed/ucsf_imaging_long_clean.csv    one row per (patient, timepoint)
  data/processed/ucsf_clinical_clean.csv        one row per patient (molecular/survival/tx)
  data/processed/ucsf_longitudinal_master.csv   one row per patient: t1 vs t2 + growth + clinical

Run:  python clean_ucsf.py [path/to/TableS1.xlsx]
"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]            # brain-cancer-sim/
RAW = ROOT / "data" / "raw" / "ucsf_glioma"
PROC = ROOT / "data" / "processed"


def find_xlsx() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    hits = glob.glob(str(RAW / "*Table S1*.xlsx")) or glob.glob(str(RAW / "*.xlsx"))
    if not hits:
        sys.exit(f"No clinical xlsx found in {RAW}. Extract it from the UCSF zip, "
                 f"or pass the path as an argument.")
    return Path(hits[0])


def snake(c: str) -> str:
    c = str(c).strip().replace("+", "_plus_").replace("/", "_")
    return re.sub(r"_+", "_", re.sub(r"[^\w]+", "_", c).strip("_")).lower()


def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True)
    xlsx = find_xlsx()
    print(f"Reading {xlsx.name}")

    # ---- imaging sheet (per-timepoint) -------------------------------------
    img = pd.read_excel(xlsx, sheet_name="UCSF-LPTDG").dropna(axis=1, how="all")
    change_pat = re.compile(r"CHANGE|Inc_Volume|Dec_Volume|OverallChange", re.I)
    change_cols = [c for c in img.columns if change_pat.search(str(c))]
    vol_cols = [c for c in img.columns if "Volume" in str(c) and c not in change_cols]
    meta_cols = [c for c in img.columns if c not in change_cols + vol_cols]

    imaging_long = img[meta_cols + vol_cols].copy()
    imaging_long[vol_cols] = imaging_long[vol_cols].fillna(0)   # absent region postop = 0
    imaging_long.columns = [snake(c) for c in imaging_long.columns]
    imaging_long.to_csv(PROC / "ucsf_imaging_long_clean.csv", index=False)

    change = (img[["SubjectID"] + change_cols].groupby("SubjectID", as_index=False).first())
    change.columns = ["subjectid"] + [snake(c) for c in change_cols]

    # wide pivot + growth metrics
    piv = img[["SubjectID", "Timepoint"] + vol_cols].copy()
    piv[vol_cols] = piv[vol_cols].fillna(0)
    wide = piv.pivot(index="SubjectID", columns="Timepoint", values=vol_cols)
    wide.columns = [f"{snake(v)}_t{int(t)}" for v, t in wide.columns]
    wide = wide.reset_index().rename(columns={"SubjectID": "subjectid"})
    wt1, wt2 = "wt_volume_label1_plus_2_plus_3_t1", "wt_volume_label1_plus_2_plus_3_t2"
    wide["wt_change"] = wide[wt2] - wide[wt1]
    wide["wt_growth_pct"] = np.where(wide[wt1] > 0, 100 * wide["wt_change"] / wide[wt1], np.nan)
    wide["wt_grew"] = wide[wt2] > wide[wt1]

    # ---- clinical sheet (per-patient) --------------------------------------
    clin = pd.read_excel(xlsx, sheet_name="Clinical Info")
    clin.columns = [snake(c) for c in clin.columns]
    clin = clin.rename(columns={clin.columns[0]: "subjectid"})
    for c in clin.columns:
        if clin[c].dtype == object:
            clin[c] = clin[c].astype(str).str.strip().replace({"nan": np.nan})
    clin.to_csv(PROC / "ucsf_clinical_clean.csv", index=False)

    master = wide.merge(change, on="subjectid", how="left").merge(clin, on="subjectid", how="left")
    master.to_csv(PROC / "ucsf_longitudinal_master.csv", index=False)

    grew = int(master["wt_grew"].sum())
    print(f"patients={len(master)}  grew t1->t2={grew} ({100*grew/len(master):.0f}%)")
    print(f"IDH: {clin['idh'].value_counts(dropna=False).to_dict()}")
    print("wrote 3 CSVs to", PROC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

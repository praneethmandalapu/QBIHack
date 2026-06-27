"""Build the barcode-keyed handoff table for Philip / Vinesh.

The merge rule: METABRIC is train-only; the TCGA barcode is the team's join key.
This scores EVERY TCGA-BRCA patient once, so a handoff is a pure lookup -- never
a live z-score of a 2-patient batch (which would be meaningless).

Each TCGA patient's 30 driver genes are z-scored against the SAVED TCGA cohort
reference (mean/sd from build_features), then run through model.pkl.

Outputs (data/processed/):
    tcga_patient_features.csv     barcode, risk, pam50, + 30 z-scored genes  <-- primary handoff
    patient_expression_top_genes.csv   barcode x 30 driver-gene z-scores
    gene_correlation_matrix.csv   30x30 gene-gene correlation (Vinesh's ODE coupling)

Run:
    python build_patient_table.py
"""

from __future__ import annotations

import _macos_omp_fix  # noqa: F401

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "brca_tcga_pan_can_atlas_2018"
PROC = ROOT / "data" / "processed"
SAVED = Path(__file__).resolve().parent / "saved"

# Philip's roster -- validate these resolve, surface their risk + subtype.
PRIMARY = ["TCGA-BH-A0BR", "TCGA-A2-A04P"]
BACKUP = ["TCGA-BH-A0BQ", "TCGA-A2-A04V", "TCGA-E2-A15C", "TCGA-AR-A1AQ",
          "TCGA-A2-A04Q", "TCGA-BH-A0BW", "TCGA-A2-A04R", "TCGA-E2-A15A"]


def _read_clinical(path: Path) -> pd.DataFrame:
    skip = sum(1 for line in open(path) if line.startswith("#"))
    return pd.read_csv(path, sep="\t", skiprows=skip)


def main() -> int:
    model = pickle.loads((SAVED / "model.pkl").read_bytes())
    genes = json.loads((PROC / "gene_list.json").read_text())["genes"]
    ref = pd.read_csv(PROC / "zscore_reference_tcga.csv", index_col=0)

    # raw TCGA expression -> log2 -> 30 genes
    print("Loading TCGA expression ...", flush=True)
    expr = pd.read_csv(RAW / "data_mrna_seq_v2_rsem.txt", sep="\t", low_memory=False)
    expr = expr.rename(columns={expr.columns[0]: "Hugo_Symbol"})
    expr = expr.drop(columns=[c for c in ("Entrez_Gene_Id",) if c in expr.columns])
    expr = expr[expr["Hugo_Symbol"].isin(genes)].set_index("Hugo_Symbol")
    expr = expr.apply(pd.to_numeric, errors="coerce").groupby(level=0).mean()
    expr = np.log2(expr.clip(lower=0) + 1.0)

    # collapse sample barcodes -> 12-char patient barcode
    expr.columns = expr.columns.str[:12]
    expr = expr.loc[:, ~expr.columns.duplicated()]

    # z-score each gene against the SAVED TCGA reference (the handoff contract)
    mu = ref["mean"].reindex(genes)
    sd = ref["sd"].reindex(genes).replace(0, np.nan)
    z = expr.reindex(genes).sub(mu, axis=0).div(sd, axis=0)  # genes x patients
    z = z.fillna(0.0).T[genes]  # patients x genes, GENE_LIST order

    # score
    risk = model.predict_proba(z.to_numpy())[:, 1]

    # subtype
    clin = _read_clinical(RAW / "data_clinical_patient.txt").set_index("PATIENT_ID")
    pam50 = clin["SUBTYPE"].reindex(z.index).fillna("NA")

    table = pd.concat([
        pd.DataFrame({"risk": np.round(risk, 5), "pam50": pam50.values}, index=z.index),
        z.round(4),
    ], axis=1)
    table.index.name = "barcode"
    table = table.sort_values("risk", ascending=False)
    table.to_csv(PROC / "tcga_patient_features.csv")
    z.round(4).rename_axis("barcode").to_csv(PROC / "patient_expression_top_genes.csv")

    # gene-gene coupling for Vinesh's ODE (from pooled training matrix)
    csv = PROC / "train_matrix.csv"
    pooled = pd.read_csv(csv, index_col="sample_id")[genes]
    corr = pooled.corr().round(4)
    corr.to_csv(PROC / "gene_correlation_matrix.csv")

    print(f"\nScored {len(table)} TCGA patients -> tcga_patient_features.csv", flush=True)
    print(f"risk: min={risk.min():.3f} median={np.median(risk):.3f} max={risk.max():.3f}",
          flush=True)

    print("\nPhilip's primary pair:", flush=True)
    for bc in PRIMARY:
        if bc in table.index:
            r = table.loc[bc]
            print(f"  OK  {bc}  risk={r['risk']:.3f}  pam50={r['pam50']}", flush=True)
        else:
            print(f"  MISSING  {bc}  -> use a backup", flush=True)
    print("Backup roster:", flush=True)
    for bc in BACKUP:
        if bc in table.index:
            r = table.loc[bc]
            print(f"  OK  {bc}  risk={r['risk']:.3f}  pam50={r['pam50']}", flush=True)
        else:
            print(f"  MISSING  {bc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

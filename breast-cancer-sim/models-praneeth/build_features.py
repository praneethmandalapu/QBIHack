"""Build the pooled cross-cohort training matrix for the OncoPulse risk model.

Pipeline
--------
1. Parse METABRIC + TCGA-BRCA clinical files into a binary disease-specific
   survival label (1 = died of disease, 0 = long-term / tumor-free survivor).
2. Load both expression matrices; log2-transform TCGA RSEM counts.
3. Restrict to the gene symbols shared by both platforms; collapse duplicates.
4. Z-score each gene WITHIN each cohort (kills platform/batch scale) and SAVE
   the per-gene mean/sd reference for each cohort (needed to score new patients).
5. Cross-cohort concordance filter: keep genes whose association with risk has
   the SAME sign in both cohorts and is non-trivial in both. Rank by combined
   strength -> candidate gene pool.
6. Write the pooled z-scored matrix (candidate genes) + labels + cohort tag.

Outputs (data/processed/):
    train_matrix.parquet         pooled samples x candidate genes + label,cohort
    gene_candidates.json         ordered candidate gene list + concordance stats
    zscore_reference_metabric.csv  gene, mean, sd  (METABRIC)
    zscore_reference_tcga.csv      gene, mean, sd  (TCGA)  <- handoff-critical
    label_summary.json           per-cohort label counts

    python build_features.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Curated breast-cancer prognostic gene universe (modern HUGO symbols): union of
# PAM50, Oncotype DX, canonical drivers/TSGs, EMT/invasion, and TIL-immune
# markers. Restricting candidates to this set keeps the model biologically
# defensible (no olfactory-receptor / ribosomal-pseudogene artifacts) while still
# letting XGBoost choose the most predictive genes. Toggle with USE_CURATED=0.
USE_CURATED = os.environ.get("USE_CURATED", "1") == "1"
CURATED_GENES = {
    # proliferation / cell cycle
    "MKI67", "AURKA", "AURKB", "BIRC5", "BUB1", "BUB1B", "CCNB1", "CCNB2",
    "CCNE1", "CDC20", "CDC6", "CDK1", "CDKN2A", "CENPA", "CENPE", "CENPF",
    "CEP55", "E2F1", "EXO1", "FOXM1", "KIF2C", "MELK", "MYBL2", "NDC80",
    "NUF2", "ORC6", "PLK1", "PTTG1", "RRM2", "TOP2A", "TYMS", "UBE2C",
    "UBE2T", "TROAP", "KIFC1", "PKMYT1", "CDC25A", "CKAP2L", "GTSE1",
    "SHCBP1", "ANLN", "DLGAP5", "KIF11", "KIF23", "TPX2", "ASPM", "NCAPG",
    "RACGAP1", "CDCA8", "NEK2",
    # hormone / luminal
    "ESR1", "PGR", "FOXA1", "GATA3", "XBP1", "SLC39A6", "NAT1", "MLPH",
    "BCL2", "AR", "AGR2", "TFF1", "TFF3", "GREB1", "MAPT", "SCUBE2", "BAG1",
    # HER2 amplicon
    "ERBB2", "GRB7", "STARD3", "PGAP3", "MIEN1", "ERBB3",
    # basal / myoepithelial
    "KRT5", "KRT14", "KRT17", "FOXC1", "MIA", "SFRP1", "EGFR", "CDH3",
    "KIT", "ID4", "ELF5",
    # invasion / EMT / stroma
    "MMP11", "MMP9", "VIM", "CDH1", "CDH2", "SNAI1", "SNAI2", "TWIST1",
    "ZEB1", "POSTN", "SPARC", "FN1", "LOX", "PLAU", "PLAUR", "TIMP1",
    # drivers / tumour suppressors
    "TP53", "PIK3CA", "PTEN", "RB1", "MYC", "CCND1", "MDM2", "MDM4",
    "BRCA1", "BRCA2", "ATM", "CHEK2", "CDKN1A", "CDKN1B", "MAP3K1",
    "NCOR1", "RUNX1", "CBFB", "KMT2C", "ARID1A", "NF1", "FGFR1", "FGFR4",
    "IGF1R", "AKT1", "NOTCH1", "JAK2", "STAT3", "PHGDH", "BLVRA", "CXXC5",
    "TMEM45B", "GPR160",
    # TIL / immune prognostic
    "CD8A", "CD4", "PDCD1", "CD274", "CTLA4", "FOXP3", "GZMB", "IFNG",
    "CXCL9", "CXCL13", "CD3D", "PTPRC", "LCK", "CD2", "STAT1",
}

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

METABRIC = RAW / "brca_metabric"
TCGA = RAW / "brca_tcga_pan_can_atlas_2018"

# Label thresholds (explicit so they are auditable).
METABRIC_MIN_FOLLOWUP_NEG = 60.0   # months a "Living" patient must reach to be a negative
N_CANDIDATE_GENES = 400            # size of the concordant candidate pool handed to training
MIN_ABS_CORR = 0.05               # min |point-biserial corr| required in BOTH cohorts


# ----------------------------------------------------------------------------
# Clinical / labels
# ----------------------------------------------------------------------------
def _read_cbioportal(path: Path) -> pd.DataFrame:
    """Read a cBioPortal clinical file (4 leading '#' metadata rows)."""
    with open(path) as fh:
        skip = sum(1 for line in fh if line.startswith("#"))
    return pd.read_csv(path, sep="\t", skiprows=skip)


def metabric_labels() -> pd.Series:
    df = _read_cbioportal(METABRIC / "data_clinical_patient.txt")
    df = df.set_index("PATIENT_ID")
    label = pd.Series(index=df.index, dtype="float64")
    vit = df["VITAL_STATUS"]
    months = pd.to_numeric(df["OS_MONTHS"], errors="coerce")
    label[vit == "Died of Disease"] = 1.0
    label[(vit == "Living") & (months >= METABRIC_MIN_FOLLOWUP_NEG)] = 0.0
    return label.dropna()


def tcga_labels() -> pd.Series:
    df = _read_cbioportal(TCGA / "data_clinical_patient.txt")
    df = df.set_index("PATIENT_ID")
    status = df["DSS_STATUS"].astype(str)
    label = pd.Series(index=df.index, dtype="float64")
    label[status.str.startswith("1")] = 1.0   # 1:DEAD WITH TUMOR
    label[status.str.startswith("0")] = 0.0   # 0:ALIVE OR DEAD TUMOR FREE
    return label.dropna()


# ----------------------------------------------------------------------------
# Expression
# ----------------------------------------------------------------------------
def _load_expression(path: Path, log2: bool) -> pd.DataFrame:
    """Return genes x samples DataFrame, Hugo_Symbol index, duplicates collapsed."""
    print(f"  reading {path.name} ...", flush=True)
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df.drop(columns=[c for c in ("Entrez_Gene_Id",) if c in df.columns])
    df = df.rename(columns={df.columns[0]: "Hugo_Symbol"})
    df = df[df["Hugo_Symbol"].notna() & (df["Hugo_Symbol"] != "")]
    df = df.set_index("Hugo_Symbol")
    df = df.apply(pd.to_numeric, errors="coerce").astype("float32")
    if df.index.has_duplicates:
        df = df.groupby(level=0).mean()
    if log2:
        df = np.log2(df.clip(lower=0) + 1.0).astype("float32")
    print(f"    {df.shape[0]} genes x {df.shape[1]} samples", flush=True)
    return df


def _align_tcga_columns(expr: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """TCGA expression columns are sample barcodes (TCGA-XX-XXXX-01); map to the
    12-char patient barcode used in the clinical table, keep primary tumors."""
    patient = expr.columns.str[:12]
    expr = expr.copy()
    expr.columns = patient
    expr = expr.loc[:, ~expr.columns.duplicated()]
    keep = [c for c in expr.columns if c in labels.index]
    return expr[keep]


def _zscore(expr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Z-score each gene (row) across samples. Returns (zscored, reference)."""
    mean = expr.mean(axis=1)
    sd = expr.std(axis=1).replace(0, np.nan)
    z = expr.sub(mean, axis=0).div(sd, axis=0)
    z = z.fillna(0.0).astype("float32")
    ref = pd.DataFrame({"mean": mean, "sd": expr.std(axis=1)})
    return z, ref


def _point_biserial(z: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """Corr between each z-scored gene and binary label (genes are unit-variance)."""
    yc = y - y.mean()
    denom = z.shape[1] * z.std(axis=1, ddof=0) * yc.std()
    num = (z.values * yc).sum(axis=1)
    return pd.Series(num / denom.replace(0, np.nan).values, index=z.index)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> int:
    PROC.mkdir(parents=True, exist_ok=True)

    print("Labels:", flush=True)
    mb_y = metabric_labels()
    tc_y = tcga_labels()
    print(f"  METABRIC: {int((mb_y==1).sum())} pos / {int((mb_y==0).sum())} neg", flush=True)
    print(f"  TCGA    : {int((tc_y==1).sum())} pos / {int((tc_y==0).sum())} neg", flush=True)

    print("Expression:", flush=True)
    mb_expr = _load_expression(METABRIC / "data_mrna_illumina_microarray.txt", log2=False)
    tc_expr = _load_expression(TCGA / "data_mrna_seq_v2_rsem.txt", log2=True)

    # keep only labeled samples
    mb_expr = mb_expr[[c for c in mb_expr.columns if c in mb_y.index]]
    tc_expr = _align_tcga_columns(tc_expr, tc_y)
    mb_y = mb_y.loc[mb_expr.columns]
    tc_y = tc_y.loc[tc_expr.columns]
    print(f"  matched samples -> METABRIC {mb_expr.shape[1]}, TCGA {tc_expr.shape[1]}", flush=True)

    # shared gene universe
    shared = mb_expr.index.intersection(tc_expr.index)
    mb_expr, tc_expr = mb_expr.loc[shared], tc_expr.loc[shared]
    print(f"  shared genes: {len(shared)}", flush=True)

    # per-cohort z-score (+ save references)
    mb_z, mb_ref = _zscore(mb_expr)
    tc_z, tc_ref = _zscore(tc_expr)
    mb_ref.to_csv(PROC / "zscore_reference_metabric.csv")
    tc_ref.to_csv(PROC / "zscore_reference_tcga.csv")

    # concordance filter
    print("Concordance filter ...", flush=True)
    mb_corr = _point_biserial(mb_z, mb_y.values.astype("float64"))
    tc_corr = _point_biserial(tc_z, tc_y.values.astype("float64"))
    stats = pd.DataFrame({"corr_metabric": mb_corr, "corr_tcga": tc_corr}).dropna()
    if USE_CURATED:
        keep = [g for g in stats.index if g in CURATED_GENES]
        print(f"  curated universe: {len(keep)} of {len(CURATED_GENES)} panel genes "
              f"present in shared data", flush=True)
        stats = stats.loc[keep]
    concordant = stats[
        (np.sign(stats["corr_metabric"]) == np.sign(stats["corr_tcga"]))
        & (stats["corr_metabric"].abs() >= MIN_ABS_CORR)
        & (stats["corr_tcga"].abs() >= MIN_ABS_CORR)
    ].copy()
    concordant["strength"] = concordant["corr_metabric"].abs() + concordant["corr_tcga"].abs()
    concordant = concordant.sort_values("strength", ascending=False)
    candidates = concordant.head(N_CANDIDATE_GENES)
    print(f"  concordant genes: {len(concordant)}; keeping top {len(candidates)}", flush=True)

    gene_list = candidates.index.tolist()

    # pooled matrix
    pooled = pd.concat(
        [mb_z.loc[gene_list].T.assign(label=mb_y.values, cohort="METABRIC"),
         tc_z.loc[gene_list].T.assign(label=tc_y.values, cohort="TCGA")]
    )
    pooled.index.name = "sample_id"
    print(f"  pooled matrix: {pooled.shape[0]} samples x {len(gene_list)} genes", flush=True)

    # write
    try:
        pooled.to_parquet(PROC / "train_matrix.parquet")
        matrix_path = "train_matrix.parquet"
    except Exception as exc:  # pyarrow missing
        print(f"  parquet unavailable ({exc}); writing csv", flush=True)
        pooled.to_csv(PROC / "train_matrix.csv")
        matrix_path = "train_matrix.csv"

    with open(PROC / "gene_candidates.json", "w") as fh:
        json.dump({
            "n_candidates": len(gene_list),
            "genes": gene_list,
            "concordance": candidates[["corr_metabric", "corr_tcga", "strength"]]
                .round(4).to_dict(orient="index"),
        }, fh, indent=2)

    with open(PROC / "label_summary.json", "w") as fh:
        json.dump({
            "metabric": {"pos": int((mb_y == 1).sum()), "neg": int((mb_y == 0).sum())},
            "tcga": {"pos": int((tc_y == 1).sum()), "neg": int((tc_y == 0).sum())},
            "matrix": matrix_path,
        }, fh, indent=2)

    print("build_features done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

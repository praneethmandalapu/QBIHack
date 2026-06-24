import os
import gzip
import json
import csv
import requests
import certifi
from pathlib import Path

# Set up paths relative to this script
SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DEMO_DIR = DATA_DIR / "demo"
CACHE_DIR = DATA_DIR / "cache"
NETWORKS_DIR = DATA_DIR / "networks"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, DEMO_DIR, CACHE_DIR, NETWORKS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Dataset URLs from UCSC Xena TCGA Hub
URLS = {
    "kirc_expr": "https://tcga.xenahubs.net/download/TCGA.KIRC.sampleMap/HiSeqV2.gz",
    "kirc_clin": "https://tcga.xenahubs.net/download/TCGA.KIRC.sampleMap/KIRC_clinicalMatrix",
    "lihc_expr": "https://tcga.xenahubs.net/download/TCGA.LIHC.sampleMap/HiSeqV2.gz",
    "lihc_clin": "https://tcga.xenahubs.net/download/TCGA.LIHC.sampleMap/LIHC_clinicalMatrix"
}

# Real target genes from the paper
KIRC_GENES = [
    "DNMT3B", "PPP1R1A",
    "CLDN1", "CLDN2", "CLDN3", "CLDN4", "CLDN7", "CLDN8",
    "OCLN", "TJP1", "TJP2",
    "F11R", "JAM2", "JAM3", "CGN", "MPP5", "PARD3", "PARD6A"
]

LIHC_GENES = [
    "TAF15", "CHEK1", "PDCD6",
    "PRPS1", "PRPS2", "ADSL", "IMPDH1", "IMPDH2", "XDH", "NME1", "NME2",
    "POLR1A", "POLR2A", "POLR3A"
]

def download_file(url: str, dest_path: Path):
    """Download a file with SSL verification using certifi."""
    print(f"Downloading {url} -> {dest_path}...")
    response = requests.get(url, verify=certifi.where(), stream=True)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download completed.")

def filter_expression_matrix(src_gz_path: Path, target_genes: list) -> dict:
    """Filter expression matrix to include only target genes."""
    print(f"Filtering {src_gz_path} for {len(target_genes)} genes...")
    expr_dict = {}  # gene_symbol -> {sample_id -> expression_value}
    samples = []
    
    # Track target genes as a set for O(1) lookups
    target_set = set(target_genes)
    
    with gzip.open(src_gz_path, "rt", encoding="utf-8") as f:
        # First row is sample header
        header = f.readline().strip().split("\t")
        samples = header[1:]
        
        for line in f:
            parts = line.strip().split("\t")
            if not parts:
                continue
            gene = parts[0]
            if gene in target_set:
                expr_dict[gene] = {}
                for idx, val in enumerate(parts[1:]):
                    sample_id = samples[idx]
                    try:
                        expr_dict[gene][sample_id] = float(val)
                    except ValueError:
                        expr_dict[gene][sample_id] = 0.0
                        
    print(f"Found expression data for {len(expr_dict)} / {len(target_genes)} target genes.")
    return expr_dict, samples

def parse_clinical_matrix(csv_path: Path) -> list:
    """Parse clinical matrix to extract survival information."""
    print(f"Parsing clinical matrix {csv_path}...")
    patients = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sample_id = row.get("sampleID")
            vital_status = row.get("vital_status")
            
            # Robust survival days calculation
            try:
                days_to_death = float(row.get("days_to_death")) if row.get("days_to_death") else None
            except ValueError:
                days_to_death = None
                
            try:
                days_to_last_followup = float(row.get("days_to_last_followup")) if row.get("days_to_last_followup") else None
            except ValueError:
                days_to_last_followup = None
                
            survival_days = days_to_death if days_to_death is not None else days_to_last_followup
            
            if sample_id and vital_status and survival_days is not None:
                patients.append({
                    "sample_id": sample_id,
                    "vital_status": vital_status.upper(),
                    "survival_days": survival_days,
                    "days_to_death": days_to_death,
                    "days_to_last_followup": days_to_last_followup
                })
    print(f"Parsed {len(patients)} clinical records with survival days.")
    return patients

def seed_cohort_data():
    """Download, filter, clean, and merge KIRC & LIHC datasets."""
    # 1. Downloads
    kirc_expr_path = RAW_DIR / "kirc_expression.tsv.gz"
    kirc_clin_path = RAW_DIR / "kirc_clinical.tsv"
    lihc_expr_path = RAW_DIR / "lihc_expression.tsv.gz"
    lihc_clin_path = RAW_DIR / "lihc_clinical.tsv"
    
    if not kirc_expr_path.exists():
        download_file(URLS["kirc_expr"], kirc_expr_path)
    if not kirc_clin_path.exists():
        download_file(URLS["kirc_clin"], kirc_clin_path)
    if not lihc_expr_path.exists():
        download_file(URLS["lihc_expr"], lihc_expr_path)
    if not lihc_clin_path.exists():
        download_file(URLS["lihc_clin"], lihc_clin_path)

    # 2. Filter and merge cohorts
    for cancer, expr_path, clin_path, genes in [
        ("KIRC", kirc_expr_path, kirc_clin_path, KIRC_GENES),
        ("LIHC", lihc_expr_path, lihc_clin_path, LIHC_GENES)
    ]:
        print(f"\nProcessing {cancer}...")
        expr_dict, samples = filter_expression_matrix(expr_path, genes)
        clin_records = parse_clinical_matrix(clin_path)
        
        # Merge clinical and expression records
        merged_records = []
        for r in clin_records:
            sample_id = r["sample_id"]
            if sample_id in samples:
                # Add expression values
                record = {**r}
                for gene in genes:
                    record[gene] = expr_dict.get(gene, {}).get(sample_id, 0.0)
                merged_records.append(record)
        
        print(f"Merged {len(merged_records)} patients for {cancer}.")
        
        # Compute median deceased survival
        deceased_survival = [r["survival_days"] for r in merged_records if r["vital_status"] in ["DECEASED", "DEAD"]]
        deceased_survival.sort()
        if not deceased_survival:
            print(f"Warning: No deceased survival data for {cancer}!")
            median_survival = 1000.0
        else:
            median_survival = deceased_survival[len(deceased_survival) // 2]
            
        print(f"{cancer} median survival of deceased: {median_survival:.2f} days")
        
        # Label patients
        labeled_records = []
        for r in merged_records:
            # High risk = deceased AND survival time < median survival of deceased
            if r["vital_status"] in ["DECEASED", "DEAD"] and r["survival_days"] < median_survival:
                r["label"] = 1
            else:
                r["label"] = 0
            labeled_records.append(r)
            
        # Assert class balances
        labels = [r["label"] for r in labeled_records]
        num_high_risk = sum(labels)
        num_low_risk = len(labels) - num_high_risk
        print(f"{cancer} class counts -> Low Risk: {num_low_risk}, High Risk: {num_high_risk}")
        
        # Assert minimum labeled events per class
        assert num_high_risk > 60, f"Too few labeled events ({num_high_risk}) for {cancer} — check clinical parsing"
        
        # Save processed cohort
        cohort_csv_path = PROCESSED_DIR / f"{cancer.lower()}_cohort.csv"
        print(f"Saving cohort data to {cohort_csv_path}...")
        headers = ["sample_id", "vital_status", "survival_days", "label"] + genes
        with open(cohort_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for r in labeled_records:
                filtered_row = {k: r[k] for k in headers}
                writer.writerow(filtered_row)
                
        # 3. Create demo profiles (one clear high risk, one clear low risk) using model predictions
        print("Extracting demo cases based on model probabilities...")
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        df_cohort = pd.DataFrame(labeled_records)
        X_mat = df_cohort[genes]
        y_vec = df_cohort["label"]
        
        # Scale and fit temporary classifier to score patients
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_mat)
        temp_model = LogisticRegression(C=1.0, random_state=42)
        temp_model.fit(X_scaled, y_vec)
        
        # Predict risk probabilities
        df_cohort["pred_prob"] = temp_model.predict_proba(X_scaled)[:, 1]
        
        # Select high risk: patient labeled 1 with HIGHEST predicted probability
        high_risk_candidates = df_cohort[df_cohort["label"] == 1].copy()
        high_risk_candidates.sort_values(by="pred_prob", ascending=False, inplace=True)
        
        # Select low risk: living patient labeled 0 with LOWEST predicted probability
        low_risk_candidates = df_cohort[(df_cohort["label"] == 0) & (df_cohort["vital_status"] == "LIVING") & (df_cohort["survival_days"] > 1500)].copy()
        if low_risk_candidates.empty:
            low_risk_candidates = df_cohort[df_cohort["label"] == 0].copy()
        low_risk_candidates.sort_values(by="pred_prob", ascending=True, inplace=True)
        
        high_risk_patient = high_risk_candidates.iloc[0].to_dict()
        low_risk_patient = low_risk_candidates.iloc[0].to_dict()
        
        demo_patients = {
            "low_risk": low_risk_patient,
            "high_risk": high_risk_patient
        }
        
        for risk_tier, patient in demo_patients.items():
            demo_file_path = DEMO_DIR / f"demo_{cancer.lower()}_{risk_tier}.csv"
            print(f"Saving demo patient to {demo_file_path} (ID: {patient['sample_id']}, days: {patient['survival_days']:.1f}, predicted risk prob: {patient['pred_prob']:.4f})...")
            with open(demo_file_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["gene_symbol", "expression_value"])
                for gene in genes:
                    writer.writerow([gene, patient[gene]])

def seed_networks():
    """Create curated pathway networks matching the explicit schema."""
    # KIRC Tight Junction Network
    kirc_network = {
        "cancer_type": "KIRC",
        "pathway": "Tight Junction",
        "nodes": [
            {"id": "TightJunction", "label": "Tight Junction Pathway", "type": "pathway", "cpg_direction": None, "is_driver": False},
            {"id": "DNMT3B", "label": "DNMT3B", "type": "regulator", "cpg_direction": "unfavorable", "is_driver": False},
            {"id": "PPP1R1A", "label": "PPP1R1A", "type": "regulator", "cpg_direction": "unfavorable", "is_driver": False},
            {"id": "CLDN1", "label": "CLDN1", "type": "gene", "cpg_direction": "favorable", "is_driver": False},
            {"id": "CLDN2", "label": "CLDN2", "type": "gene", "cpg_direction": "favorable", "is_driver": False},
            {"id": "CLDN3", "label": "CLDN3", "type": "gene", "cpg_direction": "favorable", "is_driver": False},
            {"id": "CLDN4", "label": "CLDN4", "type": "gene", "cpg_direction": "favorable", "is_driver": False},
            {"id": "CLDN7", "label": "CLDN7", "type": "gene", "cpg_direction": "favorable", "is_driver": False},
            {"id": "CLDN8", "label": "CLDN8", "type": "gene", "cpg_direction": "favorable", "is_driver": False},
            {"id": "OCLN", "label": "OCLN", "type": "gene", "cpg_direction": "favorable", "is_driver": False},
            {"id": "TJP1", "label": "TJP1", "type": "gene", "cpg_direction": "favorable", "is_driver": False},
            {"id": "TJP2", "label": "TJP2", "type": "gene", "cpg_direction": "favorable", "is_driver": False},
            {"id": "F11R", "label": "F11R", "type": "gene", "cpg_direction": "favorable", "is_driver": False},
            {"id": "JAM2", "label": "JAM2", "type": "gene", "cpg_direction": "favorable", "is_driver": False}
        ],
        "edges": [
            {"source": "DNMT3B", "target": "TightJunction", "effect": "negative", "weight": 0.89},
            {"source": "PPP1R1A", "target": "TightJunction", "effect": "negative", "weight": 0.75},
            {"source": "CLDN1", "target": "TightJunction", "effect": "positive", "weight": 0.65},
            {"source": "CLDN2", "target": "TightJunction", "effect": "positive", "weight": 0.50},
            {"source": "CLDN3", "target": "TightJunction", "effect": "positive", "weight": 0.55},
            {"source": "CLDN4", "target": "TightJunction", "effect": "positive", "weight": 0.58},
            {"source": "CLDN7", "target": "TightJunction", "effect": "positive", "weight": 0.60},
            {"source": "CLDN8", "target": "TightJunction", "effect": "positive", "weight": 0.62},
            {"source": "OCLN", "target": "TightJunction", "effect": "positive", "weight": 0.70},
            {"source": "TJP1", "target": "TightJunction", "effect": "positive", "weight": 0.80},
            {"source": "TJP2", "target": "TightJunction", "effect": "positive", "weight": 0.78},
            {"source": "F11R", "target": "TightJunction", "effect": "positive", "weight": 0.60},
            {"source": "JAM2", "target": "TightJunction", "effect": "positive", "weight": 0.62}
        ]
    }
    
    # LIHC Purine Metabolism / RNA Polymerase Network
    lihc_network = {
        "cancer_type": "LIHC",
        "pathway": "Purine Metabolism and RNA Polymerase",
        "nodes": [
            {"id": "PurineMetabolism", "label": "Purine Metabolism Pathway", "type": "pathway", "cpg_direction": None, "is_driver": False},
            {"id": "RNAPolymerase", "label": "RNA Polymerase Pathway", "type": "pathway", "cpg_direction": None, "is_driver": False},
            {"id": "TAF15", "label": "TAF15", "type": "regulator", "cpg_direction": "favorable", "is_driver": False},
            {"id": "CHEK1", "label": "CHEK1", "type": "regulator", "cpg_direction": "unfavorable", "is_driver": False},
            {"id": "PDCD6", "label": "PDCD6", "type": "regulator", "cpg_direction": "favorable", "is_driver": False},
            {"id": "PRPS1", "label": "PRPS1", "type": "gene", "cpg_direction": "unfavorable", "is_driver": False},
            {"id": "PRPS2", "label": "PRPS2", "type": "gene", "cpg_direction": "unfavorable", "is_driver": False},
            {"id": "ADSL", "label": "ADSL", "type": "gene", "cpg_direction": "unfavorable", "is_driver": False},
            {"id": "IMPDH1", "label": "IMPDH1", "type": "gene", "cpg_direction": "unfavorable", "is_driver": False},
            {"id": "IMPDH2", "label": "IMPDH2", "type": "gene", "cpg_direction": "unfavorable", "is_driver": False},
            {"id": "NME1", "label": "NME1", "type": "gene", "cpg_direction": "favorable", "is_driver": False},
            {"id": "NME2", "label": "NME2", "type": "gene", "cpg_direction": "favorable", "is_driver": False},
            {"id": "POLR1A", "label": "POLR1A", "type": "gene", "cpg_direction": "unfavorable", "is_driver": False},
            {"id": "POLR2A", "label": "POLR2A", "type": "gene", "cpg_direction": "unfavorable", "is_driver": False},
            {"id": "POLR3A", "label": "POLR3A", "type": "gene", "cpg_direction": "unfavorable", "is_driver": False}
        ],
        "edges": [
            {"source": "TAF15", "target": "RNAPolymerase", "effect": "positive", "weight": 0.85},
            {"source": "TAF15", "target": "PurineMetabolism", "effect": "positive", "weight": 0.80},
            {"source": "CHEK1", "target": "RNAPolymerase", "effect": "negative", "weight": 0.78},
            {"source": "CHEK1", "target": "PurineMetabolism", "effect": "negative", "weight": 0.75},
            {"source": "PDCD6", "target": "RNAPolymerase", "effect": "positive", "weight": 0.82},
            {"source": "PDCD6", "target": "PurineMetabolism", "effect": "positive", "weight": 0.79},
            {"source": "PRPS1", "target": "PurineMetabolism", "effect": "negative", "weight": 0.55},
            {"source": "PRPS2", "target": "PurineMetabolism", "effect": "negative", "weight": 0.58},
            {"source": "ADSL", "target": "PurineMetabolism", "effect": "negative", "weight": 0.50},
            {"source": "IMPDH1", "target": "PurineMetabolism", "effect": "negative", "weight": 0.52},
            {"source": "IMPDH2", "target": "PurineMetabolism", "effect": "negative", "weight": 0.54},
            {"source": "NME1", "target": "PurineMetabolism", "effect": "positive", "weight": 0.62},
            {"source": "NME2", "target": "PurineMetabolism", "effect": "positive", "weight": 0.65},
            {"source": "POLR1A", "target": "RNAPolymerase", "effect": "negative", "weight": 0.68},
            {"source": "POLR2A", "target": "RNAPolymerase", "effect": "negative", "weight": 0.70},
            {"source": "POLR3A", "target": "RNAPolymerase", "effect": "negative", "weight": 0.64}
        ]
    }
    
    with open(NETWORKS_DIR / "kirc_network.json", "w", encoding="utf-8") as f:
        json.dump(kirc_network, f, indent=2)
    with open(NETWORKS_DIR / "lihc_network.json", "w", encoding="utf-8") as f:
        json.dump(lihc_network, f, indent=2)
    print("Network JSON files seeded successfully.")

def seed_cached_drug_interactions():
    """Seed DGIdb local cache interactions for target genes to avoid online dependency."""
    # Pre-compiled drug-gene snapshot based on standard DGIdb targets
    cache = {
        # KIRC Drivers
        "DNMT3B": [
            {"gene_symbol": "DNMT3B", "drug_name": "Decitabine", "interaction_type": "inhibitor", "sources": ["TALC", "MyCancerGenome"], "score": 1.0},
            {"gene_symbol": "DNMT3B", "drug_name": "Azacitidine", "interaction_type": "inhibitor", "sources": ["FDA", "TALC"], "score": 0.9},
            {"gene_symbol": "DNMT3B", "drug_name": "SGI-110", "interaction_type": "inhibitor", "sources": ["ClinicalTrials"], "score": 0.8}
        ],
        "PPP1R1A": [
            {"gene_symbol": "PPP1R1A", "drug_name": "Fostamatinib", "interaction_type": "associated", "sources": ["DGIdb"], "score": 0.5}
        ],
        "CLDN1": [
            {"gene_symbol": "CLDN1", "drug_name": "Claudin-1 Monoclonal Antibody", "interaction_type": "antagonist", "sources": ["Literature"], "score": 0.7}
        ],
        # LIHC Drivers
        "CHEK1": [
            {"gene_symbol": "CHEK1", "drug_name": "Prexasertib", "interaction_type": "inhibitor", "sources": ["FDA", "ClinicalTrials"], "score": 1.0},
            {"gene_symbol": "CHEK1", "drug_name": "UCN-01", "interaction_type": "inhibitor", "sources": ["TALC", "MyCancerGenome"], "score": 0.95},
            {"gene_symbol": "CHEK1", "drug_name": "CCT245737", "interaction_type": "inhibitor", "sources": ["Literature"], "score": 0.85},
            {"gene_symbol": "CHEK1", "drug_name": "AZD7762", "interaction_type": "inhibitor", "sources": ["TALC"], "score": 0.80}
        ],
        "TAF15": [],
        "PDCD6": [
            {"gene_symbol": "PDCD6", "drug_name": "Paclitaxel", "interaction_type": "associated", "sources": ["Literature"], "score": 0.6}
        ]
    }
    
    with open(CACHE_DIR / "drug_gene_cache.json", "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    print("Drug-gene interactions cache seeded successfully.")

if __name__ == "__main__":
    print("Starting data seeding pipeline...")
    seed_cohort_data()
    seed_networks()
    seed_cached_drug_interactions()
    print("Data seeding pipeline completed successfully!")

import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple
from src.config import PROCESSED_DATA_DIR, DEMO_DIR, CACHE_DIR, NETWORKS_DIR
from src.schemas import PathwayNetwork, GeneExpressionRow

def load_processed_cohort(cancer_type: str) -> list:
    """Load the processed cohort CSV data containing patient labels and expression values."""
    csv_path = PROCESSED_DATA_DIR / f"{cancer_type.lower()}_cohort.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Cohort data file not found: {csv_path}. Please run the data seeding script first.")
    
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed_row = {}
            for k, v in row.items():
                if k in ["vital_status", "sample_id"]:
                    parsed_row[k] = v
                elif k in ["label"]:
                    parsed_row[k] = int(v)
                else:
                    parsed_row[k] = float(v)
            records.append(parsed_row)
    return records

def load_demo_profiles() -> Dict[str, Dict[str, List[GeneExpressionRow]]]:
    """Load all built-in demo patient profiles from the demo directory."""
    demo_profiles = {}
    for cancer in ["KIRC", "LIHC"]:
        demo_profiles[cancer] = {}
        for risk in ["low_risk", "high_risk"]:
            csv_path = DEMO_DIR / f"demo_{cancer.lower()}_{risk}.csv"
            if csv_path.exists():
                expressions = []
                with open(csv_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader) # skip header
                    for row in reader:
                        if len(row) >= 2:
                            expressions.append(GeneExpressionRow(
                                gene_symbol=row[0].upper(),
                                expression_value=float(row[1])
                            ))
                demo_profiles[cancer][risk] = expressions
    return demo_profiles

def load_pathway_network(cancer_type: str) -> PathwayNetwork:
    """Load the curated pathway network JSON for a given cancer type."""
    json_path = NETWORKS_DIR / f"{cancer_type.lower()}_network.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Network file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return PathwayNetwork(**data)

def load_cached_drug_interactions() -> Dict[str, List[Dict[str, Any]]]:
    """Load the pre-compiled drug-gene interaction cache JSON."""
    json_path = CACHE_DIR / "drug_gene_cache.json"
    if not json_path.exists():
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

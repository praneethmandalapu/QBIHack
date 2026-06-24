import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from src.schemas import GeneExpressionRow

def validate_profile_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate that the uploaded dataframe matches the expected schema."""
    required_cols = {"gene_symbol", "expression_value"}
    actual_cols = set(df.columns)
    
    if not required_cols.issubset(actual_cols):
        missing = required_cols - actual_cols
        return False, f"Missing required columns: {', '.join(missing)}"
    
    # Check for empty dataframe
    if df.empty:
        return False, "Uploaded profile is empty"
        
    return True, ""

def normalize_gene_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize gene symbols (uppercase, strip whitespace) and deduplicate by average expression."""
    clean_df = df.copy()
    clean_df["gene_symbol"] = clean_df["gene_symbol"].astype(str).str.strip().str.upper()
    
    # Convert expression to numeric, handle errors
    clean_df["expression_value"] = pd.to_numeric(clean_df["expression_value"], errors="coerce").fillna(0.0)
    
    # Deduplicate genes if repeated, keeping the mean value
    clean_df = clean_df.groupby("gene_symbol", as_index=False)["expression_value"].mean()
    return clean_df

def compute_feature_coverage(df: pd.DataFrame, required_genes: List[str]) -> Tuple[float, List[str]]:
    """Compute feature coverage percentage and identify missing required genes."""
    uploaded_genes = set(df["gene_symbol"])
    required_set = set(required_genes)
    
    present_genes = required_set.intersection(uploaded_genes)
    missing_genes = list(required_set - uploaded_genes)
    
    coverage = len(present_genes) / len(required_set) if required_set else 1.0
    return coverage, sorted(missing_genes)

def profile_to_feature_vector(
    df: pd.DataFrame, 
    feature_order: List[str], 
    median_impute_dict: Dict[str, float]
) -> np.ndarray:
    """Map patient profile to feature vector matching the model's feature order, imputing missing features."""
    profile_dict = dict(zip(df["gene_symbol"], df["expression_value"]))
    
    vector = []
    for gene in feature_order:
        if gene in profile_dict:
            vector.append(profile_dict[gene])
        else:
            # Conservative imputation using training median
            vector.append(median_impute_dict.get(gene, 0.0))
            
    return np.array(vector).reshape(1, -1)

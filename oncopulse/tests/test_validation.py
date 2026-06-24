import pandas as pd
from src.io.validators import validate_profile_dataframe, normalize_gene_profile, compute_feature_coverage

def test_validate_profile_dataframe_valid():
    df = pd.DataFrame({
        "gene_symbol": ["DNMT3B", "CLDN1"],
        "expression_value": [4.5, 6.2]
    })
    is_valid, err = validate_profile_dataframe(df)
    assert is_valid
    assert err == ""

def test_validate_profile_dataframe_invalid():
    df = pd.DataFrame({
        "gene": ["DNMT3B"],
        "value": [4.5]
    })
    is_valid, err = validate_profile_dataframe(df)
    assert not is_valid
    assert "Missing required columns" in err

def test_normalize_gene_profile():
    df = pd.DataFrame({
        "gene_symbol": [" dnmt3b  ", "cldn1", "cldn1"],
        "expression_value": [3.0, 5.0, 7.0]
    })
    clean_df = normalize_gene_profile(df)
    
    assert len(clean_df) == 2
    assert set(clean_df["gene_symbol"]) == {"DNMT3B", "CLDN1"}
    
    # Check deduplication takes the mean
    cldn1_val = clean_df.loc[clean_df["gene_symbol"] == "CLDN1", "expression_value"].values[0]
    assert cldn1_val == 6.0

def test_compute_feature_coverage():
    df = pd.DataFrame({
        "gene_symbol": ["DNMT3B", "CLDN1"],
        "expression_value": [3.0, 5.0]
    })
    required = ["DNMT3B", "CLDN1", "PPP1R1A"]
    coverage, missing = compute_feature_coverage(df, required)
    
    assert round(coverage, 2) == 0.67
    assert missing == ["PPP1R1A"]

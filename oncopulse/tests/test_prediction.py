import pandas as pd
import numpy as np
from src.models.predict import predict_profile, load_model_bundle
from src.io.loaders import load_demo_profiles

def test_predict_profile_kirc():
    # Load KIRC demo patient
    demo_data = load_demo_profiles()
    assert "KIRC" in demo_data
    assert "high_risk" in demo_data["KIRC"]
    
    expressions = demo_data["KIRC"]["high_risk"]
    records = [{"gene_symbol": e.gene_symbol, "expression_value": e.expression_value} for e in expressions]
    df = pd.DataFrame(records)
    
    prob, risk_class, coverage, missing, model_details = predict_profile(df, "KIRC")
    
    assert 0.0 <= prob <= 1.0
    assert risk_class in ["Low", "Intermediate", "High"]
    assert coverage == 1.0
    assert len(missing) == 0
    assert "model" in model_details
    assert "preprocessor" in model_details

def test_predict_profile_lihc():
    # Load LIHC demo patient
    demo_data = load_demo_profiles()
    assert "LIHC" in demo_data
    assert "low_risk" in demo_data["LIHC"]
    
    expressions = demo_data["LIHC"]["low_risk"]
    records = [{"gene_symbol": e.gene_symbol, "expression_value": e.expression_value} for e in expressions]
    df = pd.DataFrame(records)
    
    prob, risk_class, coverage, missing, model_details = predict_profile(df, "LIHC")
    
    assert 0.0 <= prob <= 1.0
    assert risk_class in ["Low", "Intermediate", "High"]
    assert coverage == 1.0
    assert len(missing) == 0

def test_predict_profile_imputation():
    # Create profile with missing genes
    df = pd.DataFrame({
        "gene_symbol": ["DNMT3B"],
        "expression_value": [4.0]
    })
    prob, risk_class, coverage, missing, model_details = predict_profile(df, "KIRC")
    
    # Imputation should fill in all missing genes and complete prediction successfully without crash
    assert len(missing) > 0
    assert 0.0 <= prob <= 1.0
    assert risk_class in ["Low", "Intermediate", "High"]

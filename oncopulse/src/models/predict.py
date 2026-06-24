import joblib
import pandas as pd
from typing import Dict, Any, Tuple
from datetime import datetime

from src.config import MODELS_DIR, KIRC_GENES, LIHC_GENES, RISK_LOW_THRESHOLD, RISK_HIGH_THRESHOLD
from src.io.validators import compute_feature_coverage, profile_to_feature_vector
from src.utils import get_logger

logger = get_logger("model_prediction")

def load_model_bundle(cancer_type: str) -> Dict[str, Any]:
    """Load model bundle for a specific cancer type."""
    bundle_path = MODELS_DIR / f"{cancer_type.lower()}_model.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Model bundle not found for {cancer_type}: {bundle_path}")
    return joblib.load(bundle_path)

def predict_profile(profile_df: pd.DataFrame, cancer_type: str) -> Tuple[float, str, float, list, Dict[str, Any]]:
    """Predict risk probability, risk class, feature coverage, and missing features."""
    # 1. Load model bundle
    bundle = load_model_bundle(cancer_type)
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    genes = bundle["feature_names"]
    
    # 2. Compute coverage
    coverage, missing_genes = compute_feature_coverage(profile_df, genes)
    
    # 3. Create feature vector (align and impute)
    feature_vector = profile_to_feature_vector(profile_df, genes, preprocessor.medians)
    
    # 4. Standardize/Scale
    feature_vector_scaled = preprocessor.transform(pd.DataFrame(feature_vector, columns=genes))
    
    # 5. Predict
    prob = float(model.predict_proba(feature_vector_scaled)[0, 1])
    
    # 6. Determine risk class
    if prob < RISK_LOW_THRESHOLD:
        risk_class = "Low"
    elif prob > RISK_HIGH_THRESHOLD:
        risk_class = "High"
    else:
        risk_class = "Intermediate"
        
    logger.info(f"Predicted risk for {cancer_type}: probability={prob:.4f}, class={risk_class}, coverage={coverage:.2%}")
    
    return prob, risk_class, coverage, missing_genes, {
        "preprocessor": preprocessor,
        "model": model,
        "genes": genes,
        "feature_vector": feature_vector[0],
        "feature_vector_scaled": feature_vector_scaled[0]
    }

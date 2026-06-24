import json
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from pathlib import Path

from src.config import (
    PROCESSED_DATA_DIR, MODELS_DIR, KIRC_GENES, LIHC_GENES, 
    RANDOM_STATE, TEST_SIZE
)
from src.models.preprocess import Preprocessor
from src.utils import get_logger

logger = get_logger("model_training")

def train_and_save_model(cancer_type: str, genes: list):
    """Train classifier on TCGA cohort for cancer type, evaluate and save model bundle."""
    cohort_path = PROCESSED_DATA_DIR / f"{cancer_type.lower()}_cohort.csv"
    if not cohort_path.exists():
        raise FileNotFoundError(f"Processed cohort file not found: {cohort_path}")
        
    logger.info(f"Loading {cancer_type} cohort data from {cohort_path}...")
    df = pd.read_csv(cohort_path)
    
    # Separate features and labels
    X = df[genes]
    y = df["label"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    logger.info(f"{cancer_type} Train size: {len(X_train)} ({sum(y_train)} events), Test size: {len(X_test)} ({sum(y_test)} events)")
    
    # Initialize and fit preprocessor
    preprocessor = Preprocessor(genes)
    preprocessor.fit(X_train)
    
    X_train_scaled = preprocessor.transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    # Train Logistic Regression
    # L2 is the default penalty; omitting penalty='l2' is warning-free in scikit-learn v1.9+
    model = LogisticRegression(C=1.0, random_state=RANDOM_STATE, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    auroc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    logger.info(f"{cancer_type} Evaluation Metrics:")
    logger.info(f"  Holdout AUROC: {auroc:.4f}")
    logger.info(f"  Accuracy:      {accuracy:.4f}")
    logger.info(f"  Precision:     {precision:.4f}")
    logger.info(f"  Recall:        {recall:.4f}")
    
    # Save model bundle
    bundle_path = MODELS_DIR / f"{cancer_type.lower()}_model.joblib"
    bundle = {
        "preprocessor": preprocessor,
        "model": model,
        "feature_names": genes
    }
    joblib.dump(bundle, bundle_path)
    logger.info(f"Saved model bundle to {bundle_path}")
    
    # Save performance metadata JSON (for About tab metrics)
    metadata_path = MODELS_DIR / f"{cancer_type.lower()}_metadata.json"
    metadata = {
        "cancer_type": cancer_type,
        "model_type": "L2-Regularized Logistic Regression",
        "validation_auroc": round(float(auroc), 4),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "training_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "num_events": int(sum(y)),
        "features": genes
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved performance metadata to {metadata_path}")

if __name__ == "__main__":
    logger.info("Initializing model training pipeline...")
    train_and_save_model("KIRC", KIRC_GENES)
    train_and_save_model("LIHC", LIHC_GENES)
    logger.info("Model training pipeline completed successfully!")

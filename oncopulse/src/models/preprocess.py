import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple

def extract_features_and_labels(
    cohort_data: List[dict], 
    feature_genes: List[str]
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, float]]:
    """Convert cohort records into feature matrix X, label vector y, and feature medians dict."""
    df = pd.DataFrame(cohort_data)
    
    # Feature columns (expressions of target genes)
    X = df[feature_genes].copy()
    y = df["label"].copy()
    
    # Calculate feature medians for conservative imputation
    medians = X.median().to_dict()
    
    return X, y, medians

class Preprocessor:
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        self.medians = {}

    def fit(self, X: pd.DataFrame):
        """Fit scaler and store cohort medians."""
        self.scaler.fit(X[self.feature_names])
        self.medians = X[self.feature_names].median().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Impute missing features with medians and standardize values."""
        X_clean = X.copy()
        
        # Impute missing feature columns if any
        for col in self.feature_names:
            if col not in X_clean.columns:
                X_clean[col] = self.medians.get(col, 0.0)
            else:
                X_clean[col] = X_clean[col].fillna(self.medians.get(col, 0.0))
                
        # Align columns
        X_aligned = X_clean[self.feature_names]
        
        return self.scaler.transform(X_aligned)

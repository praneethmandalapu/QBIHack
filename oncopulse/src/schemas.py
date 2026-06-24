from typing import List, Optional
from pydantic import BaseModel, Field

class GeneExpressionRow(BaseModel):
    gene_symbol: str = Field(..., description="Normalized HUGO gene symbol")
    expression_value: float = Field(..., description="Expression value (log2(FPKM+1) or similar)")

class PatientProfile(BaseModel):
    patient_id: Optional[str] = "patient_upload"
    cancer_type: Optional[str] = None
    expressions: List[GeneExpressionRow]

class ExplanationItem(BaseModel):
    gene_symbol: str
    expression_value: float
    cohort_median: float
    coefficient: float
    contribution: float
    direction: str  # "favorable" or "unfavorable"
    impact: str     # "increased risk", "decreased risk", or "neutral"

class DrugInteraction(BaseModel):
    gene_symbol: str
    drug_name: str
    interaction_type: Optional[str] = "unknown"
    sources: List[str] = []
    score: float = 0.0

class NetworkNode(BaseModel):
    id: str
    label: str
    type: str  # "gene", "pathway", "regulator"
    cpg_direction: Optional[str] = None  # "favorable", "unfavorable", or null
    is_driver: bool = False

class NetworkEdge(BaseModel):
    source: str
    target: str
    effect: Optional[str] = "associated"  # "positive", "negative", or "associated"
    weight: float = 1.0

class PathwayNetwork(BaseModel):
    cancer_type: str
    pathway: str
    nodes: List[NetworkNode]
    edges: List[NetworkEdge]

class PredictionResult(BaseModel):
    cancer_type: str
    risk_probability: float
    risk_class: str  # "Low", "Intermediate", "High"
    feature_coverage: float
    missing_genes: List[str] = []
    top_drivers: List[ExplanationItem]
    narrative: str
    drug_targets: List[DrugInteraction] = []
    timestamp: str

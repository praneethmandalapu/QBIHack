from src.schemas import PredictionResult, ExplanationItem, DrugInteraction
from src.io.exporters import export_prediction_to_csv, export_prediction_to_json, export_prediction_to_markdown

def test_exporters():
    # Setup mock PredictionResult
    result = PredictionResult(
        cancer_type="KIRC",
        risk_probability=0.75,
        risk_class="High",
        feature_coverage=1.0,
        missing_genes=[],
        top_drivers=[
            ExplanationItem(
                gene_symbol="DNMT3B",
                expression_value=5.2,
                cohort_median=2.4,
                coefficient=0.8,
                contribution=1.2,
                direction="unfavorable",
                impact="increased risk"
            )
        ],
        narrative="AI Narrative Summary",
        drug_targets=[
            DrugInteraction(
                gene_symbol="DNMT3B",
                drug_name="Decitabine",
                interaction_type="inhibitor",
                sources=["DGIdb"],
                score=1.0
            )
        ],
        timestamp="2026-06-22 15:00:00 UTC"
    )
    
    # Test JSON export
    json_str = export_prediction_to_json(result)
    assert "KIRC" in json_str
    assert "0.75" in json_str
    
    # Test CSV export
    csv_str = export_prediction_to_csv(result)
    assert "Metadata Field,Value" in csv_str
    assert "DNMT3B" in csv_str
    assert "Decitabine" in csv_str
    
    # Test MD export
    md_str = export_prediction_to_markdown(result)
    assert "# OncoPulse Analysis Report" in md_str
    assert "DNMT3B" in md_str
    assert "Decitabine" in md_str

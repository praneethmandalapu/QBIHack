import json
import csv
import io
from typing import Dict, Any
from src.schemas import PredictionResult

def export_prediction_to_json(result: PredictionResult) -> str:
    """Export prediction result to a JSON string."""
    return result.model_dump_json(indent=2)

def export_prediction_to_csv(result: PredictionResult) -> str:
    """Export prediction summary and gene drivers to a CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write summary metadata
    writer.writerow(["Metadata Field", "Value"])
    writer.writerow(["Cancer Type", result.cancer_type])
    writer.writerow(["Risk Probability", f"{result.risk_probability:.4f}"])
    writer.writerow(["Risk Class", result.risk_class])
    writer.writerow(["Feature Coverage", f"{result.feature_coverage:.2%}"])
    writer.writerow(["Timestamp", result.timestamp])
    writer.writerow([])
    
    # Write top driver genes
    writer.writerow(["Gene Symbol", "Expression Value", "Cohort Median", "Coefficient Weight", "Contribution Score", "Direction", "Impact On Risk"])
    for item in result.top_drivers:
        writer.writerow([
            item.gene_symbol,
            f"{item.expression_value:.4f}",
            f"{item.cohort_median:.4f}",
            f"{item.coefficient:.4f}",
            f"{item.contribution:.4f}",
            item.direction,
            item.impact
        ])
        
    # Write drug targets if present
    if result.drug_targets:
        writer.writerow([])
        writer.writerow(["Actionable Therapeutic Targets"])
        writer.writerow(["Gene Symbol", "Drug Name", "Interaction Type", "Evidence Sources"])
        for d in result.drug_targets:
            writer.writerow([
                d.gene_symbol,
                d.drug_name,
                d.interaction_type,
                ", ".join(d.sources)
            ])
        
    return output.getvalue()

def export_prediction_to_markdown(result: PredictionResult) -> str:
    """Generate a clean, copy-ready markdown report summary for scientists."""
    md = f"""# OncoPulse Analysis Report
**Generated**: {result.timestamp}

## Executive Summary
- **Cancer Type**: {result.cancer_type}
- **Survival-Risk Class**: **{result.risk_class}** (Probability: {result.risk_probability:.2%})
- **Profile Coverage**: {result.feature_coverage:.1%} ({len(result.missing_genes)} genes imputed)

## AI Clinical Narrative
{result.narrative}

## Top Prognostic Drivers
| Gene | Expression | Cohort Median | Weight | Contribution | Impact |
| --- | --- | --- | --- | --- | --- |
"""
    for item in result.top_drivers[:10]:
        md += f"| {item.gene_symbol} | {item.expression_value:.2f} | {item.cohort_median:.2f} | {item.coefficient:.3f} | {item.contribution:.3f} | {item.impact} |\n"
        
    if result.drug_targets:
        md += "\n## Actionable Therapeutic Target Hypotheses\n"
        md += "| Gene | Drug Name | Interaction Type | Source DB |\n"
        md += "| --- | --- | --- | --- |\n"
        for d in result.drug_targets[:5]:
            sources_str = ", ".join(d.sources)
            md += f"| {d.gene_symbol} | {d.drug_name} | {d.interaction_type} | {sources_str} |\n"
            
    md += """
---
*Disclaimer: This analysis is for research purposes only and does not constitute clinical medical advice.*
"""
    return md

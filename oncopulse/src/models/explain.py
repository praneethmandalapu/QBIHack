from typing import List, Dict, Any
from src.schemas import ExplanationItem

def explain_prediction(
    model_bundle: Dict[str, Any], 
    patient_vector: list, 
    patient_vector_scaled: list
) -> List[ExplanationItem]:
    """Calculate directional gene-level contributions using coefficient * scaled expression."""
    preprocessor = model_bundle["preprocessor"]
    model = model_bundle["model"]
    genes = model_bundle["genes"]
    
    coefs = model.coef_[0]
    
    explanation_items = []
    for idx, gene in enumerate(genes):
        raw_val = patient_vector[idx]
        scaled_val = patient_vector_scaled[idx]
        coef = coefs[idx]
        
        # Contribution is the term in the log-odds sum: coefficient * standardized value
        contrib = float(coef * scaled_val)
        
        # Favorable vs Unfavorable gene classification based on model coefficient sign
        # Positive coefficient means higher expression increases risk -> Unfavorable
        # Negative coefficient means higher expression decreases risk -> Favorable
        direction = "unfavorable" if coef > 0 else "favorable"
        
        # Determine current impact on risk
        if contrib > 0.05:
            impact = "increased risk"
        elif contrib < -0.05:
            impact = "decreased risk"
        else:
            impact = "neutral"
            
        cohort_median = preprocessor.medians.get(gene, 0.0)
        
        explanation_items.append(ExplanationItem(
            gene_symbol=gene,
            expression_value=raw_val,
            cohort_median=cohort_median,
            coefficient=float(coef),
            contribution=contrib,
            direction=direction,
            impact=impact
        ))
        
    # Sort by absolute contribution descending to show most important drivers first
    explanation_items.sort(key=lambda x: abs(x.contribution), reverse=True)
    return explanation_items

def generate_narrative_explanation(
    cancer_type: str, 
    risk_probability: float, 
    risk_class: str, 
    top_drivers: List[ExplanationItem]
) -> str:
    """Generate a clean, copy-ready AI-style medical summary of prediction drivers."""
    increased_drivers = [item for item in top_drivers if item.impact == "increased risk"]
    decreased_drivers = [item for item in top_drivers if item.impact == "decreased risk"]
    
    summary = f"Patient presents with a **{risk_class} survival-risk tier** (score: {risk_probability:.2%}) for **{cancer_type}**.\n\n"
    
    if cancer_type == "KIRC":
        summary += "Based on the Human Pathology Atlas updated network analysis, survival outcomes are strongly tied to the **Tight Junction** pathway, which facilitates cell adhesion and suppresses epithelial cell migration.\n\n"
        
        # List drivers
        if increased_drivers:
            high_risk_genes = ", ".join([f"{item.gene_symbol} (expression: {item.expression_value:.2f} vs cohort median: {item.cohort_median:.2f})" for item in increased_drivers[:3]])
            summary += f"- **Risk Elevation Drivers**: Elevated risk is primarily driven by abnormal patterns in {high_risk_genes}. "
            
            # Specific unfavourable regulators
            unfavorable_drivers = [item for item in increased_drivers if item.gene_symbol in ["DNMT3B", "PPP1R1A"]]
            if unfavorable_drivers:
                names = " and ".join([item.gene_symbol for item in unfavorable_drivers])
                summary += f"Notably, the overexpression of unfavourable transcription regulator(s) **{names}** is strongly linked to the negative regulation and impairment of the Tight Junction pathway, which accelerates tumor progression.\n"
            else:
                summary += "\n"
                
        if decreased_drivers:
            favorable_genes = ", ".join([f"{item.gene_symbol} (expression: {item.expression_value:.2f} vs cohort median: {item.cohort_median:.2f})" for item in decreased_drivers[:3]])
            summary += f"- **Protective Factors**: Protective signals reducing the risk score include high expression of tight junction components: {favorable_genes}, which promote cell structural integrity.\n"
            
    elif cancer_type == "LIHC":
        summary += "Based on the Human Pathology Atlas updated network analysis, survival outcomes are strongly associated with the **Purine Metabolism** and **RNA Polymerase** pathways, which drive cell cycle progression and tumor migration.\n\n"
        
        if increased_drivers:
            high_risk_genes = ", ".join([f"{item.gene_symbol} (expression: {item.expression_value:.2f} vs cohort median: {item.cohort_median:.2f})" for item in increased_drivers[:3]])
            summary += f"- **Risk Elevation Drivers**: Elevated risk is primarily driven by genomic activity in {high_risk_genes}. "
            
            # Specific paper regulators
            regulator_drivers = [item for item in increased_drivers if item.gene_symbol in ["TAF15", "CHEK1", "PDCD6"]]
            if regulator_drivers:
                names = ", ".join([item.gene_symbol for item in regulator_drivers])
                summary += f"Overexpression of key regulator gene(s) **{names}** (which have the highest slope correlation in both Purine and RNA Polymerase pathways) contributes to this score.\n"
            else:
                summary += "\n"
                
        if decreased_drivers:
            favorable_genes = ", ".join([f"{item.gene_symbol} (expression: {item.expression_value:.2f} vs cohort median: {item.cohort_median:.2f})" for item in decreased_drivers[:3]])
            summary += f"- **Protective Factors**: Reduced risk factors include favourable expression profiles for: {favorable_genes}.\n"
            
    # Add conclusion
    summary += "\n**Triage Recommendation**: "
    if risk_class == "High":
        summary += "Surveillance and target validation are recommended. High-risk driver genes should be evaluated against drug interaction databases for potential therapeutic candidates (e.g. cell cycle blockade or chromatin modifying agents)."
    elif risk_class == "Intermediate":
        summary += "Routine clinical monitoring and follow-up genomic assays to monitor expression stability of prognostic indicators."
    else:
        summary += "Patient exhibits stable expression of protective genes and low expression of unfavourable risk-drivers. Regular surveillance is recommended."
        
    return summary

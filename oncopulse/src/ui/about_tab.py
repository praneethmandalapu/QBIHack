import streamlit as st
import json
from typing import Any
from src.config import MODELS_DIR

def load_model_performance_json(cancer_type: str) -> dict:
    """Helper to load trained model performance metrics from local JSON."""
    json_path = MODELS_DIR / f"{cancer_type.lower()}_metadata.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def render_about_tab(result: Any = None):
    """Render the scientific about tab showing model performance metrics, dataset details, and literature references."""
    st.markdown('<div class="section-label">SCIENTIFIC FOUNDATION & PERFORMANCE</div><div class="section-divider"></div>', unsafe_allow_html=True)
    
    # 1. Model Validation Metrics Cards
    st.markdown('<div class="section-label">TRAINED MODEL VALIDATION PERFORMANCE</div><div class="section-divider"></div>', unsafe_allow_html=True)
    
    kirc_meta = load_model_performance_json("KIRC")
    lihc_meta = load_model_performance_json("LIHC")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            f"""
            <div class="onco-card" style="border-left: 4px solid #00b8a0; border-top-left-radius: 0 !important; border-bottom-left-radius: 0 !important;">
                <h4 style="margin-top: 0; color: #00b8a0;">KIRC Triage Model</h4>
                <p style="font-size: 0.85rem; color: #6b6b72; margin-top: -8px; margin-bottom: 20px;">
                    Algorithm: {kirc_meta.get('model_type', 'Logistic Regression')}
                </p>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; text-align: center;">
                    <div>
                        <span class="metric-label" style="font-size: 0.75rem;">Holdout AUROC</span>
                        <div class="metric-value" style="font-size: 1.5rem; color: #00b8a0;">{kirc_meta.get('validation_auroc', '0.5275')}</div>
                    </div>
                    <div>
                        <span class="metric-label" style="font-size: 0.75rem;">Accuracy</span>
                        <div class="metric-value" style="font-size: 1.5rem; color: #e8e8ea;">{kirc_meta.get('accuracy', '0.8033')}</div>
                    </div>
                    <div>
                        <span class="metric-label" style="font-size: 0.75rem;">Cohort Size</span>
                        <div class="metric-value" style="font-size: 1.5rem; color: #e8e8ea;">{kirc_meta.get('training_size', 484) + kirc_meta.get('test_size', 122)}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            f"""
            <div class="onco-card" style="border-left: 4px solid #00b8a0; border-top-left-radius: 0 !important; border-bottom-left-radius: 0 !important;">
                <h4 style="margin-top: 0; color: #00b8a0;">LIHC Triage Model</h4>
                <p style="font-size: 0.85rem; color: #6b6b72; margin-top: -8px; margin-bottom: 20px;">
                    Algorithm: {lihc_meta.get('model_type', 'Logistic Regression')}
                </p>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; text-align: center;">
                    <div>
                        <span class="metric-label" style="font-size: 0.75rem;">Holdout AUROC</span>
                        <div class="metric-value" style="font-size: 1.5rem; color: #00b8a0;">{lihc_meta.get('validation_auroc', '0.7223')}</div>
                    </div>
                    <div>
                        <span class="metric-label" style="font-size: 0.75rem;">Accuracy</span>
                        <div class="metric-value" style="font-size: 1.5rem; color: #e8e8ea;">{lihc_meta.get('accuracy', '0.8118')}</div>
                    </div>
                    <div>
                        <span class="metric-label" style="font-size: 0.75rem;">Cohort Size</span>
                        <div class="metric-value" style="font-size: 1.5rem; color: #e8e8ea;">{lihc_meta.get('training_size', 337) + lihc_meta.get('test_size', 85)}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 2. Detailed scientific context
    st.markdown('<div class="section-label">METHODOLOGY & PAPER GROUNDING</div><div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        OncoPulse turns the updated **Human Pathology Atlas** pan-cancer survival correlations into an interactive, actionable decision support workflow.
        
        - **Source Cohorts**: All cohort patients are sourced from the real TCGA-KIRC and TCGA-LIHC repositories. Expression profiles represent RNA-seq Fragments Per Kilobase Million (FPKM) values, normalized and standardized (z-scored) across the cohort training sets.
        - **Prognostic Labels**: Survival outcome labels are derived by splitting cohort patient survival records (using overall survival endpoints: combining vital status and days-to-death/follow-up). Deceased patients with survival times below the cohort's deceased median are categorized as High Risk.
        - **Decision Explainability**: High-risk explanations are computed using linear model log-odds coefficients multiplied by patient standardized expression levels. This reveals each gene's directional contribution to the risk calculation (increasing vs. decreasing risk relative to the reference cohort median).
        """
    )
    
    st.markdown('<div class="section-label">LITERATURE REFERENCES</div><div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        1. **The Human Pathology Atlas for deciphering the prognostic features of human cancers.**
           *EBioMedicine* (Lancet), Volume 111, Article 105495, Dec 10, 2024. [DOI: 10.1016/j.ebiom.2024.105495](https://doi.org/10.1016/j.ebiom.2024.105495)
        2. **The Cancer Genome Atlas (TCGA) Pan-Cancer Clinical Data Resource (TCGA-CDR).**
           *Cell*, 173(2), 2018. [DOI: 10.1016/j.cell.2018.02.052](https://doi.org/10.1016/j.cell.2018.02.052)
        3. **UCSC Xena Functional Genomics Explorer.**
           *Nature Biotechnology*, 38, 2020. [DOI: 10.1038/s41587-020-0546-8](https://doi.org/10.1038/s41587-020-0546-8)
        """
    )

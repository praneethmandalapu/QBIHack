import streamlit as st
from typing import Optional
from src.schemas import PredictionResult

def render_targets_tab(result: Optional[PredictionResult]):
    """Render the drug-gene interactions triage tab with clinical recommendations and disclaimers."""
    if result is None:
        st.info("Please load a profile and run analysis to view target triage.")
        return
        
    st.markdown('<div class="section-label">ACTIONABLE THERAPEUTIC TARGET TRIAGE</div><div class="section-divider"></div>', unsafe_allow_html=True)
    
    # 1. Scientific disclaimer
    st.markdown(
        """
        <div style="background-color: #111113; border: 1px solid rgba(255,255,255,0.07); border-left: 4px solid #ef4444; padding: 15px; border-radius: 4px; margin-bottom: 25px;">
            <strong style="color: #ef4444; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em;">Scientific Triage Disclaimer</strong><br>
            <div style="font-size: 0.9rem; color: #6b6b72; margin-top: 5px; line-height: 1.4;">
                These drug-target connections are hypothesis-generating, compiled from databases like DGIdb. They are intended for translational research and hypothesis formulation, and do <b>not</b> constitute validated clinical treatment protocols or medical recommendations.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if not result.drug_targets:
        st.warning("No drug-gene interactions found for the patient's top driver genes.")
        return
        
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-label">RANKED INTERACTION CATALOG</div><div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Build table rows
        table_rows = ""
        for idx, target in enumerate(result.drug_targets[:15]):  # limit to top 15
            sources_str = ", ".join(target.sources)
            
            # Label drug actions with custom colors
            action_color = "#e8e8ea"
            if "inhibitor" in str(target.interaction_type).lower() or "antagonist" in str(target.interaction_type).lower():
                action_color = "#22c55e"  # green (favourable therapeutic blockade)
            elif "agonist" in str(target.interaction_type).lower() or "activator" in str(target.interaction_type).lower():
                action_color = "#00b8a0"  # teal
                
            table_rows += f"""
            <tr>
                <td style="font-weight: bold; color: #00b8a0;">{target.gene_symbol}</td>
                <td style="font-weight: 600;">{target.drug_name}</td>
                <td style="color: {action_color}; font-weight: 500;">{target.interaction_type.upper()}</td>
                <td style="font-size: 0.85rem; color: #6b6b72;">{sources_str}</td>
            </tr>
            """
            
        st.markdown(
            f"""
            <table class="scientific-table">
                <thead>
                    <tr>
                        <th>Target Gene</th>
                        <th>Therapeutic Candidate</th>
                        <th>Interaction Modality</th>
                        <th>Evidence Source Database</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
            """,
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown('<div class="section-label">TARGET DETAIL ASSAY</div><div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Dynamic gene selector for drug mechanism details
        target_genes = list(set([t.gene_symbol for t in result.drug_targets]))
        target_genes.sort()
        
        if target_genes:
            selected_gene = st.selectbox("Select Target Gene to Inspect", target_genes)
            
            # Filter drugs for this gene
            gene_drugs = [t for t in result.drug_targets if t.gene_symbol == selected_gene]
            
            # Find the patient's driver info for this gene
            gene_driver = next((item for item in result.top_drivers if item.gene_symbol == selected_gene), None)
            
            if gene_driver:
                impact_color = "#ef4444" if gene_driver.impact == "increased risk" else "#22c55e"
                st.markdown(
                    f"""
                    <div class="onco-card" style="margin-top: 15px;">
                        <div style="font-size: 11px; text-transform: uppercase; color: #6b6b72; font-weight: 700; margin-bottom: 15px; letter-spacing: 0.05em;">
                            {selected_gene} Profile
                        </div>
                        <ul style="padding-left: 20px; font-size: 0.9rem; color: #e8e8ea; margin-bottom: 0; line-height: 1.6;">
                            <li>Patient Expression: <b>{gene_driver.expression_value:.2f}</b></li>
                            <li>Cohort Median: <b>{gene_driver.cohort_median:.2f}</b></li>
                            <li>Prognostic Type: <b style="text-transform: uppercase;">{gene_driver.direction}</b></li>
                            <li>Risk Influence: <b style="color: {impact_color}; text-transform: uppercase;">{gene_driver.impact}</b></li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            st.markdown(f"<div style='font-size: 11px; text-transform: uppercase; color: #6b6b72; font-weight: 700; margin-bottom: 10px; margin-top: 15px;'>Identified Candidates for {selected_gene}</div>", unsafe_allow_html=True)
            for d in gene_drugs[:4]:
                mod_color = "#22c55e" if "inhibitor" in str(d.interaction_type).lower() or "antagonist" in str(d.interaction_type).lower() else "#00b8a0"
                st.markdown(
                    f"""
                    <div style="background-color: #18181c; border: 1px solid rgba(255,255,255,0.07); padding: 12px; border-radius: 4px; margin-bottom: 10px; font-size: 0.9rem;">
                        <div style="font-weight: 600; color: #e8e8ea;">{d.drug_name}</div>
                        <div style="font-size: 0.8rem; color: #6b6b72; margin-top: 4px;">
                            Modality: <span style="color: {mod_color}; text-transform: uppercase; font-weight: 600;">{d.interaction_type}</span> | Sources: {', '.join(d.sources)}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

import streamlit as st
from typing import Optional
from src.schemas import PredictionResult
from src.io.loaders import load_pathway_network
from src.bio.network import annotate_network_with_patient_drivers, render_network_plotly

def render_network_tab(result: Optional[PredictionResult]):
    """Render the pathway network visualization tab."""
    if result is None:
        st.info("Please load a profile and run analysis to view the regulatory network.")
        return
        
    st.markdown('<div class="section-label">PATHWAY REGULATORY NETWORK</div><div class="section-divider"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Load and render the network
        with st.spinner("Compiling network topology..."):
            network = load_pathway_network(result.cancer_type)
            annotated_net = annotate_network_with_patient_drivers(network, result.top_drivers)
            fig = render_network_plotly(annotated_net, result.top_drivers)
            st.plotly_chart(fig, use_container_width=True)
            
    with col2:
        st.markdown('<div class="section-label">LEGEND & BIOLOGY</div><div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Color Legend HTML Block (using the exact specification)
        st.markdown("""
        <div style="display:flex; flex-direction:column; gap:12px; margin-bottom:20px; background-color:#18181c; border:1px solid rgba(255,255,255,0.07); padding:16px; border-radius:4px;">
          <div style="font-size: 10px; text-transform: uppercase; color: #6b6b72; font-weight: 700; margin-bottom: 4px; letter-spacing: 0.05em;">Visual Code</div>
          <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:10px;height:10px;border-radius:50%;background:#ef4444;"></div>
            <span style="font-size:11px;color:#6b6b72;letter-spacing:0.06em;text-transform:uppercase;">Risk-increasing</span>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:10px;height:10px;border-radius:50%;background:#22c55e;"></div>
            <span style="font-size:11px;color:#6b6b72;letter-spacing:0.06em;text-transform:uppercase;">Risk-reducing</span>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:10px;height:10px;border-radius:50%;background:#00b8a0;border:2px solid #00b8a0;"></div>
            <span style="font-size:11px;color:#6b6b72;letter-spacing:0.06em;text-transform:uppercase;">Active driver</span>
          </div>
          <div style="display:flex; align-items:center; gap:8px;">
            <div style="width:10px;height:10px;border-radius:50%;background:#3a3a42;"></div>
            <span style="font-size:11px;color:#6b6b72;letter-spacing:0.06em;text-transform:uppercase;">Pathway node</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Specific Biological Pathway Details
        if result.cancer_type == "KIRC":
            st.markdown(
                """
                **Tight Junction Pathway Regulatory Context:**
                - **Claudin family genes** (e.g. `CLDN1`, `CLDN2`) encode structural membrane proteins that form cell-to-cell barriers in epithelial tissues.
                - Higher expression of these barrier genes correlates with **favorable** survival outcomes in renal tumors by preventing cell migration.
                - Unfavorable transcription factors like **`DNMT3B`** repress the Tight Junction pathway, which degrades barrier integrity and triggers metastatic dissemination.
                """
            )
        else:
            st.markdown(
                """
                **Purine Metabolism & RNA Polymerase Regulatory Context:**
                - In LIHC, **Purine Metabolism** drives nucleotide synthesis, accelerating rapid cell division.
                - **RNA Polymerase** complexes control transcriptional throughput in hepatic cancer cells.
                - Regulators like **`TAF15`**, **`CHEK1`**, and **`PDCD6`** have the highest slope correlation with activities of both pathways, representing the major prognostic switches in HCC progression.
                """
            )
            
        # List patient's top driver nodes
        st.markdown("**Active Driver Nodes in this Case:**")
        active_drivers = [item.gene_symbol for item in result.top_drivers if abs(item.contribution) > 0.05]
        if active_drivers:
            st.markdown(", ".join([f"`{g}`" for g in active_drivers[:6]]))
        else:
            st.markdown("*No active risk drivers detected in this profile.*")
            
        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #6b6b72; margin-top: 15px; font-style: italic;">
                Tip: Hover over the network nodes to inspect patient-level expression levels and coefficients.
            </div>
            """,
            unsafe_allow_html=True
        )

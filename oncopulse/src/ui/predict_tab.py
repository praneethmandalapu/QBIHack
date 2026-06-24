import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional
from src.schemas import PredictionResult
from src.io.exporters import export_prediction_to_csv, export_prediction_to_json, export_prediction_to_markdown
from src.config import PROCESSED_DATA_DIR

def render_predict_tab(result: Optional[PredictionResult]):
    """Render the clinical prediction dashboard tab with Vercel-like high-fidelity spectrum cards."""
    if result is None:
        st.markdown(
            """
            <div class="onco-card">
                <h3 style="margin-top: 0; color: #00b8a0; font-family: 'Satoshi', sans-serif;">Clinical Intake Required</h3>
                <p style="color: #6b6b72; font-size: 0.95rem;">Please select a patient profile or upload a custom transcriptomic dataset in the sidebar, then click "Analyze Profile" to retrieve survival-risk metrics.</p>
                <div style="border-left: 3px solid #00b8a0; padding-left: 15px; margin: 20px 0; color: #6b6b72; font-size: 0.9rem;">
                    <strong>Input requirements:</strong>
                    <ul style="margin-top: 5px; padding-left: 18px; margin-bottom: 0;">
                        <li>Gene expression profile (CSV/TSV)</li>
                        <li>Minimally including columns: <code>gene_symbol</code> and <code>expression_value</code></li>
                        <li>Expression inputs will be aligned to curated confidence prognostic genes (CPGs)</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # Section header
    st.markdown('<div class="section-label">SURVIVAL RISK STRATIFICATION</div><div class="section-divider"></div>', unsafe_allow_html=True)
    
    prob = result.risk_probability
    risk_class = result.risk_class
    risk_class_lower = risk_class.lower()
    
    # Calculate alignment statistics
    total_genes = len(result.top_drivers)
    missing_count = len(result.missing_genes)
    present_genes = total_genes - missing_count
    coverage = result.feature_coverage
    
    # Render Unified Clinical Readout Card
    pin_left = prob * 100
    
    st.markdown(
        f"""
        <div class="onco-card">
            <div style="font-size: 11px; text-transform: uppercase; color: #6b6b72; font-weight: 700; margin-bottom: 15px; letter-spacing: 0.05em;">
                {result.cancer_type} Cohort · TCGA · Real Patient Assay
            </div>
            <div style="display: flex; align-items: baseline; gap: 20px; margin-bottom: 25px;">
                <span style="font-size: 3.2rem; font-weight: 700; color: #ffffff; letter-spacing: -0.03em; line-height: 1;">{prob:.1%}</span>
                <span class="status-indicator status-{risk_class_lower}">{risk_class} Risk</span>
            </div>
            
            <div class="spectrum-container">
                <div class="spectrum-bar">
                    <div class="spectrum-pin" style="left: {pin_left}%;"></div>
                </div>
                <div class="spectrum-labels">
                    <span>Low Risk (&lt;35%)</span>
                    <span>Intermediate (35% - 65%)</span>
                    <span>High Risk (&gt;65%)</span>
                </div>
            </div>
            
            <div style="font-size: 12px; color: #6b6b72; padding-top: 18px; border-top: 1px solid rgba(255,255,255,0.07); display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 15px;">
                <div>
                    <span style="display: block; font-weight: 600; color: #e8e8ea; margin-bottom: 2px;">Profile Coverage</span>
                    <span>{coverage:.1%} ({present_genes} / {total_genes} genes present in profile)</span>
                </div>
                <div>
                    <span style="display: block; font-weight: 600; color: #e8e8ea; margin-bottom: 2px;">Imputation Assay</span>
                    <span>{missing_count} genes imputed with cohort training medians</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Case metadata row (monochrome cards)
    st.markdown('<div class="section-label">COHORT CLINICAL METRICS</div><div class="section-divider"></div>', unsafe_allow_html=True)
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    top_driver_symbol = result.top_drivers[0].gene_symbol if result.top_drivers else "N/A"
    
    with m_col1:
        st.markdown(
            f"""
            <div class="onco-card" style="padding: 15px; text-align: center;">
                <span class="metric-label">Target Cohort</span>
                <div class="metric-value" style="font-size: 1.5rem; margin-top: 2px;">{result.cancer_type}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col2:
        st.markdown(
            f"""
            <div class="onco-card" style="padding: 15px; text-align: center;">
                <span class="metric-label">Triage Class</span>
                <div class="metric-value" style="font-size: 1.5rem; margin-top: 2px;">{result.risk_class}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col3:
        st.markdown(
            f"""
            <div class="onco-card" style="padding: 15px; text-align: center;">
                <span class="metric-label">Gene Alignment</span>
                <div class="metric-value" style="font-size: 1.5rem; margin-top: 2px;">{result.feature_coverage:.1%}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m_col4:
        st.markdown(
            f"""
            <div class="onco-card" style="padding: 15px; text-align: center;">
                <span class="metric-label">Primary Driver</span>
                <div class="metric-value" style="font-size: 1.5rem; margin-top: 2px;">{top_driver_symbol}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 3. Kaplan-Meier Survival Curve Visualization
    cohort_path = PROCESSED_DATA_DIR / f"{result.cancer_type.lower()}_cohort.csv"
    if cohort_path.exists():
        try:
            df_cohort = pd.read_csv(cohort_path)
            
            # Helper to compute KM curve
            def get_km_data(df_group):
                df_sorted = df_group.sort_values(by="survival_days")
                unique_times = df_sorted["survival_days"].unique()
                
                times = [0.0]
                survival = [1.0]
                s = 1.0
                n_at_risk = len(df_sorted)
                
                for t in unique_times:
                    d = df_sorted[(df_sorted["survival_days"] == t) & (df_sorted["vital_status"].isin(["DECEASED", "DEAD"]))].shape[0]
                    c = df_sorted[(df_sorted["survival_days"] == t) & (~df_sorted["vital_status"].isin(["DECEASED", "DEAD"]))].shape[0]
                    
                    if n_at_risk > 0:
                        s_next = s * (1.0 - d / n_at_risk)
                    else:
                        s_next = s
                    
                    times.append(t)
                    survival.append(s_next)
                    
                    s = s_next
                    n_at_risk -= (d + c)
                    
                return np.array(times) / 365.25, np.array(survival)
                
            low_risk_df = df_cohort[df_cohort["label"] == 0]
            high_risk_df = df_cohort[df_cohort["label"] == 1]
            
            t_low, s_low = get_km_data(low_risk_df)
            t_high, s_high = get_km_data(high_risk_df)
            
            med_low = next((t for t, s in zip(t_low, s_low) if s <= 0.5), None)
            med_high = next((t for t, s in zip(t_high, s_high) if s <= 0.5), None)
            
            st.markdown('<div class="section-label">COHORT CLINICAL SURVIVAL ESTIMATOR (KAPLAN-MEIER)</div><div class="section-divider"></div>', unsafe_allow_html=True)
            
            plot_col, info_col = st.columns([3, 1])
            
            with plot_col:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=t_low, y=s_low,
                    mode='lines',
                    line=dict(color='#22c55e', width=2.5, shape='hv'),
                    name='Cohort Low Risk Stratum'
                ))
                
                fig.add_trace(go.Scatter(
                    x=t_high, y=s_high,
                    mode='lines',
                    line=dict(color='#ef4444', width=2.5, shape='hv'),
                    name='Cohort High Risk Stratum'
                ))
                
                max_time = max(t_low.max(), t_high.max()) if len(t_low) and len(t_high) else 5.0
                fig.add_shape(type="line",
                    x0=0, y0=0.5, x1=max_time, y1=0.5,
                    line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dash")
                )
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family='Satoshi, sans-serif', color='#e8e8ea'),
                    xaxis=dict(
                        title=dict(text="Timeline (Years)", font=dict(color='#a1a1aa', size=11)),
                        tickfont=dict(color='#a1a1aa'),
                        gridcolor='rgba(255,255,255,0.05)',
                        zerolinecolor='rgba(255,255,255,0.1)'
                    ),
                    yaxis=dict(
                        title=dict(text="Overall Survival Probability", font=dict(color='#a1a1aa', size=11)),
                        tickfont=dict(color='#a1a1aa'),
                        gridcolor='rgba(255,255,255,0.05)',
                        range=[0, 1.05]
                    ),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99,
                        bgcolor="rgba(24, 24, 28, 0.8)",
                        bordercolor="rgba(255, 255, 255, 0.07)",
                        borderwidth=1,
                        font=dict(size=10)
                    ),
                    margin=dict(l=50, r=20, t=10, b=40),
                    height=320
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            with info_col:
                med_low_str = f"{med_low:.2f} yrs" if med_low is not None else "Not Reached"
                med_high_str = f"{med_high:.2f} yrs" if med_high is not None else "Not Reached"
                
                patient_badge = f"<span class='status-indicator status-{result.risk_class.lower()}' style='margin-bottom:0;'>{result.risk_class} Risk</span>"
                
                st.markdown(
                    f"""
                    <div class="onco-card" style="padding: 15px; height: 100%; display: flex; flex-direction: column; justify-content: center; gap: 12px;">
                        <div>
                            <span class="metric-label" style="font-size: 0.75rem;">Low-Risk Median Survival</span>
                            <div style="font-size: 1.4rem; font-weight: 700; color: #22c55e; margin-top:2px;">{med_low_str}</div>
                        </div>
                        <div style="border-top: 1px solid rgba(255,255,255,0.07); padding-top: 10px;">
                            <span class="metric-label" style="font-size: 0.75rem;">High-Risk Median Survival</span>
                            <div style="font-size: 1.4rem; font-weight: 700; color: #ef4444; margin-top:2px;">{med_high_str}</div>
                        </div>
                        <div style="border-top: 1px solid rgba(255,255,255,0.07); padding-top: 10px;">
                            <span class="metric-label" style="font-size: 0.75rem;">Patient Stratification</span>
                            <div style="margin-top:5px;">{patient_badge}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        except Exception as ex:
            st.error(f"Failed to generate Kaplan-Meier estimator: {ex}")
            
    # Export reports block
    st.markdown('<div class="section-label">EXPORT SUMMARY DATA</div><div class="section-divider"></div>', unsafe_allow_html=True)
    exp_col1, exp_col2, exp_col3 = st.columns(3)
    
    with exp_col1:
        csv_data = export_prediction_to_csv(result)
        st.download_button(
            label="Download CSV dataset",
            data=csv_data,
            file_name=f"oncopulse_{result.cancer_type.lower()}_report.csv",
            mime="text/csv",
            use_container_width=True
        )
    with exp_col2:
        json_data = export_prediction_to_json(result)
        st.download_button(
            label="Download JSON payload",
            data=json_data,
            file_name=f"oncopulse_{result.cancer_type.lower()}_payload.json",
            mime="application/json",
            use_container_width=True
        )
    with exp_col3:
        md_data = export_prediction_to_markdown(result)
        with st.expander("Copy Markdown report block", expanded=False):
            st.code(md_data, language="markdown")

import streamlit as st
import plotly.graph_objects as go
from typing import Optional
from src.schemas import PredictionResult

def render_explain_tab(result: Optional[PredictionResult]):
    """Render the risk driver explainability tab with interactive Plotly contributions chart, narrative summary, and details table."""
    if result is None:
        st.info("Please load a profile and run analysis to view explanation.")
        return
        
    st.markdown('<div class="section-label">MODEL DECISION DRIVER EXPLANATION</div><div class="section-divider"></div>', unsafe_allow_html=True)
    
    # 1. Clinical Narrative Block
    st.markdown('<div class="section-label">AI CLINICAL TRIAGE NARRATIVE</div><div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="onco-card" style="border-left: 4px solid #00b8a0; background-color: #18181c; border-top-left-radius: 0; border-bottom-left-radius: 0;">
            {result.narrative}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 2. Interactive Plotly Horizontal Bar Chart
    st.markdown('<div class="section-label">DRIVER GENE CONTRIBUTIONS</div><div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Extract data from top_drivers
    drivers = result.top_drivers[:10]  # Show top 10 for readability
    # Invert order for Plotly bottom-to-top rendering
    drivers = list(reversed(drivers))
    
    genes = [item.gene_symbol for item in drivers]
    contributions = [item.contribution for item in drivers]
    
    # Colors: Red for positive (risk increasing), Green for negative (risk reducing)
    colors = ["#ef4444" if val > 0 else "#22c55e" for val in contributions]
    
    hover_texts = []
    for item in drivers:
        hover_texts.append(
            f"<b>Gene:</b> {item.gene_symbol}<br>"
            f"<b>Expression:</b> {item.expression_value:.2f} (median: {item.cohort_median:.2f})<br>"
            f"<b>Coefficient:</b> {item.coefficient:.3f}<br>"
            f"<b>Contribution:</b> {item.contribution:+.3f}<br>"
            f"<b>Impact:</b> {item.impact.upper()}"
        )
        
    fig = go.Figure(
        data=[
            go.Bar(
                x=contributions,
                y=genes,
                orientation='h',
                marker_color=colors,
                hovertext=hover_texts,
                hoverinfo='text'
            )
        ]
    )
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Satoshi, sans-serif', color='#e8e8ea'),
        xaxis=dict(
            title=dict(text="Contribution Score (Log-Odds Impact)", font=dict(color='#6b6b72', size=12)),
            tickfont=dict(color='#6b6b72'),
            gridcolor='rgba(255,255,255,0.05)',
            zerolinecolor='rgba(255,255,255,0.1)',
            zerolinewidth=2
        ),
        yaxis=dict(
            tickfont=dict(color='#e8e8ea', size=11),
            gridcolor='rgba(255,255,255,0.05)'
        ),
        margin=dict(l=100, r=20, t=20, b=50),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. Comprehensive Gene Features Table
    st.markdown('<div class="section-label">GENE-LEVEL FEATURE BREAKDOWN</div><div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Create manual HTML table with custom CSS styling
    table_rows = ""
    for item in result.top_drivers:
        badge_style = ""
        if item.impact == "increased risk":
            badge_style = "color: #ef4444; font-weight: bold;"
        elif item.impact == "decreased risk":
            badge_style = "color: #22c55e; font-weight: bold;"
        else:
            badge_style = "color: #6b6b72;"
            
        direction_style = "color: #22c55e;" if item.direction == "favorable" else "color: #ef4444;"
        
        table_rows += f"""
        <tr>
            <td style="font-weight: 600;">{item.gene_symbol}</td>
            <td>{item.expression_value:.2f}</td>
            <td>{item.cohort_median:.2f}</td>
            <td style="{direction_style}">{item.direction.upper()}</td>
            <td style="font-weight: 500;">{item.contribution:+.3f}</td>
            <td style="{badge_style}">{item.impact.upper()}</td>
        </tr>
        """
        
    st.markdown(
        f"""
        <table class="scientific-table">
            <thead>
                <tr>
                    <th>Gene Symbol</th>
                    <th>Patient Expression</th>
                    <th>Cohort Median</th>
                    <th>Prognostic Type</th>
                    <th>Log-Odds Contribution</th>
                    <th>Risk Impact</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        """,
        unsafe_allow_html=True
    )

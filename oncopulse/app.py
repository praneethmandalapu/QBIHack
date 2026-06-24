import streamlit as st
import pandas as pd
from datetime import datetime

# Set wide page layout as first Streamlit command
st.set_page_config(
    page_title="OncoPulse — Survival Risk Stratification",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.ui.layout import apply_custom_css
from src.io.loaders import load_demo_profiles
from src.io.validators import validate_profile_dataframe, normalize_gene_profile
from src.models.predict import predict_profile
from src.models.explain import explain_prediction, generate_narrative_explanation
from src.bio.drug_lookup import get_drug_interactions, rank_drug_interactions
from src.schemas import PredictionResult

# Import tab modules
from src.ui.predict_tab import render_predict_tab
from src.ui.explain_tab import render_explain_tab
from src.ui.targets_tab import render_targets_tab
from src.ui.network_tab import render_network_tab
from src.ui.about_tab import render_about_tab

# Initialize session state
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "raw_profile" not in st.session_state:
    st.session_state.raw_profile = None
if "cancer_type" not in st.session_state:
    st.session_state.cancer_type = "KIRC"

# Apply premium clinical custom CSS
apply_custom_css()

# Header Area (Logo mark + wordmark + subtitle)
logo_svg_html = """
<div style="display: flex; align-items: center; margin-bottom: 2px; padding-top: 10px;">
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin-right: 12px;">
        <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12" stroke="#00b8a0" stroke-width="2" stroke-linecap="round"/>
        <path d="M7 12H10L12 6L14 18L16 12H19" stroke="#e8e8ea" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    </svg>
    <span style="font-family: 'Satoshi', sans-serif; font-weight: 600; font-size: 1.8rem; color: #ffffff; letter-spacing: -0.02em;">OncoPulse</span>
</div>
<div style="font-family: 'Satoshi', sans-serif; font-size: 0.95rem; color: #6b6b72; margin-bottom: 25px; margin-top: 2px;">
    Survival risk stratification · KIRC & LIHC
</div>
"""
st.markdown(logo_svg_html, unsafe_allow_html=True)

# Load demo profiles
demo_profiles = load_demo_profiles()

# Sidebar Layout
with st.sidebar:
    # Segmented label & hairline divider
    st.markdown('<div class="section-label">Cohort Intake</div><div class="section-divider"></div>', unsafe_allow_html=True)
    
    # 1. Cancer Cohort Selection
    cohort = st.radio(
        "COHORT",
        ["KIRC", "LIHC"],
        horizontal=True,
        label_visibility="visible"
    )
    
    # Track cohort changes
    if cohort != st.session_state.cancer_type:
        st.session_state.cancer_type = cohort
        st.session_state.analysis_result = None
        st.session_state.raw_profile = None
        
    st.markdown('<div class="section-label">Profile Source</div><div class="section-divider"></div>', unsafe_allow_html=True)
    
    # 2. Input Mode Selector
    input_mode = st.radio(
        "Input Mode Select",
        ["Demo Datasets", "Upload Expression Profile"],
        index=0,
        label_visibility="collapsed"
    )
    
    selected_profile_df = None
    
    if input_mode == "Demo Datasets":
        options = {
            "● Low Risk Cohort Profile": "low_risk",
            "● High Risk Cohort Profile": "high_risk"
        }
        selected_demo_key = st.selectbox("Cohort Profile Selected", list(options.keys()), label_visibility="collapsed")
        risk_key = options[selected_demo_key]
        
        # Load profile
        expressions = demo_profiles.get(cohort, {}).get(risk_key, [])
        if expressions:
            records = [{"gene_symbol": e.gene_symbol, "expression_value": e.expression_value} for e in expressions]
            selected_profile_df = pd.DataFrame(records)
            st.session_state.raw_profile = selected_profile_df
            
    else:
        uploaded_file = st.file_uploader(
            "Upload Gene Expression Profile",
            type=["csv", "tsv", "txt"],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            try:
                sep = "\t" if (uploaded_file.name.endswith(".tsv") or uploaded_file.name.endswith(".txt")) else ","
                df = pd.read_csv(uploaded_file, sep=sep)
                
                is_valid, err_msg = validate_profile_dataframe(df)
                if is_valid:
                    selected_profile_df = df
                    st.session_state.raw_profile = selected_profile_df
                else:
                    st.error(f"Validation Error: {err_msg}")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                
    st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
    
    # 3. Dominant CTA Button (Teal style, no emoji)
    run_btn = st.button(
        "Analyze Profile",
        use_container_width=True
    )
    
if run_btn:
    if selected_profile_df is not None:
        with st.spinner("Processing expression profile..."):
            # 1. Normalize
            clean_df = normalize_gene_profile(selected_profile_df)
            
            # 2. Predict risk
            prob, risk_class, coverage, missing, model_details = predict_profile(clean_df, cohort)
            
            # 3. Compute driver explanations
            explanation_items = explain_prediction(model_details, model_details["feature_vector"], model_details["feature_vector_scaled"])
            
            # 4. Generate narrative
            narrative = generate_narrative_explanation(cohort, prob, risk_class, explanation_items)
            
            # 5. Fetch drug interactions
            top_genes = [item.gene_symbol for item in explanation_items[:5]]
            raw_drugs = get_drug_interactions(top_genes)
            ranked_drugs = rank_drug_interactions(raw_drugs, explanation_items)
            
            # 6. Save result payload (no success banner rendered)
            st.session_state.analysis_result = PredictionResult(
                cancer_type=cohort,
                risk_probability=prob,
                risk_class=risk_class,
                feature_coverage=coverage,
                missing_genes=missing,
                top_drivers=explanation_items,
                narrative=narrative,
                drug_targets=ranked_drugs,
                timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            )
    else:
        st.sidebar.error("Load a valid profile before analyzing.")

# Main content tabs (No emojis, Tracked caps)
tab_predict, tab_explain, tab_targets, tab_network, tab_method = st.tabs([
    "PREDICT", 
    "EXPLAIN", 
    "TARGETS", 
    "NETWORK", 
    "METHOD"
])

result = st.session_state.analysis_result

# Render tabs
with tab_predict:
    render_predict_tab(result)
with tab_explain:
    render_explain_tab(result)
with tab_targets:
    render_targets_tab(result)
with tab_network:
    render_network_tab(result)
with tab_method:
    render_about_tab(result)

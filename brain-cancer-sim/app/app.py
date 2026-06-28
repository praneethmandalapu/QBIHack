"""Main Streamlit shell for the brain cancer simulation app."""

import streamlit as st

from tabs import explain_tab, predict_tab, simulate_tab

st.set_page_config(page_title="Brain Cancer Sim", layout="wide")
st.title("Brain Tumor Growth Simulation")

tab_predict, tab_explain, tab_simulate = st.tabs(["Predict", "Explain", "Simulate"])

with tab_predict:
    predict_tab.render()

with tab_explain:
    explain_tab.render()

with tab_simulate:
    simulate_tab.render()

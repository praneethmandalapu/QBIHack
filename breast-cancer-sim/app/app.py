"""Main Streamlit shell for the breast cancer simulation app."""

import streamlit as st

from tabs import explain_tab, predict_tab, simulate_tab

st.set_page_config(page_title="Breast Cancer Sim", layout="wide")
st.title("Breast Cancer Risk & Growth Simulation")

tab_predict, tab_explain, tab_simulate = st.tabs(["Predict", "Explain", "Simulate"])

with tab_predict:
    predict_tab.render()

with tab_explain:
    explain_tab.render()

with tab_simulate:
    simulate_tab.render()

"""Streamlit tab: SHAP-based model explanations."""

import streamlit as st


def render() -> None:
    st.header("Model Explanation")
    st.info("SHAP feature attributions for the selected patient.")

"""Streamlit tab: model explanations."""

import streamlit as st


def render() -> None:
    st.header("Model Explanation")
    st.info("Feature attributions for the selected glioma patient.")

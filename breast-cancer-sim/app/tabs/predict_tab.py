"""Streamlit tab: genomic/clinical risk prediction."""

import streamlit as st


def render() -> None:
    st.header("Risk Prediction")
    st.info("Upload METABRIC-style features or use demo cohort.")

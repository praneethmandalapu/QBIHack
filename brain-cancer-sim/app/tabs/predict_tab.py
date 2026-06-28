"""Streamlit tab: molecular / clinical risk prediction."""

import streamlit as st


def render() -> None:
    st.header("Risk Prediction")
    st.info("IDH / MGMT / grade-based risk — wire to genomics model in Phase 2.")

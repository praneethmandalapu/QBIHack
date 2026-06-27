"""Streamlit tab: 3D tumor growth simulation and interventions."""

import streamlit as st


def render() -> None:
    st.header("Tumor Simulation")
    st.info("Load TCIA volume, run PDE growth, and apply drug interventions.")

"""Streamlit tab: 3D tumor growth simulation."""

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "simulation"))
sys.path.insert(0, str(ROOT / "simulation" / "solver"))
sys.path.insert(0, str(ROOT / "visualization"))

from render_3d import downsample, growth_analytics, render_growth_curve, render_sequence  # noqa: E402
from run_growth import run_growth  # noqa: E402
from tumor_pde_solver import dummy_volume  # noqa: E402


def render() -> None:
    st.header("Tumor Simulation")
    st.caption("Dummy volume demo until imaging pipeline delivers real glioma data.")

    timesteps = st.slider("Timesteps", min_value=10, max_value=60, value=30)
    if st.button("Run simulation"):
        vol = dummy_volume(shape=(40, 40, 40))
        frames = run_growth(vol, params={"risk_multiplier": 1.2}, timesteps=timesteps)
        ds = [downsample(f, 2) for f in frames]
        analytics = growth_analytics(ds, days_per_step=7.0)

        st.plotly_chart(render_sequence(ds), use_container_width=True)
        st.plotly_chart(render_growth_curve(analytics), use_container_width=True)
        st.metric("RECIST", analytics["recist"])
        st.metric("Volume change", f"{analytics['diameter_change_pct']:+.0f}%")

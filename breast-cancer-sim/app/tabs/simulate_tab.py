"""Streamlit tab: breast tumor growth simulation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT / "simulation-vinesh-philip-chandan"
VINESH = HANDOFF / "vinesh"
VIZ = ROOT / "visualization-jasim"
sys.path.insert(0, str(HANDOFF))
sys.path.insert(0, str(VINESH))
sys.path.insert(0, str(VIZ))

from growth_interventions import apply_drug  # noqa: E402
from render_3d import downsample, render_sequence  # noqa: E402
from run_growth import run_growth  # noqa: E402

MANIFEST_PATH = ROOT / "data" / "processed" / "raw-extract-philip-chandan" / "manifest.json"


def _load_cases() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cases = []
    for entry in manifest.get("volumes", []):
        if entry.get("timepoint") != "baseline":
            continue
        path = ROOT / entry["pde_npy"]
        if path.exists():
            cases.append({**entry, "pde_path": path})
    return cases


@st.cache_data(show_spinner=False)
def _simulate(
    baseline_path: str,
    risk_multiplier: float,
    drug: str,
    dose: float,
    timesteps: int,
    iso: float,
) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    baseline = np.load(baseline_path).astype(np.float32)
    params = apply_drug(baseline, drug, dose, params={"risk_multiplier": risk_multiplier})
    frames = run_growth(baseline, params=params, timesteps=timesteps)
    display_frames = [downsample(frame, 2) for frame in frames]
    burden = np.array([float(frame.sum()) for frame in frames])
    volume = np.array([float((frame >= iso).sum()) for frame in frames])
    analytics = {
        "burden": burden,
        "volume": volume,
        "change_pct": float(100.0 * (burden[-1] - burden[0]) / burden[0]) if burden[0] else 0.0,
    }
    return frames, display_frames, analytics


def _render_curve(analytics: dict) -> go.Figure:
    steps = np.arange(len(analytics["burden"]))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=analytics["burden"],
            mode="lines",
            name="Integrated burden",
            line=dict(color="#00b8a0", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=analytics["volume"],
            mode="lines",
            name="Volume at iso",
            line=dict(color="#f59e0b", width=2, dash="dot"),
            yaxis="y2",
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=8, r=8, t=24, b=8),
        paper_bgcolor="#111113",
        plot_bgcolor="#111113",
        font=dict(color="#e8e8ea"),
        xaxis=dict(title="step", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="burden", gridcolor="rgba(255,255,255,0.08)"),
        yaxis2=dict(title="voxels", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    return fig


def render() -> None:
    st.header("Tumor Simulation")
    cases = _load_cases()
    if not cases:
        st.warning("No breast PDE baseline volumes found in data/processed/pde-input-vinesh.")
        return

    labels = [f"{case['subtype']} | {case['tcga_id']}" for case in cases]
    selected = st.selectbox("Case", labels, index=0)
    case = cases[labels.index(selected)]

    defaults = {"Luminal A": 0.7, "Basal-like": 1.6}
    subtype = case.get("subtype", "")
    default_risk = defaults.get(subtype, 1.2)

    c1, c2, c3, c4 = st.columns(4)
    risk = c1.slider("Risk", 0.3, 2.5, float(default_risk), 0.1)
    drug = c2.selectbox("Intervention", ["none", "hormone", "chemo", "radiation"])
    dose = c3.slider("Dose", 0.0, 1.0, 0.0, 0.1)
    timesteps = c4.slider("Steps", 10, 80, 50, 5)
    iso = st.slider("Iso", 0.05, 0.4, 0.15, 0.01)

    if st.button("Run growth", type="primary"):
        with st.spinner("Running PDE growth..."):
            frames, display_frames, analytics = _simulate(
                str(case["pde_path"]),
                risk,
                drug,
                dose,
                timesteps,
                iso,
            )

        m1, m2, m3 = st.columns(3)
        m1.metric("Initial burden", f"{analytics['burden'][0]:,.0f}")
        m2.metric("Final burden", f"{analytics['burden'][-1]:,.0f}")
        m3.metric("Change", f"{analytics['change_pct']:+.0f}%")

        st.plotly_chart(render_sequence(display_frames, iso=iso), use_container_width=True)
        st.plotly_chart(_render_curve(analytics), use_container_width=True)

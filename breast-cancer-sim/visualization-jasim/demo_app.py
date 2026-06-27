"""OncoPulse — 3D Tumor Growth Digital Twin (Person 3 / jasim).

A self-contained clinical dashboard for the visualization layer. Runs on
synthetic data today; swap make_*_sequence() for Person 2's solve_growth()
and nothing else changes.

    cd breast-cancer-sim/visualization-jasim
    streamlit run demo_app.py
"""

from __future__ import annotations

import numpy as np
import streamlit as st

import color_maps as cm
from render_3d import (
    downsample,
    growth_analytics,
    make_dummy_sequence,
    make_treatment_sequence,
    render_cutaway,
    render_growth_curve,
    render_layers,
    render_sequence,
    render_slices,
    render_volumetric,
)

st.set_page_config(page_title="OncoPulse · Tumor Digital Twin",
                   page_icon="🧬", layout="wide")

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #
st.markdown(f"""
<style>
  .stApp {{ background:
      radial-gradient(1200px 600px at 80% -10%, #142c4d 0%, {cm.BG_DEEP} 55%);
      color: {cm.TEXT}; }}
  #MainMenu, footer {{ visibility: hidden; }}
  .hero {{ padding: 18px 22px; border-radius: 16px; margin-bottom: 14px;
      background: linear-gradient(100deg, rgba(34,211,238,0.14), rgba(168,85,247,0.12));
      border: 1px solid rgba(34,211,238,0.25); }}
  .hero h1 {{ margin:0; font-size: 28px; letter-spacing:.3px;
      background: linear-gradient(90deg,{cm.ACCENT},{cm.ACCENT_2});
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
  .hero p {{ margin:4px 0 0; color:{cm.TEXT_DIM}; font-size:13px; }}
  .kpi {{ background:{cm.BG_PANEL}; border:1px solid {cm.GRID};
      border-radius:14px; padding:14px 16px; height:100%; }}
  .kpi .lab {{ color:{cm.TEXT_DIM}; font-size:12px; text-transform:uppercase;
      letter-spacing:.6px; }}
  .kpi .val {{ font-size:26px; font-weight:700; margin-top:4px; }}
  .kpi .sub {{ font-size:12px; margin-top:2px; }}
  .pill {{ display:inline-block; padding:3px 10px; border-radius:999px;
      font-size:12px; font-weight:600; }}
  [data-testid="stSidebar"] {{ background:{cm.BG_PANEL}; }}
  .block-container {{ padding-top: 1.2rem; }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🧬 OncoPulse — 3D Tumor Growth Digital Twin</h1>
  <p>Patient-specific reaction–diffusion tumor evolution · volumetric rendering ·
     quantitative burden tracking &amp; RECIST response — visualization layer (Person 3).</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚙️ Scenario")
    scenario = st.radio("Clinical scenario", ["Untreated growth", "On therapy"], index=1)
    grid = st.select_slider("Grid resolution", [32, 48, 64], value=64)
    steps = st.slider("Timesteps", 8, 36, 28)
    spacing_mm = st.slider("Voxel spacing (mm)", 0.5, 2.0, 1.0, 0.1)
    days_per_step = st.slider("Days per timestep", 1, 14, 7)

    therapy_start, response = 12, 0.6
    if scenario == "On therapy":
        therapy_start = st.slider("Therapy starts at step", 4, steps - 2, min(12, steps - 4))
        response = st.slider("Therapy response", 0.1, 1.0, 0.6, 0.05)

    st.markdown("### 🎛️ Render mode")
    mode = st.selectbox("3D view",
                        ["Volumetric (hero)", "Tissue layers", "Cutaway core",
                         "Animated growth"])
    factor = st.select_slider("Display downsample", [1, 2, 4], value=2)
    st.caption("Synthetic data — swap for `solve_growth()` when Person 2 ships.")


@st.cache_data(show_spinner="Simulating tumor evolution…")
def build(scenario, grid, steps, therapy_start, response, factor):
    shape = (grid, grid, grid)
    if scenario == "On therapy":
        seq = make_treatment_sequence(n=steps, shape=shape,
                                      therapy_start=therapy_start, response=response)
    else:
        seq = make_dummy_sequence(n=steps, shape=shape)
    disp = [downsample(f, factor) for f in seq]
    return seq, disp


full_seq, disp_seq = build(scenario, grid, steps, therapy_start, response, factor)
spacing = (spacing_mm, spacing_mm, spacing_mm)
analytics = growth_analytics(full_seq, spacing=spacing, days_per_step=days_per_step)

# selected timestep for the static hero views
peak = analytics["peak_index"]
t_idx = st.slider("Timestep to inspect", 0, len(disp_seq) - 1,
                  len(disp_seq) - 1, help="Drives the 3D hero view, slices, and metrics.")
frame_full = full_seq[t_idx]
frame_disp = disp_seq[t_idx]
m = analytics["series"][t_idx]
base_vol = analytics["volume_mm3"][0] or 1.0

# --------------------------------------------------------------------------- #
# KPI row
# --------------------------------------------------------------------------- #
recist = analytics["recist"]
recist_color = {"Progressive Disease": cm.BAD, "Stable Disease": cm.PROLIFERATING,
                "Partial Response": cm.GOOD, "Complete Response": cm.GOOD}.get(recist, cm.TEXT)
vol_delta = (m["total_mm3"] - base_vol) / base_vol * 100
dt = analytics["doubling_time_days"]


def kpi(col, label, value, sub="", sub_color=cm.TEXT_DIM, val_color=cm.TEXT):
    col.markdown(f"""<div class="kpi"><div class="lab">{label}</div>
      <div class="val" style="color:{val_color}">{value}</div>
      <div class="sub" style="color:{sub_color}">{sub}</div></div>""",
                 unsafe_allow_html=True)


c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Tumor volume", f"{m['total_mm3']/1000:.2f} cm³",
    f"{vol_delta:+.0f}% vs baseline", cm.BAD if vol_delta > 0 else cm.GOOD)
kpi(c2, "Max diameter", f"{m['max_diameter_mm']:.1f} mm",
    f"{analytics['diameter_change_pct']:+.0f}% overall")
kpi(c3, "Necrotic fraction", f"{m['necrotic_fraction']*100:.0f}%",
    "dead core / viable", cm.HYPOXIC)
kpi(c4, "Volume doubling", "—" if np.isnan(dt) else f"{dt:.0f} d",
    f"<span class='pill' style='background:{recist_color}22;color:{recist_color}'>{recist}</span>",
    cm.TEXT_DIM)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Main: 3D view + growth curve
# --------------------------------------------------------------------------- #
left, right = st.columns([1.35, 1])

with left:
    st.markdown(f"##### 🫀 3D tumor — *step {t_idx} · day {t_idx*days_per_step}*")
    if mode == "Volumetric (hero)":
        fig3d = render_volumetric(frame_disp)
    elif mode == "Tissue layers":
        fig3d = render_layers(frame_disp)
    elif mode == "Cutaway core":
        fig3d = render_cutaway(frame_disp)
    else:
        fig3d = render_sequence(disp_seq)
    st.plotly_chart(fig3d, use_container_width=True,
                    config={"displayModeBar": False})

with right:
    st.markdown("##### 📈 Tumor burden over time")
    st.plotly_chart(render_growth_curve(analytics), use_container_width=True,
                    config={"displayModeBar": False})
    st.markdown("##### 🧠 Clinical read")
    trend = "regressing under therapy" if vol_delta < 0 else "progressing"
    st.markdown(
        f"<div class='kpi'>Lesion is <b style='color:{recist_color}'>{recist}</b> "
        f"({trend}). Necrotic core now <b>{m['necrotic_fraction']*100:.0f}%</b> "
        f"of viable mass; longest axis <b>{m['longest_axis_mm']:.1f} mm</b>. "
        f"{'Doubling time ' + format(dt, '.0f') + ' days.' if not np.isnan(dt) else ''}"
        "</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Radiology slices
# --------------------------------------------------------------------------- #
st.markdown("##### 🩻 Radiology slices — orthogonal mid-planes")
st.plotly_chart(render_slices(frame_full, spacing), use_container_width=True,
                config={"displayModeBar": False})

st.caption("OncoPulse · visualization layer · Plotly + Streamlit · "
           "data contract: float32 (D,H,W) density ∈ [0,1] from solve_growth().")

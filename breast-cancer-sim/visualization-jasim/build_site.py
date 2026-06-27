"""Build the OncoPulse bespoke showcase site (Person 3 / jasim).

Embeds the REAL Plotly figures from render_3d.py into a hand-designed,
award-level static page. Run:

    ../.venv/Scripts/python.exe build_site.py
    ../.venv/Scripts/python.exe -m http.server 8080 --directory site
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import render_3d as r  # noqa: E402

SITE = Path(__file__).parent / "site"
SITE.mkdir(exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. Simulate + analyze (treatment scenario: grow, then respond)
# --------------------------------------------------------------------------- #
SEQ = r.make_treatment_sequence(28, (64, 64, 64), therapy_start=12, response=0.6)
A = r.growth_analytics(SEQ, spacing=(1.0, 1.0, 1.0), days_per_step=7)
PK = A["peak_index"]
DISP = [r.downsample(f, 2) for f in SEQ]
peak_m = A["series"][PK]


def theme(fig, three_d: bool = True):
    """Make a figure sit on the bespoke dark scene (transparent, dimmed grids)."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9aa6b6", family="JetBrains Mono, monospace"),
        margin=dict(l=0, r=0, t=8, b=0),
    )
    if three_d:
        fig.update_scenes(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showbackground=False, gridcolor="rgba(236,231,223,0.06)",
                       zerolinecolor="rgba(236,231,223,0.10)", color="#5d6470"),
            yaxis=dict(showbackground=False, gridcolor="rgba(236,231,223,0.06)",
                       zerolinecolor="rgba(236,231,223,0.10)", color="#5d6470"),
            zaxis=dict(showbackground=False, gridcolor="rgba(236,231,223,0.06)",
                       zerolinecolor="rgba(236,231,223,0.10)", color="#5d6470"),
        )
    else:
        fig.update_xaxes(gridcolor="rgba(236,231,223,0.06)", zerolinecolor="rgba(236,231,223,0.10)")
        fig.update_yaxes(gridcolor="rgba(236,231,223,0.06)", zerolinecolor="rgba(236,231,223,0.10)")
    return fig


FIGS = {
    "base_iso": theme(r._render_isosurface(DISP[0], r.ISO_LEVEL)),  # time-scrubbed view
    "volumetric": theme(r.render_volumetric(DISP[PK])),
    "layers": theme(r.render_layers(DISP[PK])),
    "cutaway": theme(r.render_cutaway(DISP[PK])),
    "growth": theme(r.render_growth_curve(A), three_d=False),
    "slices": theme(r.render_slices(SEQ[PK], (1.0, 1.0, 1.0)), three_d=False),
}

# Per-frame data for the timeline: only the `value` field changes between
# frames (x/y/z grid is shared by the base isosurface), so this stays compact.
VALUES = [np.round(d.ravel(), 2).tolist() for d in DISP]
SERIES = [{
    "day": i * 7,
    "vol": round(m["total_mm3"] / 1000, 2),
    "diam": round(m["max_diameter_mm"], 1),
    "necro": round(m["necrotic_fraction"] * 100),
} for i, m in enumerate(A["series"])]
MAXVOL_MM3 = max(m["total_mm3"] for m in A["series"])

NUMS = {
    "peak_vol_cm3": round(peak_m["total_mm3"] / 1000, 2),
    "max_diam_mm": round(peak_m["max_diameter_mm"], 1),
    "necrotic_pct": round(peak_m["necrotic_fraction"] * 100),
    "doubling_d": None if peak_m is None else (
        None if (A["doubling_time_days"] != A["doubling_time_days"]) else round(A["doubling_time_days"])),
    "recist": A["recist"],
    "diam_change": round(A["diameter_change_pct"]),
    "day_peak": int(PK * 7),
    "n_steps": len(SEQ),
    "voxels": int(64 ** 3),
}

# --------------------------------------------------------------------------- #
# 2. HTML template (tokens replaced below — kept out of f-strings on purpose)
# --------------------------------------------------------------------------- #
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OncoPulse — Tumor Growth Digital Twin</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' fill='%230a0c10'/%3E%3Cpath d='M2 16h7l3-9 4 18 3-9h11' fill='none' stroke='%23ff7a45' stroke-width='2.4' stroke-linejoin='round' stroke-linecap='round'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
:root{
  --ink:#0a0c10; --ink-2:#0f131b; --panel:#10151f;
  --bone:#ece7df; --muted:#7d8696; --line:rgba(236,231,223,.10);
  --signal:#ff7a45; --signal-rgb:255,122,69; --crit:#ff3b54; --vital:#46e0b0;
  --font-d:'Space Grotesk',sans-serif; --font-m:'JetBrains Mono',monospace; --font-s:'Instrument Serif',serif;
  --ease:cubic-bezier(.23,1,.32,1); --ease-io:cubic-bezier(.65,.01,.05,.99);
  --step-0:clamp(1rem,.91rem+.43vw,1.15rem);
  --step-1:clamp(1.2rem,1.07rem+.63vw,1.5rem);
  --step-2:clamp(1.44rem,1.26rem+.89vw,1.95rem);
  --step-3:clamp(1.9rem,1.5rem+1.6vw,2.8rem);
  --s-l:clamp(2rem,1.8rem+.9vw,2.5rem); --s-xl:clamp(3rem,2.7rem+1.3vw,3.75rem);
  --s-2xl:clamp(4rem,3.6rem+1.7vw,5.5rem); --s-3xl:clamp(6rem,5.4rem+2.6vw,8rem);
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{
  background:var(--ink); color:var(--bone); font-family:var(--font-d);
  font-size:var(--step-0); line-height:1.6; overflow-x:hidden;
  -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale;
}
::selection{background:var(--signal); color:var(--ink)}
::-webkit-scrollbar{width:10px}
::-webkit-scrollbar-track{background:var(--ink)}
::-webkit-scrollbar-thumb{background:#1c2430;border-radius:0}
a{color:inherit;text-decoration:none}
.mono{font-family:var(--font-m)}
.over{font-family:var(--font-m); font-size:.74rem; letter-spacing:.32em;
  text-transform:uppercase; color:var(--muted)}
.wrap{max-width:1280px; margin:0 auto; padding:0 clamp(1.2rem,4vw,3rem)}

/* ambient radiance */
.glow{position:fixed; inset:-10% -10% auto -10%; height:70vh; z-index:0; pointer-events:none;
  background:radial-gradient(60% 70% at 70% 0%, rgba(var(--signal-rgb),.16) 0%, transparent 60%);}
.grain::after{content:'';position:fixed;inset:0;z-index:1;pointer-events:none;opacity:.5;mix-blend-mode:overlay;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='3'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.045'/%3E%3C/svg%3E");}

/* ---------- loader ---------- */
#loader{position:fixed;inset:0;z-index:9999;background:var(--ink);display:grid;place-items:center;
  transition:opacity .6s var(--ease),visibility .6s}
#loader.done{opacity:0;visibility:hidden}
#loader svg{width:min(60vw,420px)}
#loader path{stroke:var(--signal);stroke-width:2.5;fill:none;stroke-linecap:round;stroke-linejoin:round;
  stroke-dasharray:1400;stroke-dashoffset:1400;animation:draw 1.3s var(--ease-io) forwards}
.loader-tag{position:absolute;bottom:8vh;font-family:var(--font-m);font-size:.74rem;letter-spacing:.3em;color:var(--muted)}
@keyframes draw{to{stroke-dashoffset:0}}

/* ---------- nav ---------- */
nav{position:fixed;top:0;left:0;right:0;z-index:200;display:flex;align-items:center;justify-content:space-between;
  padding:.85rem clamp(1.2rem,4vw,3rem);
  background:rgba(10,12,16,.55);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:.6rem;font-weight:700;letter-spacing:-.02em;font-size:1.05rem}
.brand svg{width:30px;height:18px}
.brand b{color:var(--signal);font-weight:700}
.nlinks{display:flex;gap:2rem;align-items:center;font-family:var(--font-m);font-size:.78rem;letter-spacing:.08em}
.nlink{position:relative;color:var(--muted);text-transform:uppercase;transition:color .3s}
.nlink::after{content:'';position:absolute;left:0;bottom:-5px;width:100%;height:1px;background:var(--signal);
  transform:scaleX(0);transform-origin:right;transition:transform .35s var(--ease)}
.nlink:hover{color:var(--bone)} .nlink:hover::after{transform:scaleX(1);transform-origin:left}
.status{display:flex;align-items:center;gap:.5rem;font-family:var(--font-m);font-size:.72rem;
  color:var(--vital);letter-spacing:.12em}
.dot{width:7px;height:7px;border-radius:50%;background:var(--vital);box-shadow:0 0 0 0 rgba(70,224,176,.5);
  animation:ping 2s var(--ease-io) infinite}
@keyframes ping{0%{box-shadow:0 0 0 0 rgba(70,224,176,.45)}70%{box-shadow:0 0 0 9px rgba(70,224,176,0)}100%{box-shadow:0 0 0 0 rgba(70,224,176,0)}}
@media(max-width:780px){.nlinks{display:none}}

/* ---------- hero ---------- */
.hero{position:relative;z-index:2;min-height:100vh;display:flex;flex-direction:column;justify-content:center;
  padding-top:6rem;padding-bottom:3rem}
.hero .over{margin-bottom:1.4rem}
.title{position:relative;font-weight:700;letter-spacing:-.045em;line-height:.84;
  font-size:clamp(3.6rem,15vw,12rem);text-transform:uppercase}
.title .pulseword{color:var(--bone)}
.ecg{position:absolute;left:0;right:0;top:52%;height:120px;width:100%;z-index:-1;overflow:visible;pointer-events:none}
.ecg path{fill:none;stroke:var(--signal);stroke-width:2;opacity:.85;
  filter:drop-shadow(0 0 6px rgba(var(--signal-rgb),.6));
  stroke-dasharray:2600;stroke-dashoffset:2600;animation:trace 3.2s var(--ease-io) .8s forwards}
@keyframes trace{to{stroke-dashoffset:0}}
.lede{max-width:46rem;margin-top:1.8rem;color:#c4c1ba;font-size:var(--step-1);line-height:1.5;text-wrap:pretty}
.lede b{color:var(--bone);font-weight:600}
.cta-row{display:flex;gap:1rem;margin-top:2.2rem;flex-wrap:wrap}
.btn{font-family:var(--font-m);font-size:.82rem;letter-spacing:.06em;text-transform:uppercase;
  padding:.95rem 1.6rem;border:1px solid var(--bone);position:relative;overflow:hidden;z-index:1;
  transition:color .4s var(--ease);cursor:pointer;background:transparent;color:var(--bone)}
.btn::before{content:'';position:absolute;inset:0;background:var(--bone);transform:scaleX(0);transform-origin:left;
  transition:transform .4s var(--ease);z-index:-1}
.btn:hover{color:var(--ink)} .btn:hover::before{transform:scaleX(1)}
.btn--signal{border-color:var(--signal);color:var(--signal)}
.btn--signal::before{background:var(--signal)} .btn--signal:hover{color:var(--ink)}
.hero-ticker{margin-top:auto;border-top:1px solid var(--line);border-bottom:1px solid var(--line);
  display:flex;gap:3rem;padding:.9rem 0;font-family:var(--font-m);font-size:.78rem;color:var(--muted);
  overflow:hidden;mask-image:linear-gradient(90deg,transparent,#000 6%,#000 94%,transparent)}
.hero-ticker .row{display:flex;gap:3rem;white-space:nowrap;animation:tick 26s linear infinite}
.hero-ticker b{color:var(--signal)}

@keyframes tick{from{transform:translateX(0)}to{transform:translateX(-50%)}}

/* ---------- section scaffold ---------- */
section{position:relative;z-index:2}
.divider{display:flex;align-items:center;gap:1.2rem;padding:var(--s-2xl) 0 var(--s-l)}
.divider .num{font-family:var(--font-m);color:var(--signal);font-size:.8rem;letter-spacing:.2em}
.divider h2{font-size:var(--step-3);font-weight:600;letter-spacing:-.03em;line-height:1}
.divider .ln{flex:1;height:1px;background:var(--line)}
.reveal{opacity:0;transform:translateY(2.5rem);transition:opacity .9s var(--ease),transform .9s var(--ease)}
.reveal.vis{opacity:1;transform:none}

/* ---------- instrument cluster ---------- */
.cluster{display:grid;grid-template-columns:repeat(12,1fr);gap:1px;background:var(--line);border:1px solid var(--line)}
.cell{background:var(--ink);padding:1.5rem 1.4rem;display:flex;flex-direction:column;gap:.5rem}
.cell .lab{font-family:var(--font-m);font-size:.68rem;letter-spacing:.18em;text-transform:uppercase;color:var(--muted)}
.cell .val{font-family:var(--font-m);font-weight:700;font-size:clamp(1.7rem,4vw,2.6rem);letter-spacing:-.02em;line-height:1}
.cell .val .u{font-size:.9rem;color:var(--muted);margin-left:.25rem}
.cell .sub{font-family:var(--font-m);font-size:.72rem;color:var(--muted)}
.cell.c-wide{grid-column:span 4} .cell.c-mid{grid-column:span 4} .cell.c-twin{grid-column:span 2}
.up{color:var(--crit)} .down{color:var(--vital)}
.pill{display:inline-block;padding:.25rem .7rem;font-family:var(--font-m);font-size:.72rem;letter-spacing:.04em;
  border:1px solid currentColor;margin-top:.2rem;width:max-content}
@media(max-width:880px){.cell.c-wide,.cell.c-mid{grid-column:span 6}.cell.c-twin{grid-column:span 6}}

/* ---------- 3D + telemetry split ---------- */
.split{display:grid;grid-template-columns:1.55fr 1fr;gap:1.4rem;align-items:stretch}
@media(max-width:980px){.split{grid-template-columns:1fr}}
.panel{border:1px solid var(--line);background:linear-gradient(180deg,var(--panel),var(--ink));position:relative}
.panel .ph{display:flex;align-items:center;justify-content:space-between;padding:.9rem 1.1rem;border-bottom:1px solid var(--line)}
.panel .ph .t{font-family:var(--font-m);font-size:.74rem;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.modes{display:flex;gap:.4rem}
.mode{font-family:var(--font-m);font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
  padding:.35rem .7rem;border:1px solid var(--line);cursor:pointer;transition:all .3s var(--ease);background:transparent}
.mode:hover{color:var(--bone);border-color:var(--muted)}
.mode.on{color:var(--ink);background:var(--signal);border-color:var(--signal)}
.stage{position:relative;height:clamp(360px,52vh,560px)}
#viz3d{width:100%;height:100%}
.scan{position:absolute;inset:0;pointer-events:none;overflow:hidden;mix-blend-mode:screen}
.scan::before{content:'';position:absolute;left:0;right:0;height:34%;
  background:linear-gradient(180deg,transparent,rgba(var(--signal-rgb),.10),transparent);
  animation:sweep 4.5s var(--ease-io) infinite}
@keyframes sweep{0%{top:-34%}100%{top:100%}}
.reticle{position:absolute;top:.8rem;left:.8rem;font-family:var(--font-m);font-size:.66rem;color:var(--muted);
  letter-spacing:.1em;line-height:1.7}
.panel .cap{padding:.8rem 1.1rem;border-top:1px solid var(--line);font-family:var(--font-m);font-size:.72rem;color:var(--muted)}

/* ---------- timeline ---------- */
.timeline{display:flex;align-items:center;gap:1rem;padding:.85rem 1.1rem;border-top:1px solid var(--line)}
.play{flex:0 0 auto;width:38px;height:38px;border:1px solid var(--signal);background:transparent;color:var(--signal);
  font-family:var(--font-m);font-size:.85rem;cursor:pointer;display:grid;place-items:center;transition:all .3s var(--ease)}
.play:hover{background:var(--signal);color:var(--ink)} .play.on{background:var(--signal);color:var(--ink)}
.tl{flex:1;-webkit-appearance:none;appearance:none;height:2px;background:var(--line);outline:none;cursor:pointer}
.tl::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:14px;height:14px;border-radius:50%;
  background:var(--signal);box-shadow:0 0 0 4px rgba(var(--signal-rgb),.18);cursor:pointer;transition:box-shadow .2s}
.tl::-webkit-slider-thumb:hover{box-shadow:0 0 0 7px rgba(var(--signal-rgb),.25)}
.tl::-moz-range-thumb{width:14px;height:14px;border:none;border-radius:50%;background:var(--signal);cursor:pointer}
.tl::-moz-range-progress{background:var(--signal);height:2px}
.tlread{flex:0 0 auto;font-size:.74rem;color:var(--muted);letter-spacing:.06em;min-width:13rem;text-align:right}
.tlread b{color:var(--signal)}
@media(max-width:520px){.tlread{display:none}}
.tele-body{padding:1rem .8rem}
.read{padding:1.1rem;border-top:1px solid var(--line);font-size:.95rem;color:#c4c1ba;line-height:1.55}
.read b{color:var(--bone)} .read .crit{color:var(--crit)} .read .ok{color:var(--vital)}

/* ---------- radiology band ---------- */
.band{width:100vw;margin-left:calc(-50vw + 50%);background:var(--ink-2);border-top:1px solid var(--line);
  border-bottom:1px solid var(--line);padding:var(--s-2xl) 0;margin-top:var(--s-2xl)}
#slices{width:100%;height:300px}
.band .meta{display:flex;gap:2.5rem;flex-wrap:wrap;font-family:var(--font-m);font-size:.74rem;color:var(--muted);margin-top:1.2rem}
.band .meta b{color:var(--bone)}

/* ---------- method ---------- */
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:1.4rem}
@media(max-width:880px){.steps{grid-template-columns:1fr}}
.step{border-top:2px solid var(--signal);padding-top:1.1rem}
.step .n{font-family:var(--font-m);font-size:.8rem;color:var(--signal);letter-spacing:.2em}
.step h3{font-size:var(--step-1);font-weight:600;margin:.6rem 0 .5rem;letter-spacing:-.02em}
.step p{color:var(--muted);font-size:.95rem}
.pullquote{font-family:var(--font-s);font-style:italic;font-size:clamp(1.6rem,4.5vw,3rem);line-height:1.2;
  letter-spacing:-.01em;max-width:38rem;margin:var(--s-2xl) 0;color:var(--bone)}
.pullquote span{color:var(--signal)}

/* ---------- footer ---------- */
footer{position:relative;z-index:2;border-top:1px solid var(--line);margin-top:var(--s-3xl);
  padding:var(--s-2xl) 0 var(--s-l)}
.foot-mark{font-weight:700;letter-spacing:-.04em;line-height:.85;text-transform:uppercase;
  font-size:clamp(3rem,12vw,9rem);color:#161b25}
.foot-mark b{color:var(--signal)}
.credits{display:grid;grid-template-columns:repeat(5,1fr);gap:1rem;margin-top:2rem;
  font-family:var(--font-m);font-size:.74rem;color:var(--muted)}
.credits .role{color:var(--signal);letter-spacing:.1em;text-transform:uppercase;font-size:.66rem;margin-bottom:.3rem}
.credits b{color:var(--bone);font-weight:500}
.foot-base{display:flex;justify-content:space-between;flex-wrap:wrap;gap:1rem;margin-top:2.5rem;
  font-family:var(--font-m);font-size:.72rem;color:var(--muted)}
@media(max-width:780px){.credits{grid-template-columns:repeat(2,1fr)}}

@media (prefers-reduced-motion: reduce){
  *,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;
    transition-duration:.01ms!important;scroll-behavior:auto!important}
  .reveal{opacity:1;transform:none}
}
</style>
</head>
<body class="grain">
<div class="glow"></div>

<div id="loader">
  <svg viewBox="0 0 700 200" preserveAspectRatio="xMidYMid meet">
    <path d="M0 100 H180 l22 -64 l30 130 l26 -150 l24 168 l22 -84 H400 l18 -40 l20 80 H700"/>
  </svg>
  <div class="loader-tag">INITIALIZING DIGITAL TWIN…</div>
</div>

<nav>
  <a class="brand" href="#top">
    <svg viewBox="0 0 60 30"><path d="M2 15h14l4-11 6 24 5-26 5 28 4-15h13" fill="none" stroke="#ff7a45" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>
    Onco<b>Pulse</b>
  </a>
  <div class="nlinks">
    <a class="nlink" href="#vitals">Vitals</a>
    <a class="nlink" href="#twin">Digital Twin</a>
    <a class="nlink" href="#imaging">Imaging</a>
    <a class="nlink" href="#method">Method</a>
  </div>
  <div class="status"><span class="dot"></span>SIM&nbsp;LIVE</div>
</nav>

<main id="top" class="wrap">

  <!-- HERO -->
  <header class="hero">
    <div class="over">Patient-specific tumor growth digital twin · Case OP-0427</div>
    <h1 class="title">
      ONCO<span class="pulseword">PULSE</span>
      <svg class="ecg" viewBox="0 0 1200 120" preserveAspectRatio="none">
        <path d="M0 60 H320 l26 -44 l34 92 l30 -104 l28 116 l26 -56 H720 l22 -30 l24 60 H1200"/>
      </svg>
    </h1>
    <p class="lede">A reaction–diffusion model evolves a <b>real MRI-derived lesion</b> through
       __N_STEPS__ timesteps — then renders it as a living volumetric twin with quantified
       burden, necrotic-core dynamics, and <b>RECIST</b> response under therapy.</p>
    <div class="cta-row">
      <a class="btn btn--signal" href="#twin">Enter the twin →</a>
      <a class="btn" href="#method">How it works</a>
    </div>
    <div class="hero-ticker" aria-hidden="true">
      <div class="row">
        <span>MODALITY <b>DCE-MRI</b></span><span>SITE <b>L-BREAST · UOQ</b></span>
        <span>GRID <b>64³ VOXELS</b></span><span>PEAK Ø <b>__MAXD__ mm</b></span>
        <span>RESPONSE <b>__RECIST__</b></span><span>ENGINE <b>REACTION–DIFFUSION PDE</b></span>
        <span>MODALITY <b>DCE-MRI</b></span><span>SITE <b>L-BREAST · UOQ</b></span>
        <span>GRID <b>64³ VOXELS</b></span><span>PEAK Ø <b>__MAXD__ mm</b></span>
        <span>RESPONSE <b>__RECIST__</b></span><span>ENGINE <b>REACTION–DIFFUSION PDE</b></span>
      </div>
    </div>
  </header>

  <!-- VITALS -->
  <section id="vitals">
    <div class="divider reveal"><span class="num">01</span><h2>Live readout</h2><span class="ln"></span></div>
    <div class="cluster reveal">
      <div class="cell c-wide">
        <span class="lab">Peak tumor volume</span>
        <span class="val"><span class="cu" data-to="__PEAKVOL__" data-dec="2">0</span><span class="u">cm³</span></span>
        <span class="sub up">▲ __DIAMCH__% diameter vs baseline</span>
      </div>
      <div class="cell c-mid">
        <span class="lab">Max diameter</span>
        <span class="val"><span class="cu" data-to="__MAXD__" data-dec="1">0</span><span class="u">mm</span></span>
        <span class="sub">RECIST long-axis · day __DAYPEAK__</span>
      </div>
      <div class="cell c-twin">
        <span class="lab">Necrotic</span>
        <span class="val" style="color:var(--crit)"><span class="cu" data-to="__NECRO__" data-dec="0">0</span><span class="u">%</span></span>
        <span class="sub">dead core</span>
      </div>
      <div class="cell c-twin">
        <span class="lab">Doubling</span>
        <span class="val"><span class="cu" data-to="__DOUBLE__" data-dec="0">0</span><span class="u">d</span></span>
        <span class="sub">pre-therapy</span>
      </div>
      <div class="cell c-wide">
        <span class="lab">Response assessment</span>
        <span class="val" style="font-size:clamp(1.3rem,3vw,1.9rem);color:var(--vital)">__RECIST__</span>
        <span class="pill" style="color:var(--vital)">RECIST 1.1 · ON THERAPY</span>
      </div>
      <div class="cell c-mid">
        <span class="lab">Simulation</span>
        <span class="val"><span class="cu" data-to="__N_STEPS__" data-dec="0">0</span><span class="u">steps</span></span>
        <span class="sub">__VOXELS__ voxels / frame</span>
      </div>
      <div class="cell c-mid">
        <span class="lab">Engine</span>
        <span class="val" style="font-size:clamp(1.1rem,2.4vw,1.5rem)">PDE · 3D</span>
        <span class="sub">reaction–diffusion + drug term</span>
      </div>
    </div>
  </section>

  <!-- DIGITAL TWIN -->
  <section id="twin">
    <div class="divider reveal"><span class="num">02</span><h2>The digital twin</h2><span class="ln"></span></div>
    <div class="split">
      <div class="panel reveal">
        <div class="ph">
          <span class="t">3D lesion · time-resolved</span>
          <div class="modes">
            <button class="mode on" data-fig="growth">Growth</button>
            <button class="mode" data-fig="volumetric">Volumetric</button>
            <button class="mode" data-fig="layers">Layers</button>
            <button class="mode" data-fig="cutaway">Cutaway</button>
          </div>
        </div>
        <div class="stage">
          <div class="reticle">CASE OP-0427<br>DCE-MRI · T1<br>ISO 0.50</div>
          <div id="viz3d"></div>
          <div class="scan"></div>
        </div>
        <div class="timeline" id="timeline">
          <button class="play" id="play" aria-label="Play timeline">▶</button>
          <input class="tl" id="tl" type="range" min="0" max="__NFRAMES_MAX__" value="0" step="1" aria-label="Timestep">
          <div class="tlread" id="tlread">DAY 00 · <b>0.00 cm³</b></div>
        </div>
        <div class="cap">Scrub or play to evolve the lesion · drag to orbit · rim → viable → hypoxic → necrotic core</div>
      </div>
      <div class="panel reveal">
        <div class="ph"><span class="t">Tumor burden telemetry</span><span class="t" style="color:var(--signal)">mm³ / day</span></div>
        <div class="tele-body"><div id="growth" style="width:100%;height:300px"></div></div>
        <div class="read">
          Lesion is <b class="ok">__RECIST__</b>. After aggressive growth (<b>doubling __DOUBLE__ d</b>),
          therapy at day __DAYTHER__ drives regression — longest axis
          <b class="ok">__DIAMCH__%</b> from peak, necrotic core <b>__NECRO__%</b> of viable mass.
        </div>
      </div>
    </div>
  </section>

  <!-- IMAGING -->
  <section id="imaging">
    <div class="band">
      <div class="wrap">
        <div class="divider reveal" style="padding-top:0"><span class="num">03</span><h2>Radiology slices</h2><span class="ln"></span></div>
        <div class="reveal"><div id="slices"></div></div>
        <div class="meta reveal">
          <span>PLANE <b>AXIAL · CORONAL · SAGITTAL</b></span>
          <span>SOURCE <b>simulated density field</b></span>
          <span>WINDOW <b>0.00 – 1.00</b></span>
          <span>SPACING <b>1.0 mm³ / voxel</b></span>
        </div>
      </div>
    </div>
  </section>

  <!-- METHOD -->
  <section id="method">
    <div class="divider reveal"><span class="num">04</span><h2>From scan to simulation</h2><span class="ln"></span></div>
    <div class="steps">
      <div class="step reveal"><div class="n">01 · ACQUIRE</div><h3>MRI → volume</h3>
        <p>Matched TCGA-BRCA DICOM is segmented into a clean 64³ density field — the patient-specific starting state.</p></div>
      <div class="step reveal"><div class="n">02 · SIMULATE</div><h3>Reaction–diffusion PDE</h3>
        <p>A 3D solver advances proliferation and diffusion, modulated by a genomics-derived risk multiplier and a drug-response term.</p></div>
      <div class="step reveal"><div class="n">03 · VISUALIZE</div><h3>Living digital twin</h3>
        <p>Each timestep is rendered volumetrically with quantified burden, necrotic dynamics, and RECIST response — this layer.</p></div>
    </div>
    <p class="pullquote reveal">Not a picture of a tumor. A <span>simulation you can interrogate</span> — frame by frame, axis by axis.</p>
  </section>

</main>

<footer>
  <div class="wrap">
    <div class="foot-mark reveal">ONCO<b>PULSE</b></div>
    <div class="credits reveal">
      <div><div class="role">P1 · Genomics ML</div><b>Praneeth</b><br>XGBoost · SHAP risk</div>
      <div><div class="role">P2 · Tumor mechanics</div><b>Vinesh</b><br>3D PDE solver</div>
      <div><div class="role">P3 · Visualization</div><b>Jasim</b><br>Volumetric twin</div>
      <div><div class="role">P4 · Systems · LLM</div><b>Vihari</b><br>Streamlit · narrative</div>
      <div><div class="role">P5 · Radiomics</div><b>Philip · Chandan</b><br>TCIA pipeline</div>
    </div>
    <div class="foot-base">
      <span>ONCOPULSE · TUMOR GROWTH DIGITAL TWIN</span>
      <span>VISUALIZATION LAYER · BUILT WITH PLOTLY + VANILLA JS</span>
    </div>
  </div>
</footer>

<script>
const FIGS = {
  base_iso: __FIG_BASEISO__,
  volumetric: __FIG_VOLUMETRIC__,
  layers: __FIG_LAYERS__,
  cutaway: __FIG_CUTAWAY__,
  growth: __FIG_GROWTH__,
  slices: __FIG_SLICES__
};
const VALUES = __VALUES__;     // per-frame density fields for the isosurface
const SERIES = __SERIES__;     // per-frame {day, vol, diam, necro}
const MAXVOL = __MAXVOL__;     // y-range for the curve marker (mm³)
const CFG = {responsive:true, displayModeBar:false, staticPlot:false};
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;

// ----- loader dismiss -----
window.addEventListener('load', ()=>{ setTimeout(()=>document.getElementById('loader').classList.add('done'), reduce?100:1500); });

// ----- camera auto-orbit -----
let spinTimer=null, angle=0, playing=false, mode='growth';
function startSpin(){
  if(reduce||playing){return;}
  clearInterval(spinTimer);
  const gd = document.getElementById('viz3d');
  spinTimer = setInterval(()=>{
    angle += 0.0045*Math.PI;
    const r=1.8;
    Plotly.relayout(gd, {'scene.camera.eye':{x:r*Math.cos(angle), y:r*Math.sin(angle), z:0.85}});
  }, 60);
}
function stopSpin(){ clearInterval(spinTimer); }

// ----- timeline -----
let curIdx=0, playTimer=null;
function gotoFrame(i){
  curIdx = Math.max(0, Math.min(i, SERIES.length-1));
  Plotly.restyle('viz3d', {value:[VALUES[curIdx]]}, [0]);
  const s = SERIES[curIdx];
  document.getElementById('tlread').innerHTML =
    `DAY ${String(s.day).padStart(2,'0')} · <b>${s.vol.toFixed(2)} cm³</b> · Ø ${s.diam.toFixed(0)} mm`;
  Plotly.relayout('growth', {shapes:[{type:'line', xref:'x', yref:'y',
    x0:s.day, x1:s.day, y0:0, y1:MAXVOL, line:{color:'#ff7a45', width:1.6, dash:'dot'}}]});
  document.getElementById('tl').value = curIdx;
}
function play(){
  if(playing){ stopPlay(); return; }
  playing = true; stopSpin();
  const pb=document.getElementById('play'); pb.textContent='❚❚'; pb.classList.add('on');
  if(curIdx >= SERIES.length-1) gotoFrame(0);
  playTimer = setInterval(()=>{
    if(curIdx >= SERIES.length-1){ stopPlay(); return; }
    gotoFrame(curIdx+1);
  }, reduce?0:150);
}
function stopPlay(){
  playing=false; clearInterval(playTimer);
  const pb=document.getElementById('play'); if(pb){ pb.textContent='▶'; pb.classList.remove('on'); }
  startSpin();
}

// ----- mode switching -----
function setMode(name, btn){
  document.querySelectorAll('.mode').forEach(x=>x.classList.remove('on'));
  btn.classList.add('on');
  stopPlay(); mode=name;
  const tl=document.getElementById('timeline');
  if(name==='growth'){
    tl.style.display='';
    Plotly.react('viz3d', FIGS.base_iso.data, FIGS.base_iso.layout, CFG).then(()=>{ gotoFrame(curIdx); startSpin(); });
  } else {
    tl.style.display='none';
    Plotly.react('viz3d', FIGS[name].data, FIGS[name].layout, CFG).then(startSpin);
  }
}

document.addEventListener('DOMContentLoaded', ()=>{
  Plotly.newPlot('viz3d', FIGS.base_iso.data, FIGS.base_iso.layout, CFG).then(()=>{ gotoFrame(0); startSpin(); });
  Plotly.newPlot('growth', FIGS.growth.data, FIGS.growth.layout, CFG).then(()=>gotoFrame(0));
  Plotly.newPlot('slices', FIGS.slices.data, FIGS.slices.layout, CFG);

  const stage = document.querySelector('.stage');
  stage.addEventListener('mouseenter', stopSpin);
  stage.addEventListener('mouseleave', startSpin);

  document.getElementById('play').addEventListener('click', play);
  document.getElementById('tl').addEventListener('input', e=>{ stopPlay(); gotoFrame(parseInt(e.target.value,10)); });

  document.querySelectorAll('.mode').forEach(b=>{
    b.addEventListener('click', ()=> setMode(b.dataset.fig, b));
  });
});

// ----- scroll reveals -----
const io = new IntersectionObserver((es)=>{es.forEach(e=>{if(e.isIntersecting){e.target.classList.add('vis');io.unobserve(e.target);}})},{threshold:.12,rootMargin:'0px 0px -40px 0px'});
document.querySelectorAll('.reveal').forEach(el=>io.observe(el));

// ----- count-up KPIs -----
function countUp(el){
  const to=parseFloat(el.dataset.to), dec=parseInt(el.dataset.dec||'0'); const dur=1100; let t0=null;
  if(reduce){ el.textContent=to.toFixed(dec); return; }
  function step(ts){ if(!t0)t0=ts; const p=Math.min((ts-t0)/dur,1); const e=1-Math.pow(1-p,3);
    el.textContent=(to*e).toFixed(dec); if(p<1)requestAnimationFrame(step); }
  requestAnimationFrame(step);
}
const cio = new IntersectionObserver((es)=>{es.forEach(e=>{if(e.isIntersecting){countUp(e.target);cio.unobserve(e.target);}})},{threshold:.6});
document.querySelectorAll('.cu').forEach(el=>cio.observe(el));

window.addEventListener('resize', ()=>{ ['viz3d','growth','slices'].forEach(id=>{const g=document.getElementById(id); if(g&&g.data)Plotly.Plots.resize(g);}); });
</script>
</body>
</html>"""

# --------------------------------------------------------------------------- #
# 3. Token replacement + write
# --------------------------------------------------------------------------- #
double_txt = "—" if NUMS["doubling_d"] is None else str(NUMS["doubling_d"])
html = TEMPLATE
replacements = {
    "__FIG_BASEISO__": FIGS["base_iso"].to_json(),
    "__FIG_VOLUMETRIC__": FIGS["volumetric"].to_json(),
    "__FIG_LAYERS__": FIGS["layers"].to_json(),
    "__FIG_CUTAWAY__": FIGS["cutaway"].to_json(),
    "__FIG_GROWTH__": FIGS["growth"].to_json(),
    "__FIG_SLICES__": FIGS["slices"].to_json(),
    "__VALUES__": json.dumps(VALUES),
    "__SERIES__": json.dumps(SERIES),
    "__MAXVOL__": str(round(MAXVOL_MM3, 1)),
    "__NFRAMES_MAX__": str(len(SERIES) - 1),
    "__PEAKVOL__": str(NUMS["peak_vol_cm3"]),
    "__MAXD__": str(NUMS["max_diam_mm"]),
    "__NECRO__": str(NUMS["necrotic_pct"]),
    "__DOUBLE__": double_txt,
    "__RECIST__": NUMS["recist"],
    "__DIAMCH__": str(NUMS["diam_change"]),
    "__DAYPEAK__": str(NUMS["day_peak"]),
    "__DAYTHER__": str(12 * 7),
    "__N_STEPS__": str(NUMS["n_steps"]),
    "__VOXELS__": f"{NUMS['voxels']:,}",
}
for k, v in replacements.items():
    html = html.replace(k, v)

out = SITE / "index.html"
out.write_text(html, encoding="utf-8")
size_mb = out.stat().st_size / 1e6
print(f"wrote {out}  ({size_mb:.1f} MB)")
print(f"scenario: peak day {NUMS['day_peak']} · {NUMS['recist']} · "
      f"vol {NUMS['peak_vol_cm3']} cm3 · doubling {double_txt} d")

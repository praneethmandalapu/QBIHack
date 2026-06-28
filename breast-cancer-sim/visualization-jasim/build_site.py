"""Build the OncoPulse real-data viewer site (Person 3 / jasim).

Renders Philip/Chandan's real TCGA-BRCA volumes (Luminal A + Basal-like,
baseline + follow-up) in a hand-designed, award-level page with a case /
timepoint / render-mode selector. 3D traces are built client-side from
embedded value arrays so all four volumes switch instantly.

    ../.venv/Scripts/python.exe build_site.py
    ../.venv/Scripts/python.exe -m http.server 8080 --directory site
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import color_maps as cm  # noqa: E402
import render_3d as r  # noqa: E402

SITE = Path(__file__).parent / "site"
SITE.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# 1. Load the real volumes (via Philip's manifest) + derive display data
# --------------------------------------------------------------------------- #
SLUGS = [
    "luminal_a_TCGA-AR-A1AX_baseline", "luminal_a_TCGA-AR-A1AX_followup",
    "basal_TCGA-AR-A1AQ_baseline", "basal_TCGA-AR-A1AQ_followup",
]
CASES = {
    "luminal_a": {"label": "Luminal A", "tcga": "TCGA-AR-A1AX",
                  "baseline": "luminal_a_TCGA-AR-A1AX_baseline",
                  "followup": "luminal_a_TCGA-AR-A1AX_followup"},
    "basal": {"label": "Basal-like", "tcga": "TCGA-AR-A1AQ",
              "baseline": "basal_TCGA-AR-A1AQ_baseline",
              "followup": "basal_TCGA-AR-A1AQ_followup"},
}

vols, entries = {}, {}
for s in SLUGS:
    v, e = r.load_pde_volume(s)          # normalized 64³ float32 in [0,1]
    vols[s], entries[s] = v, e


def theme_2d(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#9aa6b6", family="JetBrains Mono, monospace"),
                      margin=dict(l=0, r=0, t=8, b=0))
    fig.update_xaxes(gridcolor="rgba(236,231,223,0.06)", zerolinecolor="rgba(236,231,223,0.10)")
    fig.update_yaxes(gridcolor="rgba(236,231,223,0.06)", zerolinecolor="rgba(236,231,223,0.10)")
    return fig


# 3D value arrays (downsampled to 32³ for fast client-side rendering)
_d, _h, _w = 32, 32, 32
zz, yy, xx = np.mgrid[0:_d, 0:_h, 0:_w]
GRID_X, GRID_Y, GRID_Z = xx.ravel().tolist(), yy.ravel().tolist(), zz.ravel().tolist()
VOLS = {s: np.round(r.downsample(vols[s], 2).ravel(), 2).tolist() for s in SLUGS}

# radiology slices (full-res), one figure per volume
SLICES = {s: theme_2d(r.render_slices(vols[s], (1.0, 1.0, 1.0))) for s in SLUGS}


def enhancing_pct(v):
    return round(100.0 * float((v > 0.5).mean()), 1)


META = {}
for s in SLUGS:
    e = entries[s]
    META[s] = {
        "subtype": e["subtype"], "tcga": e["tcga_id"], "tp": e["timepoint"],
        "date": e["study_date"], "frac": enhancing_pct(vols[s]),
        "matrix": "×".join(str(x) for x in e["shape"]),
        "spacing": " × ".join(f"{x:.2f}" for x in e["spacing_mm"]),
    }

LAYOUT3D = {
    "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
    "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
    "font": {"color": "#9aa6b6", "family": "JetBrains Mono, monospace"},
    "scene": {
        "bgcolor": "rgba(0,0,0,0)", "aspectmode": "data",
        "camera": {"eye": {"x": 1.6, "y": 1.5, "z": 1.0}},
        "xaxis": {"showbackground": False, "gridcolor": "rgba(236,231,223,0.06)",
                  "zerolinecolor": "rgba(236,231,223,0.10)", "color": "#5d6470", "title": {"text": ""}},
        "yaxis": {"showbackground": False, "gridcolor": "rgba(236,231,223,0.06)",
                  "zerolinecolor": "rgba(236,231,223,0.10)", "color": "#5d6470", "title": {"text": ""}},
        "zaxis": {"showbackground": False, "gridcolor": "rgba(236,231,223,0.06)",
                  "zerolinecolor": "rgba(236,231,223,0.10)", "color": "#5d6470", "title": {"text": ""}},
    },
}

# --------------------------------------------------------------------------- #
# 2. HTML template (tokens replaced below)
# --------------------------------------------------------------------------- #
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OncoPulse — TCGA-BRCA Tumor Viewer</title>
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
  --step-3:clamp(1.9rem,1.5rem+1.6vw,2.8rem);
  --s-l:clamp(2rem,1.8rem+.9vw,2.5rem); --s-xl:clamp(3rem,2.7rem+1.3vw,3.75rem);
  --s-2xl:clamp(4rem,3.6rem+1.7vw,5.5rem); --s-3xl:clamp(6rem,5.4rem+2.6vw,8rem);
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{background:var(--ink); color:var(--bone); font-family:var(--font-d);
  font-size:var(--step-0); line-height:1.6; overflow-x:hidden;
  -webkit-font-smoothing:antialiased; -moz-osx-font-smoothing:grayscale;}
::selection{background:var(--signal); color:var(--ink)}
::-webkit-scrollbar{width:10px}::-webkit-scrollbar-track{background:var(--ink)}
::-webkit-scrollbar-thumb{background:#1c2430;border-radius:0}
a{color:inherit;text-decoration:none}
.over{font-family:var(--font-m); font-size:.74rem; letter-spacing:.32em; text-transform:uppercase; color:var(--muted)}
.wrap{max-width:1280px; margin:0 auto; padding:0 clamp(1.2rem,4vw,3rem)}
.glow{position:fixed; inset:-10% -10% auto -10%; height:70vh; z-index:0; pointer-events:none;
  background:radial-gradient(60% 70% at 70% 0%, rgba(var(--signal-rgb),.16) 0%, transparent 60%);}
.grain::after{content:'';position:fixed;inset:0;z-index:1;pointer-events:none;opacity:.5;mix-blend-mode:overlay;
  background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='3'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.045'/%3E%3C/svg%3E");}

#loader{position:fixed;inset:0;z-index:9999;background:var(--ink);display:grid;place-items:center;
  transition:opacity .6s var(--ease),visibility .6s}
#loader.done{opacity:0;visibility:hidden}
#loader svg{width:min(60vw,420px)}
#loader path{stroke:var(--signal);stroke-width:2.5;fill:none;stroke-linecap:round;stroke-linejoin:round;
  stroke-dasharray:1400;stroke-dashoffset:1400;animation:draw 1.3s var(--ease-io) forwards}
.loader-tag{position:absolute;bottom:8vh;font-family:var(--font-m);font-size:.74rem;letter-spacing:.3em;color:var(--muted)}
@keyframes draw{to{stroke-dashoffset:0}}

nav{position:fixed;top:0;left:0;right:0;z-index:200;display:flex;align-items:center;justify-content:space-between;
  padding:.85rem clamp(1.2rem,4vw,3rem);background:rgba(10,12,16,.55);
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border-bottom:1px solid var(--line)}
.brand{display:flex;align-items:center;gap:.6rem;font-weight:700;letter-spacing:-.02em;font-size:1.05rem}
.brand svg{width:30px;height:18px}.brand b{color:var(--signal);font-weight:700}
.nlinks{display:flex;gap:2rem;align-items:center;font-family:var(--font-m);font-size:.78rem;letter-spacing:.08em}
.nlink{position:relative;color:var(--muted);text-transform:uppercase;transition:color .3s}
.nlink::after{content:'';position:absolute;left:0;bottom:-5px;width:100%;height:1px;background:var(--signal);
  transform:scaleX(0);transform-origin:right;transition:transform .35s var(--ease)}
.nlink:hover{color:var(--bone)} .nlink:hover::after{transform:scaleX(1);transform-origin:left}
.status{display:flex;align-items:center;gap:.5rem;font-family:var(--font-m);font-size:.72rem;color:var(--vital);letter-spacing:.12em}
.dot{width:7px;height:7px;border-radius:50%;background:var(--vital);box-shadow:0 0 0 0 rgba(70,224,176,.5);animation:ping 2s var(--ease-io) infinite}
@keyframes ping{0%{box-shadow:0 0 0 0 rgba(70,224,176,.45)}70%{box-shadow:0 0 0 9px rgba(70,224,176,0)}100%{box-shadow:0 0 0 0 rgba(70,224,176,0)}}
@media(max-width:780px){.nlinks{display:none}}

.hero{position:relative;z-index:2;min-height:100vh;display:flex;flex-direction:column;justify-content:center;padding-top:6rem;padding-bottom:3rem}
.hero .over{margin-bottom:1.4rem}
.title{position:relative;font-weight:700;letter-spacing:-.045em;line-height:.84;font-size:clamp(3.6rem,15vw,12rem);text-transform:uppercase}
.title .pulseword{color:var(--bone)}
.ecg{position:absolute;left:0;right:0;top:52%;height:120px;width:100%;z-index:-1;overflow:visible;pointer-events:none}
.ecg path{fill:none;stroke:var(--signal);stroke-width:2;opacity:.85;filter:drop-shadow(0 0 6px rgba(var(--signal-rgb),.6));
  stroke-dasharray:2600;stroke-dashoffset:2600;animation:trace 3.2s var(--ease-io) .8s forwards}
@keyframes trace{to{stroke-dashoffset:0}}
.lede{max-width:48rem;margin-top:1.8rem;color:#c4c1ba;font-size:var(--step-1);line-height:1.5;text-wrap:pretty}
.lede b{color:var(--bone);font-weight:600}
.cta-row{display:flex;gap:1rem;margin-top:2.2rem;flex-wrap:wrap}
.btn{font-family:var(--font-m);font-size:.82rem;letter-spacing:.06em;text-transform:uppercase;padding:.95rem 1.6rem;
  border:1px solid var(--bone);position:relative;overflow:hidden;z-index:1;transition:color .4s var(--ease);cursor:pointer;background:transparent;color:var(--bone)}
.btn::before{content:'';position:absolute;inset:0;background:var(--bone);transform:scaleX(0);transform-origin:left;transition:transform .4s var(--ease);z-index:-1}
.btn:hover{color:var(--ink)} .btn:hover::before{transform:scaleX(1)}
.btn--signal{border-color:var(--signal);color:var(--signal)}.btn--signal::before{background:var(--signal)}.btn--signal:hover{color:var(--ink)}
.hero-ticker{margin-top:auto;border-top:1px solid var(--line);border-bottom:1px solid var(--line);display:flex;gap:3rem;
  padding:.9rem 0;font-family:var(--font-m);font-size:.78rem;color:var(--muted);overflow:hidden;
  mask-image:linear-gradient(90deg,transparent,#000 6%,#000 94%,transparent)}
.hero-ticker .row{display:flex;gap:3rem;white-space:nowrap;animation:tick 26s linear infinite}
.hero-ticker b{color:var(--signal)}
@keyframes tick{from{transform:translateX(0)}to{transform:translateX(-50%)}}

section{position:relative;z-index:2}
.divider{display:flex;align-items:center;gap:1.2rem;padding:var(--s-2xl) 0 var(--s-l)}
.divider .num{font-family:var(--font-m);color:var(--signal);font-size:.8rem;letter-spacing:.2em}
.divider h2{font-size:var(--step-3);font-weight:600;letter-spacing:-.03em;line-height:1}
.divider .ln{flex:1;height:1px;background:var(--line)}
.reveal{opacity:0;transform:translateY(2.5rem);transition:opacity .9s var(--ease),transform .9s var(--ease)}
.reveal.vis{opacity:1;transform:none}

.cluster{display:grid;grid-template-columns:repeat(12,1fr);gap:1px;background:var(--line);border:1px solid var(--line)}
.cell{background:var(--ink);padding:1.5rem 1.4rem;display:flex;flex-direction:column;gap:.5rem}
.cell .lab{font-family:var(--font-m);font-size:.68rem;letter-spacing:.18em;text-transform:uppercase;color:var(--muted)}
.cell .val{font-family:var(--font-m);font-weight:700;font-size:clamp(1.7rem,4vw,2.6rem);letter-spacing:-.02em;line-height:1}
.cell .val .u{font-size:.9rem;color:var(--muted);margin-left:.25rem}
.cell .sub{font-family:var(--font-m);font-size:.72rem;color:var(--muted)}
.cell .sub b{color:var(--bone)}
.cell.c-5{grid-column:span 5} .cell.c-4{grid-column:span 4} .cell.c-3{grid-column:span 3}
@media(max-width:880px){.cell.c-5,.cell.c-4,.cell.c-3{grid-column:span 6}}
@media(max-width:520px){.cell.c-5,.cell.c-4,.cell.c-3{grid-column:span 12}}

.split{display:grid;grid-template-columns:1.55fr 1fr;gap:1.4rem;align-items:stretch}
@media(max-width:980px){.split{grid-template-columns:1fr}}
.panel{border:1px solid var(--line);background:linear-gradient(180deg,var(--panel),var(--ink));position:relative}
.panel .ph{display:flex;align-items:center;justify-content:space-between;padding:.9rem 1.1rem;border-bottom:1px solid var(--line);flex-wrap:wrap;gap:.6rem}
.panel .ph .t{font-family:var(--font-m);font-size:.74rem;letter-spacing:.16em;text-transform:uppercase;color:var(--muted)}
.modes{display:flex;gap:.4rem;flex-wrap:wrap}
.mode{font-family:var(--font-m);font-size:.7rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
  padding:.35rem .7rem;border:1px solid var(--line);cursor:pointer;transition:all .3s var(--ease);background:transparent}
.mode:hover{color:var(--bone);border-color:var(--muted)}
.mode.on{color:var(--ink);background:var(--signal);border-color:var(--signal)}
.selbar{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;padding:.7rem 1.1rem;border-bottom:1px solid var(--line)}
.seg-l{font-family:var(--font-m);font-size:.66rem;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);margin-right:.2rem}
.seg-l.sp{margin-left:.8rem}
.stage{position:relative;height:clamp(360px,52vh,560px)}
#viz3d{width:100%;height:100%}
.scan{position:absolute;inset:0;pointer-events:none;overflow:hidden;mix-blend-mode:screen}
.scan::before{content:'';position:absolute;left:0;right:0;height:34%;
  background:linear-gradient(180deg,transparent,rgba(var(--signal-rgb),.10),transparent);animation:sweep 4.5s var(--ease-io) infinite}
@keyframes sweep{0%{top:-34%}100%{top:100%}}
.reticle{position:absolute;top:.8rem;left:.8rem;font-family:var(--font-m);font-size:.66rem;color:var(--muted);letter-spacing:.1em;line-height:1.7}
.panel .cap{padding:.8rem 1.1rem;border-top:1px solid var(--line);font-family:var(--font-m);font-size:.72rem;color:var(--muted)}
.tele-body{padding:1rem .8rem}
.read{padding:1.1rem;border-top:1px solid var(--line);font-size:.95rem;color:#c4c1ba;line-height:1.55}
.read b{color:var(--bone)} .read .ok{color:var(--vital)}

.band{width:100vw;margin-left:calc(-50vw + 50%);background:var(--ink-2);border-top:1px solid var(--line);
  border-bottom:1px solid var(--line);padding:var(--s-2xl) 0;margin-top:var(--s-2xl)}
#slices{width:100%;height:300px}
.band .meta{display:flex;gap:2.5rem;flex-wrap:wrap;font-family:var(--font-m);font-size:.74rem;color:var(--muted);margin-top:1.2rem}
.band .meta b{color:var(--bone)}

.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:1.4rem}
@media(max-width:880px){.steps{grid-template-columns:1fr}}
.step{border-top:2px solid var(--signal);padding-top:1.1rem}
.step .n{font-family:var(--font-m);font-size:.8rem;color:var(--signal);letter-spacing:.2em}
.step h3{font-size:var(--step-1);font-weight:600;margin:.6rem 0 .5rem;letter-spacing:-.02em}
.step p{color:var(--muted);font-size:.95rem}
.pullquote{font-family:var(--font-s);font-style:italic;font-size:clamp(1.6rem,4.5vw,3rem);line-height:1.2;
  letter-spacing:-.01em;max-width:40rem;margin:var(--s-2xl) 0;color:var(--bone)}
.pullquote span{color:var(--signal)}

footer{position:relative;z-index:2;border-top:1px solid var(--line);margin-top:var(--s-3xl);padding:var(--s-2xl) 0 var(--s-l)}
.foot-mark{font-weight:700;letter-spacing:-.04em;line-height:.85;text-transform:uppercase;font-size:clamp(3rem,12vw,9rem);color:#161b25}
.foot-mark b{color:var(--signal)}
.credits{display:grid;grid-template-columns:repeat(5,1fr);gap:1rem;margin-top:2rem;font-family:var(--font-m);font-size:.74rem;color:var(--muted)}
.credits .role{color:var(--signal);letter-spacing:.1em;text-transform:uppercase;font-size:.66rem;margin-bottom:.3rem}
.credits b{color:var(--bone);font-weight:500}
.foot-base{display:flex;justify-content:space-between;flex-wrap:wrap;gap:1rem;margin-top:2.5rem;font-family:var(--font-m);font-size:.72rem;color:var(--muted)}
@media(max-width:780px){.credits{grid-template-columns:repeat(2,1fr)}}

@media (prefers-reduced-motion: reduce){
  *,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important;scroll-behavior:auto!important}
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
  <div class="loader-tag">LOADING TCGA-BRCA VOLUMES…</div>
</div>

<nav>
  <a class="brand" href="#top">
    <svg viewBox="0 0 60 30"><path d="M2 15h14l4-11 6 24 5-26 5 28 4-15h13" fill="none" stroke="#ff7a45" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>
    Onco<b>Pulse</b>
  </a>
  <div class="nlinks">
    <a class="nlink" href="#readout">Case</a>
    <a class="nlink" href="#twin">Volume</a>
    <a class="nlink" href="#imaging">Imaging</a>
    <a class="nlink" href="#method">Method</a>
  </div>
  <div class="status"><span class="dot"></span>REAL&nbsp;DATA</div>
</nav>

<main id="top" class="wrap">

  <header class="hero">
    <div class="over">Real TCGA-BRCA DCE-MRI · patient-derived tumor viewer</div>
    <h1 class="title">
      ONCO<span class="pulseword">PULSE</span>
      <svg class="ecg" viewBox="0 0 1200 120" preserveAspectRatio="none">
        <path d="M0 60 H320 l26 -44 l34 92 l30 -104 l28 116 l26 -56 H720 l22 -30 l24 60 H1200"/>
      </svg>
    </h1>
    <p class="lede">Two breast-cancer subtypes from the <b>TCGA-BRCA</b> archive — <b>Luminal A</b>
       and <b>Basal-like</b> — pulled from diagnostic DCE-MRI, resampled to a 64³ field, and
       rendered as interactive 3D volumes. Switch case, timepoint, and render mode below.</p>
    <div class="cta-row">
      <a class="btn btn--signal" href="#twin">Browse the volumes →</a>
      <a class="btn" href="#method">How it's built</a>
    </div>
    <div class="hero-ticker" aria-hidden="true">
      <div class="row">
        <span>ARCHIVE <b>TCGA-BRCA</b></span><span>CASES <b>2</b></span>
        <span>SUBTYPES <b>LUMINAL A · BASAL</b></span><span>MODALITY <b>DCE-MRI</b></span>
        <span>GRID <b>64³ ISOTROPIC</b></span><span>PIPELINE <b>DICOM → RESAMPLE → RENDER</b></span>
        <span>ARCHIVE <b>TCGA-BRCA</b></span><span>CASES <b>2</b></span>
        <span>SUBTYPES <b>LUMINAL A · BASAL</b></span><span>MODALITY <b>DCE-MRI</b></span>
        <span>GRID <b>64³ ISOTROPIC</b></span><span>PIPELINE <b>DICOM → RESAMPLE → RENDER</b></span>
      </div>
    </div>
  </header>

  <!-- CASE READOUT -->
  <section id="readout">
    <div class="divider reveal"><span class="num">01</span><h2>Case readout</h2><span class="ln"></span></div>
    <div class="cluster reveal">
      <div class="cell c-5"><span class="lab">Subtype</span>
        <span class="val" style="font-size:clamp(1.5rem,3.4vw,2.3rem)" id="m-subtype">—</span>
        <span class="sub">TCGA <b id="m-tcga">—</b></span></div>
      <div class="cell c-4"><span class="lab">Timepoint</span>
        <span class="val" style="font-size:clamp(1.5rem,3.4vw,2.3rem)" id="m-tp">—</span>
        <span class="sub">study date <b id="m-date">—</b></span></div>
      <div class="cell c-3"><span class="lab">Model grid</span>
        <span class="val">64³</span><span class="sub">isotropic voxels</span></div>
      <div class="cell c-5"><span class="lab">Enhancing fraction</span>
        <span class="val"><span id="m-frac">—</span><span class="u">%</span></span>
        <span class="sub">high-signal · windowed</span></div>
      <div class="cell c-4"><span class="lab">Acquired matrix</span>
        <span class="val" style="font-size:clamp(1.2rem,2.6vw,1.7rem)" id="m-matrix">—</span>
        <span class="sub"><b id="m-spacing">—</b> spacing</span></div>
      <div class="cell c-3"><span class="lab">Modality</span>
        <span class="val" style="font-size:clamp(1.2rem,2.6vw,1.7rem)">DCE-MRI</span>
        <span class="sub">T1 · contrast</span></div>
    </div>
  </section>

  <!-- VOLUME -->
  <section id="twin">
    <div class="divider reveal"><span class="num">02</span><h2>The volume</h2><span class="ln"></span></div>
    <div class="split">
      <div class="panel reveal">
        <div class="ph">
          <span class="t">3D volume</span>
          <div class="modes">
            <button class="mode on" data-mode="volumetric">Volumetric</button>
            <button class="mode" data-mode="isosurface">Isosurface</button>
            <button class="mode" data-mode="cutaway">Cutaway</button>
          </div>
        </div>
        <div class="selbar">
          <span class="seg-l">Case</span>
          <button class="mode on" data-case="luminal_a">Luminal A</button>
          <button class="mode" data-case="basal">Basal-like</button>
          <span class="seg-l sp">Timepoint</span>
          <button class="mode on" data-tp="baseline">Baseline</button>
          <button class="mode" data-tp="followup">Follow-up</button>
        </div>
        <div class="stage">
          <div class="reticle" id="reticle">—</div>
          <div id="viz3d"></div>
          <div class="scan"></div>
        </div>
        <div class="cap">Drag to orbit · auto-rotating · windowed MR intensity, rendered by value</div>
      </div>
      <div class="panel reveal">
        <div class="ph"><span class="t">Longitudinal</span><span class="t" style="color:var(--signal)">baseline → follow-up</span></div>
        <div class="tele-body"><div id="compare" style="width:100%;height:300px"></div></div>
        <div class="read" id="read">—</div>
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
          <span>SOURCE <b>windowed MR intensity</b></span>
          <span>WINDOW <b>0.00 – 1.00</b></span>
          <span>MODEL GRID <b>64³ @ 1.0 mm</b></span>
        </div>
      </div>
    </div>
  </section>

  <!-- METHOD -->
  <section id="method">
    <div class="divider reveal"><span class="num">04</span><h2>From archive to viewer</h2><span class="ln"></span></div>
    <div class="steps">
      <div class="step reveal"><div class="n">01 · ACQUIRE</div><h3>TCGA-BRCA DICOM</h3>
        <p>Matched baseline + follow-up DCE-MRI studies pulled from the TCIA public archive for each subtype.</p></div>
      <div class="step reveal"><div class="n">02 · EXTRACT</div><h3>Resample to 64³</h3>
        <p>Philip &amp; Chandan's pipeline registers and resamples each study to a 64³ isotropic field normalized to [0, 1].</p></div>
      <div class="step reveal"><div class="n">03 · VISUALIZE</div><h3>Interactive 3D</h3>
        <p>Rendered by value as volumetric / isosurface / cutaway, with orthogonal radiology slices — this layer.</p></div>
    </div>
    <p class="pullquote reveal">Real scans. Real subtypes. <span>Interrogate the volume</span> — case by case, plane by plane.</p>
  </section>

</main>

<footer>
  <div class="wrap">
    <div class="foot-mark reveal">ONCO<b>PULSE</b></div>
    <div class="credits reveal">
      <div><div class="role">P1 · Genomics ML</div><b>Praneeth</b><br>XGBoost · SHAP risk</div>
      <div><div class="role">P2 · Tumor mechanics</div><b>Vinesh</b><br>3D PDE solver</div>
      <div><div class="role">P3 · Visualization</div><b>Jasim</b><br>3D viewer</div>
      <div><div class="role">P4 · Systems · LLM</div><b>Vihari</b><br>Streamlit · narrative</div>
      <div><div class="role">P5 · Radiomics</div><b>Philip · Chandan</b><br>TCIA pipeline</div>
    </div>
    <div class="foot-base">
      <span>ONCOPULSE · TCGA-BRCA TUMOR VIEWER</span>
      <span>VISUALIZATION LAYER · PLOTLY + VANILLA JS</span>
    </div>
  </div>
</footer>

<script>
const VOLS=__VOLS__, GRID={x:__X__,y:__Y__,z:__Z__}, CSCALE=__CSCALE__, OPAC=__OPAC__,
      LAYOUT3D=__LAYOUT3D__, SLICES=__SLICES__, META=__META__, CASES=__CASES__;
const CFG={responsive:true, displayModeBar:false};
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
let cur = {case:'luminal_a', tp:'baseline', mode:'volumetric'};
function slug(){ return CASES[cur.case][cur.tp]; }

window.addEventListener('load', ()=>{ setTimeout(()=>document.getElementById('loader').classList.add('done'), reduce?100:1400); });

// camera auto-orbit
let spinTimer=null, angle=0;
function startSpin(){ if(reduce)return; clearInterval(spinTimer); const gd=document.getElementById('viz3d');
  spinTimer=setInterval(()=>{ angle+=0.0045*Math.PI; const R=1.8;
    Plotly.relayout(gd,{'scene.camera.eye':{x:R*Math.cos(angle),y:R*Math.sin(angle),z:0.85}}); }, 60); }
function stopSpin(){ clearInterval(spinTimer); }

function trace3D(){
  const val=VOLS[slug()]; const caps={x:{show:false},y:{show:false},z:{show:false}};
  if(cur.mode==='isosurface')
    return {type:'isosurface',x:GRID.x,y:GRID.y,z:GRID.z,value:val,isomin:0.5,isomax:1.0,
      surface:{count:3},colorscale:CSCALE,showscale:false,opacity:0.6,caps};
  let v=val;
  if(cur.mode==='cutaway') v=val.map((d,i)=> GRID.y[i]>=16 ? 0 : d);
  return {type:'volume',x:GRID.x,y:GRID.y,z:GRID.z,value:v,isomin:0.15,isomax:1.0,
    colorscale:CSCALE,opacityscale:OPAC,surface:{count:18},showscale:false,caps};
}
function draw3D(){ Plotly.react('viz3d',[trace3D()],LAYOUT3D,CFG).then(startSpin); }
function drawSlices(){ const f=SLICES[slug()]; Plotly.react('slices',f.data,f.layout,CFG); }
function drawCompare(){
  const c=CASES[cur.case], b=META[c.baseline], f=META[c.followup];
  const tr={type:'bar',x:['Baseline','Follow-up'],y:[b.frac,f.frac],width:0.52,
    marker:{color:['#46e0b0','#ff7a45']},text:[b.frac.toFixed(0)+'%',f.frac.toFixed(0)+'%'],
    textposition:'outside',textfont:{color:'#ece7df',family:'JetBrains Mono'},hoverinfo:'y'};
  const ly={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',height:300,
    margin:{l:40,r:12,t:22,b:34},font:{color:'#9aa6b6',family:'JetBrains Mono'},
    yaxis:{title:'enhancing %',range:[0,Math.max(60,b.frac,f.frac)*1.2],gridcolor:'rgba(236,231,223,0.06)',zerolinecolor:'rgba(236,231,223,0.10)'},
    xaxis:{color:'#9aa6b6'}};
  Plotly.react('compare',[tr],ly,CFG);
}
function monthsBetween(d1,d2){ const a=new Date(d1),b=new Date(d2); return Math.round((b-a)/(1000*60*60*24*30.44)); }

function updateUI(){
  const s=slug(), m=META[s], c=CASES[cur.case];
  document.getElementById('m-subtype').textContent=m.subtype;
  document.getElementById('m-tcga').textContent=m.tcga;
  document.getElementById('m-tp').textContent= cur.tp==='baseline'?'Baseline':'Follow-up';
  document.getElementById('m-date').textContent=m.date;
  document.getElementById('m-frac').textContent=m.frac.toFixed(0);
  document.getElementById('m-matrix').textContent=m.matrix;
  document.getElementById('m-spacing').textContent=m.spacing+' mm';
  document.getElementById('reticle').innerHTML=`${m.tcga}<br>${m.subtype}<br>${cur.tp.toUpperCase()}`;
  const b=META[c.baseline], f=META[c.followup], iv=monthsBetween(b.date,f.date);
  document.getElementById('read').innerHTML=
    `<b>${c.label}</b> (${c.tcga}) imaged <b>${b.date}</b> and <b>${f.date}</b> — a <b>${iv}-month</b> interval. `+
    `Enhancing fraction (windowed): <b class="ok">${b.frac.toFixed(0)}%</b> → <b class="ok">${f.frac.toFixed(0)}%</b>.`;
}
function refresh(){ updateUI(); draw3D(); drawSlices(); drawCompare(); }
function setActive(sel,btn){ document.querySelectorAll(sel).forEach(x=>x.classList.remove('on')); btn.classList.add('on'); }

document.addEventListener('DOMContentLoaded', ()=>{
  Plotly.newPlot('viz3d',[trace3D()],LAYOUT3D,CFG).then(startSpin);
  drawSlices(); drawCompare(); updateUI();
  const stage=document.querySelector('.stage');
  stage.addEventListener('mouseenter', stopSpin);
  stage.addEventListener('mouseleave', startSpin);
  document.querySelectorAll('[data-case]').forEach(b=>b.addEventListener('click',()=>{ cur.case=b.dataset.case; setActive('[data-case]',b); refresh(); }));
  document.querySelectorAll('[data-tp]').forEach(b=>b.addEventListener('click',()=>{ cur.tp=b.dataset.tp; setActive('[data-tp]',b); refresh(); }));
  document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{ cur.mode=b.dataset.mode; setActive('[data-mode]',b); draw3D(); }));
});

const io=new IntersectionObserver((es)=>{es.forEach(e=>{if(e.isIntersecting){e.target.classList.add('vis');io.unobserve(e.target);}})},{threshold:.12,rootMargin:'0px 0px -40px 0px'});
document.querySelectorAll('.reveal').forEach(el=>io.observe(el));
window.addEventListener('resize', ()=>{ ['viz3d','slices','compare'].forEach(id=>{const g=document.getElementById(id); if(g&&g.data)Plotly.Plots.resize(g);}); });
</script>
</body>
</html>"""

# --------------------------------------------------------------------------- #
# 3. Token replacement + write
# --------------------------------------------------------------------------- #
slices_js = "{" + ",".join(f"{json.dumps(s)}:{SLICES[s].to_json()}" for s in SLUGS) + "}"
replacements = {
    "__VOLS__": json.dumps(VOLS),
    "__X__": json.dumps(GRID_X), "__Y__": json.dumps(GRID_Y), "__Z__": json.dumps(GRID_Z),
    "__CSCALE__": json.dumps(cm.density_colorscale()),
    "__OPAC__": json.dumps(cm.density_opacityscale()),
    "__LAYOUT3D__": json.dumps(LAYOUT3D),
    "__SLICES__": slices_js,
    "__META__": json.dumps(META),
    "__CASES__": json.dumps(CASES),
}
html = TEMPLATE
for k, v in replacements.items():
    html = html.replace(k, v)

out = SITE / "index.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
print("volumes:", ", ".join(f"{s.split('_TCGA')[0]}/{META[s]['tp']}={META[s]['frac']}%" for s in SLUGS))

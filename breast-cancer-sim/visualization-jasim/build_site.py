"""Build the OncoPulse two-cancer viewer (Person 3 / jasim).

A tabbed static site:
  • BRAIN  — longitudinal glioma growth (Vinesh's calibrated PDE frame stacks,
             seeded from a real UCSF baseline) + the real 298-patient cohort.
  • BREAST — PDE growth from real TCGA-BRCA baseline density (scrub/play player).

3D traces are built client-side from embedded value arrays so everything
switches instantly. Run:
    ../.venv/Scripts/python.exe build_site.py
    ../.venv/Scripts/python.exe -m http.server 8080 --directory site
"""

from __future__ import annotations

import csv
import json
import statistics as stx
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import color_maps as cm  # noqa: E402
import render_3d as r  # noqa: E402
import make_brain_frames as bf  # noqa: E402
import make_breast_frames as brf  # noqa: E402

HERE = Path(__file__).resolve().parent
BREAST = HERE.parent                      # breast-cancer-sim/
REPO = BREAST.parent                      # qbihack/
BRAIN = REPO / "brain-cancer-sim"
FRAMES = BREAST / "data/processed/brain-frames-jasim"
BREAST_FRAMES = BREAST / "data/processed/breast-frames-jasim"
SITE = HERE / "site"
SITE.mkdir(exist_ok=True)

# shared 32³ display grid (both diseases downsample to this)
_g = 32
zz, yy, xx = np.mgrid[0:_g, 0:_g, 0:_g]
GRID_X, GRID_Y, GRID_Z = xx.ravel().tolist(), yy.ravel().tolist(), zz.ravel().tolist()

RISK_DIR = HERE / "risk"
BRAIN_RISK_DIR = BRAIN / "visualization-jasim/risk"


def load_risk_lookup(path: Path) -> dict[str, float]:
    """patient_id -> risk score from visualization-jasim/risk/patients.csv."""
    if not path.is_file():
        return {}
    out: dict[str, float] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            pid = (row.get("patient_id") or "").strip()
            raw = (row.get("risk") or "").strip()
            if pid and raw:
                try:
                    out[pid] = round(float(raw), 5)
                except ValueError:
                    pass
    return out


def load_breast_patient_rows(path: Path) -> dict[str, dict[str, str]]:
    """tcga_id -> handoff row from visualization-jasim/risk/patients.csv."""
    if not path.is_file():
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            pid = (row.get("patient_id") or "").strip()
            if pid:
                out[pid] = row
    return out


BREAST_RISK = load_risk_lookup(RISK_DIR / "patients.csv")
BREAST_PATIENTS = load_breast_patient_rows(RISK_DIR / "patients.csv")
BRAIN_RISK = load_risk_lookup(BRAIN_RISK_DIR / "patients.csv")


def theme_2d(fig):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#9aa6b6", family="JetBrains Mono, monospace"),
                      margin=dict(l=0, r=0, t=8, b=0))
    fig.update_xaxes(gridcolor="rgba(236,231,223,0.06)", zerolinecolor="rgba(236,231,223,0.10)")
    fig.update_yaxes(gridcolor="rgba(236,231,223,0.06)", zerolinecolor="rgba(236,231,223,0.10)")
    return fig


# --------------------------------------------------------------------------- #
# BRAIN — longitudinal glioma growth (one real patient per IDH regime)
# --------------------------------------------------------------------------- #
FEATURES_CSV = BRAIN / "data/processed/brain_patient_features.csv"
_rows = list(csv.DictReader(FEATURES_CSV.open(newline="")))
BRAIN_PATIENTS = {r["subjectid"]: r for r in _rows}
_wt = [x for x in _rows if x["idh"] == "WT"]
_mut = [x for x in _rows if x["idh"] and x["idh"] != "WT"]


def _grew_pct(g):
    v = [x for x in g if x["actually_grew"] in ("True", "False")]
    return round(100 * sum(1 for x in v if x["actually_grew"] == "True") / len(v)) if v else 0


COHORT = {
    "n": len(_rows), "wt": len(_wt), "mut": len(_mut),
    "wt_grew": _grew_pct(_wt), "mut_grew": _grew_pct(_mut),
    "wt_gm": round(stx.median(float(x["growth_multiplier"]) for x in _wt), 2),
    "mut_gm": round(stx.median(float(x["growth_multiplier"]) for x in _mut), 2),
    "g4": sum(1 for x in _rows if x["grade"] == "4.0"),
    "g3": sum(1 for x in _rows if x["grade"] == "3.0"),
    "g2": sum(1 for x in _rows if x["grade"] == "2.0"),
}


def _patient_grade(patient_id: str) -> str:
    raw = (BRAIN_PATIENTS.get(patient_id) or {}).get("grade", "")
    if not raw:
        return "—"
    try:
        return str(int(float(raw)))
    except ValueError:
        return raw


REGIME_META = {
    "aggressive": {"label": "IDH-wildtype", "tag": "aggressive", "idh": "WT",
                   "gm": COHORT["wt_gm"], "grew": COHORT["wt_grew"]},
    "indolent": {"label": "IDH-mutant", "tag": "indolent", "idh": "mutant",
                 "gm": COHORT["mut_gm"], "grew": COHORT["mut_grew"]},
}
REGIMES = bf.regime_config()
bf.ensure_frames(FRAMES)
BRAIN: dict[str, dict[str, dict]] = {"aggressive": {}, "indolent": {}}
BRAIN_SLICES: dict[str, dict[str, object]] = {"aggressive": {}, "indolent": {}}
BRAIN_PICKERS: dict[str, list[dict]] = {"aggressive": [], "indolent": []}
BRAIN_DEFAULTS = {k: v["default"] for k, v in REGIMES.items()}

for regime, block in REGIMES.items():
    meta = REGIME_META[regime]
    for pid in block["patients"]:
        arr = np.load(bf.frame_path(FRAMES, regime, pid))   # (T,Z,Y,X)
        frame_meta_path = bf.meta_path(FRAMES, regime, pid)
        frame_meta = json.loads(frame_meta_path.read_text()) if frame_meta_path.is_file() else {}
        interval = float(frame_meta.get("interval_days") or 180)
        T = arr.shape[0]
        disp = [r.downsample(arr[i], 2) for i in range(T)]
        burden = [float((arr[i] > bf.BURDEN_THR).sum()) for i in range(T)]
        b0 = burden[0] or 1.0
        growth = frame_meta.get("real_growth_pct")
        BRAIN[regime][pid] = {
            **meta,
            "patient_id": pid,
            "grade": _patient_grade(pid),
            "risk": BRAIN_RISK.get(pid),
            "real_growth_pct": growth,
            "values": [np.round(d.ravel(), 2).tolist() for d in disp],
            "idx": [round(100 * b / b0, 1) for b in burden],
            "days": [round(i * interval / (T - 1)) for i in range(T)],
            "n": T, "peak": round(100 * burden[-1] / b0),
        }
        BRAIN_SLICES[regime][pid] = theme_2d(r.render_slices(arr[-1], (1.0, 1.0, 1.0)))
        growth_label = f"{growth:+.0f}%" if growth is not None else "—"
        BRAIN_PICKERS[regime].append({
            "id": pid,
            "label": f"UCSF {pid} · grade {_patient_grade(pid)} · {growth_label} WT",
        })

# --------------------------------------------------------------------------- #
# BREAST — PDE growth from real TCGA-BRCA baseline density
# --------------------------------------------------------------------------- #
brf.ensure_frames(BREAST_FRAMES)
BREAST_GROWTH = {}
for key, sc in brf.SCENARIOS.items():
    arr = np.load(brf.frame_path(BREAST_FRAMES, key))   # (T,Z,Y,X)
    frame_meta = json.loads(brf.meta_path(BREAST_FRAMES, key).read_text())
    interval = float(frame_meta["interval_days"])
    T = arr.shape[0]
    disp = [r.downsample(arr[i], 2) for i in range(T)]
    burden = [float(arr[i].sum()) for i in range(T)]
    volume = [int((arr[i] >= brf.ISO).sum()) for i in range(T)]
    b0 = burden[0] or 1.0
    v0 = volume[0] or 1
    BREAST_GROWTH[key] = {
        "label": sc["label"],
        "subtype": sc["label"],
        "tcga": sc["tcga"],
        "baseline": frame_meta.get("baseline_date") or "—",
        "slug": frame_meta.get("slug", ""),
        "risk": sc["risk"],
        "genomic_risk": BREAST_RISK.get(sc["tcga"]),
        "interval": round(interval),
        "n": T,
        "iso": brf.ISO,
        "days": [round(i * interval / (T - 1)) for i in range(T)],
        "idx": [round(100 * b / b0, 1) for b in burden],
        "burden": [round(b, 1) for b in burden],
        "volume": volume,
        "growth": round(100 * (burden[-1] - burden[0]) / b0, 1),
        "volumeGrowth": round(100 * (volume[-1] - volume[0]) / v0, 1),
        "peak": round(100 * burden[-1] / b0),
        "maxDensity": round(float(arr.max()), 3),
        "color": sc["color"],
        "modelNote": (
            "Prototype Fisher-KPP growth from copied breast PDE baseline density; "
            "not clinically validated."
        ),
        "values": [np.round(d.ravel(), 2).tolist() for d in disp],
    }

LAYOUT3D = {
    "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)",
    "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
    "font": {"color": "#9aa6b6", "family": "JetBrains Mono, monospace"},
    "scene": {"bgcolor": "rgba(0,0,0,0)", "aspectmode": "data",
              "camera": {"eye": {"x": 1.6, "y": 1.5, "z": 1.0}},
              "xaxis": {"showbackground": False, "gridcolor": "rgba(236,231,223,0.06)", "zerolinecolor": "rgba(236,231,223,0.10)", "color": "#5d6470", "title": {"text": ""}},
              "yaxis": {"showbackground": False, "gridcolor": "rgba(236,231,223,0.06)", "zerolinecolor": "rgba(236,231,223,0.10)", "color": "#5d6470", "title": {"text": ""}},
              "zaxis": {"showbackground": False, "gridcolor": "rgba(236,231,223,0.06)", "zerolinecolor": "rgba(236,231,223,0.10)", "color": "#5d6470", "title": {"text": ""}}},
}

TEMPLATE = Path(HERE / "_template.html").read_text(encoding="utf-8")

slices_brain = "{" + ",".join(
    f"{json.dumps(regime)}:{{{','.join(f'{json.dumps(pid)}:{fig.to_json()}' for pid, fig in by_pid.items())}}}"
    for regime, by_pid in BRAIN_SLICES.items()
) + "}"
brain_payload = {
    regime: {pid: {kk: vv for kk, vv in payload.items()} for pid, payload in by_pid.items()}
    for regime, by_pid in BRAIN.items()
}
breast_payload = {k: {kk: vv for kk, vv in v.items()} for k, v in BREAST_GROWTH.items()}

repl = {
    "__GRID_X__": json.dumps(GRID_X), "__GRID_Y__": json.dumps(GRID_Y), "__GRID_Z__": json.dumps(GRID_Z),
    "__CSCALE__": json.dumps(cm.density_colorscale()),
    "__BRAIN_CSCALE__": json.dumps(cm.orange_intensity_colorscale()),
    "__BRAIN_VOL_CSCALE__": json.dumps(cm.orange_volumetric_colorscale()),
    "__OPAC__": json.dumps(cm.density_opacityscale()),
    "__BRAIN_OPAC__": json.dumps(cm.brain_opacityscale()),
    "__LAYOUT3D__": json.dumps(LAYOUT3D),
    "__BRAIN__": json.dumps(brain_payload), "__BRAIN_SLICES__": slices_brain,
    "__BRAIN_DEFAULTS__": json.dumps(BRAIN_DEFAULTS), "__BRAIN_PICKERS__": json.dumps(BRAIN_PICKERS),
    "__COHORT__": json.dumps(COHORT),
    "__BREAST_GROWTH__": json.dumps(breast_payload),
}
html = TEMPLATE
for k, v in repl.items():
    html = html.replace(k, v)

out = SITE / "index.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
print(f"brain: {len(BRAIN['aggressive'])} WT + {len(BRAIN['indolent'])} mut cases · "
      f"defaults peak {BRAIN['aggressive'][BRAIN_DEFAULTS['aggressive']]['peak']}% / "
      f"{BRAIN['indolent'][BRAIN_DEFAULTS['indolent']]['peak']}%")
print(f"cohort: n={COHORT['n']} WT grew {COHORT['wt_grew']}% (GM {COHORT['wt_gm']}) · mut grew {COHORT['mut_grew']}% (GM {COHORT['mut_gm']})")
print("breast:", ", ".join(
    f"{BREAST_GROWTH[k]['label'][:3]}/{BREAST_GROWTH[k]['tcga']} peak={BREAST_GROWTH[k]['peak']} idx"
    for k in BREAST_GROWTH
))

"""Build the OncoPulse two-cancer viewer (Person 3 / jasim).

A tabbed static site:
  • BRAIN  — longitudinal glioma growth (Vinesh's calibrated PDE frame stacks,
             seeded from a real UCSF baseline) + the real 298-patient cohort.
  • BREAST — 3D snapshot of real TCGA-BRCA volumes (no growth; multiple views).

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

HERE = Path(__file__).resolve().parent
BREAST = HERE.parent                      # breast-cancer-sim/
REPO = BREAST.parent                      # qbihack/
BRAIN = REPO / "brain-cancer-sim"
FRAMES = BREAST / "data/processed/brain-frames-jasim"
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


BREAST_RISK = load_risk_lookup(RISK_DIR / "patients.csv")
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
BRAIN_META = {
    "aggressive": {"label": "IDH-wildtype", "tag": "aggressive", "idh": "WT",
                   "patient_id": bf.SCENARIOS["aggressive"]["patient_id"],
                   "grade": "4", "gm": 1.57, "grew": 70},
    "indolent": {"label": "IDH-mutant", "tag": "indolent", "idh": "mutant",
                 "patient_id": bf.SCENARIOS["indolent"]["patient_id"],
                 "grade": "2", "gm": 0.92, "grew": 55},
}
bf.ensure_frames(FRAMES)   # regenerate gitignored demo stacks (100118 + 100002)
BRAIN = {}
BRAIN_SLICES = {}
for key, m in BRAIN_META.items():
    pid = m["patient_id"]
    arr = np.load(bf.frame_path(FRAMES, key))   # (T,Z,Y,X)
    meta_path = bf.meta_path(FRAMES, key)
    frame_meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    interval = float(frame_meta.get("interval_days") or 180)
    T = arr.shape[0]
    disp = [r.downsample(arr[i], 2) for i in range(T)]
    # Burden index at the same floor as the 3D isosurface (0.2).
    burden = [float((arr[i] > bf.BURDEN_THR).sum()) for i in range(T)]
    b0 = burden[0] or 1.0
    BRAIN[key] = {
        **m,
        "risk": BRAIN_RISK.get(pid),
        "real_growth_pct": frame_meta.get("real_growth_pct"),
        "values": [np.round(d.ravel(), 2).tolist() for d in disp],
        "idx": [round(100 * b / b0, 1) for b in burden],     # burden index, baseline = 100
        "days": [round(i * interval / (T - 1)) for i in range(T)],
        "n": T, "peak": round(100 * burden[-1] / b0),
    }
    BRAIN_SLICES[key] = theme_2d(r.render_slices(arr[-1], (1.0, 1.0, 1.0)))

# real UCSF cohort summary (298 patients)
_rows = list(csv.DictReader(open(REPO / "brain-cancer-sim/data/processed/brain_patient_features.csv")))
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

# --------------------------------------------------------------------------- #
# BREAST — real TCGA-BRCA snapshot volumes (no growth)
# --------------------------------------------------------------------------- #
BR_SLUGS = ["luminal_a_TCGA-AR-A1AX_baseline", "luminal_a_TCGA-AR-A1AX_followup",
            "basal_TCGA-AR-A1AQ_baseline", "basal_TCGA-AR-A1AQ_followup"]
BR_CASES = {
    "luminal_a": {"label": "Luminal A", "tcga": "TCGA-AR-A1AX",
                  "study1": "luminal_a_TCGA-AR-A1AX_baseline", "study2": "luminal_a_TCGA-AR-A1AX_followup"},
    "basal": {"label": "Basal-like", "tcga": "TCGA-AR-A1AQ",
              "study1": "basal_TCGA-AR-A1AQ_baseline", "study2": "basal_TCGA-AR-A1AQ_followup"},
}
BR_VOLS, BR_SLICES, BR_META = {}, {}, {}
for s in BR_SLUGS:
    v, e = r.load_pde_volume(s)
    BR_VOLS[s] = np.round(r.downsample(v, 2).ravel(), 2).tolist()
    BR_SLICES[s] = theme_2d(r.render_slices(v, (1.0, 1.0, 1.0)))
    BR_META[s] = {"subtype": e["subtype"], "tcga": e["tcga_id"], "study": e["study_date"],
                  "frac": round(100.0 * float((v > 0.5).mean()), 1),
                  "matrix": "×".join(str(x) for x in e["shape"]),
                  "spacing": " × ".join(f"{x:.2f}" for x in e["spacing_mm"]),
                  "risk": BREAST_RISK.get(e["tcga_id"])}

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

slices_brain = "{" + ",".join(f"{json.dumps(k)}:{BRAIN_SLICES[k].to_json()}" for k in BRAIN_SLICES) + "}"
slices_breast = "{" + ",".join(f"{json.dumps(s)}:{BR_SLICES[s].to_json()}" for s in BR_SLUGS) + "}"
brain_payload = {k: {kk: vv for kk, vv in v.items()} for k, v in BRAIN.items()}

repl = {
    "__GRID_X__": json.dumps(GRID_X), "__GRID_Y__": json.dumps(GRID_Y), "__GRID_Z__": json.dumps(GRID_Z),
    "__CSCALE__": json.dumps(cm.density_colorscale()),
    "__BRAIN_CSCALE__": json.dumps(cm.orange_intensity_colorscale()),
    "__OPAC__": json.dumps(cm.density_opacityscale()),
    "__LAYOUT3D__": json.dumps(LAYOUT3D),
    "__BRAIN__": json.dumps(brain_payload), "__BRAIN_SLICES__": slices_brain, "__COHORT__": json.dumps(COHORT),
    "__BR_VOLS__": json.dumps(BR_VOLS), "__BR_SLICES__": slices_breast,
    "__BR_META__": json.dumps(BR_META), "__BR_CASES__": json.dumps(BR_CASES),
}
html = TEMPLATE
for k, v in repl.items():
    html = html.replace(k, v)

out = SITE / "index.html"
out.write_text(html, encoding="utf-8")
print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
print(f"brain: aggressive peak {BRAIN['aggressive']['peak']}% · indolent peak {BRAIN['indolent']['peak']}%")
print(f"cohort: n={COHORT['n']} WT grew {COHORT['wt_grew']}% (GM {COHORT['wt_gm']}) · mut grew {COHORT['mut_grew']}% (GM {COHORT['mut_gm']})")
print("breast:", ", ".join(f"{BR_META[s]['subtype'][:3]}/{BR_META[s]['study']}={BR_META[s]['frac']}%" for s in BR_SLUGS))

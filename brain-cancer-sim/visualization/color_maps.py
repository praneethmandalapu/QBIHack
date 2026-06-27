"""Clinical color system for tumor visualization.

Ported from breast-cancer-sim/visualization-jasim/color_maps.py @ QBIHack 5119ad6.
"""

from __future__ import annotations

NECROTIC = "#8B0000"
VIABLE = "#FF4500"
HEALTHY = "#90EE90"

PROLIFERATING = "#ffd166"
VIABLE_TUMOR = "#ff6b35"
HYPOXIC = "#c41e3a"
NECROTIC_CORE = "#5a0a14"

BG_DEEP = "#070b16"
BG_PANEL = "#0e1626"
GRID = "#1b2740"
ACCENT = "#22d3ee"
ACCENT_2 = "#a855f7"
TEXT = "#e6edf7"
TEXT_DIM = "#8aa0c0"
GOOD = "#34d399"
BAD = "#f87171"


def tissue_colormap():
    return {"necrotic": NECROTIC, "viable": VIABLE, "healthy": HEALTHY}


def clinical_palette() -> dict:
    return {
        "proliferating": PROLIFERATING,
        "viable": VIABLE_TUMOR,
        "hypoxic": HYPOXIC,
        "necrotic": NECROTIC_CORE,
    }


def density_colorscale():
    return [
        [0.00, "#10243f"],
        [0.40, PROLIFERATING],
        [0.62, VIABLE_TUMOR],
        [0.82, HYPOXIC],
        [1.00, NECROTIC_CORE],
    ]


def density_opacityscale():
    return [
        [0.00, 0.00],
        [0.30, 0.02],
        [0.50, 0.12],
        [0.70, 0.35],
        [0.85, 0.65],
        [1.00, 0.95],
    ]

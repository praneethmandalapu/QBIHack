"""Clinical color system for tumor visualization (Person 3 / jasim).

Backwards compatible: NECROTIC / VIABLE / HEALTHY and tissue_colormap() are
preserved. Everything below adds a perceptually-ordered clinical palette plus
ready-made Plotly colorscales and opacity transfer functions.
"""

from __future__ import annotations

# --- legacy constants (kept so existing imports don't break) ----------------
NECROTIC = "#8B0000"
VIABLE = "#FF4500"
HEALTHY = "#90EE90"

# --- upgraded clinical palette ---------------------------------------------
# Tissue compartments, ordered by malignancy (outer -> inner).
PROLIFERATING = "#ffd166"   # amber  – actively dividing outer rim
VIABLE_TUMOR = "#ff6b35"    # orange – bulk viable tumor
HYPOXIC = "#c41e3a"         # crimson – oxygen-starved transition zone
NECROTIC_CORE = "#5a0a14"   # dark maroon – dead core

# UI / theme accents
BG_DEEP = "#070b16"         # app background
BG_PANEL = "#0e1626"        # card / scene background
GRID = "#1b2740"            # subtle gridlines
ACCENT = "#22d3ee"          # cyan highlight
ACCENT_2 = "#a855f7"        # violet secondary
TEXT = "#e6edf7"
TEXT_DIM = "#8aa0c0"
GOOD = "#34d399"            # response / shrinkage
BAD = "#f87171"             # progression / growth


def tissue_colormap():
    """Legacy three-class palette (kept for compatibility)."""
    return {"necrotic": NECROTIC, "viable": VIABLE, "healthy": HEALTHY}


def clinical_palette() -> dict:
    """Full four-compartment clinical palette."""
    return {
        "proliferating": PROLIFERATING,
        "viable": VIABLE_TUMOR,
        "hypoxic": HYPOXIC,
        "necrotic": NECROTIC_CORE,
    }


def density_colorscale():
    """Plotly colorscale mapping density [0,1] -> tissue color.

    Low density = healthy/edge (amber), high density = necrotic core (maroon).
    Designed to read as a glowing tumor against a dark scene.
    """
    return [
        [0.00, "#10243f"],          # ambient / background tissue
        [0.40, PROLIFERATING],      # proliferating rim
        [0.62, VIABLE_TUMOR],       # viable bulk
        [0.82, HYPOXIC],            # hypoxic zone
        [1.00, NECROTIC_CORE],      # necrotic core
    ]


def density_opacityscale():
    """Opacity transfer function for go.Volume.

    Empty space stays invisible; tissue ramps up so the dense core glows
    through the translucent rim — the 'volumetric' look judges remember.
    """
    return [
        [0.00, 0.00],
        [0.30, 0.02],
        [0.50, 0.12],
        [0.70, 0.35],
        [0.85, 0.65],
        [1.00, 0.95],
    ]

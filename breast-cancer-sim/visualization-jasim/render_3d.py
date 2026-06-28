"""3D rendering + analytics for tumor density volumes (Person 3 / jasim).

Contract with Person 2 (vinesh) `solve_growth`:
    - each frame: np.float32, shape (D, H, W) = (Z, Y, X), slices-first
    - values: single density field in [0, 1]
    - render by value, isosurface at 0.5
    - solve_growth() -> list[np.ndarray], one frame per timestep

Public API (stable):
    render_volume, render_sequence, to_labels, make_dummy_sequence, downsample

Upgraded "hero" API:
    render_volumetric   - go.Volume with clinical transfer function
    render_layers       - nested isosurfaces (rim / viable / hypoxic / necrotic)
    render_cutaway      - clipped volume revealing the necrotic core
    render_slices       - radiology-style axial/coronal/sagittal triptych
    tumor_metrics       - quantitative burden for one frame (mm^3, fractions...)
    growth_analytics    - per-timestep series + doubling time + RECIST status
    render_growth_curve - tumor-burden-over-time chart
    make_treatment_sequence - grow-then-respond scenario for demos

Default backend is Plotly so Person 4 (vihari) can `st.plotly_chart(fig)`.
"""

from __future__ import annotations

import numpy as np

import color_maps as cm
from color_maps import HEALTHY, NECROTIC, VIABLE, tissue_colormap

ISO_LEVEL = 0.5            # tumor boundary
VIABLE_THRESH = 0.5        # density >= this is tumor (viable)
HYPOXIC_THRESH = 0.7       # transition / hypoxic zone
NECROTIC_THRESH = 0.85     # dense necrotic core
DEFAULT_SPACING = (1.0, 1.0, 1.0)   # mm per voxel (Z, Y, X) — overwrite from DICOM


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def to_labels(
    frame: np.ndarray,
    viable_thresh: float = VIABLE_THRESH,
    necrotic_thresh: float = NECROTIC_THRESH,
) -> np.ndarray:
    """Density field -> uint8 label map {0: healthy, 1: viable, 2: necrotic}."""
    labels = np.zeros(frame.shape, dtype=np.uint8)
    labels[frame >= viable_thresh] = 1
    labels[frame >= necrotic_thresh] = 2
    return labels


def downsample(frame: np.ndarray, factor: int = 2) -> np.ndarray:
    """Stride-decimate a volume for fast display of heavy real arrays."""
    if factor <= 1:
        return frame
    return frame[::factor, ::factor, ::factor]


def normalize_intensity(vol: np.ndarray, hi_pct: float = 99.5) -> np.ndarray:
    """Display-window a real MR-derived field so its bright region maps to ~1.

    Philip's `pde_npy` volumes are normalized MR intensities, not saturating
    tumor density: most tissue sits at 0.1-0.5 and the dynamic range varies per
    case. A fixed isosurface at 0.5 would over/under-shoot. This contrast-stretch
    rescales each volume by its own high percentile (background stays 0), so the
    tissue structure renders legibly without changing the source files.
    """
    nz = vol[vol > 0]
    if nz.size == 0:
        return vol.astype(np.float32)
    hi = float(np.percentile(nz, hi_pct))
    if hi <= 0:
        return vol.astype(np.float32)
    return np.clip(vol / hi, 0.0, 1.0).astype(np.float32)


def load_pde_volume(slug: str, data_root: str | None = None, normalize: bool = True):
    """Load a Philip/Chandan PDE-ready volume by slug via the manifest.

    Returns (volume, entry). `volume` is float32 (Z,Y,X) in [0,1]; `entry` is the
    manifest record (subtype, timepoint, tcga_id, ...). Loads `pde_npy` only —
    never the full-resolution `raw_npy`.
    """
    import json
    from pathlib import Path

    root = Path(data_root) if data_root else Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "data/processed/raw-extract-philip-chandan/manifest.json").read_text())
    entry = next(v for v in manifest["volumes"] if v["slug"] == slug)
    vol = np.load(root / entry["pde_npy"]).astype(np.float32)
    if normalize:
        vol = normalize_intensity(vol)
    return vol, entry


def make_dummy_sequence(
    n: int = 24,
    shape: tuple[int, int, int] = (64, 64, 64),
    seed: int = 0,
) -> list[np.ndarray]:
    """Stand-in for solve_growth(): a growing tumor with a denser core."""
    rng = np.random.default_rng(seed)
    D, H, W = shape
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
    cz, cy, cx = D / 2, H / 2, W / 2
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)

    # a little anisotropy + lobulation so it doesn't look like a perfect ball
    lobe = 1.0 + 0.18 * np.sin(3 * np.arctan2(yy - cy, xx - cx))

    frames: list[np.ndarray] = []
    max_r = 0.45 * min(shape)
    for t in range(n):
        radius = max_r * (0.25 + 0.75 * t / max(n - 1, 1)) * lobe
        frame = np.clip(1.0 - dist / radius, 0.0, 1.0).astype(np.float32)
        # sharpen the core so a necrotic centre emerges as it grows
        frame = frame ** (1.0 - 0.25 * t / max(n - 1, 1))
        frame += rng.normal(0, 0.015, size=shape).astype(np.float32)
        frames.append(np.clip(frame, 0.0, 1.0).astype(np.float32))
    return frames


def make_treatment_sequence(
    n: int = 28,
    shape: tuple[int, int, int] = (64, 64, 64),
    therapy_start: int = 12,
    response: float = 0.6,
    seed: int = 0,
) -> list[np.ndarray]:
    """Grow, then regress after `therapy_start` — a clinical response scenario.

    `response` in [0,1]: fraction of growth reversed per step after therapy.
    Lets the demo show the tumor shrinking under a drug before Person 2's
    intervention engine exists.
    """
    rng = np.random.default_rng(seed)
    D, H, W = shape
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
    cz, cy, cx = D / 2, H / 2, W / 2
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    lobe = 1.0 + 0.18 * np.sin(3 * np.arctan2(yy - cy, xx - cx))
    max_r = 0.45 * min(shape)

    frames: list[np.ndarray] = []
    radius_scale = 0.25
    for t in range(n):
        if t < therapy_start:
            radius_scale = 0.25 + 0.75 * t / max(therapy_start, 1)
        else:
            # exponential shrinkage of the *radius* under therapy
            radius_scale *= (1.0 - 0.18 * response)
        radius = max_r * max(radius_scale, 0.05) * lobe
        frame = np.clip(1.0 - dist / radius, 0.0, 1.0).astype(np.float32)
        frame = frame ** 0.9
        frame += rng.normal(0, 0.015, size=shape).astype(np.float32)
        frames.append(np.clip(frame, 0.0, 1.0).astype(np.float32))
    return frames


# --------------------------------------------------------------------------- #
# Quantitative analytics  (this is what turns a picture into a "digital twin")
# --------------------------------------------------------------------------- #
def tumor_metrics(frame: np.ndarray, spacing: tuple[float, float, float] = DEFAULT_SPACING) -> dict:
    """Quantify one frame: volumes (mm^3), fractions, and max diameter (mm)."""
    voxel_mm3 = float(np.prod(spacing))
    viable_mask = frame >= VIABLE_THRESH
    hypoxic_mask = frame >= HYPOXIC_THRESH
    necrotic_mask = frame >= NECROTIC_THRESH

    viable_vox = int(viable_mask.sum())
    necrotic_vox = int(necrotic_mask.sum())
    total_mm3 = viable_vox * voxel_mm3

    # max 3D diameter from the viable bounding box (RECIST-ish longest axis)
    if viable_vox > 0:
        zs, ys, xs = np.where(viable_mask)
        extent = np.array([
            (zs.max() - zs.min()) * spacing[0],
            (ys.max() - ys.min()) * spacing[1],
            (xs.max() - xs.min()) * spacing[2],
        ])
        max_diameter = float(np.linalg.norm(extent))
        longest_axis = float(extent.max())
    else:
        max_diameter = longest_axis = 0.0

    return {
        "total_mm3": total_mm3,
        "viable_mm3": (viable_vox - necrotic_vox) * voxel_mm3,
        "necrotic_mm3": necrotic_vox * voxel_mm3,
        "hypoxic_mm3": int(hypoxic_mask.sum()) * voxel_mm3,
        "necrotic_fraction": (necrotic_vox / viable_vox) if viable_vox else 0.0,
        "max_diameter_mm": max_diameter,
        "longest_axis_mm": longest_axis,
    }


def growth_analytics(
    volumes: list[np.ndarray],
    spacing: tuple[float, float, float] = DEFAULT_SPACING,
    days_per_step: float = 7.0,
) -> dict:
    """Per-timestep series + derived clinical readouts.

    Returns days, volume_mm3, diameter_mm, necrotic_fraction arrays plus
    doubling_time_days, growth_rate, and RECIST response category.
    """
    series = [tumor_metrics(v, spacing) for v in volumes]
    days = np.arange(len(volumes)) * days_per_step
    vol = np.array([m["total_mm3"] for m in series])
    diam = np.array([m["max_diameter_mm"] for m in series])
    necf = np.array([m["necrotic_fraction"] for m in series])

    # doubling time from baseline to peak (exponential fit V = V0 * 2^(t/Td))
    peak = int(np.argmax(vol))
    if peak > 0 and vol[0] > 0 and vol[peak] > vol[0]:
        doubling = (days[peak] - days[0]) / np.log2(vol[peak] / vol[0])
    else:
        doubling = float("nan")

    # RECIST 1.1 on longest diameter: baseline -> current
    base_d = diam[0] if diam[0] > 0 else (diam[diam > 0][0] if (diam > 0).any() else 1.0)
    cur_d = diam[-1]
    nadir = float(np.min(diam)) if len(diam) else cur_d
    delta_pct = (cur_d - base_d) / base_d * 100 if base_d else 0.0
    if cur_d >= 1.2 * nadir and cur_d > base_d:
        recist = "Progressive Disease"
    elif cur_d == 0:
        recist = "Complete Response"
    elif cur_d <= 0.7 * base_d:
        recist = "Partial Response"
    else:
        recist = "Stable Disease"

    return {
        "days": days,
        "volume_mm3": vol,
        "diameter_mm": diam,
        "necrotic_fraction": necf,
        "peak_index": peak,
        "doubling_time_days": float(doubling),
        "diameter_change_pct": float(delta_pct),
        "recist": recist,
        "series": series,
    }


# --------------------------------------------------------------------------- #
# Scene styling
# --------------------------------------------------------------------------- #
def _dark_scene(fig, title: str | None = None):
    fig.update_layout(
        paper_bgcolor=cm.BG_PANEL,
        plot_bgcolor=cm.BG_PANEL,
        font=dict(color=cm.TEXT, family="Inter, Segoe UI, sans-serif"),
        margin=dict(l=0, r=0, t=30 if title else 0, b=0),
        title=dict(text=title or "", x=0.02, font=dict(size=15, color=cm.TEXT_DIM)),
        scene=dict(
            xaxis=dict(title="X (mm)", backgroundcolor=cm.BG_PANEL,
                       gridcolor=cm.GRID, zerolinecolor=cm.GRID, color=cm.TEXT_DIM),
            yaxis=dict(title="Y (mm)", backgroundcolor=cm.BG_PANEL,
                       gridcolor=cm.GRID, zerolinecolor=cm.GRID, color=cm.TEXT_DIM),
            zaxis=dict(title="Z (mm)", backgroundcolor=cm.BG_PANEL,
                       gridcolor=cm.GRID, zerolinecolor=cm.GRID, color=cm.TEXT_DIM),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.5, z=1.1)),
        ),
    )
    return fig


# --------------------------------------------------------------------------- #
# Rendering  (legacy-compatible)
# --------------------------------------------------------------------------- #
def render_volume(volume, colormap=None, backend: str = "plotly", iso: float = ISO_LEVEL):
    """Render one density frame as an interactive 3D object (returns a Figure)."""
    if backend == "plotly":
        return _render_isosurface(volume, iso)
    if backend == "pyvista":
        return _render_pyvista(volume, iso)
    raise ValueError(f"unknown backend: {backend!r}")


def _render_isosurface(volume: np.ndarray, iso: float):
    import plotly.graph_objects as go

    D, H, W = volume.shape
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
    fig = go.Figure(go.Isosurface(
        x=xx.ravel(), y=yy.ravel(), z=zz.ravel(), value=volume.ravel(),
        isomin=iso, isomax=1.0, surface_count=3,
        colorscale=cm.density_colorscale(),
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=0.6, showscale=False,
    ))
    return _dark_scene(fig)


# --------------------------------------------------------------------------- #
# Rendering  (hero / upgraded)
# --------------------------------------------------------------------------- #
def render_volumetric(volume: np.ndarray, opacity_gain: float = 1.0):
    """True volumetric render with a clinical transfer function (the hero shot)."""
    import plotly.graph_objects as go

    D, H, W = volume.shape
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
    opac = [[p, min(1.0, a * opacity_gain)] for p, a in cm.density_opacityscale()]
    fig = go.Figure(go.Volume(
        x=xx.ravel(), y=yy.ravel(), z=zz.ravel(), value=volume.ravel(),
        isomin=0.15, isomax=1.0,
        colorscale=cm.density_colorscale(),
        opacityscale=opac,
        surface_count=18,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorbar=dict(title="density", thickness=12, len=0.6,
                      tickfont=dict(color=cm.TEXT_DIM)),
    ))
    return _dark_scene(fig)


def render_layers(volume: np.ndarray):
    """Nested isosurfaces: proliferating rim -> viable -> hypoxic -> necrotic."""
    import plotly.graph_objects as go

    D, H, W = volume.shape
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
    layers = [
        (VIABLE_THRESH, cm.PROLIFERATING, 0.18, "Proliferating rim"),
        (0.62, cm.VIABLE_TUMOR, 0.30, "Viable tumor"),
        (HYPOXIC_THRESH, cm.HYPOXIC, 0.55, "Hypoxic zone"),
        (NECROTIC_THRESH, cm.NECROTIC_CORE, 0.95, "Necrotic core"),
    ]
    fig = go.Figure()
    for lo, color, op, name in layers:
        fig.add_trace(go.Isosurface(
            x=xx.ravel(), y=yy.ravel(), z=zz.ravel(), value=volume.ravel(),
            isomin=lo, isomax=min(lo + 0.12, 1.0), surface_count=1,
            colorscale=[[0, color], [1, color]], showscale=False,
            opacity=op, name=name, showlegend=True,
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))
    fig = _dark_scene(fig)
    fig.update_layout(legend=dict(bgcolor="rgba(14,22,38,0.6)", font=dict(color=cm.TEXT)))
    return fig


def render_cutaway(volume: np.ndarray, plane: str = "y"):
    """Clip the volume on a half-space to expose the necrotic core."""
    D, H, W = volume.shape
    clipped = volume.copy()
    if plane == "x":
        clipped[:, :, W // 2:] = 0.0
    elif plane == "z":
        clipped[D // 2:, :, :] = 0.0
    else:  # y
        clipped[:, H // 2:, :] = 0.0
    return render_volumetric(clipped, opacity_gain=1.15)


def render_slices(volume: np.ndarray, spacing: tuple[float, float, float] = DEFAULT_SPACING):
    """Radiology triptych: axial / coronal / sagittal mid-planes (2D)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    D, H, W = volume.shape
    axial = volume[D // 2, :, :]
    coronal = volume[:, H // 2, :]
    sagittal = volume[:, :, W // 2]
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=("Axial", "Coronal", "Sagittal"),
                        horizontal_spacing=0.04)
    for col, img in enumerate([axial, coronal, sagittal], start=1):
        fig.add_trace(go.Heatmap(
            z=img, colorscale=cm.density_colorscale(), zmin=0, zmax=1,
            showscale=(col == 3),
            colorbar=dict(thickness=10, len=0.9, tickfont=dict(color=cm.TEXT_DIM)),
        ), row=1, col=col)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor=None)
    fig.update_layout(
        paper_bgcolor=cm.BG_PANEL, plot_bgcolor=cm.BG_PANEL,
        font=dict(color=cm.TEXT_DIM), margin=dict(l=4, r=4, t=24, b=4), height=240,
    )
    for ann in fig.layout.annotations:
        ann.font.color = cm.TEXT
        ann.font.size = 13
    return fig


def render_growth_curve(analytics: dict):
    """Tumor-burden-over-time chart with therapy/peak markers."""
    import plotly.graph_objects as go

    days = analytics["days"]
    vol = analytics["volume_mm3"]
    diam = analytics["diameter_mm"]
    peak = analytics["peak_index"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=days, y=vol, mode="lines", name="Tumor volume (mm³)",
        line=dict(color=cm.ACCENT, width=3),
        fill="tozeroy", fillcolor="rgba(34,211,238,0.10)",
    ))
    fig.add_trace(go.Scatter(
        x=days, y=diam, mode="lines", name="Max diameter (mm)",
        line=dict(color=cm.VIABLE_TUMOR, width=2, dash="dot"), yaxis="y2",
    ))
    fig.add_trace(go.Scatter(
        x=[days[peak]], y=[vol[peak]], mode="markers+text",
        marker=dict(color=cm.ACCENT_2, size=11, line=dict(color="white", width=1)),
        text=["peak"], textposition="top center",
        textfont=dict(color=cm.TEXT_DIM), showlegend=False,
    ))
    fig.update_layout(
        paper_bgcolor=cm.BG_PANEL, plot_bgcolor=cm.BG_PANEL,
        font=dict(color=cm.TEXT), height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="days", gridcolor=cm.GRID, zerolinecolor=cm.GRID),
        yaxis=dict(title="volume (mm³)", gridcolor=cm.GRID, zerolinecolor=cm.GRID),
        yaxis2=dict(title="diameter (mm)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def render_sequence(volumes: list[np.ndarray], iso: float = ISO_LEVEL):
    """Animated Plotly figure (Play button + timestep slider) over all frames."""
    import plotly.graph_objects as go

    base = _render_isosurface(volumes[0], iso)
    frames = [go.Frame(data=_render_isosurface(v, iso).data, name=str(i))
              for i, v in enumerate(volumes)]
    base.frames = frames
    base.update_layout(
        sliders=[dict(
            active=0, currentvalue=dict(prefix="timestep ", font=dict(color=cm.ACCENT)),
            bgcolor=cm.GRID, bordercolor=cm.GRID, font=dict(color=cm.TEXT_DIM),
            steps=[dict(method="animate", label=str(i),
                        args=[[str(i)], dict(mode="immediate",
                              frame=dict(duration=0, redraw=True))])
                   for i in range(len(volumes))],
        )],
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.02, y=0.02,
            bgcolor=cm.BG_DEEP, font=dict(color=cm.TEXT),
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(fromcurrent=True,
                           frame=dict(duration=120, redraw=True))]),
                dict(label="❚❚ Pause", method="animate",
                     args=[[None], dict(mode="immediate",
                           frame=dict(duration=0, redraw=False))]),
            ],
        )],
    )
    return base


def _render_pyvista(volume: np.ndarray, iso: float):
    """Optional high-fidelity local render (needs pyvista + a GL context)."""
    import pyvista as pv

    grid = pv.ImageData(dimensions=np.array(volume.shape) + 1)
    grid.cell_data["density"] = volume.flatten(order="F")
    contour = grid.cell_data_to_point_data().contour([iso], scalars="density")
    pl = pv.Plotter()
    pl.add_mesh(contour, color=VIABLE, opacity=0.7)
    return pl


if __name__ == "__main__":
    seq = make_treatment_sequence()
    a = growth_analytics(seq, days_per_step=7.0)
    print(f"frames={len(seq)} shape={seq[0].shape} dtype={seq[0].dtype}")
    print(f"doubling_time={a['doubling_time_days']:.1f}d  "
          f"diam_change={a['diameter_change_pct']:+.0f}%  RECIST={a['recist']}")
    ds = [downsample(f, 2) for f in seq]
    render_sequence(ds).write_html("tumor_preview.html")
    render_volumetric(ds[a["peak_index"]]).write_html("tumor_volumetric.html")
    print("wrote tumor_preview.html + tumor_volumetric.html")

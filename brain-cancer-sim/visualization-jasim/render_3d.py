"""3D rendering + analytics for tumor density volumes.

Ported from breast-cancer-sim/visualization-jasim/render_3d.py @ QBIHack 5119ad6.
Contract with solver `solve_growth`: float32 (Z,Y,X) density in [0, 1].
"""

from __future__ import annotations

import numpy as np

import color_maps as cm
from color_maps import VIABLE, tissue_colormap

ISO_LEVEL = 0.5
VIABLE_THRESH = 0.5
HYPOXIC_THRESH = 0.7
NECROTIC_THRESH = 0.85
DEFAULT_SPACING = (1.0, 1.0, 1.0)


def to_labels(
    frame: np.ndarray,
    viable_thresh: float = VIABLE_THRESH,
    necrotic_thresh: float = NECROTIC_THRESH,
) -> np.ndarray:
    labels = np.zeros(frame.shape, dtype=np.uint8)
    labels[frame >= viable_thresh] = 1
    labels[frame >= necrotic_thresh] = 2
    return labels


def downsample(frame: np.ndarray, factor: int = 2) -> np.ndarray:
    if factor <= 1:
        return frame
    return frame[::factor, ::factor, ::factor]


def make_dummy_sequence(
    n: int = 24,
    shape: tuple[int, int, int] = (64, 64, 64),
    seed: int = 0,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    D, H, W = shape
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
    cz, cy, cx = D / 2, H / 2, W / 2
    dist = np.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
    lobe = 1.0 + 0.18 * np.sin(3 * np.arctan2(yy - cy, xx - cx))
    frames: list[np.ndarray] = []
    max_r = 0.45 * min(shape)
    for t in range(n):
        radius = max_r * (0.25 + 0.75 * t / max(n - 1, 1)) * lobe
        frame = np.clip(1.0 - dist / radius, 0.0, 1.0).astype(np.float32)
        frame = frame ** (1.0 - 0.25 * t / max(n - 1, 1))
        frame += rng.normal(0, 0.015, size=shape).astype(np.float32)
        frames.append(np.clip(frame, 0.0, 1.0).astype(np.float32))
    return frames


def tumor_metrics(frame: np.ndarray, spacing: tuple[float, float, float] = DEFAULT_SPACING) -> dict:
    voxel_mm3 = float(np.prod(spacing))
    viable_mask = frame >= VIABLE_THRESH
    hypoxic_mask = frame >= HYPOXIC_THRESH
    necrotic_mask = frame >= NECROTIC_THRESH
    viable_vox = int(viable_mask.sum())
    necrotic_vox = int(necrotic_mask.sum())
    total_mm3 = viable_vox * voxel_mm3
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
    series = [tumor_metrics(v, spacing) for v in volumes]
    days = np.arange(len(volumes)) * days_per_step
    vol = np.array([m["total_mm3"] for m in series])
    diam = np.array([m["max_diameter_mm"] for m in series])
    necf = np.array([m["necrotic_fraction"] for m in series])
    peak = int(np.argmax(vol))
    if peak > 0 and vol[0] > 0 and vol[peak] > vol[0]:
        doubling = (days[peak] - days[0]) / np.log2(vol[peak] / vol[0])
    else:
        doubling = float("nan")
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


def render_volume(volume, colormap=None, backend: str = "plotly", iso: float = ISO_LEVEL):
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


def render_volumetric(volume: np.ndarray, opacity_gain: float = 1.0):
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


def render_growth_curve(analytics: dict):
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
    import pyvista as pv

    grid = pv.ImageData(dimensions=np.array(volume.shape) + 1)
    grid.cell_data["density"] = volume.flatten(order="F")
    contour = grid.cell_data_to_point_data().contour([iso], scalars="density")
    pl = pv.Plotter()
    pl.add_mesh(contour, color=VIABLE, opacity=0.7)
    return pl


if __name__ == "__main__":
    seq = make_dummy_sequence()
    a = growth_analytics(seq, days_per_step=7.0)
    print(f"frames={len(seq)} shape={seq[0].shape} dtype={seq[0].dtype}")
    print(f"doubling_time={a['doubling_time_days']:.1f}d  RECIST={a['recist']}")
    ds = [downsample(f, 2) for f in seq]
    render_sequence(ds).write_html("tumor_preview.html")
    print("wrote tumor_preview.html")

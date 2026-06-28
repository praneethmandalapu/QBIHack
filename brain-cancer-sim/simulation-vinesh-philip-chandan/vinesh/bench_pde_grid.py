"""Benchmark PDE prep (Philip handoff) and solve_growth across grid sizes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

VINESH_DIR = Path(__file__).resolve().parent
SPIKE_ROOT = VINESH_DIR.parent
REPO_ROOT = SPIKE_ROOT.parent
sys.path.insert(0, str(SPIKE_ROOT))
sys.path.insert(0, str(VINESH_DIR))

from handoff_contract import default_grid_size, grid_size_options, spike_patient  # noqa: E402
from prepare_pde_input import (  # noqa: E402
    load_expert_mask,
    load_raw_extract,
    prepare_pde_input,
    save_pde_input,
)
from run_growth import run_growth  # noqa: E402
from spike_paths import (  # noqa: E402
    QC_PDE_PREP_VINESH,
    ensure_spike_dirs,
    pde_input_npy,
    resolve_pde_input_npy,
)
from tumor_pde_solver import total_volume  # noqa: E402


def _time_prep(slug: str, grid_size: int) -> dict:
    """Time in-memory prep + disk write for one grid size."""
    t0 = time.perf_counter()
    raw_volume, raw_metadata = load_raw_extract(slug)
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    expert_mask, mask_path = load_expert_mask(raw_metadata, raw_volume.shape)
    mask_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    pde_volume, pde_spacing = prepare_pde_input(
        raw_volume,
        raw_metadata["spacing_mm"],
        expert_mask,
        grid_size=grid_size,
    )
    prep_s = time.perf_counter() - t2

    t3 = time.perf_counter()
    npy_path, json_path = save_pde_input(
        pde_volume,
        pde_spacing,
        raw_metadata,
        segmentation_path=mask_path,
        slug=slug,
        grid_size=grid_size,
    )
    write_s = time.perf_counter() - t3

    bg = 0.0
    tumor_voxels = int((pde_volume > bg).sum())
    nbytes = pde_volume.nbytes

    return {
        "grid_size": grid_size,
        "shape": list(pde_volume.shape),
        "tumor_voxels": tumor_voxels,
        "nbytes": nbytes,
        "load_raw_s": load_s,
        "load_mask_s": mask_s,
        "prep_s": prep_s,
        "write_s": write_s,
        "prep_total_s": load_s + mask_s + prep_s + write_s,
        "npy_path": str(npy_path.relative_to(REPO_ROOT)),
        "json_path": str(json_path.relative_to(REPO_ROOT)),
    }


def _time_solve(slug: str, grid_size: int) -> dict:
    npy_path = resolve_pde_input_npy(slug, grid_size=grid_size)
    vol = np.load(npy_path)
    spacing = (1.0, 1.0, 1.0)

    t0 = time.perf_counter()
    frames = run_growth(vol, params={"spacing": spacing})
    solve_s = time.perf_counter() - t0

    v0 = total_volume(frames[0], spacing=spacing, threshold=0.0)
    vend = total_volume(frames[-1], spacing=spacing, threshold=0.5)
    frame_bytes = sum(f.nbytes for f in frames)

    return {
        "grid_size": grid_size,
        "solve_s": solve_s,
        "n_frames": len(frames),
        "frame_bytes": frame_bytes,
        "volume_t0_mm3": v0,
        "volume_tend_mm3": vend,
    }


def run_benchmark(slug: str, grid_sizes: list[int] | None = None) -> dict:
    ensure_spike_dirs()
    sizes = grid_sizes or list(grid_size_options())
    rows: list[dict] = []

    for size in sizes:
        prep = _time_prep(slug, size)
        solve = _time_solve(slug, size)
        rows.append({**prep, **solve, "slug": slug})

    return {"slug": slug, "runs": rows}


def _print_table(result: dict) -> None:
    print(f"\nPDE grid benchmark — {result['slug']}\n")
    header = (
        f"{'grid':>6}  {'shape':>14}  {'tumor vox':>10}  "
        f"{'prep (s)':>9}  {'solve (s)':>10}  {'total (s)':>10}  {'pde MB':>7}"
    )
    print(header)
    print("-" * len(header))
    for row in result["runs"]:
        total = row["prep_total_s"] + row["solve_s"]
        mb = row["nbytes"] / (1024 * 1024)
        shape = "x".join(str(n) for n in row["shape"])
        print(
            f"g{row['grid_size']:>4}  {shape:>14}  {row['tumor_voxels']:>10,}  "
            f"{row['prep_total_s']:>9.3f}  {row['solve_s']:>10.3f}  {total:>10.3f}  {mb:>7.2f}"
        )
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Time prepare_pde_input + solve_growth for contract grid size(s)."
    )
    parser.add_argument("--slug", default=spike_patient()["slug"])
    parser.add_argument(
        "--grid-size",
        type=int,
        choices=grid_size_options(),
        action="append",
        dest="grid_sizes",
        help="Repeatable; default: all options in contract",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=QC_PDE_PREP_VINESH / "grid_benchmark.json",
        help="Write machine-readable timings",
    )
    args = parser.parse_args(argv)

    sizes = args.grid_sizes or list(grid_size_options())
    result = run_benchmark(args.slug, sizes)
    _print_table(result)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()

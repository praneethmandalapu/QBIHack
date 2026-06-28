"""CLI entry point for segmentation outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SEGMENTATION_DIR = Path(__file__).resolve().parent
STRETCH_DIR = SEGMENTATION_DIR.parent / "stretch"
PHILIP_CHANDAN_DIR = SEGMENTATION_DIR.parent

sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(STRETCH_DIR))
sys.path.insert(0, str(SEGMENTATION_DIR))

from ground_truth import write_les_reference  # noqa: E402
from seg_paths import BENCHMARK_METHODS, REFERENCE_METHOD  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one segmentation step for a manifest slug.")
    parser.add_argument("--slug", required=True, help="Manifest slug (baseline for .les reference)")
    parser.add_argument(
        "--method",
        choices=(REFERENCE_METHOD, *BENCHMARK_METHODS),
        default=REFERENCE_METHOD,
        help=f"Segmentation method (default: {REFERENCE_METHOD} reference embed)",
    )
    args = parser.parse_args()

    if args.method == REFERENCE_METHOD:
        _, meta = write_les_reference(args.slug)
        print(f"Wrote .les reference for {args.slug}")
        print(f"  mask_voxels={meta['mask_voxels']:,} → {meta['mask_npy']}")
        return

    raise NotImplementedError(
        f"Automated method {args.method!r} is not implemented yet. "
        f"Write {{slug}}_{args.method}_mask.npy manually or add methods/{args.method}.py, "
        "then run run_benchmark.py."
    )


if __name__ == "__main__":
    main()

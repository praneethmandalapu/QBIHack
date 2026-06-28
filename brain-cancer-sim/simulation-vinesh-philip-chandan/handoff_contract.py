"""Load the versioned imaging ↔ solver handoff contract."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

SIM_ROOT = Path(__file__).resolve().parent
DEFAULT_CONTRACT_PATH = SIM_ROOT / "handoff_contract.json"


@lru_cache(maxsize=4)
def load_handoff_contract(path: str | None = None) -> dict[str, Any]:
    """Parse handoff_contract.json. Pass path to override the default file."""
    contract_path = Path(path) if path else DEFAULT_CONTRACT_PATH
    with contract_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def contract_version(path: str | None = None) -> str:
    return str(load_handoff_contract(path)["version"])


def spike_patient(path: str | None = None) -> dict[str, str]:
    return dict(load_handoff_contract(path)["spike_patient"])


def raw_extract_spec(path: str | None = None) -> dict[str, Any]:
    return dict(load_handoff_contract(path)["raw_extract"])


def pde_input_spec(path: str | None = None) -> dict[str, Any]:
    return dict(load_handoff_contract(path)["pde_input"])


def solver_spec(path: str | None = None) -> dict[str, Any]:
    return dict(load_handoff_contract(path)["solver"])


def default_grid_size(path: str | None = None) -> int:
    spec = pde_input_spec(path)
    if "default_grid_size" in spec:
        return int(spec["default_grid_size"])
    shape = spec["max_shape"]
    return int(shape[0])


def grid_size_options(path: str | None = None) -> tuple[int, ...]:
    spec = pde_input_spec(path)
    if "grid_size_options" in spec:
        return tuple(int(n) for n in spec["grid_size_options"])
    size = default_grid_size(path)
    return (size,)


def max_shape_for_grid(grid_size: int, path: str | None = None) -> tuple[int, int, int]:
    allowed = grid_size_options(path)
    if grid_size not in allowed:
        raise ValueError(f"grid_size must be one of {allowed}, got {grid_size}")
    return grid_size, grid_size, grid_size


def max_shape(path: str | None = None) -> tuple[int, int, int]:
    return max_shape_for_grid(default_grid_size(path), path)


def target_spacing_mm(path: str | None = None) -> list[float]:
    spacing = pde_input_spec(path)["target_spacing_mm"]
    return [float(value) for value in spacing]

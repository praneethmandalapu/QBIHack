"""Load the versioned Philip-Chandan ↔ Vinesh handoff contract."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

SPIKE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONTRACT_PATH = SPIKE_ROOT / "handoff_contract.json"


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


def max_shape(path: str | None = None) -> tuple[int, int, int]:
    shape = pde_input_spec(path)["max_shape"]
    return int(shape[0]), int(shape[1]), int(shape[2])


def target_spacing_mm(path: str | None = None) -> list[float]:
    spacing = pde_input_spec(path)["target_spacing_mm"]
    return [float(value) for value in spacing]

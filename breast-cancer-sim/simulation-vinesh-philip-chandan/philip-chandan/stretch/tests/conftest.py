"""Fixtures for stretch tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synthetic_raw_sphere() -> tuple[np.ndarray, list[float]]:
    """Bright Gaussian blob on dim background, anisotropic spacing like real MR."""
    shape = (32, 48, 48)
    spacing = [3.0, 0.86, 0.86]
    zz, yy, xx = np.indices(shape)
    cz, cy, cx = (s / 2 for s in shape)
    r2 = ((zz - cz) / 4) ** 2 + ((yy - cy) / 8) ** 2 + ((xx - cx) / 8) ** 2
    raw = 200.0 + 800.0 * np.exp(-r2)
    return raw.astype(np.float32), spacing

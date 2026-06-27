"""Read Philip-Chandan manifest.json (read-only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paths import MANIFEST_PATH


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = path or MANIFEST_PATH
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest at {manifest_path}. Run export_all_raw.py first."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_volumes(path: Path | None = None) -> list[dict[str, Any]]:
    return list(load_manifest(path)["volumes"])


def find_volume(
    *,
    slug: str | None = None,
    subtype: str | None = None,
    timepoint: str | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    volumes = load_volumes(path)
    if slug is not None:
        for entry in volumes:
            if entry["slug"] == slug:
                return entry
        raise KeyError(f"Slug not in manifest: {slug}")
    if subtype is None or timepoint is None:
        raise ValueError("Provide slug or both subtype and timepoint")
    for entry in volumes:
        if entry["subtype"] == subtype and entry["timepoint"] == timepoint:
            return entry
    raise KeyError(f"No manifest entry for {subtype!r} / {timepoint!r}")

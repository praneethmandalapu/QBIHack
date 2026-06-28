"""One-time migration: flat slug volumes → nested {tcga_id}/{timepoint} layout."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

PHILIP_CHANDAN_DIR = Path(__file__).resolve().parent
SPIKE_ROOT = PHILIP_CHANDAN_DIR.parent
REPO_ROOT = SPIKE_ROOT.parent

sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(SPIKE_ROOT))

from generate_manifest import write_manifest  # noqa: E402
from spike_paths import (  # noqa: E402
    PDE_INPUT_VINESH,
    RAW_EXTRACT_PHILIP_CHANDAN,
    pde_input_metadata,
    pde_input_npy,
    pde_input_npy_legacy,
    raw_extract_metadata,
    raw_extract_npy,
    raw_extract_npy_legacy,
    slug_to_tcga_timepoint,
)


def _move_pair(src_npy: Path, dst_npy: Path, *, dry_run: bool) -> bool:
    src_json = src_npy.with_suffix(".json")
    dst_json = dst_npy.with_suffix(".json")
    if not src_npy.is_file():
        return False
    if dst_npy.is_file():
        return False
    if dry_run:
        print(f"  would move {src_npy.name} -> {dst_npy}")
        return True
    dst_npy.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_npy), str(dst_npy))
    if src_json.is_file():
        meta = json.loads(src_json.read_text(encoding="utf-8"))
        tcga_id, timepoint = slug_to_tcga_timepoint(str(meta.get("slug", src_npy.stem)))
        meta.setdefault("tcga_id", tcga_id)
        meta.setdefault("timepoint", timepoint)
        dst_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        src_json.unlink()
    return True


def _update_pde_source_raw(slug: str, *, dry_run: bool) -> None:
    pde_json = pde_input_metadata(slug)
    legacy_pde_json = pde_input_npy_legacy(slug).with_suffix(".json")
    path = pde_json if pde_json.is_file() else legacy_pde_json
    if not path.is_file():
        return
    meta = json.loads(path.read_text(encoding="utf-8"))
    new_source = str(raw_extract_npy(slug).relative_to(REPO_ROOT))
    if meta.get("source_raw_extract") == new_source:
        return
    meta["source_raw_extract"] = new_source
    if dry_run:
        print(f"  would update source_raw_extract in {path.name}")
        return
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def migrate(*, dry_run: bool = False) -> None:
    moved = 0
    for legacy_npy in sorted(RAW_EXTRACT_PHILIP_CHANDAN.glob("*.npy")):
        slug = legacy_npy.stem
        tcga_id, timepoint = slug_to_tcga_timepoint(slug)
        dst = raw_extract_npy(slug, tcga_id=tcga_id, timepoint=timepoint)
        if _move_pair(legacy_npy, dst, dry_run=dry_run):
            moved += 1
            print(f"raw: {slug} -> {dst.relative_to(REPO_ROOT)}")

    for legacy_npy in sorted(PDE_INPUT_VINESH.glob("*.npy")):
        slug = legacy_npy.stem
        tcga_id, timepoint = slug_to_tcga_timepoint(slug)
        dst = pde_input_npy(slug, tcga_id=tcga_id, timepoint=timepoint)
        if _move_pair(legacy_npy, dst, dry_run=dry_run):
            moved += 1
            print(f"pde: {slug} -> {dst.relative_to(REPO_ROOT)}")

    slugs: set[str] = set()
    for json_path in RAW_EXTRACT_PHILIP_CHANDAN.rglob("*.json"):
        if json_path.name in ("manifest.json",):
            continue
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        slug = str(meta.get("slug") or "")
        if slug:
            slugs.add(slug)
    for slug in sorted(slugs):
        _update_pde_source_raw(slug, dry_run=dry_run)

    if not dry_run and moved:
        manifest_path = write_manifest()
        print(f"Wrote {manifest_path}")
    print(f"Done ({moved} volume pairs migrated)")


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    migrate(dry_run=dry_run)


if __name__ == "__main__":
    main()

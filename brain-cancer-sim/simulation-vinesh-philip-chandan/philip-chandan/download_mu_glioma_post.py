"""Download MU-Glioma-Post from TCIA into data/raw/mu_glioma_post/.

The imaging bundle (~11 GB NIfTI + segmentations) is distributed via TCIA Faspex
(IBM Aspera). Clinical spreadsheets are plain HTTPS and download without Aspera.

Usage (from brain-cancer-sim/):
  python simulation-vinesh-philip-chandan/philip-chandan/download_mu_glioma_post.py --metadata-only
  python simulation-vinesh-philip-chandan/philip-chandan/download_mu_glioma_post.py --imaging

Requires for --imaging:
  - Ruby >= 3.2 + ``gem install aspera-cli`` OR IBM Aspera Connect browser plugin
  - ``ascli config ascp install`` (installs the FASP transfer engine)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw" / "mu_glioma_post"
METADATA_DIR = RAW_DIR / "_metadata"

CLINICAL_XLSX = (
    "https://www.cancerimagingarchive.net/wp-content/uploads/"
    "MU-Glioma-Post_ClinicalData-July2025.xlsx"
)
SCANNER_XLSX = (
    "https://www.cancerimagingarchive.net/wp-content/uploads/MR_Scanner_data.xlsx"
)
FASPEX_IMAGING_URL = (
    "https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?"
    "context=eyJyZXNvdXJjZSI6InBhY2thZ2VzIiwidHlwZSI6ImV4dGVybmFsX2Rvd25sb2FkX3BhY2thZ2UiLCJpZCI6"
    "IjEwMzAiLCJwYXNzY29kZSI6ImYyMGMxZDY1ODgwOWM1MWJhNWE3NGIwNTY0NGE4YWNmYzVlYTBmZjUiLCJw"
    "YWNrYWdlX2lkIjoiMTAzMCIsImVtYWlsIjoiaGVscEBjYW5jZXJpbWFnaW5nYXJjaGl2ZS5uZXQifQ=="
)

TCIA_COLLECTION_PAGE = "https://www.cancerimagingarchive.net/collection/mu-glioma-post/"


def _faspex_receive_cmd(ascli: str, dest: Path) -> list[str]:
    """Build ascli argv; URL and folder are separate args (URL ends with ``==``)."""
    return [
        ascli,
        "faspex5",
        "packages",
        "receive",
        f"--url={FASPEX_IMAGING_URL}",
        f"--to-folder={dest.resolve()}",
        "1030",
    ]


def _manual_imaging_instructions() -> str:
    return f"""
TCIA Faspex often returns HTTP 500 for CLI downloads (server-side bug).
Use the browser + Aspera Connect plugin instead:

  1. Open {TCIA_COLLECTION_PAGE}
  2. Under Data Access, click Download (11gb)
  3. Install IBM Aspera Connect if prompted:
     https://www.ibm.com/products/aspera/downloads
  4. Save/extract into:
     {RAW_DIR}/

For a spike (~50 MB), download one patient folder only from the TCIA file browser.

Direct Faspex link (same bundle):
  {FASPEX_IMAGING_URL}
"""


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists): {dest.name}")
        return
    print(f"  fetching {dest.name} ...")
    with urllib.request.urlopen(url) as response:
        dest.write_bytes(response.read())


def download_metadata() -> None:
    print(f"Metadata -> {METADATA_DIR}")
    _download(CLINICAL_XLSX, METADATA_DIR / "MU-Glioma-Post_ClinicalData-July2025.xlsx")
    _download(SCANNER_XLSX, METADATA_DIR / "MR_Scanner_data.xlsx")


def _gem_bin_dirs() -> list[Path]:
    """Common directories where ``gem install aspera-cli`` puts ascli."""
    from glob import glob

    candidates: list[Path] = []
    home = Path.home()

    for pattern in (
        "/usr/local/lib/ruby/gems/*/bin",
        "/opt/homebrew/lib/ruby/gems/*/bin",
        str(home / ".gem/ruby/*/bin"),
    ):
        candidates.extend(Path(p) for p in glob(pattern))

    try:
        out = subprocess.run(
            ["ruby", "-e", "puts Gem.bindir"],
            capture_output=True,
            text=True,
            check=True,
        )
        bindir = out.stdout.strip()
        if bindir:
            candidates.append(Path(bindir))
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in candidates:
        resolved = path.resolve() if path.exists() else path
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(path)
    return ordered


def _find_ascli() -> str | None:
    found = shutil.which("ascli")
    if found:
        return found
    for bindir in _gem_bin_dirs():
        candidate = bindir / "ascli"
        if candidate.is_file():
            return str(candidate)
    return None


def _ascli_install_hint() -> str:
    bindir = _gem_bin_dirs()
    path_hint = ""
    if bindir:
        path_hint = (
            f"\nIf already installed, add Ruby gems to PATH:\n"
            f"  export PATH=\"{bindir[0]}:$PATH\"\n"
        )
    return (
        "ascli not found. Install Aspera CLI, then retry:\n"
        "  brew install ruby\n"
        "  gem install aspera-cli\n"
        "  ascli config ascp install\n"
        f"{path_hint}"
        "  python .../download_mu_glioma_post.py --imaging\n"
        "\nOr download manually from:\n"
        f"  {FASPEX_IMAGING_URL}\n"
        "Extract into data/raw/mu_glioma_post/"
    )


def download_imaging() -> int:
    ascli = _find_ascli()
    if ascli is None:
        print(_ascli_install_hint(), file=sys.stderr)
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Imaging (~11 GB) -> {RAW_DIR}")
    print(f"Using ascli: {ascli}")
    print("Starting Faspex transfer via ascli ...")
    cmd = _faspex_receive_cmd(ascli, RAW_DIR)
    print("  ", cmd[0], "faspex5 packages receive \\")
    print(f"      --url=<Faspex public URL> \\")
    print(f"      --to-folder={RAW_DIR.resolve()} \\")
    print("      1030")
    exit_code = subprocess.call(cmd)
    if exit_code != 0:
        print(_manual_imaging_instructions(), file=sys.stderr)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Download MU-Glioma-Post from TCIA.")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download clinical + scanner spreadsheets only (~125 KB)",
    )
    parser.add_argument(
        "--imaging",
        action="store_true",
        help="Download full NIfTI bundle via Aspera Faspex (~11 GB)",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Print browser download instructions for the imaging bundle",
    )
    args = parser.parse_args()

    if args.manual:
        print(_manual_imaging_instructions())
        return 0

    if not args.metadata_only and not args.imaging:
        args.metadata_only = True
        args.imaging = True

    if args.metadata_only:
        download_metadata()

    if args.imaging:
        return download_imaging()
    return 0


if __name__ == "__main__":
    sys.exit(main())

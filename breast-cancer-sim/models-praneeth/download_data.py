"""Download METABRIC + TCGA-BRCA clinical and expression data from cBioPortal.

Pulls from the cBioPortal datahub Git-LFS media endpoint (no auth required) into
breast-cancer-sim/data/raw/<study>/. Files are large (~800 MB total) and are
gitignored; re-run this script to reproduce them.

    python download_data.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
BASE = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public"

# (study, filename) pairs to fetch.
FILES = [
    ("brca_metabric", "data_clinical_patient.txt"),
    ("brca_metabric", "data_clinical_sample.txt"),
    ("brca_metabric", "data_mrna_illumina_microarray.txt"),
    ("brca_tcga_pan_can_atlas_2018", "data_clinical_patient.txt"),
    ("brca_tcga_pan_can_atlas_2018", "data_clinical_sample.txt"),
    ("brca_tcga_pan_can_atlas_2018", "data_mrna_seq_v2_rsem.txt"),
]


def _remote_size(url: str) -> int:
    resp = requests.head(url, allow_redirects=True, timeout=30)
    resp.raise_for_status()
    return int(resp.headers.get("Content-Length", 0))


def download(study: str, filename: str) -> Path:
    url = f"{BASE}/{study}/{filename}"
    dest = DATA_RAW / study / filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    remote = _remote_size(url)
    if dest.exists() and remote and dest.stat().st_size == remote:
        print(f"[skip] {study}/{filename} already complete "
              f"({remote / 1e6:.1f} MB)", flush=True)
        return dest

    print(f"[get ] {study}/{filename} ({remote / 1e6:.1f} MB)", flush=True)
    got = 0
    start = time.time()
    last = start
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB
                fh.write(chunk)
                got += len(chunk)
                now = time.time()
                if now - last > 2.0:
                    pct = (got / remote * 100) if remote else 0
                    rate = got / (now - start) / 1e6
                    print(f"       {got / 1e6:7.1f} MB  {pct:5.1f}%  "
                          f"{rate:5.1f} MB/s", flush=True)
                    last = now
    print(f"[done] {study}/{filename} {got / 1e6:.1f} MB in "
          f"{time.time() - start:.0f}s", flush=True)
    return dest


def main() -> int:
    print(f"Downloading into {DATA_RAW}", flush=True)
    for study, filename in FILES:
        download(study, filename)
    print("All downloads complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

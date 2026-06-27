#!/usr/bin/env bash
# Download Option B spike data for TCGA-AR-A1AX (gitignored — share this script, not data/).
#
# From repo root:
#   bash simulation-vinesh-philip-chandan/download_spike_data.sh
#   bash simulation-vinesh-philip-chandan/download_spike_data.sh --export-raw
#
# Baseline only (~50 MB DICOM). Adds follow-up with --include-followup.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Creating venv at breast-cancer-sim/.venv ..."
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt

exec python simulation-vinesh-philip-chandan/download_spike_data.py "$@"

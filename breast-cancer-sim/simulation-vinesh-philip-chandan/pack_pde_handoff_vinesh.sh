#!/usr/bin/env bash
# Pack expert-mask PDE baselines for Vinesh (contract v1.1.0). Run from breast-cancer-sim/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ZIP="${ROOT}/pde-handoff-vinesh.zip"
rm -f "$ZIP"
zip -r "$ZIP" \
  simulation-vinesh-philip-chandan/handoff_contract.json \
  data/processed/raw-extract-philip-chandan/manifest.json \
  data/processed/segmentations/luminal_a_TCGA-AR-A1AX_baseline_mask.nii.gz \
  data/processed/segmentations/basal_TCGA-AR-A1AQ_baseline_mask.nii.gz \
  data/processed/pde-input-vinesh/TCGA-AR-A1AX/g64/baseline.npy \
  data/processed/pde-input-vinesh/TCGA-AR-A1AX/g64/baseline.json \
  data/processed/pde-input-vinesh/TCGA-AR-A1AQ/g64/baseline.npy \
  data/processed/pde-input-vinesh/TCGA-AR-A1AQ/g64/baseline.json
echo "Wrote $ZIP"
unzip -l "$ZIP"

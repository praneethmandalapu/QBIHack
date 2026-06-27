"""Save a middle-slice PNG for visual QC of the spike case."""

from __future__ import annotations

import sys
from pathlib import Path

PHILIP_CHANDAN_DIR = Path(__file__).resolve().parent
SPIKE_ROOT = PHILIP_CHANDAN_DIR.parent
sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
sys.path.insert(0, str(SPIKE_ROOT))

from spike_paths import QC_SLICE_PLOTS_PHILIP_CHANDAN, SPIKE_PATIENT, ensure_spike_dirs  # noqa: E402
from tcia_extractor import extract_volume_for_timepoint  # noqa: E402


def save_middle_slice_plot(
    tcga_id: str,
    subtype: str,
    study_date: str,
    *,
    slug: str | None = None,
) -> Path:
    ensure_spike_dirs()
    output_slug = slug or SPIKE_PATIENT["slug"]
    out_path = QC_SLICE_PLOTS_PHILIP_CHANDAN / f"{output_slug}_mid-z.png"

    volume = extract_volume_for_timepoint(tcga_id, subtype, study_date)
    mid_z = volume.shape[0] // 2

    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(6, 6))
    axis.imshow(volume[mid_z], cmap="gray")
    axis.set_title(f"{tcga_id} {study_date} z={mid_z}")
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    patient = SPIKE_PATIENT
    out_path = save_middle_slice_plot(
        patient["tcga_id"],
        patient["subtype"],
        patient["study_date"],
        slug=patient["slug"],
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

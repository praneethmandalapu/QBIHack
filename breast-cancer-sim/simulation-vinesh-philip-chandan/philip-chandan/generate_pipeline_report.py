"""Generate PIPELINE_REPORT.pdf — Philip-Chandan imaging pipeline narrative."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from fpdf import FPDF

OUT_DIR = Path(__file__).resolve().parent
SPIKE_ROOT = OUT_DIR.parent
PDF_PATH = OUT_DIR / "PIPELINE_REPORT.pdf"

sys.path.insert(0, str(SPIKE_ROOT))

PHILIP_CHANDAN_DIR = OUT_DIR
sys.path.insert(0, str(PHILIP_CHANDAN_DIR))
from qc_otsu_plot import ensure_otsu_norm_overlay, ensure_pde_input_slice  # noqa: E402
from qc_slice_plot import ensure_overlay_plot, slice_plot_path  # noqa: E402

# Rev2 slugs referenced in cohort.json exports
SLUG_LUMA_BASELINE = "luminal_a_TCGA-AR-A1AX_baseline"
SLUG_LUMA_FOLLOWUP = "luminal_a_TCGA-AR-A1AX_followup"
SLUG_BASAL_BASELINE = "basal_TCGA-AR-A1AQ_baseline"


def qc_plot_path(slug: str) -> Path:
    """Resolve overlay QC PNG for the report, generating from raw .npy if needed."""
    path = ensure_overlay_plot(slug)
    if path is not None:
        return path
    return slice_plot_path(slug, overlay=True)


def embed_qc_figure(
    pdf: "ReportPDF",
    slug: str,
    caption: str,
    *,
    width_mm: float = 85,
) -> bool:
    """Embed a single overlay QC PNG; return False if missing."""
    path = ensure_overlay_plot(slug)
    if path is None:
        pdf.body(
            f"[QC overlay missing: {slug}. Run export_all_raw.py or ensure raw .npy exists.]"
        )
        return False
    if pdf.get_y() + width_mm > 265:
        pdf.add_page()
    pdf.set_x(pdf.l_margin)
    pdf.image(str(path), w=width_mm)
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(pdf.epw, 4, caption)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)
    return True


def embed_qc_figure_pair(
    pdf: "ReportPDF",
    left_slug: str,
    right_slug: str,
    caption: str,
) -> bool:
    """Embed two overlay QC PNGs side by side; return False if either is missing."""
    left = ensure_overlay_plot(left_slug)
    right = ensure_overlay_plot(right_slug)
    if left is None or right is None:
        pdf.body(
            "[Comparison QC overlays missing. Run export_all_raw.py --all-primary "
            "or ensure raw .npy files exist.]"
        )
        return False
    gap_mm = 5.0
    each_w = (pdf.epw - gap_mm) / 2
    if pdf.get_y() + each_w > 250:
        pdf.add_page()
    y0 = pdf.get_y()
    x0 = pdf.l_margin
    pdf.image(str(left), x=x0, y=y0, w=each_w)
    pdf.image(str(right), x=x0 + each_w + gap_mm, y=y0, w=each_w)
    pdf.set_y(y0 + each_w + 4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(pdf.epw, 4, caption)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)
    return True


def embed_image(
    pdf: "ReportPDF",
    path: Path,
    caption: str,
    *,
    width_mm: float = 85,
) -> None:
    if pdf.get_y() + width_mm > 265:
        pdf.add_page()
    pdf.set_x(pdf.l_margin)
    pdf.image(str(path), w=width_mm)
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(pdf.epw, 4, caption)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)


def embed_otsu_figure(pdf: "ReportPDF", slug: str, caption: str) -> bool:
    path = ensure_otsu_norm_overlay(slug)
    if path is None:
        pdf.body(
            f"[Otsu overlay missing: {slug}. Run export_all_raw.py then qc_otsu_plot.py.]"
        )
        return False
    embed_image(pdf, path, caption)
    return True


def embed_pde_figure(pdf: "ReportPDF", slug: str, caption: str) -> bool:
    path = ensure_pde_input_slice(slug)
    if path is None:
        pdf.body(
            f"[PDE input slice missing: {slug}. Ensure raw .npy exists for prepare_pde_input.]"
        )
        return False
    embed_image(pdf, path, caption)
    return True


def embed_pde_figure_pair(
    pdf: "ReportPDF",
    left_slug: str,
    right_slug: str,
    caption: str,
) -> bool:
    left = ensure_pde_input_slice(left_slug)
    right = ensure_pde_input_slice(right_slug)
    if left is None or right is None:
        pdf.body("[PDE input comparison missing. Ensure raw extracts exist for both slugs.]")
        return False
    gap_mm = 5.0
    each_w = (pdf.epw - gap_mm) / 2
    if pdf.get_y() + each_w > 250:
        pdf.add_page()
    y0 = pdf.get_y()
    x0 = pdf.l_margin
    pdf.image(str(left), x=x0, y=y0, w=each_w)
    pdf.image(str(right), x=x0 + each_w + gap_mm, y=y0, w=each_w)
    pdf.set_y(y0 + each_w + 4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(pdf.epw, 4, caption)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)
    return True


class ReportPDF(FPDF):
    def header(self) -> None:
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 100, 100)
        self.set_x(self.l_margin)
        self.cell(self.epw, 8, "Philip-Chandan - TCGA/TCIA Imaging Pipeline", align="R")
        self.ln(10)
        self.set_text_color(0, 0, 0)
        self.set_x(self.l_margin)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, number: str, title: str) -> None:
        self.ln(4)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(20, 60, 100)
        self.multi_cell(self.epw, 8, f"{number}. {title}")
        self.ln(2)
        self.set_text_color(0, 0, 0)

    def subsection(self, title: str) -> None:
        self.ln(2)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 11)
        self.multi_cell(self.epw, 6, title)
        self.ln(1)

    def body(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.multi_cell(self.epw, 5, text)
        self.ln(2)

    def bullet(self, text: str) -> None:
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.cell(6, 5, "-")
        self.multi_cell(self.epw - 6, 5, text)
        self.ln(1)

    def table(self, headers: list[str], rows: list[list[str]], col_widths: list[int]) -> None:
        line_h = 5
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(230, 240, 250)
        self.set_x(self.l_margin)
        for i, header in enumerate(headers):
            self.cell(col_widths[i], 7, header, border=1, fill=True)
        self.ln()
        self.set_font("Helvetica", "", 8)
        for row in rows:
            line_sets = [
                self.multi_cell(
                    col_widths[i],
                    line_h,
                    cell,
                    dry_run=True,
                    output="LINES",
                )
                for i, cell in enumerate(row)
            ]
            row_h = line_h * max(1, max(len(lines) for lines in line_sets))
            if self.get_y() + row_h > 275:
                self.add_page()
            y0 = self.get_y()
            x0 = self.l_margin
            for i, lines in enumerate(line_sets):
                self.set_xy(x0 + sum(col_widths[:i]), y0)
                self.multi_cell(col_widths[i], line_h, "\n".join(lines), border=1)
            self.set_xy(x0, y0 + row_h)
        self.ln(3)


def build_report() -> FPDF:
    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(20, 60, 100)
    pdf.multi_cell(pdf.epw, 10, "Philip-Chandan Imaging Pipeline")
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(pdf.epw, 6, "TCGA-BRCA longitudinal MRI: discovery through raw export and Otsu handoff")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        pdf.epw,
        5,
        f"QBIHack breast-cancer-sim  |  Generated {date.today().isoformat()}",
    )
    pdf.ln(4)

    pdf.body(
        "Philip-Chandan owns the radiomics pipeline: pull real breast MRI from TCIA, "
        "validate and stack DICOM into 3D numpy arrays, and hand raw volumes to Vinesh "
        "for PDE simulation. Resample, normalize, and segmentation are Vinesh's scope "
        "(handoff_contract.json v1.0.0, Option B)."
    )

    # --- 1. Criteria ---
    pdf.section_title("1", "Patient selection criteria")
    pdf.body(
        "We needed real TCGA-BRCA cases for a Luminal A vs Basal-like demo with "
        "longitudinal MRI, not a single timepoint. Each candidate had to pass several gates:"
    )
    pdf.table(
        ["Criterion", "Why it matters"],
        [
            ["MRI on TCIA", "Many TCGA patients have genomics but no public imaging"],
            ["Longitudinal MR (>=2 study dates)", "Baseline + follow-up for growth comparison"],
            ["PAM50 subtype match", "Luminal A and Basal-like labels align with cBioPortal"],
            ["Genomics for Praneeth", "ER/PR status and survival on GDC/cBioPortal"],
            ["Usable contrast series", "Post-contrast T1 (e.g. VIBRANT) for tumor visibility"],
        ],
        [55, 125],
    )
    pdf.body(
        "Our first picks failed: TCGA-BH-A0BR and TCGA-A2-A04P had no MRI on TCIA. "
        "Only 19 of 139 TCGA-BRCA patients with MRI on TCIA have multiple MR studies. "
        "Patient selection was a real constraint."
    )
    pdf.subsection("Rev2 primary pair (cohort/cohort.json)")
    pdf.table(
        ["Subtype", "TCGA ID", "Baseline", "Follow-up"],
        [
            ["Luminal A", "TCGA-AR-A1AX", "2002-09-12", "2003-09-24 (~12 mo)"],
            ["Basal-like", "TCGA-AR-A1AQ", "2001-11-21", "2003-05-07 (~17 mo)"],
        ],
        [35, 45, 40, 60],
    )

    # --- 2. Discovery ---
    pdf.section_title("2", "Finding patients - cohort_discovery.py")
    pdf.body(
        "Before locking IDs in cohort.json, we built cohort/cohort_discovery.py to "
        "automate discovery and validation against public APIs:"
    )
    pdf.bullet("TCIA NBIA REST API - list MR series, group by study date, flag longitudinal cases, prefer contrast series")
    pdf.bullet("cBioPortal - PAM50 subtype, ER/PR, overall survival (aligned with Praneeth's genomics fields)")
    pdf.ln(2)
    pdf.subsection("CLI workflows")
    pdf.set_font("Courier", "", 8)
    for line in [
        "python .../cohort/cohort_discovery.py audit",
        "python .../cohort/cohort_discovery.py find-longitudinal --subtype \"Luminal A\"",
        "python .../cohort/cohort_discovery.py recommend-pair",
    ]:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(pdf.epw, 4, line)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.body(
        "Each patient gets a structured report: imaging availability, study span, PAM50 match, "
        "missing genomics fields, and a pass/fail ok flag. This tool justified the rev2 pivot "
        "and gave Praneeth the same TCGA barcodes for GDC queries."
    )

    # --- 3. Download ---
    pdf.section_title("3", "Download - download_tcia.py")
    pdf.body("Once patients are locked, download_tcia.py pulls DICOM into a consistent layout:")
    pdf.set_font("Courier", "", 8)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        pdf.epw,
        4,
        "data/raw/tcia/\n"
        "  luminal_a/TCGA-AR-A1AX/\n"
        "    2002-09-12/   # baseline\n"
        "    2003-09-24/   # follow-up\n"
        "  basal/TCGA-AR-A1AQ/\n"
        "    2001-11-21/\n"
        "    2003-05-07/",
    )
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)
    pdf.body("For each study date the downloader:")
    pdf.bullet("Lists MR series in collection TCGA-BRCA")
    pdf.bullet("Picks the best series - prefers post-contrast (+C, VIBRANT, T1); deprioritizes calibration/localizer scans")
    pdf.bullet("Downloads via idc-index (primary) with tcia-utils NBIA fallback")
    pdf.ln(1)
    pdf.set_font("Courier", "", 8)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(pdf.epw, 4, "python .../download_tcia.py --all-primary --longitudinal")
    pdf.ln(3)

    # --- 4. Validate ---
    pdf.section_title("4", "Validate - validate_series() in tcia_extractor.py")
    pdf.body(
        "Download alone is not enough. Before export, validate_series(dicom_dir) checks the local series:"
    )
    pdf.table(
        ["Check", "What we catch"],
        [
            ["DICOM slices present", "Empty or non-image folders"],
            ["Consistent Rows/Columns", "Mixed slice dimensions"],
            ["Unique InstanceNumber", "Duplicate or corrupted ordering"],
            ["SimpleITK read succeeds", "Metadata OK but ITK cannot stack"],
            ["Slice count, shape, spacing_mm", "Recorded in validation report for QC and sidecar JSON"],
        ],
        [55, 125],
    )
    pdf.body(
        "extract_volume() refuses to run if validation fails. Export scripts call validate_series "
        "first and abort with explicit errors."
    )
    pdf.subsection("Visual QC")
    pdf.body(
        "qc_slice_plot.py writes middle-slice PNGs per volume to "
        "data/qc/slice-plots-philip-chandan/, including optional overlay PNGs with a "
        "lime contour over high-intensity voxels (QC visualization only, not tumor segmentation)."
    )

    pdf.subsection("Figure 1 - Spike validation slice")
    embed_qc_figure(
        pdf,
        SLUG_LUMA_BASELINE,
        "Figure 1. Raw MR slice for TCGA-AR-A1AX baseline (2002-09-12) after DICOM validation "
        "and 3D stacking. Lime contour marks voxels above the 90th percentile (QC only).",
    )

    # --- 5. Export ---
    pdf.section_title("5", "Export - raw volumes for Vinesh")
    pdf.body(
        "After validation passes, we export raw 3D arrays, not PDE-ready volumes. "
        "Resample, normalize, and segmentation are Vinesh's scope per handoff_contract.json."
    )
    pdf.bullet("Single case: export_raw_extract.py")
    pdf.bullet("Batch (all primaries x timepoints): export_all_raw.py")
    pdf.ln(2)
    pdf.body("For each patient x timepoint slug (e.g. luminal_a_TCGA-AR-A1AX_baseline):")
    pdf.bullet("Resolve DICOM path from cohort.json")
    pdf.bullet("Run validate_series")
    pdf.bullet("extract_volume_with_spacing() - SimpleITK DICOM to float32 numpy, shape (Z, Y, X), native spacing preserved")
    pdf.bullet("Write .npy, JSON sidecar, and QC plot")
    pdf.ln(2)
    pdf.table(
        ["Output", "Path"],
        [
            ["Raw volume", "data/processed/raw-extract-philip-chandan/{slug}.npy"],
            ["Metadata sidecar", ".../{slug}.json (shape, spacing_mm, contract_version)"],
            ["QC plot", "data/qc/slice-plots-philip-chandan/{slug}_mid-z[-overlay].png"],
            ["Manifest", "manifest.json v1.1.0 - subtype, timepoint, slug, paths"],
        ],
        [40, 140],
    )
    pdf.subsection("Handoff contract - what we export vs what we do not")
    pdf.table(
        ["Philip-Chandan (raw extract)", "Vinesh (PDE input)"],
        [
            ["Raw MR intensities, not normalized", "Resample/crop to max 64^3, 1 mm spacing"],
            ["Spacing in JSON sidecar", "Normalize to [0, 1]"],
            ["Axis order (Z, Y, X)", "Otsu tumor segmentation"],
        ],
        [85, 95],
    )

    pdf.subsection("Figure 2 - Baseline subtype comparison")
    embed_qc_figure_pair(
        pdf,
        SLUG_LUMA_BASELINE,
        SLUG_BASAL_BASELINE,
        "Figure 2. Baseline comparison: Luminal A (TCGA-AR-A1AX) vs Basal-like "
        "(TCGA-AR-A1AQ). Lime contour highlights enhancing tissue (90th percentile, QC only).",
    )

    pdf.subsection("Figure 3 - Longitudinal Luminal A")
    embed_qc_figure_pair(
        pdf,
        SLUG_LUMA_BASELINE,
        SLUG_LUMA_FOLLOWUP,
        "Figure 3. Longitudinal Luminal A: baseline (2002-09-12) vs follow-up "
        "(2003-09-24). Contour overlay on the slice with the most enhancing voxels per volume.",
    )

    # --- 6. Otsu segmentation (Vinesh handoff) ---
    pdf.section_title("6", "Otsu tumor segmentation - prepare_pde_input.py (Vinesh)")
    pdf.body(
        "After Philip-Chandan raw export, Vinesh prepares PDE-ready volumes in "
        "vinesh/prepare_pde_input.py per handoff_contract.json v1.0.0:"
    )
    pdf.bullet("Resample raw MR to isotropic 1 mm spacing (scipy.ndimage.zoom)")
    pdf.bullet("Min-max normalize intensities to [0, 1]")
    pdf.bullet("Otsu threshold (skimage) on normalized voxels; keep continuous density inside tumor")
    pdf.bullet("Zero background; crop/pad to max 64^3 centered on tumor center of mass")
    pdf.ln(2)
    pdf.body(
        "Continuous tumor values (not a binary mask) are required so the PDE logistic growth "
        "term rho*u*(1-u) can evolve. qc_otsu_plot.py documents the Otsu step using the same "
        "pipeline as prepare_pde_input.py."
    )
    pdf.subsection("Figure 4 - Otsu on normalized resampled volume")
    embed_otsu_figure(
        pdf,
        SLUG_LUMA_BASELINE,
        "Figure 4. TCGA-AR-A1AX baseline after resample + normalize. Magenta contour is the "
        "Otsu tumor mask (voxels above threshold); background will be zeroed before crop.",
    )
    pdf.subsection("Figure 5 - PDE input after Otsu and crop")
    embed_pde_figure_pair(
        pdf,
        SLUG_LUMA_BASELINE,
        SLUG_BASAL_BASELINE,
        "Figure 5. Baseline PDE inputs (64^3, 1 mm) for Luminal A vs Basal-like. Inferno "
        "colormap shows continuous initial tumor burden; cyan contour marks tumor voxels > 0.",
    )

    # --- Summary ---
    pdf.section_title("7", "End-to-end summary")
    pdf.body(
        "We defined strict patient criteria (longitudinal MRI on TCIA, PAM50-aligned subtypes, "
        "genomics for the team), built cohort_discovery.py to search and audit candidates against "
        "TCIA and cBioPortal, locked rev2 primaries after rev1 failed imaging availability, downloaded "
        "four DICOM timepoints with contrast-aware series selection, validated each series structurally "
        "before extraction and visually via slice QC, exported four raw .npy volumes plus JSON "
        "sidecars and a manifest, and documented Vinesh Otsu segmentation into 64^3 PDE inputs "
        "for downstream simulation and UI integration."
    )
    pdf.subsection("Deliverables")
    pdf.table(
        ["Stage", "Deliverable", "Count"],
        [
            ["Discovery", "cohort.json rev2 + COHORT.md", "2 primaries + backups"],
            ["Download", "DICOM under data/raw/tcia/", "4 study folders"],
            ["Validate", "validate_series pass + QC PNGs", "4 volumes"],
            ["Export", "Raw .npy + .json + manifest.json", "4 slugs"],
            ["PDE prep", "Otsu segmentation QC + pde-input-vinesh/", "4 slugs"],
        ],
        [35, 85, 60],
    )

    pdf.ln(4)
    pdf.subsection("Exported volume shapes (rev2)")
    pdf.table(
        ["Slug", "Shape (Z,Y,X)", "Spacing (mm)"],
        [
            ["luminal_a_TCGA-AR-A1AX_baseline", "[352, 256, 256]", "[3.0, 0.8594, 0.8594]"],
            ["basal_TCGA-AR-A1AQ_baseline", "[464, 256, 256]", "[3.0, 0.859375, 0.859375]"],
            ["luminal_a_TCGA-AR-A1AX_followup", "[552, 512, 512]", "[2.2, 0.5273, 0.5273]"],
            ["basal_TCGA-AR-A1AQ_followup", "[448, 256, 256]", "[3.0, 0.9375, 0.9375]"],
        ],
        [70, 45, 65],
    )

    return pdf


def main() -> None:
    pdf = build_report()
    pdf.output(str(PDF_PATH))
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()

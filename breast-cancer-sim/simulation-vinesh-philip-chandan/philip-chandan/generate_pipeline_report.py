"""Generate PIPELINE_REPORT.pdf — Philip-Chandan imaging pipeline narrative."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from fpdf import FPDF

OUT_DIR = Path(__file__).resolve().parent
PDF_PATH = OUT_DIR / "PIPELINE_REPORT.pdf"


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
    pdf.multi_cell(pdf.epw, 6, "TCGA-BRCA longitudinal MRI: discovery through raw export")
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
        "qc_slice_plot.py writes a middle-slice PNG per volume to "
        "data/qc/slice-plots-philip-chandan/ so a human can confirm anatomy looks real before handoff."
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
            ["QC plot", "data/qc/slice-plots-philip-chandan/{slug}_mid-z.png"],
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

    # --- Summary ---
    pdf.section_title("6", "End-to-end summary")
    pdf.body(
        "We defined strict patient criteria (longitudinal MRI on TCIA, PAM50-aligned subtypes, "
        "genomics for the team), built cohort_discovery.py to search and audit candidates against "
        "TCIA and cBioPortal, locked rev2 primaries after rev1 failed imaging availability, downloaded "
        "four DICOM timepoints with contrast-aware series selection, validated each series structurally "
        "before extraction and visually via slice QC, then exported four raw .npy volumes plus JSON "
        "sidecars and a manifest for downstream PDE simulation and UI integration."
    )
    pdf.subsection("Deliverables")
    pdf.table(
        ["Stage", "Deliverable", "Count"],
        [
            ["Discovery", "cohort.json rev2 + COHORT.md", "2 primaries + backups"],
            ["Download", "DICOM under data/raw/tcia/", "4 study folders"],
            ["Validate", "validate_series pass + QC PNGs", "4 volumes"],
            ["Export", "Raw .npy + .json + manifest.json", "4 slugs"],
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

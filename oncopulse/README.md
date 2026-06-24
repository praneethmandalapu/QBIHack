# OncoPulse — Explainable Survival-Risk Triage Dashboard

OncoPulse is a clinical decision support prototype designed for precision oncology researchers. It takes a tumor gene-expression profile for **KIRC** (Kidney Renal Clear Cell Carcinoma) or **LIHC** (Liver Hepatocellular Carcinoma), predicts survival-risk probabilities, explains the prognostic driver genes, maps them to pathway context, and highlights therapeutic drug-gene interactions.

The model is trained entirely on **real clinical and genomic patient records** (539 KIRC patients and 377 LIHC patients) from The Cancer Genome Atlas (TCGA) via UCSC Xena hubs.

---

## 🧬 Scientific Foundation

OncoPulse is scientifically grounded in the updated **Human Pathology Atlas** (published in *Lancet EBioMedicine*, Jan 2025). The paper validated prognostic genes across cohorts and demonstrated that KIRC and LIHC have the strongest and most reproducible survival correlation.

- **KIRC Pathway**: Focused on the **Tight Junction** pathway (responsible for cell barrier integrity and cell migration). Overexpression of unfavourable transcription factors like **`DNMT3B`** and **`PPP1R1A`** negatively regulates/represses this pathway, leading to tumor progression and worse clinical outcomes.
- **LIHC Pathways**: Focused on **Purine Metabolism** and **RNA Polymerase** pathways. Key regulators with the highest slope values in both pathways include **`TAF15`**, **`CHEK1`**, and **`PDCD6`**, acting as major prognostic switches.

---

## 🛠️ Folder Structure

```text
oncopulse/
├── app.py                     # Streamlit main application dashboard
├── requirements.txt           # Project python dependencies
├── README.md                  # Project documentation
├── .env.example               # Environmental variables configuration
├── models/                    # Serialized models and performance metadata JSONs
│   ├── kirc_model.joblib
│   ├── kirc_metadata.json
│   ├── lihc_model.joblib
│   └── lihc_metadata.json
├── data/                      # Local data folders
│   ├── raw/                   # Raw TCGA clinical and genomic files from UCSC Xena
│   ├── processed/             # Cleaned cohort matrices
│   ├── demo/                  # Real patient profiles for low/high risk demo cases
│   ├── cache/                 # Local DGIdb drug-gene interaction snapshot
│   └── networks/              # Pathway topology JSON files
├── src/                       # Main source code package
│   ├── config.py              # Central configurations and target gene lists
│   ├── schemas.py             # Pydantic schemas for verification
│   ├── utils.py               # Shared logger and utility helpers
│   ├── io/                    # Ingestion and validation scripts
│   │   ├── loaders.py
│   │   ├── validators.py
│   │   └── exporters.py
│   ├── models/                # Machine learning scripts
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── explain.py
│   ├── bio/                   # Biological pathway and database interfaces
│   │   ├── drug_lookup.py
│   │   └── network.py
│   ├── ui/                    # Streamlit components layout
│   │   ├── layout.py
│   │   ├── predict_tab.py
│   │   ├── explain_tab.py
│   │   ├── targets_tab.py
│   │   ├── network_tab.py
│   │   └── about_tab.py
│   └── demo/
│       └── seed_demo_outputs.py # Dataset downloader and cleaner script
└── tests/                     # Unit test suite
    ├── test_validation.py
    ├── test_prediction.py
    ├── test_drug_lookup.py
    └── test_export.py
```

---

## 🚀 Installation & Local Execution

### 1. Install Dependencies
Ensure you have Python 3.10+ installed. In your terminal, run:
```bash
pip install -r requirements.txt
```

### 2. Download and Clean Real TCGA Data
Run the data preparation script to download expression matrices and clinical datasets from UCSC Xena hubs, filter them to keep target genes, compute survival times, and extract demo patients:
```bash
python src/demo/seed_demo_outputs.py
```
*Note: This script uses secure SSL verification checks (`certifi.where()`) to prevent macOS-specific connection errors.*

### 3. Train Models
Fit Logistic Regression classifiers on the cohort training sets:
```bash
PYTHONPATH=. python3 src/models/train.py
```
This script will print out holdout validation metrics (AUROC, accuracy) and save joblib/metadata bundles in `models/`.

### 4. Launch the Dashboard
Run the Streamlit application:
```bash
streamlit run app.py
```

---

## 🧪 Running Unit Tests
Execute the unit tests to verify data cleaning, model predictions, cache fallbacks, and report exports:
```bash
PYTHONPATH=. pytest tests/
```

---

## 📝 2-Minute Live Demo Walkthrough

### Step 1: Landing & Loading
- Start the app. Open the sidebar.
- Select **KIRC** under "Select Tumor Cohort".
- Select **Demo Patients** -> **High Risk Cohort Profile**.
- Click **🚀 RUN TARGET TRIAGE**.

### Step 2: Risk Triage (PREDICT Tab)
- Look at the glowing red **HIGH RISK** card.
- See the **75%+ risk probability** score.
- Review the Patient Cohort Placement bar.

### Step 3: Explain Drivers (EXPLAIN Tab)
- Navigate to the **EXPLAIN** tab.
- Review the interactive Plotly bar chart showing contributions.
- Explain: *"Elevated risk is driven by abnormal expression of unfavourable regulator DNMT3B and underexpression of barrier proteins CLDN1 and CLDN2."*

### Step 4: Actionable Hypotheses (TARGETS Tab)
- Go to the **TARGETS** tab.
- Look at the ranked therapeutic suggestions.
- Point to **Decitabine** or **Azacitidine** as known blockers targeting DNMT3B.

### Step 5: Network Topology (NETWORK Tab)
- Go to the **NETWORK** tab.
- Interact with the circular node-link graph showing how DNMT3B is linked to the Tight Junction pathway node.
- Hover over nodes to see patient-level expression values.

### Step 6: Export Results (PREDICT Tab)
- Go back to the **PREDICT** tab and click **Download CSV Report** or download the **JSON Payload** to showcase clinical export workflows.

---

## 📖 Scientific References
- **Human Pathology Atlas Update**: Meng Yuan, Cheng Zhang, et al. *EBioMedicine* (Lancet), Jan 2025. [DOI: 10.1016/j.ebiom.2024.105495](https://doi.org/10.1016/j.ebiom.2024.105495)
- **TCGA Clinical Data Resource (TCGA-CDR)**: *Cell*, 2018. [DOI: 10.1016/j.cell.2018.02.052](https://doi.org/10.1016/j.cell.2018.02.052)

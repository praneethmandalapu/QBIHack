# Brain cancer simulation — candidate datasets

Longitudinal, same-patient annotated MRI is hard to find for breast cancer. This table tracks brain tumor / MS datasets that may support growth modeling with repeated scans and segmentation.

| Dataset | Temporal | Segmentation | Disease | Growth Modeling |
|---------|----------|--------------|---------|-----------------|
| UCSF Longitudinal Glioma | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Glioma | ⭐⭐⭐⭐⭐ |
| MU-Glioma-Post | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Glioma | ⭐⭐⭐⭐⭐ |
| Yale Brain Mets | ⭐⭐⭐⭐⭐ | Variable | Metastases | ⭐⭐⭐⭐ |
| LUMIERE | ⭐⭐⭐⭐ | Auto + RANO | GBM | ⭐⭐⭐⭐⭐ |

Preferred glioma imaging cohorts for v1: **UCSF-ALPTDG**, **MU-Glioma-Post** (~11 GB on TCIA), **LUMIERE** (Figshare).
| MS Longitudinal | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | MS | ⭐⭐⭐⭐⭐ |
| MSSEG2 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | MS | ⭐⭐⭐⭐⭐ |

## Genomics cohorts (no imaging link)

| Cohort | Role | Notes |
|--------|------|-------|
| **TCGA-GBM / TCGA-LGG** | Primary genomics + optional TCIA MRI | GDC/cBioPortal; TCGA barcode join for imaging subset |
| **CGGA** | External validation (METABRIC-like) | Independent glioma expression + clinical; train TCGA → validate CGGA — see [`README.md`](README.md) TODO |

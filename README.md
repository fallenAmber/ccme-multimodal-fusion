# CCME: Cross-Modal Counterfactual Explanation

This repository contains the implementation and analysis code associated
with:

"Cross-Modal Counterfactual Explanation for Reference-Conditioned Modality
Necessity and Sufficiency in Multi-Modal Fusion Systems"

## Contents

The repository contains experiments for two RGB-thermal domains:

### Photovoltaic (PV) Inspection
- PV preprocessing and leakage-controlled splitting
- RGB-only, thermal-only, early-fusion, and late-fusion models
- Five-seed predictive evaluation
- Cross-Modal Counterfactual Explanation (CCME)
- Full-test endpoint-approximation validation
- Complementary attribution, Shapley, robustness, and modality-replacement analyses

### NII-CU Multispectral Aerial Person Detection
- RGB-thermal crop construction
- Thermal-separability and leakage audits
- RGB-only, thermal-only, early-fusion, and late-fusion models
- Five-seed predictive evaluation
- Full-test CCME modality-regime analysis
- Reference-sensitivity analysis
- Exact two-modality Shapley analysis
- Pixel-level modality replacement
- Complementary seed-42 analyses

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── pv_ccme_training_and_analysis.ipynb
│   └── nii_cu_ccme_analysis.ipynb
├── src/
│   ├── preprocessing_verified.py
│   └── dataset_utils.py
├── scripts/
│   └── dedup_split_check.py

# Country- and Class-Aware Reliability Consistency Learning for Cross-Country Road Damage Detection (CCARC)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Framework: PyTorch](https://img.shields.io/badge/framework-PyTorch%202.4.1-orange.svg)](https://pytorch.org/)
[![Model: YOLOv8s](https://img.shields.io/badge/model-YOLOv8s-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![Benchmark: RDD2022](https://img.shields.io/badge/benchmark-RDD2022-yellow.svg)](https://crddc2022.sekilab.global/)

Final-Year BSc Computer Science with AI Research Project.

---

> [!IMPORTANT]
> **Scientific Hypothesis Notice**:
> The proposed CCARC consistency-learning method, its novelty, and its anticipated performance improvements are **working scientific hypotheses**, not established conclusions. The primary objective of this project is rigorous, unbiased empirical evaluation under strict out-of-domain evaluation protocols.

---

## 1. Research Objectives

Automated Road Damage Detection (RDD) is critical for public safety, civil infrastructure maintenance, and autonomous transport systems. However, existing deep learning detectors experience severe performance degradation when deployed across geographic regions due to acute **domain shift** (varying road materials, weather, lighting, sensor optics, and camera mounting geometry) and severe **class imbalance** (abundant superficial cracks vs. scarce, severe structural damages such as potholes).

The core objectives of this research are:
1. **Investigate Cross-Country Generalization**: Systematically quantify the performance drop of state-of-the-art single-stage detectors (YOLOv8s) when tested on unseen countries.
2. **Formulate CCARC**: Formulate a provisional **Country- and Class-Aware Reliability Consistency (CCARC)** learning scheme designed to regularize feature and prediction consistency across domain-shifted views while accounting for source reliability and class imbalance.
3. **Strict Source-Only LOCO Benchmarking**: Evaluate the baseline and proposed approach under a strict Leave-One-Country-Out (LOCO) experimental protocol using the multi-country RDD2022 benchmark.
4. **Hypothesis Validation & Ablation**: Empirically accept or refute the hypothesis that reliability-weighted consistency regularization mitigates geographic domain shift without degrading in-distribution performance.

---

## 2. Benchmark Dataset: RDD2022

This research utilizes the **Crowdsensing-based Road Damage Dataset 2022 (RDD2022)**, comprising smartphone-captured and vehicle-mounted dashcam imagery across distinct countries:

### Participating Countries
- **Japan (JP)**
- **India (IN)**
- **Czech Republic (CR)**
- **Norway (NO)**
- **United States (US)**
- **China (CN)**

### Standardized 4-Class Taxonomy
To ensure cross-national consistency and fair evaluation, annotations are mapped to the four core damage categories:
| Class ID | Code | Damage Category | Description & Characteristics |
| :---: | :---: | :--- | :--- |
| **0** | **D00** | Longitudinal Crack | Cracks running parallel to the direction of travel (wheel tracks, joints). |
| **1** | **D10** | Transverse Crack | Lateral cracks running perpendicular to the direction of travel. |
| **2** | **D20** | Alligator Crack | Interconnected polygonal fatigue cracking (structural degradation). |
| **3** | **D40** | Pothole | Bowl-shaped depressions, surface separation, rutting, or severe bumps. |

---

## 3. Evaluation Protocol: Source-Only LOCO

To guarantee realistic out-of-domain testing, this project enforces a **Source-Only Leave-One-Country-Out (LOCO)** evaluation scheme:

```
Fold k:
  Target Country (Held Out):    T_k
  Source Training Countries:     S_k = { All Countries \ {T_k} }
  Protocol:                      Train exclusively on S_k; Zero-shot evaluation on T_k.
```

- **Zero Target Exposure**: No images, unlabelled frames, or annotations from the target country $T_k$ are accessible during training or validation hyperparameter tuning.
- **Evaluation Metrics**:
  - Primary: $\text{mAP@0.5}$ and $\text{mAP@0.5:0.95}$ on the unseen target country.
  - Granular: Per-class Average Precision ($AP_{D00}$, $AP_{D10}$, $AP_{D20}$, $AP_{D40}$) to diagnose class-specific degradation.
  - Robustness: Out-of-Domain Generalization Gap ($\Delta = \text{mAP}_{\text{In-Domain}} - \text{mAP}_{\text{Target}}$).

---

## 4. Provisional CCARC Methodology

The **CCARC (Country- and Class-Aware Reliability Consistency)** methodology is designed as a consistency regularization framework built upon YOLOv8s:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{YOLOv8}} + \lambda \, \mathcal{L}_{\text{CCARC}}$$

### Core Components (Working Hypotheses)
1. **Country-Aware Reliability Weighting ($w_{\text{country}}$)**:
   Measures prediction variance and feature uncertainty across source geographic distributions, attenuating noisy or outlier domain signals while emphasizing high-confidence invariant representations.
2. **Class-Aware Balancing ($\beta_{\text{class}}$)**:
   Weights consistency constraints inversely to class frequencies, preventing dominant classes (such as D00) from suppressing critical underrepresented structural damage classes (such as D40).
3. **Perturbation Consistency Regularization ($\mathcal{L}_{\text{consistency}}$)**:
   Enforces invariant prediction distributions under photometric (illumination, weather, color shift) and geometric perturbations representative of cross-country camera and asphalt variations.

*Note: The mathematical formulation and specific loss terms are systematically ablated and empirically validated in subsequent project phases.*

---

## 5. Repository Structure

```
CCARC-Road-Damage-Research/
├── configs/                # Experiment configurations & hyperparameters (YAML)
├── src/                    # Source code package
│   ├── data/               # Ingestion, RDD2022 preprocessing, augmentations, loaders
│   ├── models/             # YOLOv8s architecture, CCARC consistency wrappers
│   ├── training/           # Training pipelines, custom loss functions, LOCO loops
│   ├── evaluation/         # LOCO evaluation, mAP calculation, error diagnostics
│   └── utils/              # General helpers, seeding, device configuration, logging
├── tests/                  # Unit and integration test suite
├── notebooks/              # Exploratory data analysis and diagnostic notebooks
├── docs/                   # Research notes, thesis literature review, protocol specs
├── experiments/            # Experiment tracking, artifacts, logs
│   ├── runs/               # Training run outputs
│   └── logs/               # Detailed execution and evaluation logs
├── checkpoints/            # Model weights (git-ignored)
├── app/                    # Interactive Streamlit dashboard for research exploration
├── .env.example            # Environment configuration template
├── .gitignore              # Git ignore rules for data, weights, caches
├── requirements.txt        # Pinned dependency specifications
└── verify_env.py           # Environment verification diagnostic utility
```

---

## 6. Setup and Verification

### Prerequisites
- Python 3.12 (64-bit)
- Git 2.40+
- NVIDIA GPU with CUDA 12.x recommended (CPU supported for smoke tests and testing)

### Installation
1. Clone the repository and navigate into the project directory:
   ```bash
   cd CCARC-Road-Damage-Research
   ```
2. Activate the local virtual environment:
   ```powershell
   # Windows PowerShell
   .\.venv\Scripts\Activate.ps1
   ```
3. Install pinned dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(For CUDA support, install PyTorch wheels matching your CUDA version as annotated in `requirements.txt`.)*
4. Run the environment verification script:
   ```bash
   python verify_env.py
   ```
5. Run the test suite:
   ```bash
   pytest tests/
   ```

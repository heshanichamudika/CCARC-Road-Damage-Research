# RDD2022 Dataset Validation in Google Colab Pro

This guide documents how to execute the dataset validation and manifest generation pipeline (`src/data/validate_rdd2022.py`) in Google Colab Pro.

The architecture strictly separates compute and code from heavy dataset storage:
- **Code & Repository**: Cloned into fast local Colab VM disk (`/content/CCARC-Road-Damage-Research`).
- **Dataset Storage**: Mounted via Google Drive (`/content/drive/MyDrive/CCARC-Road-Damage-Research/datasets/RDD2022/extracted`).

---

## 1. Storage Layout in Google Drive

The extracted RDD2022 dataset is stored in Google Drive under:
```
/content/drive/MyDrive/CCARC-Road-Damage-Research/datasets/RDD2022/extracted/
├── China_Drone/
│   └── China_Drone/               # Handles nested or direct structure
│       ├── train/
│       │   ├── images/
│       │   └── annotations/xmls/
│       └── test/
│           └── images/
├── China_MotorBike/
│   └── China_MotorBike/
│       ├── train/
│       └── test/
├── Czech/
│   └── Czech/
│       ├── train/
│       └── test/
├── India/
│   └── India/
│       ├── train/
│       └── test/
├── Japan/
│   └── Japan/
│       ├── train/
│       └── test/
├── Norway/
│   └── Norway/
│       ├── train/
│       └── test/
└── United_States/
    └── United_States/
        ├── train/
        └── test/
```

---

## 2. Google Colab Execution Notebook Cells

### Cell 1: Mount Google Drive (Dataset Storage Only)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Clone or Update Project Repository in `/content`
```bash
# Clone the repository onto fast local Colab disk
%cd /content
!git clone https://github.com/heshanichamudika/CCARC-Road-Damage-Research.git
%cd /content/CCARC-Road-Damage-Research

# Verify active branch and latest commit
!git branch --show-current
!git log -1 --oneline
```

*(If repository is already cloned, run `%cd /content/CCARC-Road-Damage-Research` and `!git pull origin main`)*

### Cell 3: Install Dependencies
```bash
# Install pinned dependencies without modifying Colab's pre-installed GPU PyTorch
!pip install -r requirements-colab.txt
```

### Cell 4: Run Test Suite
```bash
# Verify test suite in Colab environment (all 15 tests should pass)
!pytest -v
```

### Cell 5: Execute Dataset Validation & Manifest Generation
```bash
# Run validation on the Google Drive dataset, outputting to local repo data/processed
!python -m src.data.validate_rdd2022 \
    --data-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/datasets/RDD2022/extracted" \
    --output-dir "data/processed/RDD2022"
```

*Tip: For an ultra-fast dry-run that checks XML annotations and directory structures without decoding pixel data across all 47,420 images, add `--skip-image-decode`.*

### Cell 6: Persist Manifest & Summary to Google Drive (Optional)
```bash
# Backup generated manifest and summary JSON to Google Drive
!mkdir -p /content/drive/MyDrive/CCARC-Road-Damage-Research/manifests/
!cp data/processed/RDD2022/rdd2022_manifest.csv /content/drive/MyDrive/CCARC-Road-Damage-Research/manifests/
!cp data/processed/RDD2022/rdd2022_summary.json /content/drive/MyDrive/CCARC-Road-Damage-Research/manifests/
```

---

## 3. Generated Artifacts

The execution produces two research artifacts in `data/processed/RDD2022/`:
1. **`rdd2022_manifest.csv`**:
   - Tabular record for all 47,420 images across all 7 subsets.
   - Contains country mappings (grouping `China_Drone` and `China_MotorBike` under sovereign country `China`).
   - Standardizes target class counts (`count_D00`, `count_D10`, `count_D20`, `count_D40`).
   - Segregates annotation status into distinct categories:
     - `labeled_target`: at least one target class present.
     - `non_target_only`: XML contains only auxiliary classes (e.g., D01, D11).
     - `empty_xml`: XML file has 0 object annotations.
     - `unlabeled_test`: official test images (excluded from LOCO evaluation: `loco_eligible = False`).
   - Flags anomalies (such as the zero-width box in `Japan_001265.xml`).

2. **`rdd2022_summary.json`**:
   - Hierarchical statistics aggregated by country, subset, and class.
   - Exact audit log of any invalid bounding boxes without modifying original files.

---

## 4. Expected Validation Metrics

| Metric | Expected Value |
| :--- | :--- |
| **Total Images** | 47,420 |
| **Train Split Images (with XML)** | 38,385 |
| **Test Split Images (Unlabeled)** | 9,035 |
| **Empty XML Files (0 Objects)** | 11,724 |
| **Non-Target Only XML Files** | 2,894 |
| **Known Bounding Box Anomaly** | 1 (`Japan_001265.xml` with zero width) |
| **Target Classes** | D00, D10, D20, D40 |
| **LOCO Evaluation Eligible Images** | 38,385 |

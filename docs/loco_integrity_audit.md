# Comprehensive LOCO Dataset Integrity and Leakage Audit Guide

This document details the independent integrity audit protocol, mathematical zero-leakage checks, label geometry verification, reconciliation arithmetic, and Google Colab execution instructions for the CCARC research project.

---

## 1. Audit Overview & Objectives

Following the generation of the six Leave-One-Country-Out (LOCO) dataset folds for RDD2022, an independent, read-only audit is required to guarantee scientific rigor before conducting training experiments.

The audit tool (`src/data/loco_auditor.py` and CLI `src/data/audit_loco_splits.py`) evaluates **11 automated checks**:

1. **`CHK_01_SIX_FOLDS` (Six Folds Configuration)**: Verifies exactly 6 folds corresponding to the 6 benchmark countries (`China`, `Czech`, `India`, `Japan`, `Norway`, `United_States`).
2. **`CHK_02_ZERO_LEAKAGE` (Partition Disjointness & Zero Leakage)**: Strictly verifies that $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$, and that zero target-country samples exist in train or val.
3. **`CHK_03_HELD_OUT_ISOLATION` (Held-Out Isolation)**: Confirms that 100% of target-country samples reside in the test split and 0% in source train/val.
4. **`CHK_04_CHINA_GROUPING` (China Subsets Unification)**: Confirms `China_Drone` and `China_MotorBike` are unified under sovereign country `China` with zero target leakage into other folds.
5. **`CHK_05_FILE_READABILITY` (File Existence & Decoding)**: Verifies that every selected image file exists on disk and is decodable by OpenCV/PIL, and every label file exists and is non-empty.
6. **`CHK_06_YOLO_LABEL_SYNTAX` (YOLO Format & Geometry)**: Parses every line of every YOLO label file:
   - Exactly 5 numeric fields per line.
   - Class ID $\in \{0, 1, 2, 3\}$.
   - Finite normalized center coordinates $0 < x_c, y_c < 1$.
   - Positive dimensions $w, h > 0$.
   - Bounding boxes strictly within normalized image boundaries $[0, 1]$.
7. **`CHK_07_SYMLINK_INTEGRITY` (Runtime Symlink Integrity)**: Verifies that symlinks in `images/{train,val,test}` and `labels/{train,val,test}` resolve to the intended source images and converted labels without dangling or broken links.
8. **`CHK_08_YAML_AND_RESOLUTION` (YAML & Ultralytics Resolution)**: Confirms `data.yaml` has valid `path`, `train`, `val`, `test`, `nc=4`, and class names mapping. Verifies Ultralytics image-to-label resolution: replacing `${sep}images${sep}` with `${sep}labels${sep}` and `.jpg` with `.txt`.
9. **`CHK_09_RECONCILIATION` (Target Sample Reconciliation)**: Reconciles the 23,767 converted-target label files against the 19,221 selected LOCO images, producing an exact breakdown of all 4,546 exclusions.
10. **`CHK_10_SUMMARY_METRICS` (Independent Recomputation)**: Independently recomputes per-fold train/val/test counts and class distributions and compares them against `loco_summary.json` and `fold_summary.json`.
11. **`CHK_11_DUPLICATES_AND_HASHES` (Duplicates & Cryptographic Hashing)**: Checks for duplicate image IDs, duplicate source paths, and (optionally) identical cryptographic image content hashes.

---

## 2. Supervised Target Reconciliation Breakdown

Under the default conservative baseline policy (`--incomplete-label-policy exclude`):
- **Total Converted Target Labels**: 23,767 images.
- **Selected Supervised LOCO Images**: 19,221 images.
- **Excluded Images Total**: 4,546 images.

The auditor breaks down the 4,546 exclusions into exact constituent categories:
1. **`incomplete_label_risk`**: Images containing valid target annotations ($D00, D10, D20, D40$) alongside auxiliary non-target annotations ($D01, D11, \text{Repair}$). Quarantined by default to prevent unannotated background region contamination during baseline supervised training.
2. **`loco_ineligible`**: Images explicitly flagged as ineligible for quantitative LOCO cross-validation (e.g. official unlabeled test images or validation-flagged entries).
3. **`non_train_split`**: Entries originating outside the official training archives.
4. **`validation_flagged`**: Entries with corrupted dimensions or excluded bounding boxes.

The auditor strictly verifies:
$$\text{total\_selected} + \sum \text{exclusion\_categories} = \text{total\_converted\_target}$$

---

## 3. Methodological Note: Near-Duplicate Road Frames

> [!NOTE]
> Exact cryptographic hashing (MD5 / SHA-256) identifies bitwise identical image files. However, in vehicle-mounted road survey datasets like RDD2022, consecutive video frames captured along the same road segment depict the same pavement distress with minute camera displacement and varying lighting. These frames produce distinct cryptographic hashes and remain undetected without perceptual hashing or GPS/route-based trajectory clustering.
> Because an entire sovereign country is held out in each LOCO fold, cross-country evaluation is strictly protected from route-level leakage between train and test splits. Researchers should nevertheless interpret image-level source validation metrics with this sequential autocorrelation in mind.

---

## 4. Google Colab Execution Instructions

The audit tool operates in **strictly read-only mode** with respect to dataset archives, converted labels, and split manifests.

- **Repository**: `/content/CCARC-Road-Damage-Research`
- **Conversion Manifest**: `/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/yolo_conversion/yolo_conversion_manifest.csv`
- **LOCO Splits Directory**: `/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/loco_splits`
- **Runtime Views Directory**: `/content/datasets/loco_views`
- **Audit Output Directory**: `/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/loco_splits/audit`

### Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Navigate to Project Repository & Update
```bash
%cd /content/CCARC-Road-Damage-Research
!git pull origin main
```

### Cell 3: Run Full Pytest Suite
```bash
!pytest -v
```

### Cell 4: Execute Comprehensive LOCO Audit
```bash
!python -m src.data.audit_loco_splits \
  --conversion-manifest "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/yolo_conversion/yolo_conversion_manifest.csv" \
  --loco-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/loco_splits" \
  --views-dir "/content/datasets/loco_views" \
  --output-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/loco_splits/audit" \
  --check-hashes
```

### Cell 5: Inspect Audit Markdown Report
```python
from IPython.display import Markdown, display
from pathlib import Path

report_path = Path("/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/loco_splits/audit/loco_audit_report.md")
if report_path.is_file():
    display(Markdown(report_path.read_text(encoding="utf-8")))
else:
    print("Audit report not found at:", report_path)
```

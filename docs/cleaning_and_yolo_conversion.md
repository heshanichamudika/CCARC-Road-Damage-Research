# RDD2022 Cleaning Policy and YOLO Label Conversion Guide

This document details the cleaning methodology, mathematical coordinate conversion, label quarantine policies, and Google Colab Pro execution procedures for converting RDD2022 Pascal VOC annotations into YOLOv8 format for the CCARC research project.

---

## 1. Class Mapping Specification

The cross-country road damage benchmark standardizes on 4 core target classes across the 6 sovereign countries:

| Class Name | YOLO Class ID | Damage Category | Description |
| :---: | :---: | :--- | :--- |
| **D00** | **0** | Longitudinal Crack | Cracks running parallel to vehicle motion (wheel paths, construction joints). |
| **D10** | **1** | Transverse Crack | Lateral cracks perpendicular to vehicle motion. |
| **D20** | **2** | Alligator Crack | Interconnected structural fatigue cracking. |
| **D40** | **3** | Pothole | Surface depressions, potholes, and severe bumps. |

Auxiliary classes present in RDD2022 subsets (such as `D01`, `D11`, `D43`, `D44`, `Repair`) are treated as **non-target categories**.

---

## 2. Cleaning and Labeling Policies

### Policy 1: Individual Invalid Object Exclusion
- If an image contains an invalid bounding box (e.g. zero-width box in `Japan_001265.xml`), **only that specific box is excluded**.
- Any valid target bounding boxes in the same image are retained and converted into YOLO format.
- Every excluded box is recorded in the conversion audit log with its image ID, country, class, original coordinates, and reason for exclusion.

### Policy 2: Strict Out-of-Bounds Rejection (Default) vs. Optional Clamping
- **Strict Rejection (Default)**: Any bounding box with coordinates outside image boundaries ($x_{\min} < 0$, $y_{\min} < 0$, $x_{\max} > W$, or $y_{\max} > H$) is strictly excluded as invalid, protecting against annotation errors.
- **Optional Clamping Mode (`--clamp-out-of-bounds`)**: When explicitly enabled by the researcher, coordinates are clamped to $[0, W]$ and $[0, H]$. The transformation is explicitly logged in `clamped_boxes_log` with both original and clamped coordinates.

### Policy 3: Quarantine for Non-Target-Only Images
- **Rationale**: 2,894 training images in RDD2022 contain exclusively auxiliary classes (e.g. `D01` or `D11` cracks). If these images were converted into empty (0-byte) YOLO label files, the detector would treat visible road damage as verified negative background, penalizing the model when it detects cracks.
- **Action**: These images are categorized as `quarantined_non_target`. By default, no YOLO label files are generated for them, preventing false-negative background poisoning.

### Policy 4: Isolation of Empty XML Images (Candidate Negatives)
- **Rationale**: 11,724 training images have XML files with zero objects. While many depict undamaged road segments, some represent skipped or unannotated frames.
- **Action**: These images are classified as `empty_xml_candidate` rather than assumed ground-truth negatives.
- The pipeline provides an explicit flag (`--include-empty-as-negatives`) allowing researchers to control whether 0-byte label files are generated for them.

### Policy 5: Incomplete-Label Risk Tracking
- When an image contains both valid target annotations (e.g., `D00`) and auxiliary non-target annotations (e.g., `D01`), the valid target boxes are converted, but the image is tagged in the manifest with `incomplete_label_risk = True` and `has_non_target_objects = True`.
- **Experimental Notice**: Unannotated auxiliary cracks in these images may produce unpenalized or false background regions during standard YOLO training. These images may require separate handling or exclusion in sensitivity and ablation studies.

### Policy 6: Zero Ground-Truth Generation for Official Test Sets
- Official test images (9,035 images) lack public annotations.
- The pipeline skips label generation for test images, categorizing them as `unlabeled_test` and setting `loco_eligible = False`.

---

## 3. Mathematical Coordinate Normalization

Pascal VOC provides pixel bounding boxes $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ on an image with dimensions $W \times H$.

### Step 1: Positivity & Geometry Check
$$w_{px} = x_{\max} - x_{\min}, \quad h_{px} = y_{\max} - y_{\min}$$
If $w_{px} \le 0$ or $h_{px} \le 0$, the box is marked invalid and excluded (e.g. zero-width box).

### Step 2: Normalization to YOLO Format
$$x_{\text{center}} = \frac{x_{\min} + x_{\max}}{2 \times W}, \quad y_{\text{center}} = \frac{y_{\min} + y_{\max}}{2 \times H}$$
$$w = \frac{w_{px}}{W}, \quad h = \frac{h_{px}}{H}$$

### Step 3: Strict Range & Floating-Point Verification
$$0 < x_{\text{center}} < 1, \quad 0 < y_{\text{center}} < 1, \quad 0 < w \le 1, \quad 0 < h \le 1$$
Boundary edge-cases (e.g., full image coverage where $w = 1.0$) are verified within precision limits ($\epsilon = 10^{-6}$).

Each line written to `<image_id>.txt` takes the standard format:
```text
<class_id> <x_center> <y_center> <width> <height>
```

---

## 4. Google Colab Pro Execution Instructions

The execution strictly isolates compute and code on local Colab storage from persistent data on Google Drive:
- **Repository Location**: `/content/CCARC-Road-Damage-Research`
- **Extracted Dataset**: `/content/drive/MyDrive/CCARC-Road-Damage-Research/datasets/RDD2022/extracted`
- **Persistent Conversion Output**: `/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/yolo_conversion`

### Cell 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Cell 2: Navigate to Project Repository in `/content`
```bash
%cd /content/CCARC-Road-Damage-Research
!git pull origin main
```

### Cell 3: Run Unit Tests
```bash
!pytest -v
```

### Cell 4: Execute Cleaning & YOLO Label Conversion
```bash
# Execute cleaning policy and YOLO conversion directly into persistent Google Drive output folder
!python -m src.data.convert_to_yolo \
    --data-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/datasets/RDD2022/extracted" \
    --output-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/yolo_conversion"
```

*Options:*
- `--include-empty-as-negatives`: Write 0-byte label files for candidate negative images.
- `--clamp-out-of-bounds`: Clamp overflow bounding boxes instead of strictly excluding them (logs original and corrected coordinates).

### Cell 5: Inspect Generated Artifacts
```bash
!ls -la /content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/yolo_conversion/
!head -n 5 /content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/yolo_conversion/yolo_conversion_manifest.csv
!cat /content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/yolo_conversion/yolo_conversion_summary.json | head -n 40
```

---

## 5. Generated Artifacts

1. **`labels/{subset}/{split}/{image_id}.txt`**:
   - Cleaned YOLO annotations formatted with standard floating-point coordinates.
   - Distinct subdirectories for each subset prevent file collisions (e.g. `China_Drone` vs `China_MotorBike`).
2. **`yolo_conversion_manifest.csv`**:
   - Row-per-image inventory recording exact label paths, conversion status, YOLO box counts, class breakdown, exclusion counts, clamped counts, and incomplete label risk flags.
3. **`yolo_conversion_summary.json`**:
   - Comprehensive audit report detailing dataset totals, per-country statistics, class distributions, an audit log of excluded bounding boxes, and an audit log of clamped bounding boxes.

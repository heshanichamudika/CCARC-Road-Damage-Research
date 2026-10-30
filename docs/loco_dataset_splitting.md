# Six-Country Leave-One-Country-Out (LOCO) Dataset Splitting Guide

This document specifies the Leave-One-Country-Out (LOCO) dataset partitioning strategy, fold matrix, candidate selection policies, lightweight dataset view generation, and Google Colab Pro execution instructions for the CCARC research project.

---

## 1. LOCO Evaluation Paradigm

Cross-country generalization is the central scientific investigation of the CCARC framework. Conventional random train/val/test splits suffer from spatial and environmental auto-correlation (e.g. identical road textures, asphalt types, weather, and camera setups present across both train and test splits).

Under **Leave-One-Country-Out (LOCO)** cross-validation:
- **6 Sovereign Folds**: Exactly 6 evaluation folds are created, corresponding to the 6 countries in RDD2022: `China`, `Czech`, `India`, `Japan`, `Norway`, and `United_States`.
- **Target Country Isolation**: In fold $k$, all images belonging to the held-out country form the **unseen target test set**. No image from this country is ever exposed during training or hyperparameter tuning.
- **Source Country Split**: The remaining 5 source countries form the training pool, which is partitioned into **Source Training (80%)** and **Source Validation (20%)** using country-aware multi-label stratification.

### Six-Country LOCO Fold Matrix

| Fold | Held-Out Target Country | Source Countries (Train + Val) |
| :---: | :--- | :--- |
| **Fold 0** | `China` (Drone + MotorBike) | Czech, India, Japan, Norway, United_States |
| **Fold 1** | `Czech` | China, India, Japan, Norway, United_States |
| **Fold 2** | `India` | China, Czech, Japan, Norway, United_States |
| **Fold 3** | `Japan` | China, Czech, India, Norway, United_States |
| **Fold 4** | `Norway` | China, Czech, India, Japan, United_States |
| **Fold 5** | `United_States` | China, Czech, India, Japan, Norway |

---

## 2. Subset Grouping and Filtering Policies

### China Subsets Unification
In RDD2022, China is provided as two distinct archives: `China_Drone` and `China_MotorBike`, representing aerial and low-angle two-wheeler perspectives respectively.
- **Sovereign Grouping**: Both subsets are mapped to country `China`.
- **Target Fold Isolation**: When China is the held-out country (Fold 0), **both** `China_Drone` and `China_MotorBike` images are held out together as the target test set.
- **Source Fold Participation**: When another country is held out (Folds 1–5), both Chinese subsets participate in the source pool under the sovereign `China` partition.
- **Identity Preservation**: Manifest records preserve the granular `subset` identifier (`China_Drone` vs `China_MotorBike`) for downstream domain shift analysis.

### Supervised Pool Selection & Cleaning Policies
1. **Target Labeled Images (`converted_target`)**: Only images containing valid target annotations (D00, D10, D20, D40) and valid YOLO label files from task T02.16 form the primary supervised pool.
2. **Official Test Images (`unlabeled_test`)**: 9,035 official evaluation images lacking public annotations are strictly excluded from all LOCO folds (`loco_eligible = False`). They are never used as ground-truth test data.
3. **Empty XML Candidates (`empty_xml_candidate`)**: 11,724 unannotated frames are excluded by default (`--include-empty-candidates` optional) to protect against unverified negative contamination.
4. **Quarantined Non-Target Images (`quarantined_non_target`)**: 2,894 images with exclusively non-target damage types (`D01`, `D11`, `Repair`) are excluded by default (`--include-quarantined-non-target` optional) to prevent false-negative background poisoning.
5. **Incomplete-Label Risk Policy (`--incomplete-label-policy`)**:
   - `exclude` (**Default**): Discards all images tagged with `incomplete_label_risk = True` (i.e. images containing both target damages and unannotated auxiliary classes) to establish a rigorous, conservative baseline without background ambiguity.
   - `include` (**Ablation Mode**): Retains these images to test model sensitivity to partial background label noise in subsequent ablation experiments.

---

## 3. Stratified Source Partitioning & Methodological Considerations

### Stratification Methodology
The source training pool (5 countries) is split into 80% train and 20% validation:
1. **Country-Aware Independence**: Each source country is partitioned independently so that every source country contributes exactly 80% of its images to `train` and 20% to `val`.
2. **Multi-Label Class Stratification**: Within each country, images are binned by the co-occurrence pattern of classes present ($D00, D10, D20, D40$). Samples within each bin are deterministically assigned to train/val to preserve class prevalence across splits.
3. **Reproducibility**: Controlled by a deterministic random seed (`--seed 42` by default).

### Methodological Limitations & Target Scope
- **Image-Level Partitioning Limitation**: In RDD2022, images were captured sequentially from moving vehicles and drones along driving trajectories. However, road section identifiers, GPS tracks, and video sequence IDs are not provided in the public annotations. Consequently, train/val partitioning must occur at the image level rather than the route level. While held-out target evaluation is strictly immune to location leakage (an entire sovereign country is held out), researchers should note that source validation metrics may experience slight optimism due to consecutive frame correlations.
- **Restricted Target Evaluation Scope**: Held-out quantitative evaluation is conducted strictly on verified target-damaged pavement segments (`converted_target`), excluding candidate negatives (`empty_xml_candidate`) and non-target damage (`quarantined_non_target`). Metrics therefore evaluate domain-shifted performance specifically on target damage types (D00, D10, D20, D40) rather than full unconstrained road environments.

### Zero-Leakage Verification
Prior to writing dataset views, the partitioning engine runs strict programmatic assertions:
$$\text{Source-Train} \cap \text{Source-Val} = \emptyset$$
$$\text{Source-Train} \cap \text{Target-Test} = \emptyset$$
$$\text{Source-Val} \cap \text{Target-Test} = \emptyset$$
$$\forall x \in \text{Source-Train} \cup \text{Source-Val}, \quad \text{country}(x) \ne \text{held\_out\_country}$$
$$\forall x \in \text{Target-Test}, \quad \text{country}(x) == \text{held\_out\_country}$$

---

## 4. Lightweight YOLO Dataset View Architecture

Ultralytics YOLO resolves label paths for an image path `<path>/images/<split>/<filename>.jpg` by searching for `<path>/labels/<split>/<filename>.txt`.

To support 6 folds without replicating tens of gigabytes of image files, the pipeline constructs **symbolic link dataset views**:
```text
/content/datasets/loco_views/ (Local Fast SSD)
├── fold_0_China/
│   ├── data.yaml
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
│   ├── images/
│   │   ├── train/  -> (symlinks to source train images)
│   │   ├── val/    -> (symlinks to source val images)
│   │   └── test/   -> (symlinks to target test images)
│   └── labels/
│       ├── train/  -> (symlinks to source train YOLO txt labels)
│       ├── val/    -> (symlinks to source val YOLO txt labels)
│       └── test/   -> (symlinks to target test YOLO txt labels)
...
└── fold_5_United_States/

/content/drive/MyDrive/.../experiments/loco_splits/ (Google Drive Persistent)
├── loco_master_manifest.csv
├── loco_summary.json
├── fold_0_China/
│   ├── data.yaml
│   ├── fold_manifest.csv
│   └── fold_summary.json
...
└── fold_5_United_States/
```

### Generated `data.yaml` Structure
```yaml
path: /content/datasets/loco_views/fold_0_China
train: images/train
val: images/val
test: images/test
nc: 4
names:
  0: D00
  1: D10
  2: D20
  3: D40
loco_target_country: China
loco_source_countries:
  - Czech
  - India
  - Japan
  - Norway
  - United_States
```

---

## 5. Google Colab Pro Execution Instructions

Compute and symlink hierarchies run on local Colab storage (`/content`), while persistent dataset files, labels, and fold manifests reside on Google Drive:
- **Repository Location**: `/content/CCARC-Road-Damage-Research`
- **YOLO Conversion Manifest**: `/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/yolo_conversion/yolo_conversion_manifest.csv`
- **Persistent Output Directory**: `/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/loco_splits`
- **Runtime Views Directory**: `/content/datasets/loco_views`

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

### Cell 3: Run Pytest Verification
```bash
!pytest -v
```

### Cell 4: Execute LOCO Dataset Partitioning (Default Conservative Baseline)
```bash
!python -m src.data.create_loco_splits \
  --manifest-path "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/yolo_conversion/yolo_conversion_manifest.csv" \
  --output-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/loco_splits" \
  --views-dir "/content/datasets/loco_views" \
  --val-ratio 0.20 \
  --seed 42 \
  --incomplete-label-policy exclude
```

### Cell 5: Inspect Master Summary and Fold Distribution
```python
import json
from pathlib import Path

summary_path = Path("/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/loco_splits/loco_summary.json")
with open(summary_path, "r", encoding="utf-8") as f:
    summary = json.load(f)

print(f"Total Candidate Samples: {summary['totals']['total_candidate_samples']:,}")
print(f"Validation Ratio:        {summary['configuration']['val_ratio']}")
print(f"Random Seed:             {summary['configuration']['seed']}")
print(f"Incomplete Label Policy: {summary['configuration']['incomplete_label_policy']}")
print(f"All Leak Checks Passed:  {summary['totals']['all_leak_checks_passed']}")
print("\nPer-Fold Image Counts:")
for fold in summary["folds"]:
    counts = fold["counts"]
    print(f"  Fold {fold['fold_idx']} ({fold['target_country']:<13}) -> Train: {counts['train']:5d} | Val: {counts['val']:5d} | Test: {counts['test']:5d}")
```

### Cell 6: Verify Ultralytics YOLODataset Compatibility
```python
from ultralytics.data.dataset import YOLODataset
import yaml

fold_china_yaml = "/content/datasets/loco_views/fold_0_China/data.yaml"
with open(fold_china_yaml, "r", encoding="utf-8") as f:
    data_dict = yaml.safe_load(f)

print("Validating Fold 0 (China) Train dataset resolution with Ultralytics:")
ds = YOLODataset(
    img_path=f"{data_dict['path']}/{data_dict['train']}",
    data=data_dict,
    task="detect"
)
print(f"Successfully loaded {len(ds)} samples into Ultralytics dataset!")
sample = ds[0]
print(f"Sample 0 bounding boxes tensor shape: {sample['bboxes'].shape}")
```

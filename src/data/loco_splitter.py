"""Six-Country Leave-One-Country-Out (LOCO) Dataset Splitting Module for RDD2022.

Implements rigorous cross-country partitioning across six benchmark countries
(China, Czech, India, Japan, Norway, United_States), ensuring zero target-domain
leakage, country-aware stratification, deterministic reproducibility, and
YOLOv8-compatible dataset views.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import yaml

from src.data.rdd2022_validator import DEFAULT_TARGET_CLASSES
from src.data.yolo_converter import TARGET_CLASS_TO_ID, ConversionStatus

# The 6 canonical evaluation countries
LOCO_COUNTRIES: Tuple[str, ...] = (
    "China",
    "Czech",
    "India",
    "Japan",
    "Norway",
    "United_States",
)

CHINA_SUBSETS: Tuple[str, ...] = ("China_Drone", "China_MotorBike")


@dataclass
class FoldSplitResult:
    """Contains partitioned records and diagnostic audit for a single LOCO fold."""

    fold_idx: int
    target_country: str
    source_countries: List[str]
    train_records: List[Dict[str, Any]]
    val_records: List[Dict[str, Any]]
    test_records: List[Dict[str, Any]]
    leak_check_passed: bool
    overlap_count: int
    train_class_counts: Dict[str, int]
    val_class_counts: Dict[str, int]
    test_class_counts: Dict[str, int]
    country_breakdown: Dict[str, Dict[str, int]]
    data_yaml_path: Optional[str] = None


def create_link_or_copy(source_path: Path, dest_path: Path) -> str:
    """Create a lightweight symlink, falling back to hardlink or copy.

    Returns:
        Link method used ('symlink', 'hardlink', or 'copy').
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() or dest_path.is_symlink():
        dest_path.unlink()

    # 1. Try relative or absolute symlink (standard on Linux / Colab)
    try:
        os.symlink(source_path.resolve(), dest_path)
        return "symlink"
    except OSError:
        pass

    # 2. Try hardlink
    try:
        os.link(source_path.resolve(), dest_path)
        return "hardlink"
    except OSError:
        pass

    # 3. Fallback to copy (e.g. Windows without developer privileges)
    shutil.copy2(source_path.resolve(), dest_path)
    return "copy"


class RDD2022LOCOSplitter:
    """Orchestrates Leave-One-Country-Out (LOCO) partitioning for RDD2022.

    Scientific Methodology Notes:
    - Zero Target Exposure: When evaluating cross-country generalization, an entire sovereign country
      (including all of its capture modalities/subsets, such as China_Drone and China_MotorBike) is
      held out as the target test set. Zero target-country images are exposed during source training or validation.
    - Limitation of Image-Level Splitting: In RDD2022, images were captured sequentially via driving
      vehicles and drones, but road segment / route identifiers are absent in the public VOC XML annotations.
      Consequently, train/val partitioning of the source pool is performed at the image level using
      country-aware class stratification. While held-out target evaluation is immune to spatial leakage,
      source validation metrics may be subject to minor spatial auto-correlation across consecutive frames.
    - Restricted Target Scope: By default, held-out evaluation is conducted strictly on verified target-damaged
      samples (`converted_target`), excluding candidate negatives (`empty_xml_candidate`) and non-target damage
      (`quarantined_non_target`). Metrics reflect performance on target damage types (D00, D10, D20, D40) rather
      than the entire continuous road distribution.
    """

    def __init__(
        self,
        val_ratio: float = 0.20,
        seed: int = 42,
        incomplete_label_policy: str = "exclude",
        include_empty_candidates: bool = False,
        include_quarantined_non_target: bool = False,
        target_classes: Sequence[str] = DEFAULT_TARGET_CLASSES,
        loco_countries: Sequence[str] = LOCO_COUNTRIES,
    ) -> None:
        """Initialize the LOCO partitioner.

        Args:
            val_ratio: Fraction of source-country data allocated to validation (default 0.20).
            seed: Deterministic random seed for reproducibility.
            incomplete_label_policy: How to handle images with incomplete_label_risk ('exclude' default for
                conservative baseline, or 'include' for ablation studies).
            include_empty_candidates: Whether to include empty XML candidates (default False).
            include_quarantined_non_target: Whether to include quarantined non-target images (default False).
            target_classes: Sequence of target classes (D00, D10, D20, D40).
            loco_countries: Sequence of 6 countries for LOCO folds.
        """
        assert 0.0 < val_ratio < 1.0, f"val_ratio must be between 0 and 1, got {val_ratio}"
        assert incomplete_label_policy in (
            "include",
            "exclude",
        ), f"Unknown incomplete_label_policy: {incomplete_label_policy}"

        self.val_ratio = val_ratio
        self.seed = seed
        self.incomplete_label_policy = incomplete_label_policy
        self.include_empty_candidates = include_empty_candidates
        self.include_quarantined_non_target = include_quarantined_non_target
        self.target_classes = tuple(target_classes)
        self.loco_countries = tuple(loco_countries)

    def filter_candidate_records(
        self, manifest_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter manifest records to identify valid supervised training samples according to policies."""
        df = manifest_df.copy()

        # Enforce official training split eligibility only (exclude unlabeled official test set)
        if "split" in df.columns:
            df = df[df["split"] == "train"]
        if "loco_eligible" in df.columns:
            df = df[df["loco_eligible"] == True]

        # Allowed conversion statuses
        allowed_statuses = {ConversionStatus.CONVERTED_TARGET}
        if self.include_empty_candidates:
            allowed_statuses.add(ConversionStatus.EMPTY_XML_CANDIDATE)
        if self.include_quarantined_non_target:
            allowed_statuses.add(ConversionStatus.QUARANTINED_NON_TARGET)

        if "conversion_status" in df.columns:
            df = df[df["conversion_status"].isin(allowed_statuses)]

        # Apply incomplete_label_policy
        if self.incomplete_label_policy == "exclude" and "incomplete_label_risk" in df.columns:
            df = df[df["incomplete_label_risk"] == False]

        return df

    def _get_stratification_key(self, row: pd.Series) -> str:
        """Construct multi-class stratification key based on present target classes."""
        present_classes = []
        for cls_name in self.target_classes:
            col = f"count_{cls_name}"
            if col in row and row[col] is not None:
                try:
                    if int(row[col]) > 0:
                        present_classes.append(cls_name)
                except (ValueError, TypeError):
                    pass

        if not present_classes:
            return "none"
        return "_".join(sorted(present_classes))

    def split_source_country(
        self,
        country_df: pd.DataFrame,
        fold_idx: int,
        country_name: str,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split a single source country's records into train and val using stratified sampling."""
        records = country_df.to_dict(orient="records")
        if not records:
            return [], []

        # Group records by class stratum
        strata: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in records:
            key = self._get_stratification_key(pd.Series(r))
            strata[key].append(r)

        train_records: List[Dict[str, Any]] = []
        val_records: List[Dict[str, Any]] = []

        # Deterministic RNG with country and fold salt
        country_salt = sum(ord(c) for c in country_name)
        rng = random.Random(self.seed + fold_idx * 1000 + country_salt)

        for stratum_key in sorted(strata.keys()):
            stratum_items = strata[stratum_key]
            # Sort deterministically by image_id
            stratum_items.sort(key=lambda x: str(x.get("image_id", "")))
            rng.shuffle(stratum_items)

            n_total = len(stratum_items)
            n_val = int(round(n_total * self.val_ratio))
            # If stratum has more than 1 item but n_val rounds to 0 and total > 4, reserve at least 1 for val
            if n_val == 0 and n_total >= 5 and self.val_ratio > 0:
                n_val = 1
            if n_val >= n_total and n_total > 1:
                n_val = n_total - 1

            val_items = stratum_items[:n_val]
            train_items = stratum_items[n_val:]

            train_records.extend(train_items)
            val_records.extend(val_items)

        return train_records, val_records

    def generate_single_fold(
        self,
        filtered_df: pd.DataFrame,
        fold_idx: int,
        target_country: str,
    ) -> FoldSplitResult:
        """Generate a single LOCO fold holding out target_country."""
        assert target_country in self.loco_countries, f"Unknown target country: {target_country}"
        source_countries = [c for c in self.loco_countries if c != target_country]

        # 1. Held-out Target Country -> 100% into Test split
        test_df = filtered_df[filtered_df["country"] == target_country]
        test_records = test_df.to_dict(orient="records")

        # 2. Source Countries -> Stratified split into Train and Val
        train_records: List[Dict[str, Any]] = []
        val_records: List[Dict[str, Any]] = []

        for src_c in source_countries:
            src_df = filtered_df[filtered_df["country"] == src_c]
            c_train, c_val = self.split_source_country(
                country_df=src_df,
                fold_idx=fold_idx,
                country_name=src_c,
            )
            train_records.extend(c_train)
            val_records.extend(c_val)

        # 3. Mathematical Leakage & Overlap Validation
        train_ids = {r["image_id"] for r in train_records}
        val_ids = {r["image_id"] for r in val_records}
        test_ids = {r["image_id"] for r in test_records}

        overlap_train_val = len(train_ids.intersection(val_ids))
        overlap_train_test = len(train_ids.intersection(test_ids))
        overlap_val_test = len(val_ids.intersection(test_ids))
        total_overlap = overlap_train_val + overlap_train_test + overlap_val_test

        leak_check_passed = (
            (total_overlap == 0)
            and all(r["country"] == target_country for r in test_records)
            and all(r["country"] != target_country for r in train_records)
            and all(r["country"] != target_country for r in val_records)
        )

        # 4. Compute class and country breakdown
        train_classes = self._count_classes(train_records)
        val_classes = self._count_classes(val_records)
        test_classes = self._count_classes(test_records)

        country_breakdown: Dict[str, Dict[str, int]] = {}
        for c in self.loco_countries:
            country_breakdown[c] = {
                "train": sum(1 for r in train_records if r["country"] == c),
                "val": sum(1 for r in val_records if r["country"] == c),
                "test": sum(1 for r in test_records if r["country"] == c),
                "total": sum(1 for r in (train_records + val_records + test_records) if r["country"] == c),
            }

        return FoldSplitResult(
            fold_idx=fold_idx,
            target_country=target_country,
            source_countries=source_countries,
            train_records=train_records,
            val_records=val_records,
            test_records=test_records,
            leak_check_passed=leak_check_passed,
            overlap_count=total_overlap,
            train_class_counts=train_classes,
            val_class_counts=val_classes,
            test_class_counts=test_classes,
            country_breakdown=country_breakdown,
        )

    def _count_classes(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """Aggregate total target bounding box annotations for a record set."""
        totals = {c: 0 for c in self.target_classes}
        for r in records:
            for c in self.target_classes:
                col = f"count_{c}"
                if col in r and r[col] is not None:
                    try:
                        totals[c] += int(r[col])
                    except (ValueError, TypeError):
                        pass
        return totals

    def build_fold_dataset_view(
        self,
        fold_result: FoldSplitResult,
        fold_dir: Path,
        view_dir: Optional[Path] = None,
    ) -> Path:
        """Create a lightweight YOLO-compatible dataset view with verified label resolution.

        Creates directory structure in target view dir (and mirrors configs to fold_dir):
            view_dir/ (defaults to fold_dir if not specified)
            ├── data.yaml
            ├── train.txt
            ├── val.txt
            ├── test.txt
            ├── images/{train,val,test}/{image_id}.jpg
            └── labels/{train,val,test}/{image_id}.txt
        """
        fold_dir = fold_dir.resolve()
        target_view_dir = view_dir.resolve() if view_dir is not None else fold_dir

        fold_dir.mkdir(parents=True, exist_ok=True)
        target_view_dir.mkdir(parents=True, exist_ok=True)

        images_dir = target_view_dir / "images"
        labels_dir = target_view_dir / "labels"

        split_map = {
            "train": fold_result.train_records,
            "val": fold_result.val_records,
            "test": fold_result.test_records,
        }

        link_methods_used: Set[str] = set()

        for split_name, recs in split_map.items():
            sp_img_dir = images_dir / split_name
            sp_lbl_dir = labels_dir / split_name
            sp_img_dir.mkdir(parents=True, exist_ok=True)
            sp_lbl_dir.mkdir(parents=True, exist_ok=True)

            # Prevent stale links or mismatched labels from previous runs
            for stale_f in sp_img_dir.iterdir():
                if stale_f.is_symlink() or stale_f.is_file():
                    try:
                        stale_f.unlink()
                    except OSError:
                        pass
            for stale_f in sp_lbl_dir.iterdir():
                if stale_f.is_symlink() or stale_f.is_file():
                    try:
                        stale_f.unlink()
                    except OSError:
                        pass

            txt_lines = []
            used_stems: Dict[str, str] = {}

            for r in recs:
                img_path = Path(r["image_path"])
                label_path_str = r.get("label_path", "")
                label_path = Path(label_path_str) if label_path_str else None

                image_ext = img_path.suffix if img_path.suffix else ".jpg"
                stem = str(r["image_id"])

                # Disambiguate identical filenames across subsets if any exist
                if stem in used_stems and used_stems[stem] != str(img_path.resolve()):
                    stem = f"{r.get('subset', 'sub')}_{stem}"
                used_stems[stem] = str(img_path.resolve())

                dest_img = sp_img_dir / f"{stem}{image_ext}"
                dest_lbl = sp_lbl_dir / f"{stem}.txt"

                # Link or copy image
                if img_path.is_file():
                    method_img = create_link_or_copy(img_path, dest_img)
                    link_methods_used.add(method_img)
                    txt_lines.append(str(dest_img.resolve()))
                else:
                    # Record placeholder path for offline manifest
                    txt_lines.append(str(img_path.resolve()))

                # Link or copy label
                if label_path and label_path.is_file():
                    method_lbl = create_link_or_copy(label_path, dest_lbl)
                    link_methods_used.add(method_lbl)

            # Write split image list file to target_view_dir and fold_dir
            split_content = "\n".join(txt_lines) + "\n"
            (target_view_dir / f"{split_name}.txt").write_text(split_content, encoding="utf-8")
            if target_view_dir != fold_dir:
                (fold_dir / f"{split_name}.txt").write_text(split_content, encoding="utf-8")

        # Write YOLO data.yaml configuration
        # Standard Ultralytics layout: path is base directory, train/val/test are relative to path
        yolo_config = {
            "path": str(target_view_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(self.target_classes),
            "names": {TARGET_CLASS_TO_ID[cls]: cls for cls in self.target_classes},
            "loco_target_country": fold_result.target_country,
            "loco_source_countries": fold_result.source_countries,
        }

        yaml_path = target_view_dir / "data.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yolo_config, f, sort_keys=False)

        if target_view_dir != fold_dir:
            fold_yaml_path = fold_dir / "data.yaml"
            with open(fold_yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(yolo_config, f, sort_keys=False)

        fold_result.data_yaml_path = str(yaml_path.resolve())

        # Write fold manifest CSV to persistent fold_dir
        all_fold_records = []
        for split_name, recs in split_map.items():
            for r in recs:
                rec_copy = dict(r)
                rec_copy["loco_fold"] = fold_result.fold_idx
                rec_copy["loco_target_country"] = fold_result.target_country
                rec_copy["loco_split"] = split_name
                all_fold_records.append(rec_copy)

        fold_manifest_path = fold_dir / "fold_manifest.csv"
        pd.DataFrame(all_fold_records).to_csv(fold_manifest_path, index=False)

        # Write fold summary JSON to persistent fold_dir
        fold_summary = {
            "fold_idx": fold_result.fold_idx,
            "target_country": fold_result.target_country,
            "source_countries": fold_result.source_countries,
            "leak_check_passed": fold_result.leak_check_passed,
            "overlap_count": fold_result.overlap_count,
            "link_methods_used": list(link_methods_used),
            "counts": {
                "train": len(fold_result.train_records),
                "val": len(fold_result.val_records),
                "test": len(fold_result.test_records),
                "total": len(fold_result.train_records)
                + len(fold_result.val_records)
                + len(fold_result.test_records),
            },
            "class_counts": {
                "train": fold_result.train_class_counts,
                "val": fold_result.val_class_counts,
                "test": fold_result.test_class_counts,
            },
            "country_breakdown": fold_result.country_breakdown,
            "data_yaml": str(yaml_path.resolve()),
            "views_dir": str(target_view_dir.resolve()),
        }

        fold_summary_path = fold_dir / "fold_summary.json"
        with open(fold_summary_path, "w", encoding="utf-8") as f:
            json.dump(fold_summary, f, indent=2)

        return yaml_path

    def run_all_folds(
        self,
        manifest_df: pd.DataFrame,
        output_dir: Path,
        views_dir: Optional[Path] = None,
        clean_output: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Generate all 6 LOCO folds, dataset views, manifests and global summary."""
        output_dir = output_dir.resolve()
        if clean_output and output_dir.exists():
            print(f"Cleaning existing LOCO output directory: {output_dir}")
            shutil.rmtree(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        if views_dir is not None:
            views_dir = views_dir.resolve()
            if clean_output and views_dir.exists():
                print(f"Cleaning existing LOCO views directory: {views_dir}")
                shutil.rmtree(views_dir)
            views_dir.mkdir(parents=True, exist_ok=True)

        print(f"Total raw manifest rows: {len(manifest_df)}")
        filtered_df = self.filter_candidate_records(manifest_df)
        print(f"Filtered candidate supervised samples: {len(filtered_df)}")

        fold_results: List[FoldSplitResult] = []
        master_rows: List[Dict[str, Any]] = filtered_df.to_dict(orient="records")

        # Map image_id to assignments across folds
        image_assignments: Dict[str, Dict[str, str]] = defaultdict(dict)

        for idx, target_country in enumerate(self.loco_countries):
            fold_name = f"fold_{idx}_{target_country}"
            fold_dir = output_dir / fold_name
            view_fold_dir = views_dir / fold_name if views_dir is not None else None
            print(f"Generating LOCO Fold {idx}: Holding out {target_country} -> {fold_dir}")

            fold_res = self.generate_single_fold(
                filtered_df=filtered_df,
                fold_idx=idx,
                target_country=target_country,
            )

            assert fold_res.leak_check_passed, f"Leakage detected in fold {idx} ({target_country})!"

            self.build_fold_dataset_view(fold_res, fold_dir=fold_dir, view_dir=view_fold_dir)
            fold_results.append(fold_res)

            # Record assignments
            for r in fold_res.train_records:
                image_assignments[r["image_id"]][f"fold_{target_country}"] = "train"
            for r in fold_res.val_records:
                image_assignments[r["image_id"]][f"fold_{target_country}"] = "val"
            for r in fold_res.test_records:
                image_assignments[r["image_id"]][f"fold_{target_country}"] = "test"

        # Construct Master LOCO Manifest
        for row in master_rows:
            iid = row["image_id"]
            for target_country in self.loco_countries:
                row[f"fold_{target_country}"] = image_assignments[iid].get(f"fold_{target_country}", "unassigned")

        master_df = pd.DataFrame(master_rows)
        master_manifest_path = output_dir / "loco_master_manifest.csv"
        master_df.to_csv(master_manifest_path, index=False)
        print(f"Generated Master LOCO Manifest: {master_manifest_path} ({len(master_df)} rows)")

        # Global Summary JSON
        global_summary = {
            "dataset_name": "RDD2022",
            "protocol": "Source_Only_Leave_One_Country_Out_LOCO",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "output_dir": str(output_dir),
            "views_dir": str(views_dir) if views_dir is not None else str(output_dir),
            "configuration": {
                "val_ratio": self.val_ratio,
                "seed": self.seed,
                "incomplete_label_policy": self.incomplete_label_policy,
                "include_empty_candidates": self.include_empty_candidates,
                "include_quarantined_non_target": self.include_quarantined_non_target,
                "target_classes": list(self.target_classes),
                "loco_countries": list(self.loco_countries),
            },
            "totals": {
                "total_candidate_samples": len(filtered_df),
                "num_folds": len(fold_results),
                "all_leak_checks_passed": all(f.leak_check_passed for f in fold_results),
            },
            "folds": [
                {
                    "fold_idx": f.fold_idx,
                    "target_country": f.target_country,
                    "source_countries": f.source_countries,
                    "leak_check_passed": f.leak_check_passed,
                    "counts": {
                        "train": len(f.train_records),
                        "val": len(f.val_records),
                        "test": len(f.test_records),
                        "total": len(f.train_records) + len(f.val_records) + len(f.test_records),
                    },
                    "class_counts": {
                        "train": f.train_class_counts,
                        "val": f.val_class_counts,
                        "test": f.test_class_counts,
                    },
                    "country_breakdown": f.country_breakdown,
                    "data_yaml": f.data_yaml_path,
                }
                for f in fold_results
            ],
        }

        summary_json_path = output_dir / "loco_summary.json"
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(global_summary, f, indent=2)
        print(f"Generated Global LOCO Summary: {summary_json_path}")

        return master_df, global_summary

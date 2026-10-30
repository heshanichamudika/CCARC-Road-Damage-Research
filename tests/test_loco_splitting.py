"""Unit and Integration Tests for Six-Country Leave-One-Country-Out (LOCO) Splitting."""

import json
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import pytest
import yaml

from ultralytics.data.dataset import YOLODataset

from src.data.loco_splitter import (
    CHINA_SUBSETS,
    LOCO_COUNTRIES,
    RDD2022LOCOSplitter,
)
from src.data.yolo_converter import ConversionStatus


def create_mock_manifest_record(
    image_id: str,
    country: str,
    subset: str,
    split: str = "train",
    conversion_status: str = ConversionStatus.CONVERTED_TARGET,
    loco_eligible: bool = True,
    incomplete_label_risk: bool = False,
    count_D00: int = 1,
    count_D10: int = 0,
    count_D20: int = 0,
    count_D40: int = 0,
    base_dir: Path | None = None,
) -> dict:
    """Helper to create a synthetic manifest record with optional real files."""
    img_path = ""
    lbl_path = ""
    if base_dir:
        img_file = base_dir / "images" / f"{image_id}.jpg"
        lbl_file = base_dir / "labels" / f"{image_id}.txt"
        img_file.parent.mkdir(parents=True, exist_ok=True)
        lbl_file.parent.mkdir(parents=True, exist_ok=True)
        if not img_file.exists():
            cv2.imwrite(str(img_file), np.zeros((100, 100, 3), dtype=np.uint8))
        if not lbl_file.exists():
            lbl_file.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        img_path = str(img_file.resolve())
        lbl_path = str(lbl_file.resolve())

    return {
        "image_id": image_id,
        "image_path": img_path,
        "xml_path": f"/dummy/xmls/{image_id}.xml",
        "label_path": lbl_path,
        "country": country,
        "subset": subset,
        "split": split,
        "width": 640,
        "height": 640,
        "conversion_status": conversion_status,
        "yolo_boxes_count": count_D00 + count_D10 + count_D20 + count_D40,
        "count_D00": count_D00,
        "count_D10": count_D10,
        "count_D20": count_D20,
        "count_D40": count_D40,
        "excluded_boxes_count": 0,
        "clamped_boxes_count": 0,
        "has_non_target_objects": False,
        "non_target_classes": "",
        "incomplete_label_risk": incomplete_label_risk,
        "validation_flags": "",
        "loco_eligible": loco_eligible,
    }


def generate_synthetic_six_country_manifest(base_dir: Path | None = None) -> pd.DataFrame:
    """Create a balanced mock manifest containing all 6 countries, subsets, and edge cases."""
    records = []

    # 1. China: 10 images in China_Drone, 10 images in China_MotorBike
    for i in range(10):
        records.append(
            create_mock_manifest_record(
                f"China_Drone_{i:04d}",
                "China",
                "China_Drone",
                base_dir=base_dir,
            )
        )
    for i in range(10):
        records.append(
            create_mock_manifest_record(
                f"China_MotorBike_{i:04d}",
                "China",
                "China_MotorBike",
                base_dir=base_dir,
            )
        )

    # 2. Czech, India, Japan, Norway, United_States: 20 images each
    for country in ["Czech", "India", "Japan", "Norway", "United_States"]:
        for i in range(20):
            records.append(
                create_mock_manifest_record(
                    f"{country}_{i:04d}",
                    country,
                    country,
                    base_dir=base_dir,
                )
            )

    # Edge cases
    # Official test image (must be excluded)
    records.append(
        create_mock_manifest_record(
            "Japan_test_01",
            "Japan",
            "Japan",
            split="test",
            conversion_status=ConversionStatus.UNLABELED_TEST,
            loco_eligible=False,
            base_dir=base_dir,
        )
    )

    # Empty XML candidate (must be excluded by default)
    records.append(
        create_mock_manifest_record(
            "India_empty_01",
            "India",
            "India",
            conversion_status=ConversionStatus.EMPTY_XML_CANDIDATE,
            base_dir=base_dir,
        )
    )

    # Quarantined non-target (must be excluded by default)
    records.append(
        create_mock_manifest_record(
            "Czech_quarantined_01",
            "Czech",
            "Czech",
            conversion_status=ConversionStatus.QUARANTINED_NON_TARGET,
            base_dir=base_dir,
        )
    )

    # Incomplete label risk image
    records.append(
        create_mock_manifest_record(
            "Norway_incomplete_01",
            "Norway",
            "Norway",
            incomplete_label_risk=True,
            base_dir=base_dir,
        )
    )

    return pd.DataFrame(records)


def test_country_isolation_and_zero_leakage(tmp_path: Path):
    """Verify that every fold strictly isolates target country in test with zero leakage into train/val."""
    df_manifest = generate_synthetic_six_country_manifest()
    splitter = RDD2022LOCOSplitter(val_ratio=0.20, seed=42)

    master_df, summary = splitter.run_all_folds(
        manifest_df=df_manifest,
        output_dir=tmp_path / "loco_out",
        clean_output=True,
    )

    assert summary["totals"]["all_leak_checks_passed"] is True
    assert summary["totals"]["num_folds"] == 6

    for fold in summary["folds"]:
        target = fold["target_country"]
        source_countries = fold["source_countries"]
        assert len(source_countries) == 5
        assert target not in source_countries

        breakdown = fold["country_breakdown"]
        # Target country MUST have 0 train and 0 val
        assert breakdown[target]["train"] == 0
        assert breakdown[target]["val"] == 0
        assert breakdown[target]["test"] > 0

        # All source countries MUST have 0 test
        for sc in source_countries:
            assert breakdown[sc]["test"] == 0
            assert breakdown[sc]["train"] > 0
            assert breakdown[sc]["val"] > 0


def test_china_subsets_grouped_together(tmp_path: Path):
    """Verify that China_Drone and China_MotorBike both move to test when China is held out."""
    df_manifest = generate_synthetic_six_country_manifest()
    splitter = RDD2022LOCOSplitter(val_ratio=0.20, seed=42)

    # Generate Fold China (idx 0)
    filtered = splitter.filter_candidate_records(df_manifest)
    china_fold = splitter.generate_single_fold(filtered, fold_idx=0, target_country="China")

    test_subsets = {r["subset"] for r in china_fold.test_records}
    assert "China_Drone" in test_subsets
    assert "China_MotorBike" in test_subsets
    assert test_subsets == {"China_Drone", "China_MotorBike"}

    train_subsets = {r["subset"] for r in china_fold.train_records}
    val_subsets = {r["subset"] for r in china_fold.val_records}
    assert "China_Drone" not in train_subsets
    assert "China_MotorBike" not in train_subsets
    assert "China_Drone" not in val_subsets
    assert "China_MotorBike" not in val_subsets


def test_unlabeled_official_test_excluded(tmp_path: Path):
    """Verify that official unlabeled test split images are never included in any fold."""
    df_manifest = generate_synthetic_six_country_manifest()
    splitter = RDD2022LOCOSplitter(val_ratio=0.20, seed=42)
    filtered = splitter.filter_candidate_records(df_manifest)

    # Japan_test_01 must be filtered out
    filtered_ids = set(filtered["image_id"])
    assert "Japan_test_01" not in filtered_ids


def test_excluded_categories_policy():
    """Verify that empty_xml_candidate and quarantined_non_target are excluded by default."""
    df_manifest = generate_synthetic_six_country_manifest()

    # 1. Default policy: excluded
    splitter_default = RDD2022LOCOSplitter()
    filtered_default = splitter_default.filter_candidate_records(df_manifest)
    default_ids = set(filtered_default["image_id"])
    assert "India_empty_01" not in default_ids
    assert "Czech_quarantined_01" not in default_ids

    # 2. Include empty candidates flag: included
    splitter_empty = RDD2022LOCOSplitter(include_empty_candidates=True)
    filtered_empty = splitter_empty.filter_candidate_records(df_manifest)
    empty_ids = set(filtered_empty["image_id"])
    assert "India_empty_01" in empty_ids


def test_incomplete_label_policy():
    """Verify configurable incomplete_label_policy ('exclude' default vs 'include' ablation)."""
    df_manifest = generate_synthetic_six_country_manifest()

    # 1. Default policy: conservative exclusion of incomplete-label risk
    splitter_default = RDD2022LOCOSplitter()
    assert splitter_default.incomplete_label_policy == "exclude"
    filtered_default = splitter_default.filter_candidate_records(df_manifest)
    assert "Norway_incomplete_01" not in set(filtered_default["image_id"])

    # 2. Explicit inclusion for ablation studies
    splitter_inc = RDD2022LOCOSplitter(incomplete_label_policy="include")
    filtered_inc = splitter_inc.filter_candidate_records(df_manifest)
    assert "Norway_incomplete_01" in set(filtered_inc["image_id"])


def test_deterministic_reproducibility():
    """Verify that LOCO splits are 100% deterministic given the same seed."""
    df_manifest = generate_synthetic_six_country_manifest()
    splitter_1 = RDD2022LOCOSplitter(seed=42)
    splitter_2 = RDD2022LOCOSplitter(seed=42)

    filtered_1 = splitter_1.filter_candidate_records(df_manifest)
    filtered_2 = splitter_2.filter_candidate_records(df_manifest)

    fold_a = splitter_1.generate_single_fold(filtered_1, fold_idx=3, target_country="Japan")
    fold_b = splitter_2.generate_single_fold(filtered_2, fold_idx=3, target_country="Japan")

    train_ids_a = [r["image_id"] for r in fold_a.train_records]
    train_ids_b = [r["image_id"] for r in fold_b.train_records]
    assert train_ids_a == train_ids_b

    val_ids_a = [r["image_id"] for r in fold_a.val_records]
    val_ids_b = [r["image_id"] for r in fold_b.val_records]
    assert val_ids_a == val_ids_b


def test_yolo_dataset_view_and_yaml_loading(tmp_path: Path):
    """Integration test: build dataset view and verify loading directly with Ultralytics YOLODataset."""
    base_dir = tmp_path / "mock_dataset"
    df_manifest = generate_synthetic_six_country_manifest(base_dir=base_dir)

    output_dir = tmp_path / "loco_output"
    splitter = RDD2022LOCOSplitter(seed=42, val_ratio=0.20, incomplete_label_policy="include")

    master_df, summary = splitter.run_all_folds(
        manifest_df=df_manifest,
        output_dir=output_dir,
        clean_output=True,
    )

    # Verify Fold Japan
    fold_japan_dir = output_dir / "fold_3_Japan"
    yaml_path = fold_japan_dir / "data.yaml"
    assert yaml_path.is_file()

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    assert data_cfg["loco_target_country"] == "Japan"
    assert "Japan" not in data_cfg["loco_source_countries"]
    assert len(data_cfg["loco_source_countries"]) == 5

    # Check images and labels directories exist
    assert (fold_japan_dir / "images" / "train").is_dir()
    assert (fold_japan_dir / "labels" / "train").is_dir()
    assert (fold_japan_dir / "images" / "test").is_dir()
    assert (fold_japan_dir / "labels" / "test").is_dir()

    # Load with Ultralytics YOLODataset to verify label resolution
    ds_train = YOLODataset(
        img_path=str(fold_japan_dir / "images" / "train"),
        data=data_cfg,
        task="detect",
    )
    assert len(ds_train) > 0
    # Check sample item loads valid bounding box tensor
    sample = ds_train[0]
    assert "bboxes" in sample
    assert sample["bboxes"].shape[0] >= 1

    ds_test = YOLODataset(
        img_path=str(fold_japan_dir / "images" / "test"),
        data=data_cfg,
        task="detect",
    )
    assert len(ds_test) == 20  # Exactly the 20 Japan images
    assert sample["bboxes"].shape[0] >= 1


def test_separate_views_directory(tmp_path: Path):
    """Verify that manifests persist in output_dir while symlink views reside in views_dir."""
    base_dir = tmp_path / "mock_dataset_sep"
    df_manifest = generate_synthetic_six_country_manifest(base_dir=base_dir)

    output_dir = tmp_path / "gdrive_persistent"
    views_dir = tmp_path / "colab_local_ssd_views"
    splitter = RDD2022LOCOSplitter(seed=42, val_ratio=0.20, incomplete_label_policy="include")

    master_df, summary = splitter.run_all_folds(
        manifest_df=df_manifest,
        output_dir=output_dir,
        views_dir=views_dir,
        clean_output=True,
    )

    # Master manifests must be in output_dir
    assert (output_dir / "loco_master_manifest.csv").is_file()
    assert (output_dir / "loco_summary.json").is_file()

    # Per-fold manifests in output_dir
    assert (output_dir / "fold_0_China" / "fold_manifest.csv").is_file()
    assert (output_dir / "fold_0_China" / "fold_summary.json").is_file()
    assert (output_dir / "fold_0_China" / "data.yaml").is_file()

    # Symlink views must be in views_dir
    fold_view = views_dir / "fold_0_China"
    assert (fold_view / "images" / "train").is_dir()
    assert (fold_view / "labels" / "train").is_dir()
    assert (fold_view / "data.yaml").is_file()

    with open(fold_view / "data.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["path"] == str(fold_view.resolve())

    # Verify Ultralytics YOLODataset loads from view
    ds = YOLODataset(
        img_path=str(fold_view / "images" / "test"),
        data=cfg,
        task="detect",
    )
    # China fold has 10 China_Drone + 10 China_MotorBike = 20 test images
    assert len(ds) == 20


def test_rerun_stale_link_cleanup(tmp_path: Path):
    """Verify that rerunning build_fold_dataset_view cleans up stale files/symlinks."""
    base_dir = tmp_path / "mock_cleanup"
    df_manifest = generate_synthetic_six_country_manifest(base_dir=base_dir)

    fold_dir = tmp_path / "fold_cleanup"
    splitter = RDD2022LOCOSplitter(seed=42, val_ratio=0.20)
    filtered = splitter.filter_candidate_records(df_manifest)
    fold_res = splitter.generate_single_fold(filtered, fold_idx=1, target_country="Czech")

    # Initial build
    splitter.build_fold_dataset_view(fold_res, fold_dir)

    # Plant a rogue stale file in images/train and labels/train
    stale_img = fold_dir / "images" / "train" / "rogue_stale_image.jpg"
    stale_lbl = fold_dir / "labels" / "train" / "rogue_stale_image.txt"
    stale_img.write_text("rogue", encoding="utf-8")
    stale_lbl.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    assert stale_img.exists()
    assert stale_lbl.exists()

    # Rerun build_fold_dataset_view
    splitter.build_fold_dataset_view(fold_res, fold_dir)

    # Rogue stale files must have been purged
    assert not stale_img.exists()
    assert not stale_lbl.exists()

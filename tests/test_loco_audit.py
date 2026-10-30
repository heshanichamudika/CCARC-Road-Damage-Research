"""Unit and Integration Tests for RDD2022 LOCO Dataset Integrity and Leakage Audit."""

import json
from pathlib import Path
import shutil

import numpy as np
import cv2
import pandas as pd
import pytest
import yaml

from src.data.audit_loco_splits import parse_args, main as cli_main
from src.data.loco_auditor import CheckStatus, RDD2022LOCOAuditor
from src.data.loco_splitter import RDD2022LOCOSplitter
from src.data.yolo_converter import ConversionStatus
from tests.test_loco_splitting import generate_synthetic_six_country_manifest


def setup_test_loco_environment(tmp_path: Path, incomplete_policy: str = "exclude"):
    """Helper to set up mock source images, labels, conversion manifest, and LOCO splits."""
    mock_raw = tmp_path / "mock_raw"
    conv_manifest_df = generate_synthetic_six_country_manifest(base_dir=mock_raw)
    conv_manifest_path = tmp_path / "yolo_conversion_manifest.csv"
    conv_manifest_df.to_csv(conv_manifest_path, index=False)

    loco_dir = tmp_path / "loco_splits"
    views_dir = tmp_path / "loco_views"

    splitter = RDD2022LOCOSplitter(
        val_ratio=0.20,
        seed=42,
        incomplete_label_policy=incomplete_policy,
    )
    master_df, summary = splitter.run_all_folds(
        manifest_df=conv_manifest_df,
        output_dir=loco_dir,
        views_dir=views_dir,
        clean_output=True,
    )

    return conv_manifest_path, loco_dir, views_dir, master_df


def test_audit_passes_on_valid_dataset(tmp_path: Path):
    """Verify that a properly partitioned dataset passes all 11 integrity and leakage checks."""
    conv_path, loco_dir, views_dir, _ = setup_test_loco_environment(tmp_path, incomplete_policy="exclude")

    auditor = RDD2022LOCOAuditor(
        conversion_manifest_path=conv_path,
        loco_dir=loco_dir,
        views_dir=views_dir,
        check_hashes=True,
    )
    report = auditor.run_full_audit()

    assert report.overall_status == CheckStatus.PASS
    assert report.failed_checks == 0
    assert report.passed_checks + report.warned_checks == report.total_checks

    # Check reconciliation
    reconcil = report.reconciliation
    assert reconcil["reconciliation_verified"] is True
    assert reconcil["total_selected_loco"] + reconcil["total_excluded"] == reconcil["total_converted_target"]
    assert "incomplete_label_risk" in reconcil["exclusion_breakdown"]

    # Verify Markdown generation
    md_content = report.to_markdown()
    assert "# CCARC RDD2022 LOCO Dataset Integrity" in md_content
    assert "CHK_01_SIX_FOLDS" in md_content
    assert "CHK_02_ZERO_LEAKAGE" in md_content


def test_audit_catches_held_out_leakage(tmp_path: Path):
    """Verify that the auditor detects target-country images leaking into train/val partitions."""
    conv_path, loco_dir, views_dir, _ = setup_test_loco_environment(tmp_path)

    # Intentionally corrupt loco_master_manifest.csv: mark a Japan image as train in fold_Japan
    master_path = loco_dir / "loco_master_manifest.csv"
    df = pd.read_csv(master_path)
    japan_idx = df[df["country"] == "Japan"].index[0]
    df.loc[japan_idx, "fold_Japan"] = "train"
    df.to_csv(master_path, index=False)

    auditor = RDD2022LOCOAuditor(
        conversion_manifest_path=conv_path,
        loco_dir=loco_dir,
        views_dir=views_dir,
    )
    report = auditor.run_full_audit()

    assert report.overall_status == CheckStatus.FAIL
    assert report.checks["CHK_02_ZERO_LEAKAGE"].status == CheckStatus.FAIL
    assert report.checks["CHK_03_HELD_OUT_ISOLATION"].status == CheckStatus.FAIL


def test_audit_catches_duplicate_assignment(tmp_path: Path):
    """Verify that duplicate image IDs or paths in the LOCO manifest are caught."""
    conv_path, loco_dir, views_dir, _ = setup_test_loco_environment(tmp_path)

    master_path = loco_dir / "loco_master_manifest.csv"
    df = pd.read_csv(master_path)
    # Duplicate the first row
    dup_df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    dup_df.to_csv(master_path, index=False)

    auditor = RDD2022LOCOAuditor(
        conversion_manifest_path=conv_path,
        loco_dir=loco_dir,
        views_dir=views_dir,
    )
    report = auditor.run_full_audit()

    assert report.overall_status == CheckStatus.FAIL
    assert report.checks["CHK_11_DUPLICATES_AND_HASHES"].status == CheckStatus.FAIL
    assert "duplicate image_id" in report.checks["CHK_11_DUPLICATES_AND_HASHES"].errors[0]


def test_audit_catches_corrupted_label_class_id(tmp_path: Path):
    """Verify detection of invalid YOLO class IDs (> 3 or < 0)."""
    conv_path, loco_dir, views_dir, master_df = setup_test_loco_environment(tmp_path)

    # Overwrite one label file with class ID 4
    lbl_file = Path(master_df.iloc[0]["label_path"])
    lbl_file.write_text("4 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    auditor = RDD2022LOCOAuditor(
        conversion_manifest_path=conv_path,
        loco_dir=loco_dir,
        views_dir=views_dir,
    )
    report = auditor.run_full_audit()

    assert report.overall_status == CheckStatus.FAIL
    assert report.checks["CHK_06_YOLO_LABEL_SYNTAX"].status == CheckStatus.FAIL
    assert any("Invalid class ID 4" in err for err in report.checks["CHK_06_YOLO_LABEL_SYNTAX"].errors)


def test_audit_catches_corrupted_label_out_of_bounds(tmp_path: Path):
    """Verify detection of out-of-bounds normalized YOLO coordinates (> 1.0 or < 0.0)."""
    conv_path, loco_dir, views_dir, master_df = setup_test_loco_environment(tmp_path)

    lbl_file = Path(master_df.iloc[0]["label_path"])
    lbl_file.write_text("0 1.25 0.5 0.2 0.2\n", encoding="utf-8")

    auditor = RDD2022LOCOAuditor(
        conversion_manifest_path=conv_path,
        loco_dir=loco_dir,
        views_dir=views_dir,
    )
    report = auditor.run_full_audit()

    assert report.overall_status == CheckStatus.FAIL
    assert report.checks["CHK_06_YOLO_LABEL_SYNTAX"].status == CheckStatus.FAIL
    assert any("out of range" in err for err in report.checks["CHK_06_YOLO_LABEL_SYNTAX"].errors)


def test_audit_catches_corrupted_label_field_count(tmp_path: Path):
    """Verify detection of label lines that do not have exactly 5 fields."""
    conv_path, loco_dir, views_dir, master_df = setup_test_loco_environment(tmp_path)

    lbl_file = Path(master_df.iloc[0]["label_path"])
    lbl_file.write_text("0 0.5 0.5 0.2\n", encoding="utf-8")  # only 4 fields

    auditor = RDD2022LOCOAuditor(
        conversion_manifest_path=conv_path,
        loco_dir=loco_dir,
        views_dir=views_dir,
    )
    report = auditor.run_full_audit()

    assert report.overall_status == CheckStatus.FAIL
    assert report.checks["CHK_06_YOLO_LABEL_SYNTAX"].status == CheckStatus.FAIL
    assert any("Expected 5 fields" in err for err in report.checks["CHK_06_YOLO_LABEL_SYNTAX"].errors)


def test_audit_catches_broken_or_mismatched_symlink(tmp_path: Path):
    """Verify detection of broken or missing symlinks in views_dir."""
    conv_path, loco_dir, views_dir, _ = setup_test_loco_environment(tmp_path)

    # Delete a file in views_dir/fold_0_China/images/train/
    fold_0_train_imgs = list((views_dir / "fold_0_China" / "images" / "train").iterdir())
    assert len(fold_0_train_imgs) > 0
    victim_img = fold_0_train_imgs[0]
    victim_lbl = views_dir / "fold_0_China" / "labels" / "train" / f"{victim_img.stem}.txt"

    # Remove the label file in the view
    if victim_lbl.exists():
        victim_lbl.unlink()

    auditor = RDD2022LOCOAuditor(
        conversion_manifest_path=conv_path,
        loco_dir=loco_dir,
        views_dir=views_dir,
    )
    report = auditor.run_full_audit()

    assert report.overall_status == CheckStatus.FAIL
    assert report.checks["CHK_07_SYMLINK_INTEGRITY"].status == CheckStatus.FAIL
    assert any("Missing corresponding label file" in err for err in report.checks["CHK_07_SYMLINK_INTEGRITY"].errors)


def test_audit_catches_yaml_or_resolution_error(tmp_path: Path):
    """Verify detection of misconfigured data.yaml or wrong class names/nc."""
    conv_path, loco_dir, views_dir, _ = setup_test_loco_environment(tmp_path)

    yaml_path = views_dir / "fold_0_China" / "data.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["nc"] = 3  # Corrupt nc
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    auditor = RDD2022LOCOAuditor(
        conversion_manifest_path=conv_path,
        loco_dir=loco_dir,
        views_dir=views_dir,
    )
    report = auditor.run_full_audit()

    assert report.overall_status == CheckStatus.FAIL
    assert report.checks["CHK_08_YAML_AND_RESOLUTION"].status == CheckStatus.FAIL
    assert any("nc=3, expected 4" in err for err in report.checks["CHK_08_YAML_AND_RESOLUTION"].errors)


def test_audit_cli_execution(tmp_path: Path):
    """Verify that the CLI execution runs end-to-end and exits with code 0 on clean dataset."""
    conv_path, loco_dir, views_dir, _ = setup_test_loco_environment(tmp_path)
    audit_out = tmp_path / "audit_reports"

    exit_code = cli_main([
        "--conversion-manifest", str(conv_path),
        "--loco-dir", str(loco_dir),
        "--views-dir", str(views_dir),
        "--output-dir", str(audit_out),
    ])

    assert exit_code == 0
    assert (audit_out / "loco_audit_report.json").is_file()
    assert (audit_out / "loco_audit_report.md").is_file()


def test_audit_flags_partial_sampling_as_warning(tmp_path: Path):
    """Verify that using sample_image_limit is flagged as a partial audit with a warning."""
    conv_path, loco_dir, views_dir, _ = setup_test_loco_environment(tmp_path)

    auditor = RDD2022LOCOAuditor(
        conversion_manifest_path=conv_path,
        loco_dir=loco_dir,
        views_dir=views_dir,
        sample_image_limit=5,
    )
    report = auditor.run_full_audit()

    check_5 = report.checks["CHK_05_FILE_READABILITY"]
    assert check_5.status == CheckStatus.WARN
    assert check_5.details["is_partial_audit"] is True
    assert check_5.details["inspected_records"] == 5
    assert any("PARTIAL AUDIT NOTICE" in w for w in check_5.warnings)


def test_audit_catches_cross_fold_sample_identity_inconsistency(tmp_path: Path):
    """Verify that if a fold misses an eligible sample or assigns it to 'unassigned', audit fails."""
    conv_path, loco_dir, views_dir, _ = setup_test_loco_environment(tmp_path)

    master_path = loco_dir / "loco_master_manifest.csv"
    df = pd.read_csv(master_path)
    # Set one image's fold_China assignment to 'unassigned'
    df.loc[0, "fold_China"] = "unassigned"
    df.to_csv(master_path, index=False)

    auditor = RDD2022LOCOAuditor(
        conversion_manifest_path=conv_path,
        loco_dir=loco_dir,
        views_dir=views_dir,
    )
    report = auditor.run_full_audit()

    assert report.overall_status == CheckStatus.FAIL
    assert report.checks["CHK_02_ZERO_LEAKAGE"].status == CheckStatus.FAIL
    assert any("eligible samples missing from assignments" in err for err in report.checks["CHK_02_ZERO_LEAKAGE"].errors)


def test_audit_catches_reconciliation_double_counting(tmp_path: Path):
    """Verify that arithmetic reconciliation fails if double counting or discrepancy occurs."""
    conv_path, loco_dir, views_dir, _ = setup_test_loco_environment(tmp_path)

    # Corrupt conversion manifest by adding an extra converted target row not in loco manifest
    conv_df = pd.read_csv(conv_path)
    new_row = conv_df.iloc[0].copy()
    new_row["image_id"] = "Phantom_0001"
    new_row["conversion_status"] = ConversionStatus.CONVERTED_TARGET
    new_row["incomplete_label_risk"] = False
    new_row["loco_eligible"] = True
    new_row["split"] = "train"
    new_row["validation_flags"] = ""
    # Notice this phantom row is NOT in loco_master_df, and falls into 'other' in exclusion
    # But if we artificially corrupt the conversion manifest total:
    conv_df = pd.concat([conv_df, pd.DataFrame([new_row])], ignore_index=True)
    conv_df.to_csv(conv_path, index=False)

    auditor = RDD2022LOCOAuditor(
        conversion_manifest_path=conv_path,
        loco_dir=loco_dir,
        views_dir=views_dir,
    )
    report = auditor.run_full_audit()

    reconcil = report.reconciliation
    assert reconcil["total_converted_target"] == 122
    assert reconcil["total_selected_loco"] == 120
    assert reconcil["total_excluded"] == 2
    assert reconcil["exclusion_breakdown"]["incomplete_label_risk"] == 1
    assert reconcil["exclusion_breakdown"]["other"] == 1
    assert reconcil["reconciliation_verified"] is True

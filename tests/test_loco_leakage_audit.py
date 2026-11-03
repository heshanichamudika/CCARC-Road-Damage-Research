"""Unit and Integration Tests for Comprehensive LOCO Data Leakage Auditor."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.loco_leakage_audit import (
    AuditCheckItem,
    ComprehensiveLOCOAuditor,
    LOCOLeakageAuditSummary,
    main as cli_main,
)
from src.data.loco_splitter import RDD2022LOCOSplitter
from tests.test_loco_splitting import generate_synthetic_six_country_manifest


def setup_mock_loco_dataset(tmp_path: Path):
    """Set up complete mock dataset with images, labels, conversion manifest, and LOCO splits."""
    mock_raw = tmp_path / "mock_raw"
    conv_manifest_df = generate_synthetic_six_country_manifest(base_dir=mock_raw)
    conv_manifest_path = tmp_path / "yolo_conversion_manifest.csv"
    conv_manifest_df.to_csv(conv_manifest_path, index=False)

    loco_dir = tmp_path / "loco_splits"
    views_dir = tmp_path / "loco_views"

    splitter = RDD2022LOCOSplitter(val_ratio=0.20, seed=42)
    master_df, summary = splitter.run_all_folds(
        manifest_df=conv_manifest_df,
        output_dir=loco_dir,
        views_dir=views_dir,
        clean_output=True,
    )

    return conv_manifest_path, loco_dir, views_dir


def test_audit_distinguishes_verification_tiers_without_dataset():
    """Verify that when run locally without dataset files, audit reports Code-Level PASS and NOT_VERIFIED for manifests/hashes."""
    auditor = ComprehensiveLOCOAuditor(
        conversion_manifest=Path("nonexistent/yolo_conversion_manifest.csv"),
        loco_dir=Path("nonexistent/loco_splits"),
    )
    report = auditor.run_comprehensive_audit()

    assert "PARTIALLY_VERIFIED" in report.overall_status
    assert report.code_level_passed == 8
    assert report.manifest_level_passed == 0
    assert report.content_hash_passed == 0
    assert report.checks_not_verified == 2
    assert report.checks_failed == 0

    assert report.checks["CHK_02_PARTITION_ISOLATION"].status == "NOT_VERIFIED"
    assert report.checks["CHK_03_DUPLICATE_DETECTION"].status == "NOT_VERIFIED"

    # All 8 code-level checks must pass
    for cid in [
        "CHK_01_PIPELINE_ARCHITECTURE",
        "CHK_04_COUNTRY_SUBSETS",
        "CHK_05_CBS_SAMPLING_POOLS",
        "CHK_06_CAL_CLASS_WEIGHTS",
        "CHK_07_RWC_BATCH_SOURCE",
        "CHK_08_EARLY_STOPPING_ISOLATION",
        "CHK_09_EVALUATION_NON_MUTATING",
        "CHK_10_ARTIFACT_TRACEABILITY",
    ]:
        assert report.checks[cid].status == "PASS"


def test_audit_fully_passes_when_dataset_manifests_provided(tmp_path: Path):
    """Verify that when real/fixture dataset manifests and images are present, all 10 checks pass across all 3 tiers."""
    conv_path, loco_dir, views_dir = setup_mock_loco_dataset(tmp_path)

    auditor = ComprehensiveLOCOAuditor(
        conversion_manifest=conv_path,
        loco_dir=loco_dir,
        views_dir=views_dir,
    )
    report = auditor.run_comprehensive_audit()

    assert "PASS" in report.overall_status
    assert report.code_level_passed == 8
    assert report.manifest_level_passed == 1
    assert report.content_hash_passed == 1
    assert report.checks_not_verified == 0
    assert report.checks_failed == 0

    assert report.checks["CHK_02_PARTITION_ISOLATION"].status == "PASS"
    assert report.checks["CHK_03_DUPLICATE_DETECTION"].status == "PASS"


def test_audit_cbs_pool_isolation_catches_target_samples():
    """Verify that Check 5 detects and excludes held-out country samples from CBS sampling."""
    auditor = ComprehensiveLOCOAuditor()
    c5 = auditor.audit_check_5_cbs_sampling_pools()

    assert c5.check_num == 5
    assert c5.status == "PASS"
    assert c5.details["held_out_samples_in_pools"] is False
    assert c5.details["held_out_samples_excluded_count"] == 2
    assert "Norway" not in c5.details["active_source_countries_in_pools"]


def test_audit_cal_source_isolation_rejects_val_test_labels():
    """Verify that Check 6 ensures CAL class weights derive strictly from train labels."""
    auditor = ComprehensiveLOCOAuditor()
    c6 = auditor.audit_check_6_cal_class_weights()

    assert c6.check_num == 6
    assert c6.status == "PASS"
    assert c6.details["val_test_contamination_detected"] is False
    assert c6.details["extracted_train_counts"]["D00"] == 10
    assert c6.details["extracted_train_counts"]["D10"] == 5


def test_audit_china_subsets_mapping():
    """Verify that Check 4 ensures China_Drone and China_MotorBike map strictly to China."""
    auditor = ComprehensiveLOCOAuditor()
    c4 = auditor.audit_check_4_country_subsets()

    assert c4.check_num == 4
    assert c4.status == "PASS"
    assert c4.details["china_drone_mapped_country"] == "China"
    assert c4.details["china_motorbike_mapped_country"] == "China"


def test_audit_cli_and_report_exports(tmp_path: Path):
    """Verify that the CLI exports machine-readable JSON and human-readable Markdown summaries."""
    out_dir = tmp_path / "leakage_report"
    code = cli_main(["--output-dir", str(out_dir)])

    assert code == 0
    json_file = out_dir / "loco_leakage_audit_report.json"
    md_file = out_dir / "loco_leakage_audit_report.md"

    assert json_file.is_file()
    assert md_file.is_file()

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "checks" in data
    assert data["code_level_passed"] == 8

    md_text = md_file.read_text(encoding="utf-8")
    assert "# CCARC Research — Comprehensive LOCO Data Leakage Audit Report" in md_text
    assert "Tier A: Code-Level" in md_text
    assert "Tier B: Manifest-Level" in md_text
    assert "Tier C: Full Image Content-Hash" in md_text

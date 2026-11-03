"""Unit and Integration Tests for CCARC LOCO Evaluation Pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from src.data.loco_splitter import DEFAULT_TARGET_CLASSES, LOCO_COUNTRIES
from src.evaluation.evaluator import (
    ExperimentTargetResult,
    LOCOBenchmarkEvaluator,
)
from src.evaluation.metrics import (
    compute_f1,
    extract_detection_metrics,
    normalize_metrics_dict,
)
from src.evaluation.report_generator import EvaluationReportGenerator
from src.evaluation.evaluate_loco import main as cli_main, parse_args


# =========================================================================
# 1. Metrics Computation Tests
# =========================================================================

def test_compute_f1_basic():
    """Verify standard F1-score computation."""
    # F1 = 2 * (0.8 * 0.6) / (0.8 + 0.6) = 0.96 / 1.4 = 0.685714...
    f1 = compute_f1(0.8, 0.6)
    assert f1 == 0.6857

    # Precision = Recall = 1.0 -> F1 = 1.0
    assert compute_f1(1.0, 1.0) == 1.0

    # Precision = Recall = 0.0 -> F1 = 0.0 (no division by zero)
    assert compute_f1(0.0, 0.0) == 0.0
    assert compute_f1(0.0, 0.5) == 0.0
    assert compute_f1(0.5, 0.0) == 0.0


def test_extract_detection_metrics_modern_ultralytics():
    """Verify metrics extraction with modern Ultralytics DetMetrics structure."""
    mock_box = SimpleNamespace(
        mp=0.75,
        mr=0.60,
        map50=0.65,
        map=0.45,
        maps=np.array([0.40, 0.50, 0.42, 0.48]),
        ap50=np.array([0.60, 0.70, 0.62, 0.68]),
        p=np.array([0.70, 0.80, 0.72, 0.78]),
        r=np.array([0.55, 0.65, 0.58, 0.62]),
        f1=np.array([0.6154, 0.7172, 0.6425, 0.6909]),
    )
    mock_val_results = SimpleNamespace(
        box=mock_box,
        fitness=0.47,
        results_dict={},
    )

    metrics = extract_detection_metrics(mock_val_results, DEFAULT_TARGET_CLASSES)

    assert metrics["precision"] == 0.75
    assert metrics["recall"] == 0.60
    assert metrics["map50"] == 0.65
    assert metrics["map50_95"] == 0.45
    assert metrics["fitness"] == 0.47
    assert 0.60 <= metrics["f1"] <= 0.70

    for idx, cls_name in enumerate(DEFAULT_TARGET_CLASSES):
        assert cls_name in metrics["per_class"]
        c_dict = metrics["per_class"][cls_name]
        assert "precision" in c_dict
        assert "recall" in c_dict
        assert "f1" in c_dict
        assert "ap50" in c_dict
        assert "map50_95" in c_dict
        assert c_dict["ap50"] == round(float(mock_box.ap50[idx]), 4)
        assert c_dict["map50_95"] == round(float(mock_box.maps[idx]), 4)


def test_extract_detection_metrics_legacy_ultralytics():
    """Verify metrics extraction with legacy Ultralytics scalar metrics."""
    mock_box = SimpleNamespace(
        p=0.5234,
        r=0.4567,
        map50=0.4891,
        map=0.2912,
        maps=[0.25, 0.30, 0.28, 0.33],
        ap50=[0.45, 0.52, 0.47, 0.51],
    )
    mock_val_results = SimpleNamespace(box=mock_box)

    metrics = extract_detection_metrics(mock_val_results, DEFAULT_TARGET_CLASSES)

    assert metrics["precision"] == 0.5234
    assert metrics["recall"] == 0.4567
    assert metrics["map50"] == 0.4891
    assert metrics["map50_95"] == 0.2912
    # Harmonic mean: 2 * (0.5234 * 0.4567) / (0.5234 + 0.4567)
    expected_f1 = compute_f1(0.5234, 0.4567)
    assert metrics["f1"] == expected_f1


def test_normalize_metrics_dict_enriches_missing_f1():
    """Verify that older saved metrics without F1 are safely enriched."""
    raw = {
        "precision": 0.60,
        "recall": 0.50,
        "map50": 0.55,
        "map50_95": 0.35,
        "per_class": {
            "D00": {"ap50": 0.50, "map50_95": 0.30},
            "D10": {"ap50": 0.60, "map50_95": 0.40},
            "D20": {"ap50": 0.52, "map50_95": 0.32},
            "D40": {"ap50": 0.58, "map50_95": 0.38},
        },
    }

    norm = normalize_metrics_dict(raw, DEFAULT_TARGET_CLASSES)
    assert norm["f1"] == compute_f1(0.60, 0.50)
    for cls_name in DEFAULT_TARGET_CLASSES:
        assert "f1" in norm["per_class"][cls_name]


# =========================================================================
# 2. Checkpoint Discovery & Status Tracking Tests
# =========================================================================

def test_evaluator_checkpoint_discovery(tmp_path: Path):
    """Verify discovery of E0 baseline and CCARC checkpoints across folder layouts."""
    base_dir = tmp_path / "baseline"
    ccarc_dir = tmp_path / "ccarc"

    # Setup E0 checkpoint
    e0_fold0 = base_dir / "fold_0_China" / "weights"
    e0_fold0.mkdir(parents=True, exist_ok=True)
    (e0_fold0 / "best.pt").write_text("dummy_weight", encoding="utf-8")

    # Setup E4 checkpoint
    e4_fold4 = ccarc_dir / "fold_4_Norway_E4" / "weights"
    e4_fold4.mkdir(parents=True, exist_ok=True)
    (e4_fold4 / "best.pt").write_text("dummy_weight_e4", encoding="utf-8")

    evaluator = LOCOBenchmarkEvaluator(
        views_dir=tmp_path / "views",
        baseline_dir=base_dir,
        ccarc_dir=ccarc_dir,
    )

    # 1. E0 Fold 0 China -> Found
    d0, ckpt0 = evaluator.locate_experiment("E0", 0, "China")
    assert ckpt0 is not None and ckpt0.is_file()
    assert ckpt0 == e0_fold0 / "best.pt"

    # 2. E4 Fold 4 Norway -> Found
    d4, ckpt4 = evaluator.locate_experiment("E4", 4, "Norway")
    assert ckpt4 is not None and ckpt4.is_file()
    assert ckpt4 == e4_fold4 / "best.pt"

    # 3. E1 Fold 2 India -> Not started yet -> None
    d2, ckpt2 = evaluator.locate_experiment("E1", 2, "India")
    assert ckpt2 is None


def test_evaluator_in_progress_and_pending_status(tmp_path: Path):
    """Verify that in-progress runs (last.pt only) and pending runs are reported without synthetic metrics."""
    ccarc_dir = tmp_path / "ccarc"
    views_dir = tmp_path / "views"

    # Simulate E4 Fold 4 active training: last.pt exists, but best.pt does not exist yet
    in_progress_dir = ccarc_dir / "fold_4_Norway_E4" / "weights"
    in_progress_dir.mkdir(parents=True, exist_ok=True)
    (in_progress_dir / "last.pt").write_text("in_progress", encoding="utf-8")

    evaluator = LOCOBenchmarkEvaluator(
        views_dir=views_dir,
        baseline_dir=tmp_path / "baseline",
        ccarc_dir=ccarc_dir,
    )

    res_in_prog = evaluator.evaluate_condition("E4", 4, "Norway")
    assert res_in_prog.status == "IN_PROGRESS"
    assert res_in_prog.metrics is None
    assert "last.pt present" in str(res_in_prog.status_detail)

    res_pending = evaluator.evaluate_condition("E2", 0, "China")
    assert res_pending.status == "PENDING"
    assert res_pending.metrics is None


def test_evaluator_rejects_intermediate_checkpoint_when_incomplete(tmp_path: Path):
    """Verify that when best.pt exists at epoch 25/100, evaluator flags IN_PROGRESS and halts inference."""
    ccarc_dir = tmp_path / "ccarc"
    views_dir = tmp_path / "views"
    exp_dir = ccarc_dir / "fold_4_Norway_E4"
    weights_dir = exp_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Both best.pt and last.pt exist during active training
    (weights_dir / "best.pt").write_text("intermediate_best", encoding="utf-8")
    (weights_dir / "last.pt").write_text("intermediate_last", encoding="utf-8")

    # args.yaml specifies 100 epochs
    with open(exp_dir / "args.yaml", "w", encoding="utf-8") as f:
        yaml.dump({"epochs": 100, "model": "yolov8s.pt"}, f)

    # results.csv only has 25 epochs (header + 25 lines)
    header = "epoch,train/box_loss,val/box_loss\n"
    rows = "".join(f"{i},0.5,0.4\n" for i in range(25))
    (exp_dir / "results.csv").write_text(header + rows, encoding="utf-8")

    evaluator = LOCOBenchmarkEvaluator(
        views_dir=views_dir,
        baseline_dir=tmp_path / "baseline",
        ccarc_dir=ccarc_dir,
    )

    with patch("src.evaluation.evaluator.YOLO") as mock_yolo:
        res = evaluator.evaluate_condition("E4", 4, "Norway")
        mock_yolo.assert_not_called()  # Must NEVER evaluate intermediate checkpoints!

        assert res.status == "IN_PROGRESS"
        assert res.metrics is None
        assert "25/100 epochs completed" in str(res.status_detail)
        assert "best.pt is intermediate" in str(res.status_detail)


def test_evaluator_rejects_smoke_test_checkpoint(tmp_path: Path):
    """Verify that checkpoints from mini-debugging smoke tests are rejected from final benchmark."""
    ccarc_dir = tmp_path / "ccarc"
    views_dir = tmp_path / "views"
    exp_dir = ccarc_dir / "fold_4_Norway_E4"
    weights_dir = exp_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "best.pt").write_text("smoke_best", encoding="utf-8")

    # Write smoke test master summary
    with open(exp_dir / "smoke_master_summary.json", "w", encoding="utf-8") as f:
        json.dump({"smoke_mode": True, "stage": "SMOKE_DEBUG_E4", "smoke_batches_executed": 3}, f)

    evaluator = LOCOBenchmarkEvaluator(
        views_dir=views_dir,
        baseline_dir=tmp_path / "baseline",
        ccarc_dir=ccarc_dir,
    )

    with patch("src.evaluation.evaluator.YOLO") as mock_yolo:
        res = evaluator.evaluate_condition("E4", 4, "Norway")
        mock_yolo.assert_not_called()

        assert res.status == "SMOKE_TEST_CHECKPOINT"
        assert res.metrics is None
        assert "smoke test" in str(res.status_detail).lower()


def test_evaluator_cached_metrics_ingestion(tmp_path: Path):
    """Verify that existing held_out_test_metrics.json is ingested without running model inference."""
    base_dir = tmp_path / "baseline"
    exp_dir = base_dir / "fold_0_China"
    exp_dir.mkdir(parents=True, exist_ok=True)

    cached_metrics = {
        "precision": 0.5812,
        "recall": 0.5143,
        "map50": 0.5489,
        "map50_95": 0.3211,
        "fitness": 0.3438,
        "per_class": {
            "D00": {"ap50": 0.50, "map50_95": 0.30},
            "D10": {"ap50": 0.60, "map50_95": 0.40},
            "D20": {"ap50": 0.52, "map50_95": 0.32},
            "D40": {"ap50": 0.58, "map50_95": 0.38},
        },
    }
    with open(exp_dir / "held_out_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "stage": "STAGE_3_HELD_OUT_TARGET_EVALUATION",
            "variant": "E0",
            "fold_idx": 0,
            "target_country": "China",
            "split": "test",
            "metrics": cached_metrics,
            "evaluated_at": "2026-11-03T12:00:00Z",
        }, f)

    evaluator = LOCOBenchmarkEvaluator(
        views_dir=tmp_path / "views",
        baseline_dir=base_dir,
        ccarc_dir=tmp_path / "ccarc",
    )

    with patch("src.evaluation.evaluator.YOLO") as mock_yolo:
        res = evaluator.evaluate_condition("E0", 0, "China", force_reevaluate=False)
        mock_yolo.assert_not_called()  # Must not launch YOLO inference

        assert res.status == "COMPLETED"
        assert res.source == "cached_test_metrics"
        assert res.metrics is not None
        assert res.metrics["map50"] == 0.5489
        assert res.metrics["f1"] == compute_f1(0.5812, 0.5143)


def test_evaluator_executes_model_val_and_saves_metrics(tmp_path: Path):
    """Verify that when best.pt exists, evaluator executes model.val(split='test') and saves metrics."""
    base_dir = tmp_path / "baseline"
    exp_dir = base_dir / "fold_1_Czech" / "weights"
    exp_dir.mkdir(parents=True, exist_ok=True)
    best_pt = exp_dir / "best.pt"
    best_pt.write_text("weights", encoding="utf-8")

    # Mock LOCO view data.yaml
    views_dir = tmp_path / "views"
    fold_view_dir = views_dir / "fold_1_Czech"
    fold_view_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = fold_view_dir / "data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.dump({
            "path": str(fold_view_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: c for i, c in enumerate(DEFAULT_TARGET_CLASSES)},
        }, f)

    evaluator = LOCOBenchmarkEvaluator(
        views_dir=views_dir,
        baseline_dir=base_dir,
        ccarc_dir=tmp_path / "ccarc",
    )

    mock_val_results = SimpleNamespace(
        box=SimpleNamespace(
            mp=0.62,
            mr=0.55,
            map50=0.59,
            map=0.38,
            maps=[0.35, 0.40, 0.36, 0.41],
            ap50=[0.55, 0.65, 0.57, 0.60],
            p=[0.60, 0.65, 0.61, 0.63],
            r=[0.52, 0.58, 0.54, 0.56],
            f1=[0.556, 0.612, 0.572, 0.592],
        ),
        fitness=0.401,
        results_dict={},
    )

    with patch("src.evaluation.evaluator.YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        mock_model.val.return_value = mock_val_results
        mock_yolo_cls.return_value = mock_model

        res = evaluator.evaluate_condition("E0", 1, "Czech")

        # Verify YOLO was invoked with split='test'
        mock_yolo_cls.assert_called_once_with(str(best_pt))
        mock_model.val.assert_called_once()
        call_kwargs = mock_model.val.call_args[1]
        assert call_kwargs["split"] == "test"
        assert call_kwargs["data"] == str(data_yaml)

        assert res.status == "COMPLETED"
        assert res.source == "evaluated_checkpoint"
        assert res.metrics["map50"] == 0.59

        # Verify held_out_test_metrics.json was saved in experiment directory
        saved_file = base_dir / "fold_1_Czech" / "held_out_test_metrics.json"
        assert saved_file.is_file()
        with open(saved_file, "r", encoding="utf-8") as f:
            saved_json = json.load(f)
        assert saved_json["split"] == "test"
        assert saved_json["metrics"]["map50"] == 0.59


# =========================================================================
# 3. Report Generation Tests (JSON, CSV, Markdown, LaTeX)
# =========================================================================

def test_report_generator_exports_all_formats(tmp_path: Path):
    """Verify that EvaluationReportGenerator creates valid JSON, CSV, MD, and TeX files."""
    dummy_benchmark_data = {
        "benchmark": "CCARC_LOCO_TEST",
        "timestamp": "2026-11-03T23:00:00Z",
        "experiments": {
            "E0_fold_0_China": {
                "variant": "E0",
                "fold_idx": 0,
                "target_country": "China",
                "status": "COMPLETED",
                "checkpoint_path": "/path/best.pt",
                "metrics": {
                    "precision": 0.65,
                    "recall": 0.55,
                    "f1": 0.5957,
                    "map50": 0.60,
                    "map50_95": 0.40,
                    "fitness": 0.42,
                    "per_class": {
                        "D00": {"ap50": 0.58, "map50_95": 0.38, "precision": 0.62, "recall": 0.52, "f1": 0.5657},
                        "D10": {"ap50": 0.68, "map50_95": 0.48, "precision": 0.72, "recall": 0.62, "f1": 0.6662},
                        "D20": {"ap50": 0.55, "map50_95": 0.35, "precision": 0.60, "recall": 0.50, "f1": 0.5455},
                        "D40": {"ap50": 0.59, "map50_95": 0.39, "precision": 0.66, "recall": 0.56, "f1": 0.6059},
                    },
                },
            },
            "E4_fold_4_Norway": {
                "variant": "E4",
                "fold_idx": 4,
                "target_country": "Norway",
                "status": "IN_PROGRESS",
                "metrics": None,
            },
        },
    }

    generator = EvaluationReportGenerator(dummy_benchmark_data)
    out_dir = tmp_path / "reports"
    exported = generator.export_all(out_dir)

    # 1. JSON
    assert exported["json"].is_file()
    with open(exported["json"], "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert "experiments" in loaded_json

    # 2. Summary CSV
    assert exported["summary_csv"].is_file()
    csv_text = exported["summary_csv"].read_text(encoding="utf-8")
    assert "variant,fold,target_country,status" in csv_text
    assert "E0,0,China,COMPLETED" in csv_text
    assert "E4,4,Norway,IN_PROGRESS" in csv_text

    # 3. Per-class CSV
    assert exported["per_class_csv"].is_file()
    per_class_text = exported["per_class_csv"].read_text(encoding="utf-8")
    assert "D00" in per_class_text
    assert "D10" in per_class_text

    # 4. Markdown Table
    assert exported["markdown"].is_file()
    md_text = exported["markdown"].read_text(encoding="utf-8")
    assert "## 1. Held-Out Target Generalization (mAP@0.5)" in md_text
    assert "0.6000" in md_text  # mAP@0.5 for China E0
    assert "*IN_PROGRESS*" in md_text

    # 5. LaTeX Table
    assert exported["latex"].is_file()
    tex_text = exported["latex"].read_text(encoding="utf-8")
    assert "\\begin{table*}" in tex_text
    assert "\\toprule" in tex_text
    assert "\\bottomrule" in tex_text
    assert "\\end{table*}" in tex_text


# =========================================================================
# 4. CLI Execution Tests
# =========================================================================

def test_cli_dry_run_and_ingest_only(tmp_path: Path):
    """Verify that CLI evaluate_loco handles dry-run and ingest-only without launching training or GPU."""
    views_dir = tmp_path / "views"
    base_dir = tmp_path / "baseline"
    ccarc_dir = tmp_path / "ccarc"
    out_dir = tmp_path / "results"

    # Create dummy experiment
    (base_dir / "fold_0_China" / "weights").mkdir(parents=True, exist_ok=True)
    (base_dir / "fold_0_China" / "weights" / "best.pt").write_text("weights", encoding="utf-8")

    # Dry-run
    code_dry = cli_main([
        "--views-dir", str(views_dir),
        "--baseline-dir", str(base_dir),
        "--ccarc-dir", str(ccarc_dir),
        "--output-dir", str(out_dir),
        "--dry-run",
    ])
    assert code_dry == 0

    # Ingest-only
    code_ingest = cli_main([
        "--views-dir", str(views_dir),
        "--baseline-dir", str(base_dir),
        "--ccarc-dir", str(ccarc_dir),
        "--output-dir", str(out_dir),
        "--ingest-only",
    ])
    assert code_ingest == 0
    assert (out_dir / "loco_benchmark_results.json").is_file()
    assert (out_dir / "loco_benchmark_summary.md").is_file()

"""Unit and Integration Tests for YOLOv8s LOCO Baseline Training Pipeline."""

import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import yaml

from src.data.loco_splitter import DEFAULT_TARGET_CLASSES, LOCO_COUNTRIES
from src.training.baseline_config import (
    BaselineTrainingConfig,
    check_gpu_environment,
    resolve_fold_data_yaml,
)
from src.training.train_loco_baseline import (
    main as cli_main,
    parse_args,
)
from src.training.yolov8_trainer import YOLOv8LOCOTrainer


def create_mock_fold_view(base_dir: Path, fold_idx: int, target_country: str) -> Path:
    """Helper to create a minimal valid LOCO fold view for dry-run testing."""
    fold_dir = base_dir / f"fold_{fold_idx}_{target_country}"
    for split in ["train", "val", "test"]:
        (fold_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (fold_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        # Create at least one dummy image in each split
        (fold_dir / "images" / split / f"{target_country}_0001.jpg").write_text("img", encoding="utf-8")
        (fold_dir / "labels" / split / f"{target_country}_0001.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    yaml_data = {
        "path": str(fold_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(DEFAULT_TARGET_CLASSES),
        "names": {i: cls for i, cls in enumerate(DEFAULT_TARGET_CLASSES)},
        "loco_target_country": target_country,
    }
    yaml_path = fold_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f)

    return yaml_path


def test_baseline_config_validation_and_serialization(tmp_path: Path):
    """Verify default hyperparameters, train kwargs conversion and YAML roundtrip."""
    config = BaselineTrainingConfig(
        data_yaml_path=str(tmp_path / "data.yaml"),
        project_dir=str(tmp_path / "experiments"),
        experiment_name="fold_0_China",
    )

    assert config.model_name == "yolov8s.pt"
    assert config.img_size == 640
    assert config.batch_size == 16
    assert config.epochs == 100
    assert config.patience == 20
    assert config.seed == 42
    assert config.num_classes == 4
    assert config.target_classes == ("D00", "D10", "D20", "D40")

    # Verify Ultralytics kwargs
    train_args = config.to_yolo_train_args()
    assert train_args["epochs"] == 100
    assert train_args["batch"] == 16
    assert train_args["imgsz"] == 640
    assert train_args["val"] is True
    assert train_args["patience"] == 20
    assert "data" in train_args

    # YAML roundtrip
    yaml_path = tmp_path / "config_snapshot.yaml"
    config.save_yaml(yaml_path)
    assert yaml_path.is_file()

    loaded_config = BaselineTrainingConfig.from_yaml(yaml_path)
    assert loaded_config.model_name == config.model_name
    assert loaded_config.epochs == config.epochs
    assert loaded_config.batch_size == config.batch_size
    assert loaded_config.seed == config.seed
    assert loaded_config.target_classes == config.target_classes


def test_hardware_environment_check():
    """Verify hardware environment discovery (CUDA, GPU count, PyTorch version)."""
    hw = check_gpu_environment()

    assert "cuda_available" in hw
    assert "pytorch_version" in hw
    assert "ultralytics_version" in hw
    assert "python_version" in hw
    assert "os_platform" in hw

    config = BaselineTrainingConfig(device="auto")
    dev = config.resolve_device()
    assert dev in ("0", "cpu")


def test_fold_selection_and_yaml_resolution(tmp_path: Path):
    """Verify resolving fold data.yaml across different directory patterns."""
    yaml_china = create_mock_fold_view(tmp_path, fold_idx=0, target_country="China")
    resolved = resolve_fold_data_yaml(tmp_path, fold_idx=0, target_country="China")
    assert resolved is not None
    assert resolved == yaml_china.resolve()

    # Non-existent fold
    assert resolve_fold_data_yaml(tmp_path, fold_idx=1, target_country="Czech") is None


def test_test_data_isolation():
    """Verify that training arguments isolate source train/val and never include test data."""
    config = BaselineTrainingConfig(
        data_yaml_path="/dummy/path/data.yaml",
        project_dir="/dummy/experiments",
        experiment_name="fold_0_China",
    )
    args = config.to_yolo_train_args()

    # The training kwargs only receive data=data_yaml_path and val=True
    # Ultralytics model.train() uses 'train' split for training and 'val' split for evaluation
    assert args["val"] is True
    assert "split" not in args  # 'split' is not an argument to train()
    # Test split is strictly evaluated in Stage 3 via model.val(split='test')


def test_dry_run_execution(tmp_path: Path):
    """Verify that dry-run mode validates fold directories, YAML, and permissions with 0 errors."""
    yaml_path = create_mock_fold_view(tmp_path / "views", fold_idx=0, target_country="China")
    out_dir = tmp_path / "experiments"

    config = BaselineTrainingConfig(
        fold_idx=0,
        target_country="China",
        data_yaml_path=str(yaml_path),
        project_dir=str(out_dir),
        experiment_name="fold_0_China",
    )
    trainer = YOLOv8LOCOTrainer(config)
    report = trainer.dry_run()

    assert report["status"] == "PASS"
    assert len(report["errors"]) == 0
    assert report["checks"]["data_yaml"] is True
    assert report["checks"]["train_dir_exists"] is True
    assert report["checks"]["val_dir_exists"] is True
    assert report["checks"]["test_dir_exists"] is True
    assert report["checks"]["output_dir_writable"] is True

    # Confirm report was saved to disk
    assert (trainer.experiment_dir / "dry_run_report.json").is_file()


def test_dry_run_fails_on_missing_yaml(tmp_path: Path):
    """Verify that dry-run mode catches missing or invalid data.yaml."""
    out_dir = tmp_path / "experiments"
    config = BaselineTrainingConfig(
        fold_idx=1,
        target_country="Czech",
        data_yaml_path=str(tmp_path / "non_existent.yaml"),
        project_dir=str(out_dir),
        experiment_name="fold_1_Czech",
    )
    trainer = YOLOv8LOCOTrainer(config)
    report = trainer.dry_run()

    assert report["status"] == "FAIL"
    assert any("data.yaml not found" in err for err in report["errors"])


def test_checkpoint_paths_and_resume_detection(tmp_path: Path):
    """Verify checkpoint paths and resume detection logic."""
    out_dir = tmp_path / "experiments"
    yaml_path = create_mock_fold_view(tmp_path / "views", fold_idx=2, target_country="India")

    config = BaselineTrainingConfig(
        fold_idx=2,
        target_country="India",
        data_yaml_path=str(yaml_path),
        project_dir=str(out_dir),
        experiment_name="fold_2_India",
        resume=True,
    )
    trainer = YOLOv8LOCOTrainer(config)

    assert trainer.best_checkpoint == trainer.weights_dir / "best.pt"
    assert trainer.last_checkpoint == trainer.weights_dir / "last.pt"

    # Simulate in-progress experiment with last.pt
    trainer.weights_dir.mkdir(parents=True, exist_ok=True)
    trainer.last_checkpoint.write_text("dummy_last", encoding="utf-8")
    report = trainer.dry_run()
    assert report["checks"]["last_checkpoint_exists"] is True
    assert report["checks"]["is_already_completed"] is False

    # Simulate completed experiment (best.pt + results.csv)
    trainer.best_checkpoint.write_text("dummy_best", encoding="utf-8")
    (trainer.experiment_dir / "results.csv").write_text("epoch,train/loss\n1,0.5\n", encoding="utf-8")
    report_completed = trainer.dry_run()
    assert report_completed["checks"]["is_already_completed"] is True


def test_metric_extraction_schema():
    """Verify detection metric extraction from mock validation results."""
    config = BaselineTrainingConfig()
    trainer = YOLOv8LOCOTrainer(config)

    # Mock Ultralytics Results object
    mock_box = SimpleNamespace(
        p=0.6521,
        r=0.5843,
        map50=0.6120,
        map=0.3845,
        ap50=[0.65, 0.60, 0.55, 0.65],
        maps=[0.42, 0.38, 0.32, 0.42],
    )
    mock_results = SimpleNamespace(box=mock_box, fitness=0.4072)

    metrics = trainer._extract_metrics(mock_results)
    assert metrics["precision"] == 0.6521
    assert metrics["recall"] == 0.5843
    assert metrics["map50"] == 0.6120
    assert metrics["map50_95"] == 0.3845
    assert metrics["fitness"] == 0.4072

    per_class = metrics["per_class"]
    for cls_name in ["D00", "D10", "D20", "D40"]:
        assert cls_name in per_class
        assert "ap50" in per_class[cls_name]
        assert "map50_95" in per_class[cls_name]


def test_cli_dry_run_execution(tmp_path: Path):
    """Verify CLI launcher executes dry-run mode and returns exit code 0."""
    views_dir = tmp_path / "loco_views"
    out_dir = tmp_path / "experiments"
    create_mock_fold_view(views_dir, fold_idx=0, target_country="China")

    exit_code = cli_main([
        "--fold", "0",
        "--views-dir", str(views_dir),
        "--output-dir", str(out_dir),
        "--dry-run",
    ])

    assert exit_code == 0
    report_file = out_dir / "fold_0_China" / "dry_run_report.json"
    assert report_file.is_file()
    with open(report_file, "r", encoding="utf-8") as f:
        rep = json.load(f)
    assert rep["status"] == "PASS"


def test_sgd_optimizer_configuration():
    """Verify explicit SGD optimizer and default learning hyperparameters."""
    config = BaselineTrainingConfig()
    assert config.optimizer == "SGD"
    assert config.lr0 == 0.01
    assert config.lrf == 0.01
    assert config.momentum == 0.937
    assert config.weight_decay == 0.0005

    train_args = config.to_yolo_train_args()
    assert train_args["optimizer"] == "SGD"
    assert train_args["lr0"] == 0.01
    assert train_args["lrf"] == 0.01
    assert train_args["momentum"] == 0.937
    assert train_args["weight_decay"] == 0.0005


def test_explicit_held_out_evaluation_separation(tmp_path: Path, monkeypatch):
    """Verify that held-out target evaluation is strictly decoupled and skipped unless requested."""
    yaml_path = create_mock_fold_view(tmp_path / "views", fold_idx=0, target_country="China")
    out_dir = tmp_path / "experiments"

    config = BaselineTrainingConfig(
        fold_idx=0,
        target_country="China",
        data_yaml_path=str(yaml_path),
        project_dir=str(out_dir),
        experiment_name="fold_0_China",
    )
    trainer = YOLOv8LOCOTrainer(config)

    # Mock stage 1 training and stage 2 val to avoid downloading weights or running GPU
    monkeypatch.setattr(trainer, "train_stage", lambda force_restart=False: {"status": "MOCK_TRAIN_SUCCESS"})
    monkeypatch.setattr(
        trainer,
        "evaluate_source_val_stage",
        lambda checkpoint_path=None: {"stage": "STAGE_2_SOURCE_VALIDATION", "metrics": {"map50": 0.65}},
    )
    target_eval_called = False

    def mock_eval_target(checkpoint_path=None):
        nonlocal target_eval_called
        target_eval_called = True
        return {"stage": "STAGE_3_HELD_OUT_TARGET_EVALUATION", "metrics": {"map50": 0.45}}

    monkeypatch.setattr(trainer, "evaluate_held_out_target_stage", mock_eval_target)

    # 1. Default run: eval_target is False -> Stage 3 must be SKIPPED
    res_default = trainer.run_fold(dry_run=False, eval_target=False)
    assert not target_eval_called
    assert res_default["stage_3_held_out_target"]["status"] == "SKIPPED_NOT_REQUESTED"
    assert "Target test data remained unexposed" in res_default["stage_3_held_out_target"]["message"]

    # 2. Explicit run: eval_target is True -> Stage 3 must run
    res_explicit = trainer.run_fold(dry_run=False, eval_target=True)
    assert target_eval_called
    assert res_explicit["stage_3_held_out_target"]["stage"] == "STAGE_3_HELD_OUT_TARGET_EVALUATION"
    assert res_explicit["stage_3_held_out_target"]["metrics"]["map50"] == 0.45


def test_persistent_checkpoint_synchronization(tmp_path: Path):
    """Verify real-time synchronization between fast local storage and persistent Google Drive directory."""
    local_dir = tmp_path / "local_content" / "fold_0_China"
    persistent_dir = tmp_path / "drive_experiments" / "fold_0_China"

    config = BaselineTrainingConfig(
        project_dir=str(tmp_path / "drive_experiments"),
        local_run_dir=str(tmp_path / "local_content"),
        experiment_name="fold_0_China",
    )
    trainer = YOLOv8LOCOTrainer(config)

    # Simulate artifacts in local storage
    trainer.local_weights_dir.mkdir(parents=True, exist_ok=True)
    (trainer.local_weights_dir / "last.pt").write_text("dummy_last_local", encoding="utf-8")
    (trainer.local_weights_dir / "best.pt").write_text("dummy_best_local", encoding="utf-8")
    (trainer.local_experiment_dir / "results.csv").write_text("epoch,train_loss\n1,0.5\n", encoding="utf-8")
    (trainer.local_experiment_dir / "args.yaml").write_text("epochs: 100\n", encoding="utf-8")

    # Trigger sync
    trainer.sync_to_persistent()

    # Verify mirrored to persistent directory
    assert (trainer.persistent_weights_dir / "last.pt").read_text(encoding="utf-8") == "dummy_last_local"
    assert (trainer.persistent_weights_dir / "best.pt").read_text(encoding="utf-8") == "dummy_best_local"
    assert (trainer.persistent_experiment_dir / "results.csv").read_text(encoding="utf-8") == "epoch,train_loss\n1,0.5\n"
    assert (trainer.persistent_experiment_dir / "args.yaml").read_text(encoding="utf-8") == "epochs: 100\n"


def test_safe_resume_from_persistent_storage(tmp_path: Path):
    """Verify resuming training restores checkpoints from persistent storage into a fresh local directory."""
    config = BaselineTrainingConfig(
        project_dir=str(tmp_path / "drive_experiments"),
        local_run_dir=str(tmp_path / "local_content"),
        experiment_name="fold_0_China",
        resume=True,
    )
    trainer = YOLOv8LOCOTrainer(config)

    # Populate persistent directory as if previous runtime crashed or disconnected
    trainer.persistent_weights_dir.mkdir(parents=True, exist_ok=True)
    (trainer.persistent_weights_dir / "last.pt").write_text("drive_last_checkpoint", encoding="utf-8")
    (trainer.persistent_experiment_dir / "results.csv").write_text("epoch,train_loss\n1,0.4\n", encoding="utf-8")

    # Trigger restore
    trainer.sync_from_persistent()

    # Check that local directory has been restored
    assert (trainer.local_weights_dir / "last.pt").is_file()
    assert (trainer.local_weights_dir / "last.pt").read_text(encoding="utf-8") == "drive_last_checkpoint"
    assert (trainer.local_experiment_dir / "results.csv").read_text(encoding="utf-8") == "epoch,train_loss\n1,0.4\n"


def test_reproducibility_metadata_recording(tmp_path: Path):
    """Verify saving reproducibility metadata records frozen augmentation, optimizer, and hardware settings."""
    config = BaselineTrainingConfig(
        project_dir=str(tmp_path / "experiments"),
        experiment_name="fold_3_Japan",
        fold_idx=3,
        target_country="Japan",
    )
    trainer = YOLOv8LOCOTrainer(config)
    meta_path = trainer.save_reproducibility_metadata()

    assert meta_path.is_file()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["protocol"] == "YOLOv8s_LOCO_Source_Only_Baseline"
    assert meta["fold_idx"] == 3
    assert meta["target_country"] == "Japan"
    assert meta["optimization"]["optimizer"] == "SGD"
    assert meta["optimization"]["momentum"] == 0.937
    assert meta["optimization"]["weight_decay"] == 0.0005
    assert meta["augmentations"]["policy"] == "FROZEN_STANDARD_YOLOV8_BASELINE"
    assert meta["model"]["classes"] == ["D00", "D10", "D20", "D40"]
    assert "hardware" in meta

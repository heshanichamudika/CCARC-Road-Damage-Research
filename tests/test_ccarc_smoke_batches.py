"""Unit and Integration Tests for CCARC GPU Mini-Debugging Mode (--smoke-batches).

Tests:
1. CLI parsing of --smoke-batches (valid positive int, rejection of 0, rejection of negative).
2. CCARCTrainingConfig validation of smoke_batches.
3. CCARCDetectionTrainer initialization with smoke_batches (argument stripping, accumulate=1).
4. Accumulate enforcement in preprocess_batch.
5. Optimizer step detection and flag recording.
6. Diagnostics capture in _on_smoke_batch_end (losses, CAL, RWC, GPU memory, gradient check, CBS dist).
7. Clean early stopping after exactly N batches via CCARCSmokeBatchStop.
8. Epoch early stopping to prevent epoch validation.
9. train_stage handling of CCARCSmokeBatchStop (no best.pt overwrite, smoke report written).
10. run_fold early exit in smoke mode without running source_val or held_out stages.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.training.ccarc_config import CCARCTrainingConfig
from src.training.ccarc_trainer import (
    CCARCDetectionTrainer,
    CCARCLOCOTrainer,
    CCARCSmokeBatchStop,
    CCARCCompoundLoss,
)
from src.training.train_loco_ccarc import parse_args


class TestSmokeBatchesCLI:
    """Tests for CLI parsing and validation of --smoke-batches."""

    def test_cli_smoke_batches_default_none(self):
        args = parse_args(["--fold", "0", "--variant", "E4"])
        assert args.smoke_batches is None

    def test_cli_smoke_batches_positive_int(self):
        args = parse_args(["--fold", "0", "--variant", "E4", "--smoke-batches", "5"])
        assert args.smoke_batches == 5

    def test_cli_smoke_batches_rejection_zero(self):
        with pytest.raises(SystemExit):
            parse_args(["--fold", "0", "--variant", "E4", "--smoke-batches", "0"])

    def test_cli_smoke_batches_rejection_negative(self):
        with pytest.raises(SystemExit):
            parse_args(["--fold", "0", "--variant", "E4", "--smoke-batches", "-3"])


class TestSmokeBatchesConfig:
    """Tests for CCARCTrainingConfig smoke_batches validation."""

    def test_config_smoke_batches_valid(self):
        cfg = CCARCTrainingConfig(experiment_variant="E4", smoke_batches=3)
        assert cfg.smoke_batches == 3

    def test_config_smoke_batches_zero_raises(self):
        with pytest.raises(ValueError, match="smoke_batches must be a positive integer"):
            CCARCTrainingConfig(experiment_variant="E4", smoke_batches=0)

    def test_config_smoke_batches_negative_raises(self):
        with pytest.raises(ValueError, match="smoke_batches must be a positive integer"):
            CCARCTrainingConfig(experiment_variant="E4", smoke_batches=-2)

    def test_config_smoke_batches_not_leaked_to_yolo_args(self):
        cfg = CCARCTrainingConfig(experiment_variant="E4", smoke_batches=5)
        yolo_args = cfg.to_yolo_train_args()
        assert "smoke_batches" not in yolo_args


class TestSmokeBatchesTrainer:
    """Tests for CCARCDetectionTrainer smoke-batches mechanisms."""

    def test_overrides_strip_smoke_batches(self):
        """Verify smoke_batches is popped and never reaches Ultralytics get_cfg."""
        overrides = {
            "model": "yolov8n.yaml",
            "data": "coco8.yaml",
            "epochs": 1,
            "smoke_batches": 3,
            "ccarc_config": CCARCTrainingConfig(experiment_variant="E0", smoke_batches=3),
        }
        with patch("ultralytics.engine.trainer.check_det_dataset"):
            with patch("ultralytics.models.yolo.detect.train.DetectionTrainer.get_dataset", return_value=(None, None)):
                trainer = CCARCDetectionTrainer(overrides=overrides)
                assert trainer.smoke_batches == 3
                assert "smoke_batches" not in trainer.args

    def test_accumulate_forced_to_one(self):
        """Verify accumulate=1 is enforced in __init__ and preprocess_batch."""
        cfg = CCARCTrainingConfig(experiment_variant="E0", smoke_batches=2)
        overrides = {
            "model": "yolov8n.yaml",
            "epochs": 1,
            "batch": 4,
            "smoke_batches": 2,
            "ccarc_config": cfg,
        }
        with patch("ultralytics.models.yolo.detect.train.DetectionTrainer.get_dataset", return_value=(None, None)):
            trainer = CCARCDetectionTrainer(overrides=overrides)
            assert trainer.accumulate == 1
            assert trainer.args.nbs == 4

            # Simulate warmup setting accumulate to > 1
            trainer.accumulate = 4
            dummy_batch = {"im_file": ["/path/Norway_001.jpg", "/path/Japan_002.jpg"]}
            with patch("ultralytics.models.yolo.detect.train.DetectionTrainer.preprocess_batch", return_value=dummy_batch):
                out = trainer.preprocess_batch(dummy_batch)
                assert trainer.accumulate == 1
                assert trainer._last_opt_step_ran is False
                assert trainer._current_batch == dummy_batch

    def test_optimizer_step_sets_flag(self):
        """Verify optimizer_step sets _last_opt_step_ran = True."""
        cfg = CCARCTrainingConfig(experiment_variant="E0", smoke_batches=2)
        overrides = {"model": "yolov8n.yaml", "epochs": 1, "smoke_batches": 2, "ccarc_config": cfg}
        with patch("ultralytics.models.yolo.detect.train.DetectionTrainer.get_dataset", return_value=(None, None)):
            trainer = CCARCDetectionTrainer(overrides=overrides)
            trainer._last_opt_step_ran = False
            with patch("ultralytics.engine.trainer.BaseTrainer.optimizer_step"):
                trainer.optimizer_step()
                assert trainer._last_opt_step_ran is True

    def test_on_smoke_batch_end_diagnostics_and_stopping(self):
        """Verify diagnostics capture and CCARCSmokeBatchStop early halt on batch N."""
        cfg = CCARCTrainingConfig(experiment_variant="E4", smoke_batches=2)
        overrides = {"model": "yolov8n.yaml", "epochs": 1, "smoke_batches": 2, "ccarc_config": cfg}
        with patch("ultralytics.models.yolo.detect.train.DetectionTrainer.get_dataset", return_value=(None, None)):
            trainer = CCARCDetectionTrainer(overrides=overrides)
            trainer.loss = torch.tensor(2.4567)
            trainer.loss_items = torch.tensor([0.8123, 1.2345, 0.4099])
            trainer._last_opt_step_ran = True
            trainer._current_batch = {
                "im_file": ["/data/Norway_0001.jpg", "/data/Norway_0002.jpg", "/data/Japan_0003.jpg"]
            }

            # Batch 1: should record diagnostics and NOT raise
            trainer._on_smoke_batch_end(trainer)
            assert trainer.smoke_batch_counter == 1
            assert len(trainer.smoke_diagnostics) == 1
            d1 = trainer.smoke_diagnostics[0]
            assert d1["batch"] == 1
            assert d1["total_loss"] == 2.4567
            assert d1["optimizer_step_ran"] is True
            assert d1["cbs_country_distribution"] == {"Norway": 2, "Japan": 1}

            # Batch 2: should record diagnostics and RAISE CCARCSmokeBatchStop
            with pytest.raises(CCARCSmokeBatchStop) as exc_info:
                trainer._on_smoke_batch_end(trainer)

            assert len(exc_info.value.diagnostics) == 2
            assert exc_info.value.diagnostics[1]["batch"] == 2

    def test_on_smoke_epoch_end_prevents_validation(self):
        """Verify _on_smoke_epoch_end raises CCARCSmokeBatchStop if epoch ends early."""
        cfg = CCARCTrainingConfig(experiment_variant="E0", smoke_batches=5)
        overrides = {"model": "yolov8n.yaml", "epochs": 1, "smoke_batches": 5, "ccarc_config": cfg}
        with patch("ultralytics.models.yolo.detect.train.DetectionTrainer.get_dataset", return_value=(None, None)):
            trainer = CCARCDetectionTrainer(overrides=overrides)
            trainer.smoke_batch_counter = 2
            trainer.smoke_diagnostics = [{"batch": 1}, {"batch": 2}]

            with pytest.raises(CCARCSmokeBatchStop) as exc_info:
                trainer._on_smoke_epoch_end(trainer)

            assert len(exc_info.value.diagnostics) == 2


class TestSmokeBatchesPipeline:
    """Tests for LOCO trainer smoke batch integration and reporting."""

    def test_train_stage_catches_smoke_stop_and_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = CCARCTrainingConfig(
                experiment_variant="E4",
                fold_idx=4,
                target_country="Norway",
                project_dir=str(tmp_path),
                smoke_batches=2,
            )
            loco_trainer = CCARCLOCOTrainer(cfg)

            # Mock YOLO model.train to simulate raising CCARCSmokeBatchStop
            dummy_diags = [
                {"batch": 1, "total_loss": 3.12, "optimizer_step_ran": True},
                {"batch": 2, "total_loss": 2.89, "optimizer_step_ran": True},
            ]

            with patch("src.training.ccarc_trainer.YOLO") as mock_yolo_cls:
                mock_model = MagicMock()
                mock_model.train.side_effect = CCARCSmokeBatchStop(diagnostics=dummy_diags)
                mock_yolo_cls.return_value = mock_model

                summary = loco_trainer.train_stage()

                assert summary["status"] == "SMOKE_BATCHES_COMPLETE"
                assert summary["smoke_batches_executed"] == 2
                assert len(summary["diagnostics"]) == 2

                # Verify smoke_batches_report.json written
                report_file = loco_trainer.persistent_experiment_dir / "smoke_batches_report.json"
                assert report_file.is_file()
                with open(report_file, "r") as f:
                    saved_data = json.load(f)
                assert saved_data["status"] == "SMOKE_BATCHES_COMPLETE"
                assert saved_data["smoke_batches_requested"] == 2

                # Verify best.pt was NOT created or marked as complete
                assert not (loco_trainer.persistent_weights_dir / "best.pt").exists()
                assert not (loco_trainer.persistent_experiment_dir / "results.csv").exists()

    def test_run_fold_in_smoke_mode_skips_validation_stages(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cfg = CCARCTrainingConfig(
                experiment_variant="E4",
                fold_idx=4,
                target_country="Norway",
                project_dir=str(tmp_path),
                smoke_batches=1,
            )
            loco_trainer = CCARCLOCOTrainer(cfg)

            with patch.object(loco_trainer, "dry_run", return_value={"status": "PASS"}):
                with patch.object(
                    loco_trainer,
                    "train_stage",
                    return_value={"status": "SMOKE_BATCHES_COMPLETE", "smoke_batches_executed": 1},
                ):
                    with patch.object(loco_trainer, "evaluate_source_val_stage") as mock_val:
                        with patch.object(loco_trainer, "evaluate_held_out_target_stage") as mock_test:
                            master = loco_trainer.run_fold(dry_run=False, eval_target=True)

                            # Verify validation and target testing were never called
                            mock_val.assert_not_called()
                            mock_test.assert_not_called()

                            assert master["smoke_mode"] is True
                            assert master["stage_1_train"]["status"] == "SMOKE_BATCHES_COMPLETE"
                            smoke_master = loco_trainer.persistent_experiment_dir / "smoke_master_summary.json"
                            assert smoke_master.is_file()

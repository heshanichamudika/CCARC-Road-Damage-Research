"""Tests for CCARC Ablation Experiment Orchestrator and Safety Controller."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.training.ablation_orchestrator import (
    COLAB_BASELINE_DIR,
    COLAB_CCARC_DIR,
    COLAB_LOCAL_RUN_DIR,
    COLAB_VIEWS_DIR,
    VARIANT_SPECIFICATIONS,
    AblationOrchestrator,
    ExperimentStatus,
    parse_args,
)
from src.training.ccarc_config import CCARCTrainingConfig


class TestVariantSpecifications:
    """Verify variant specifications and component flag consistency."""

    def test_all_five_variants_defined(self):
        assert set(VARIANT_SPECIFICATIONS.keys()) == {"E0", "E1", "E2", "E3", "E4"}

    def test_component_flags_match_ccarc_config(self):
        """Verify orchestrator specs match CCARCTrainingConfig.apply_variant behavior."""
        for variant in ("E0", "E1", "E2", "E3", "E4"):
            cfg = CCARCTrainingConfig(experiment_variant=variant)
            spec = VARIANT_SPECIFICATIONS[variant]
            assert cfg.cbs_enabled == spec["cbs_enabled"], f"{variant} CBS mismatch"
            assert cfg.cal_enabled == spec["cal_enabled"], f"{variant} CAL mismatch"
            assert cfg.rwc_enabled == spec["rwc_enabled"], f"{variant} RWC mismatch"

    def test_e2_discrepancy_documented(self):
        """Confirm E2 explicitly documents CAL-only vs cumulative CBS+CAL."""
        e2_spec = VARIANT_SPECIFICATIONS["E2"]
        assert e2_spec["cbs_enabled"] is False
        assert e2_spec["cal_enabled"] is True
        assert e2_spec["rwc_enabled"] is False
        assert "discrepancy_note" in e2_spec
        assert "CAL-only" in e2_spec["discrepancy_note"]


class TestOrchestratorInitialization:
    """Test parameter handling, defaults, and input validation."""

    def test_comparable_hyperparameter_defaults(self, tmp_path):
        orch = AblationOrchestrator(
            views_dir=tmp_path / "views",
            ccarc_dir=tmp_path / "ccarc",
            variants=["E1"],
            folds=[4],
        )
        assert orch.epochs == 100
        assert orch.batch_size == 16
        assert orch.img_size == 640
        assert orch.workers == 2
        assert orch.patience == 20
        assert orch.optimizer == "SGD"
        assert orch.lr0 == 0.01
        assert orch.device == "0"

    def test_custom_hyperparameters(self, tmp_path):
        orch = AblationOrchestrator(
            views_dir=tmp_path / "views",
            ccarc_dir=tmp_path / "ccarc",
            variants=["E1", "E2"],
            folds=[4],
            epochs=50,
            batch_size=8,
            workers=4,
            patience=10,
            optimizer="AdamW",
            lr0=0.005,
            device="cuda:0",
        )
        assert orch.variants == ["E1", "E2"]
        assert orch.folds == [4]
        assert orch.epochs == 50
        assert orch.batch_size == 8
        assert orch.workers == 4
        assert orch.patience == 10
        assert orch.optimizer == "AdamW"
        assert orch.lr0 == 0.005
        assert orch.device == "cuda:0"

    def test_invalid_variant_raises_error(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid variant 'E99'"):
            AblationOrchestrator(
                views_dir=tmp_path,
                variants=["E99"],
            )

    def test_invalid_fold_raises_error(self, tmp_path):
        with pytest.raises(ValueError, match="Invalid fold index '7'"):
            AblationOrchestrator(
                views_dir=tmp_path,
                folds=[7],
            )

    def test_colab_paths_switch(self):
        orch = AblationOrchestrator(colab_paths=True)
        assert orch.views_dir == COLAB_VIEWS_DIR
        assert orch.ccarc_dir == COLAB_CCARC_DIR
        assert orch.baseline_dir == COLAB_BASELINE_DIR
        assert orch.local_run_dir == COLAB_LOCAL_RUN_DIR


class TestStatusDetectionAndSafety:
    """Test experiment status detection and checkpoint protection rules."""

    def test_pending_experiment_when_dir_missing(self, tmp_path):
        orch = AblationOrchestrator(
            views_dir=tmp_path / "views",
            ccarc_dir=tmp_path / "ccarc",
            variants=["E1"],
            folds=[4],
        )
        status, action, reason, best_ckpt, last_ckpt, done_eps, total_eps = (
            orch.inspect_experiment_status("E1", 4)
        )
        assert status == "PENDING"
        assert action == "TRAIN"
        assert best_ckpt is None
        assert last_ckpt is None
        assert done_eps == 0

    def test_completed_experiment_with_summary_is_protected(self, tmp_path):
        ccarc_dir = tmp_path / "ccarc"
        exp_dir = ccarc_dir / "fold_4_Norway_E1"
        weights_dir = exp_dir / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_text("fake_weights")

        # Write completion summary
        summary_path = exp_dir / "train_stage_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"completed": True, "epochs": 100}, f)

        orch = AblationOrchestrator(
            views_dir=tmp_path / "views",
            ccarc_dir=ccarc_dir,
            variants=["E1"],
            folds=[4],
        )
        status, action, reason, best_ckpt, last_ckpt, done_eps, total_eps = (
            orch.inspect_experiment_status("E1", 4)
        )
        assert status == "COMPLETED"
        assert action == "SKIP"
        assert "Protected" in reason
        assert best_ckpt == weights_dir / "best.pt"

    def test_early_stopped_run_is_recognized_as_completed(self, tmp_path):
        """Early stopping writes train_stage_summary.json even with fewer epochs."""
        ccarc_dir = tmp_path / "ccarc"
        exp_dir = ccarc_dir / "fold_4_Norway_E2"
        weights_dir = exp_dir / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_text("best_weights")
        (weights_dir / "last.pt").write_text("last_weights")

        # 45 epochs recorded in results.csv
        results_csv = exp_dir / "results.csv"
        with open(results_csv, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss\n")
            for i in range(45):
                f.write(f"{i},0.4,0.3\n")

        # Early stopping completed cleanly, emitting train_stage_summary.json
        summary_path = exp_dir / "train_stage_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"stage": "STAGE_1_TRAINING_E2", "completed": True}, f)

        orch = AblationOrchestrator(
            views_dir=tmp_path / "views",
            ccarc_dir=ccarc_dir,
            variants=["E2"],
            folds=[4],
        )
        status, action, reason, best_ckpt, last_ckpt, done_eps, total_eps = (
            orch.inspect_experiment_status("E2", 4)
        )
        assert status == "COMPLETED"
        assert action == "SKIP"
        assert done_eps == 45
        assert "early stopping" in reason

    def test_interrupted_run_not_mistaken_for_completed(self, tmp_path):
        """Interrupted run has results.csv and last.pt, but NO summary json."""
        ccarc_dir = tmp_path / "ccarc"
        exp_dir = ccarc_dir / "fold_4_Norway_E4"
        weights_dir = exp_dir / "weights"
        weights_dir.mkdir(parents=True)
        (weights_dir / "best.pt").write_text("intermediate_best")
        (weights_dir / "last.pt").write_text("intermediate_last")

        # 35 epochs out of 100 in results.csv, NO summary json
        results_csv = exp_dir / "results.csv"
        with open(results_csv, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss\n")
            for i in range(35):
                f.write(f"{i},0.5,0.4\n")

        # Default allow_resume=False: Must be BLOCKED
        orch = AblationOrchestrator(
            views_dir=tmp_path / "views",
            ccarc_dir=ccarc_dir,
            variants=["E4"],
            folds=[4],
            allow_resume=False,
        )
        status, action, reason, best_ckpt, last_ckpt, done_eps, total_eps = (
            orch.inspect_experiment_status("E4", 4)
        )
        assert status == "IN_PROGRESS"
        assert action == "BLOCKED"
        assert done_eps == 35
        assert "Interrupted or active run" in reason

        # allow_resume=True: Must be RESUME
        orch_res = AblationOrchestrator(
            views_dir=tmp_path / "views",
            ccarc_dir=ccarc_dir,
            variants=["E4"],
            folds=[4],
            allow_resume=True,
        )
        status, action, reason, best_ckpt, last_ckpt, done_eps, total_eps = (
            orch_res.inspect_experiment_status("E4", 4)
        )
        assert status == "IN_PROGRESS"
        assert action == "RESUME"
        assert done_eps == 35
        assert "Resume enabled" in reason

    def test_interrupted_run_missing_last_pt_cannot_resume(self, tmp_path):
        """Interrupted run without last.pt cannot resume and must be reported as INCOMPLETE."""
        ccarc_dir = tmp_path / "ccarc"
        exp_dir = ccarc_dir / "fold_4_Norway_E3"
        weights_dir = exp_dir / "weights"
        weights_dir.mkdir(parents=True)
        # Only intermediate best.pt, last.pt is missing
        (weights_dir / "best.pt").write_text("intermediate_best")

        results_csv = exp_dir / "results.csv"
        with open(results_csv, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss\n")
            for i in range(20):
                f.write(f"{i},0.5,0.4\n")

        orch = AblationOrchestrator(
            views_dir=tmp_path / "views",
            ccarc_dir=ccarc_dir,
            variants=["E3"],
            folds=[4],
            allow_resume=True,
        )
        status, action, reason, best_ckpt, last_ckpt, done_eps, total_eps = (
            orch.inspect_experiment_status("E3", 4)
        )
        assert status == "INCOMPLETE"
        assert action == "BLOCKED"
        assert last_ckpt is None
        assert "last.pt is missing" in reason

    def test_smoke_test_run_detected_and_cleared_for_full_training(self, tmp_path):
        ccarc_dir = tmp_path / "ccarc"
        exp_dir = ccarc_dir / "fold_4_Norway_E2"
        exp_dir.mkdir(parents=True)
        (exp_dir / "smoke_master_summary.json").write_text('{"smoke": true}')

        orch = AblationOrchestrator(
            views_dir=tmp_path / "views",
            ccarc_dir=ccarc_dir,
            variants=["E2"],
            folds=[4],
        )
        status, action, reason, _, _, _, _ = orch.inspect_experiment_status("E2", 4)
        assert status == "SMOKE_TEST_CHECKPOINT"
        assert action == "TRAIN"


class TestCommandGeneration:
    """Test CLI command generation for both local and Google Colab environments."""

    def test_build_command_contains_all_comparable_hyperparameters(self, tmp_path):
        orch = AblationOrchestrator(
            views_dir="/custom/views",
            ccarc_dir="/custom/ccarc",
            variants=["E1"],
            folds=[4],
            epochs=100,
            batch_size=16,
            img_size=640,
            workers=2,
            patience=20,
            optimizer="SGD",
            lr0=0.01,
            device="0",
        )
        cmd = orch.build_command("E1", 4, "TRAIN")
        assert "python -m src.training.train_loco_ccarc" in cmd
        assert '--views-dir "/custom/views"' in cmd
        assert '--output-dir "/custom/ccarc"' in cmd
        assert "--fold 4" in cmd
        assert "--variant E1" in cmd
        assert "--epochs 100" in cmd
        assert "--batch-size 16" in cmd
        assert "--img-size 640" in cmd
        assert "--workers 2" in cmd
        assert "--patience 20" in cmd
        assert "--optimizer SGD" in cmd
        assert "--lr0 0.01" in cmd
        assert "--device 0" in cmd
        assert "--all-stages" in cmd
        assert "--resume" not in cmd

    def test_build_command_with_resume(self, tmp_path):
        orch = AblationOrchestrator(
            views_dir="/custom/views",
            ccarc_dir="/custom/ccarc",
            variants=["E4"],
            folds=[4],
            allow_resume=True,
        )
        cmd = orch.build_command("E4", 4, "RESUME")
        assert "--resume" in cmd

    def test_build_command_for_baseline_e0(self, tmp_path):
        orch = AblationOrchestrator(
            views_dir="/custom/views",
            baseline_dir="/custom/baseline",
            variants=["E0"],
            folds=[1],
        )
        cmd = orch.build_command("E0", 1, "TRAIN")
        assert "python -m src.training.train_loco_baseline" in cmd
        assert '--output-dir "/custom/baseline"' in cmd
        assert "--variant" not in cmd

    def test_colab_mode_commands(self):
        orch = AblationOrchestrator(colab_paths=True, variants=["E3"], folds=[4])
        cmd = orch.build_command("E3", 4, "TRAIN")
        assert COLAB_VIEWS_DIR.as_posix() in cmd
        assert COLAB_CCARC_DIR.as_posix() in cmd
        assert COLAB_LOCAL_RUN_DIR.as_posix() in cmd


class TestPlanAndReporting:
    """Test plan creation, report writing, and dry-run execution."""

    def test_plan_generates_all_combinations(self, tmp_path):
        orch = AblationOrchestrator(
            views_dir=tmp_path / "views",
            ccarc_dir=tmp_path / "ccarc",
            variants=["E1", "E2", "E3"],
            folds=[3, 4],
        )
        plan = orch.plan()
        assert len(plan) == 6
        assert {p.variant for p in plan} == {"E1", "E2", "E3"}
        assert {p.fold_idx for p in plan} == {3, 4}

    def test_reports_written_without_launching_training(self, tmp_path):
        report_dir = tmp_path / "reports"
        orch = AblationOrchestrator(
            views_dir=tmp_path / "views",
            ccarc_dir=tmp_path / "ccarc",
            variants=["E1"],
            folds=[4],
        )
        plan = orch.plan()
        json_file, md_file = orch.write_reports(report_dir, plan)

        assert json_file.is_file()
        assert md_file.is_file()

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["plan"]) == 1
        assert data["plan"][0]["target_country"] == "Norway"
        assert data["workers"] == 2
        assert data["optimizer"] == "SGD"
        assert data["lr0"] == 0.01

        md_text = md_file.read_text(encoding="utf-8")
        assert "CCARC Ablation Experiment Orchestration Report" in md_text
        assert "Norway" in md_text
        assert "E1" in md_text
        assert "Workers:** 2" in md_text
        assert "SGD" in md_text

    def test_dry_run_execution(self, tmp_path):
        orch = AblationOrchestrator(
            views_dir=tmp_path / "views",
            ccarc_dir=tmp_path / "ccarc",
            variants=["E1"],
            folds=[4],
        )
        plan = orch.plan()
        res = orch.execute_plan(plan, dry_run=True)
        assert res["dry_run"] is True
        assert len(res["executed"]) == 1
        assert res["executed"][0]["dry_run"] is True
        assert len(res["failed"]) == 0


class TestCliArgParsing:
    """Test CLI argument parser."""

    def test_cli_defaults(self):
        args = parse_args([])
        assert args.variants == ["E1", "E2", "E3"]
        assert args.folds == [4]
        assert args.epochs == 100
        assert args.batch_size == 16
        assert args.img_size == 640
        assert args.workers == 2
        assert args.patience == 20
        assert args.optimizer == "SGD"
        assert args.lr0 == 0.01
        assert args.device == "0"
        assert args.execute is False

    def test_cli_overrides(self):
        args = parse_args([
            "--variants", "E2", "E4",
            "--folds", "0", "1",
            "--epochs", "50",
            "--workers", "4",
            "--patience", "15",
            "--optimizer", "AdamW",
            "--lr0", "0.005",
            "--device", "cuda:1",
            "--colab",
            "--resume",
        ])
        assert args.variants == ["E2", "E4"]
        assert args.folds == [0, 1]
        assert args.epochs == 50
        assert args.workers == 4
        assert args.patience == 15
        assert args.optimizer == "AdamW"
        assert args.lr0 == 0.005
        assert args.device == "cuda:1"
        assert args.colab is True
        assert args.resume is True

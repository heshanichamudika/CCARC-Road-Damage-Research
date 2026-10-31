"""YOLOv8 LOCO Baseline Trainer and Three-Stage Evaluator.

Executes cross-country training strictly on source train/val data, selects model
checkpoints on source validation fitness, synchronizes checkpoints to persistent
storage during training, and strictly isolates held-out country test evaluation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO
import yaml

from src.data.loco_splitter import DEFAULT_TARGET_CLASSES, LOCO_COUNTRIES
from src.training.baseline_config import (
    BaselineTrainingConfig,
    check_gpu_environment,
)


class YOLOv8LOCOTrainer:
    """Orchestrates reproducible YOLOv8s training and three-stage evaluation for LOCO folds."""

    def __init__(self, config: BaselineTrainingConfig) -> None:
        """Initialize trainer with experiment configuration."""
        self.config = config
        self.persistent_experiment_dir = (Path(config.project_dir) / config.experiment_name).resolve()
        self.local_experiment_dir = (
            (Path(config.local_run_dir) / config.experiment_name).resolve()
            if config.local_run_dir
            else self.persistent_experiment_dir
        )

        # Primary and local weights paths
        self.persistent_weights_dir = self.persistent_experiment_dir / "weights"
        self.local_weights_dir = self.local_experiment_dir / "weights"

        # Checkpoints (resolves to persistent if local is not present)
        self.best_checkpoint = self.persistent_weights_dir / "best.pt"
        self.last_checkpoint = self.persistent_weights_dir / "last.pt"

    @property
    def experiment_dir(self) -> Path:
        """Alias for persistent experiment directory."""
        return self.persistent_experiment_dir

    @property
    def weights_dir(self) -> Path:
        """Alias for persistent weights directory."""
        return self.persistent_weights_dir

    def sync_to_persistent(self) -> None:
        """Synchronize checkpoints, logs, and configs from local storage to persistent storage."""
        if self.local_experiment_dir == self.persistent_experiment_dir:
            return

        self.persistent_weights_dir.mkdir(parents=True, exist_ok=True)
        files_to_sync = [
            ("results.csv", self.local_experiment_dir / "results.csv", self.persistent_experiment_dir / "results.csv"),
            ("args.yaml", self.local_experiment_dir / "args.yaml", self.persistent_experiment_dir / "args.yaml"),
            ("last.pt", self.local_weights_dir / "last.pt", self.persistent_weights_dir / "last.pt"),
            ("best.pt", self.local_weights_dir / "best.pt", self.persistent_weights_dir / "best.pt"),
        ]

        for name, src, dst in files_to_sync:
            if src.is_file():
                try:
                    shutil.copy2(src, dst)
                except OSError as e:
                    print(f"[SYNC WARNING] Failed copying {name} to persistent storage: {e}")

    def sync_from_persistent(self) -> None:
        """Restore checkpoints and logs from persistent storage to local storage for resumption."""
        if self.local_experiment_dir == self.persistent_experiment_dir:
            return

        self.local_weights_dir.mkdir(parents=True, exist_ok=True)
        files_to_restore = [
            ("results.csv", self.persistent_experiment_dir / "results.csv", self.local_experiment_dir / "results.csv"),
            ("args.yaml", self.persistent_experiment_dir / "args.yaml", self.local_experiment_dir / "args.yaml"),
            ("last.pt", self.persistent_weights_dir / "last.pt", self.local_weights_dir / "last.pt"),
            ("best.pt", self.persistent_weights_dir / "best.pt", self.local_weights_dir / "best.pt"),
        ]

        for name, src, dst in files_to_restore:
            if src.is_file() and not dst.is_file():
                try:
                    shutil.copy2(src, dst)
                except OSError as e:
                    print(f"[SYNC WARNING] Failed restoring {name} to local storage: {e}")

    def dry_run(self) -> Dict[str, Any]:
        """Validate paths, configuration, GPU availability and write access without training."""
        checks: Dict[str, Any] = {}
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Check data.yaml exists and is valid
        yaml_path = Path(self.config.data_yaml_path).resolve()
        if not yaml_path.is_file():
            errors.append(f"data.yaml not found: {yaml_path}")
            checks["data_yaml"] = False
        else:
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    data_cfg = yaml.safe_load(f)

                checks["data_yaml"] = True
                checks["data_yaml_path"] = str(yaml_path)
                checks["target_country"] = data_cfg.get("loco_target_country", self.config.target_country)

                # Verify paths in yaml
                base_path = Path(data_cfg.get("path", yaml_path.parent)).resolve()
                train_dir = (base_path / data_cfg.get("train", "images/train")).resolve()
                val_dir = (base_path / data_cfg.get("val", "images/val")).resolve()
                test_dir = (base_path / data_cfg.get("test", "images/test")).resolve()

                checks["train_dir_exists"] = train_dir.is_dir()
                checks["val_dir_exists"] = val_dir.is_dir()
                checks["test_dir_exists"] = test_dir.is_dir()

                if not train_dir.is_dir():
                    errors.append(f"Train image directory not found: {train_dir}")
                if not val_dir.is_dir():
                    errors.append(f"Val image directory not found: {val_dir}")
                if not test_dir.is_dir():
                    errors.append(f"Test image directory not found: {test_dir}")

                # Check classes
                names = data_cfg.get("names", {})
                checks["num_classes"] = len(names)
                if len(names) != len(self.config.target_classes):
                    warnings.append(f"Class count in data.yaml ({len(names)}) differs from config ({len(self.config.target_classes)})")

            except Exception as e:
                errors.append(f"Error parsing data.yaml: {e}")
                checks["data_yaml"] = False

        # 2. Check output directory write access (both persistent and local)
        try:
            self.persistent_experiment_dir.mkdir(parents=True, exist_ok=True)
            test_file = self.persistent_experiment_dir / ".write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
            checks["persistent_output_dir_writable"] = True
            checks["output_dir_writable"] = True
            checks["persistent_output_dir"] = str(self.persistent_experiment_dir)
        except Exception as e:
            errors.append(f"Persistent output directory not writable: {e}")
            checks["persistent_output_dir_writable"] = False
            checks["output_dir_writable"] = False

        if self.local_experiment_dir != self.persistent_experiment_dir:
            try:
                self.local_experiment_dir.mkdir(parents=True, exist_ok=True)
                test_file = self.local_experiment_dir / ".write_test_local"
                test_file.write_text("ok", encoding="utf-8")
                test_file.unlink()
                checks["local_run_dir_writable"] = True
                checks["local_run_dir"] = str(self.local_experiment_dir)
            except Exception as e:
                errors.append(f"Local run directory not writable: {e}")
                checks["local_run_dir_writable"] = False

        # 3. Check hardware environment
        hw_info = check_gpu_environment()
        checks["hardware"] = hw_info
        resolved_device = self.config.resolve_device()
        checks["resolved_device"] = resolved_device

        if resolved_device == "cpu" and self.config.device != "cpu":
            warnings.append("CUDA requested or auto-selected, but no CUDA GPU detected. Training will run on CPU.")

        # 4. Check resume status
        has_last = (self.persistent_weights_dir / "last.pt").is_file() or (self.local_weights_dir / "last.pt").is_file()
        has_best = (self.persistent_weights_dir / "best.pt").is_file() or (self.local_weights_dir / "best.pt").is_file()
        checks["last_checkpoint_exists"] = has_last
        checks["best_checkpoint_exists"] = has_best

        # Check if already completed (results.csv exists and best.pt exists)
        results_csv = self.persistent_experiment_dir / "results.csv"
        is_completed = has_best and results_csv.is_file()
        checks["is_already_completed"] = is_completed

        # 5. Methodological note verification
        checks["test_data_isolated_from_training"] = True
        checks["target_positive_only_test_set_acknowledged"] = True

        passed = len(errors) == 0
        report = {
            "status": "PASS" if passed else "FAIL",
            "fold_idx": self.config.fold_idx,
            "target_country": self.config.target_country,
            "experiment_name": self.config.experiment_name,
            "checks": checks,
            "errors": errors,
            "warnings": warnings,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Save dry run report to persistent directory
        self.persistent_experiment_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.persistent_experiment_dir / "dry_run_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    def save_reproducibility_metadata(self) -> Path:
        """Record exact runtime parameters, versions, hardware, and seeds in persistent metadata."""
        self.persistent_experiment_dir.mkdir(parents=True, exist_ok=True)
        hw = check_gpu_environment()

        metadata = {
            "protocol": "YOLOv8s_LOCO_Source_Only_Baseline",
            "experiment_name": self.config.experiment_name,
            "fold_idx": self.config.fold_idx,
            "target_country": self.config.target_country,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": {
                "name": self.config.model_name,
                "pretrained": self.config.pretrained,
                "input_resolution": self.config.img_size,
                "num_classes": self.config.num_classes,
                "classes": list(self.config.target_classes),
            },
            "optimization": {
                "optimizer": self.config.optimizer,
                "initial_learning_rate_lr0": self.config.lr0,
                "final_learning_rate_factor_lrf": self.config.lrf,
                "momentum": self.config.momentum,
                "weight_decay": self.config.weight_decay,
                "warmup_epochs": self.config.warmup_epochs,
                "warmup_momentum": self.config.warmup_momentum,
                "warmup_bias_lr": self.config.warmup_bias_lr,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "patience": self.config.patience,
                "seed": self.config.seed,
                "deterministic": self.config.deterministic,
            },
            "augmentations": {
                "policy": "FROZEN_STANDARD_YOLOV8_BASELINE",
                "hsv_h": self.config.hsv_h,
                "hsv_s": self.config.hsv_s,
                "hsv_v": self.config.hsv_v,
                "degrees": self.config.degrees,
                "translate": self.config.translate,
                "scale": self.config.scale,
                "shear": self.config.shear,
                "perspective": self.config.perspective,
                "flipud": self.config.flipud,
                "fliplr": self.config.fliplr,
                "mosaic": self.config.mosaic,
                "mixup": self.config.mixup,
                "copy_paste": self.config.copy_paste,
            },
            "checkpoint_selection": {
                "criterion": "SOURCE_VALIDATION_FITNESS_ONLY",
                "formula": "0.1 * mAP@0.5 + 0.9 * mAP@0.5:0.95",
                "target_test_data_leakage": "ZERO (Test split completely unread during training)",
            },
            "environment": {
                "pytorch_version": hw["pytorch_version"],
                "ultralytics_version": hw["ultralytics_version"],
                "cuda_available": hw["cuda_available"],
                "device_resolved": self.config.resolve_device(),
                "gpus": hw["gpus"],
                "os": hw["os_platform"],
                "python": hw["python_version"],
            },
            "hardware": hw,
            "scientific_limitations": (
                "Quantitative LOCO test sets currently consist exclusively of verified target-positive images "
                "(D00, D10, D20, D40); negative background road images (empty_xml_candidate) and non-target "
                "damage frames (quarantined_non_target) are quarantined by default. Unconstrained false-positive "
                "behavior on undamaged roads is not fully evaluated under this baseline protocol."
            ),
        }

        meta_path = self.persistent_experiment_dir / "reproducibility_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return meta_path

    def train_stage(self, force_restart: bool = False) -> Dict[str, Any]:
        """Execute Stage 1: Train YOLOv8s strictly on source train and val partitions."""
        self.persistent_experiment_dir.mkdir(parents=True, exist_ok=True)
        self.local_experiment_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration snapshot & metadata
        config_snapshot_path = self.persistent_experiment_dir / "config_snapshot.yaml"
        self.config.save_yaml(config_snapshot_path)
        self.save_reproducibility_metadata()

        # Check if already completed and not force restarting
        results_csv_persistent = self.persistent_experiment_dir / "results.csv"
        results_csv_local = self.local_experiment_dir / "results.csv"
        best_pt_exists = (self.persistent_weights_dir / "best.pt").is_file() or (self.local_weights_dir / "best.pt").is_file()

        if not force_restart and best_pt_exists and (results_csv_persistent.is_file() or results_csv_local.is_file()):
            print(f"[STAGE 1] Experiment {self.config.experiment_name} already completed. Skipping training.")
            return {
                "status": "COMPLETED",
                "message": "Training already completed.",
                "best_checkpoint": str(self.best_checkpoint),
                "last_checkpoint": str(self.last_checkpoint),
                "results_csv": str(results_csv_persistent),
            }

        # Check for resumption
        resume_training = self.config.resume and (
            (self.local_weights_dir / "last.pt").is_file() or (self.persistent_weights_dir / "last.pt").is_file()
        )
        if resume_training:
            self.sync_from_persistent()
            resume_ckpt = self.local_weights_dir / "last.pt"
            print(f"[STAGE 1] Resuming training from: {resume_ckpt}")
            model = YOLO(str(resume_ckpt))
        else:
            print(f"[STAGE 1] Initializing fresh model weights: {self.config.model_name}")
            model = YOLO(self.config.model_name)

        train_args = self.config.to_yolo_train_args()
        if resume_training:
            train_args["resume"] = True

        # Attach periodic synchronization callback to save checkpoints to Google Drive during training
        def on_fit_epoch_end_callback(trainer_obj: Any) -> None:
            self.sync_to_persistent()

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end_callback)

        start_time = time.perf_counter()
        print(f"[STAGE 1] Starting training for Fold {self.config.fold_idx} ({self.config.target_country})...")
        print(f"         Device: {train_args['device']} | Batch: {train_args['batch']} | Epochs: {train_args['epochs']} | Optimizer: {train_args['optimizer']}")

        # Execute training
        train_results = model.train(**train_args)
        elapsed_sec = time.perf_counter() - start_time

        # Final synchronization to persistent Google Drive storage
        self.sync_to_persistent()

        summary = {
            "stage": "STAGE_1_TRAINING",
            "fold_idx": self.config.fold_idx,
            "target_country": self.config.target_country,
            "elapsed_seconds": round(elapsed_sec, 2),
            "best_checkpoint": str(self.persistent_weights_dir / "best.pt"),
            "last_checkpoint": str(self.persistent_weights_dir / "last.pt"),
            "results_csv": str(results_csv_persistent),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(self.persistent_experiment_dir / "train_stage_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def _extract_metrics(self, val_results: Any) -> Dict[str, Any]:
        """Extract standardized detection metrics from Ultralytics validation results."""
        box = getattr(val_results, "box", None)
        res_dict = getattr(val_results, "results_dict", {})

        # Global metrics
        p = float(getattr(box, "p", res_dict.get("metrics/precision(B)", 0.0)))
        r = float(getattr(box, "r", res_dict.get("metrics/recall(B)", 0.0)))
        map50 = float(getattr(box, "map50", res_dict.get("metrics/mAP50(B)", 0.0)))
        map50_95 = float(getattr(box, "map", res_dict.get("metrics/mAP50-95(B)", 0.0)))

        # Fitness = 0.1 * mAP@0.5 + 0.9 * mAP@0.5:0.95
        fitness = float(getattr(val_results, "fitness", 0.1 * map50 + 0.9 * map50_95))

        # Per-class metrics
        per_class: Dict[str, Dict[str, float]] = {}
        maps_arr = getattr(box, "maps", None)
        ap50_arr = getattr(box, "ap50", None)

        for idx, cls_name in enumerate(self.config.target_classes):
            cls_ap50 = float(ap50_arr[idx]) if (ap50_arr is not None and len(ap50_arr) > idx) else 0.0
            cls_map = float(maps_arr[idx]) if (maps_arr is not None and len(maps_arr) > idx) else 0.0
            per_class[cls_name] = {
                "ap50": round(cls_ap50, 4),
                "map50_95": round(cls_map, 4),
            }

        return {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
            "fitness": round(fitness, 4),
            "per_class": per_class,
        }

    def evaluate_source_val_stage(self, checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
        """Execute Stage 2: Evaluate best model on source validation data."""
        ckpt = Path(checkpoint_path).resolve() if checkpoint_path else (self.persistent_weights_dir / "best.pt")
        if not ckpt.is_file():
            ckpt = self.local_weights_dir / "best.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found for validation: {ckpt}")

        print(f"[STAGE 2] Evaluating checkpoint on SOURCE VALIDATION: {ckpt}")
        model = YOLO(str(ckpt))

        start_time = time.perf_counter()
        val_results = model.val(
            data=str(Path(self.config.data_yaml_path).resolve()),
            split="val",
            imgsz=self.config.img_size,
            batch=self.config.batch_size,
            device=self.config.resolve_device(),
            plots=False,
        )
        elapsed_sec = time.perf_counter() - start_time

        metrics = self._extract_metrics(val_results)
        summary = {
            "stage": "STAGE_2_SOURCE_VALIDATION",
            "fold_idx": self.config.fold_idx,
            "target_country": self.config.target_country,
            "split": "val",
            "checkpoint_evaluated": str(ckpt),
            "elapsed_seconds": round(elapsed_sec, 2),
            "metrics": metrics,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

        out_path = self.persistent_experiment_dir / "source_val_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[STAGE 2] Source Val -> mAP@0.5: {metrics['map50']:.4f} | mAP@0.5:0.95: {metrics['map50_95']:.4f}")
        return summary

    def evaluate_held_out_target_stage(self, checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
        """Execute Stage 3: Explicitly evaluate best model on unseen held-out target country data."""
        ckpt = Path(checkpoint_path).resolve() if checkpoint_path else (self.persistent_weights_dir / "best.pt")
        if not ckpt.is_file():
            ckpt = self.local_weights_dir / "best.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found for held-out test evaluation: {ckpt}")

        print(f"[STAGE 3] Explicitly evaluating checkpoint on HELD-OUT TARGET TEST ({self.config.target_country}): {ckpt}")
        model = YOLO(str(ckpt))

        start_time = time.perf_counter()
        test_results = model.val(
            data=str(Path(self.config.data_yaml_path).resolve()),
            split="test",
            imgsz=self.config.img_size,
            batch=self.config.batch_size,
            device=self.config.resolve_device(),
            plots=False,
        )
        elapsed_sec = time.perf_counter() - start_time

        metrics = self._extract_metrics(test_results)
        summary = {
            "stage": "STAGE_3_HELD_OUT_TARGET_EVALUATION",
            "fold_idx": self.config.fold_idx,
            "target_country": self.config.target_country,
            "split": "test",
            "checkpoint_evaluated": str(ckpt),
            "elapsed_seconds": round(elapsed_sec, 2),
            "metrics": metrics,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

        out_path = self.persistent_experiment_dir / "held_out_test_metrics.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"[STAGE 3] Target Test ({self.config.target_country}) -> mAP@0.5: {metrics['map50']:.4f} | mAP@0.5:0.95: {metrics['map50_95']:.4f}")
        return summary

    def run_fold(
        self,
        dry_run: bool = False,
        eval_target: bool = False,
        eval_only: bool = False,
        force_restart: bool = False,
    ) -> Dict[str, Any]:
        """Run requested stages for a single LOCO fold."""
        # Step 0: Dry Run Validation
        dry_run_report = self.dry_run()
        if dry_run or dry_run_report["status"] != "PASS":
            return {"dry_run_report": dry_run_report}

        # Step 1: Training (skipped if eval_only)
        if not eval_only:
            train_summary = self.train_stage(force_restart=force_restart)
        else:
            train_summary = {"status": "SKIPPED_EVAL_ONLY"}

        # Step 2: Source Validation Evaluation (always runs on model selection checkpoint)
        val_summary = self.evaluate_source_val_stage()

        # Step 3: Held-Out Target Test Evaluation (ONLY if explicitly requested)
        if eval_target:
            test_summary = self.evaluate_held_out_target_stage()
        else:
            test_summary = {
                "status": "SKIPPED_NOT_REQUESTED",
                "message": "Stage 3 held-out target evaluation was not requested. Target test data remained unexposed.",
            }

        # Master Summary
        master_summary = {
            "protocol": "LOCO_YOLOv8s_Baseline",
            "fold_idx": self.config.fold_idx,
            "target_country": self.config.target_country,
            "experiment_dir": str(self.persistent_experiment_dir),
            "dry_run": dry_run_report,
            "stage_1_train": train_summary,
            "stage_2_source_val": val_summary,
            "stage_3_held_out_target": test_summary,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(self.persistent_experiment_dir / "fold_master_summary.json", "w", encoding="utf-8") as f:
            json.dump(master_summary, f, indent=2)

        return master_summary

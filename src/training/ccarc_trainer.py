"""CCARC LOCO Trainer and Three-Stage Evaluator for YOLOv8s.

Integrates Country-Balanced Sampling (CBS), Class-Aware Learning (CAL), and
Reliability-Weighted Consistency (RWC) into an Ultralytics-compatible training pipeline.
Preserves standard YOLOv8 inference architecture, strictly isolates held-out country
test data, supports safe resumption, and isolates outputs from baseline runs.
"""

from __future__ import annotations

from copy import copy
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import LOGGER, RANK
import yaml

from src.data.loco_splitter import DEFAULT_TARGET_CLASSES, LOCO_COUNTRIES
from src.models.ccarc_consistency import (
    ReliabilityConsistencyConfig,
    ReliabilityConsistencyLoss,
)
from src.models.ccarc_loss import (
    ClassAwareLearningConfig,
    ClassWeightedBCEWithLogitsLoss,
    attach_class_aware_loss,
    compute_class_weights,
    extract_source_train_class_counts,
    resolve_detection_model,
)
from src.models.ccarc_perturbations import (
    PhotometricPerturbation,
    PhotometricPerturbationConfig,
)
from src.training.baseline_config import check_gpu_environment
from src.training.ccarc_config import CCARCTrainingConfig
from src.training.ccarc_sampler import (
    CCARCInfiniteDataLoader,
    CountryBalancedBatchSampler,
    PIN_MEMORY,
    load_manifest_country_lookup,
    seed_worker,
)


class CCARCCompoundLoss:
    """Compound loss callable wrapping Supervised Detection Loss + Reliability Consistency Regularizer.

    Defined at module level to ensure full pickle-compatibility with torch.save / multiprocessing.
    """

    def __init__(
        self,
        raw_model: Any,
        perturbation: PhotometricPerturbation,
        rwc_loss: ReliabilityConsistencyLoss,
        epoch: int = 0,
    ) -> None:
        self.raw_model = raw_model
        self.perturb_module = perturbation
        self.rwc_module = rwc_loss
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        """Update current training epoch for RWC ramp-up schedule."""
        self.epoch = epoch

    def __call__(
        self, batch: Dict[str, Any], preds: Optional[Any] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute dual weak/strong forward passes and compute combined loss."""
        img = batch["img"]
        # Ensure model criterion is initialized
        if getattr(self.raw_model, "criterion", None) is None:
            self.raw_model.criterion = self.raw_model.init_criterion()

        # Generate weak view (teacher) and strong view (student)
        x_w, x_s = self.perturb_module.generate_pair(img)

        # Teacher forward pass (detached gradients)
        with torch.no_grad():
            teacher_preds = self.raw_model.forward(x_w)

        # Student forward pass (maintains autograd graph)
        student_preds = self.raw_model.forward(x_s)

        # Supervised detection loss on student predictions
        sup_loss, loss_items = self.raw_model.criterion(student_preds, batch)

        # Reliability consistency loss using synchronized epoch
        rwc_result = self.rwc_module(teacher_preds, student_preds, current_epoch=self.epoch)
        con_loss = rwc_result["loss"]

        # Scale consistency loss by batch size to align magnitude with supervised loss
        batch_sz = img.shape[0]

        # In Ultralytics 8.4+, criterion returns loss * batch_size as a 1D vector (box, cls, dfl),
        # whereas in Ultralytics <=8.3, criterion returns loss.sum() * batch_size as a scalar.
        if isinstance(sup_loss, torch.Tensor) and sup_loss.ndim > 0:
            sup_loss_scalar = sup_loss.sum()
        else:
            sup_loss_scalar = sup_loss

        total_loss = sup_loss_scalar + con_loss * batch_sz

        return total_loss, loss_items


class CCARCDetectionTrainer(DetectionTrainer):
    """Ultralytics DetectionTrainer subclass implementing CCARC proposed components.

    Hooks:
    1. CBS: Overrides get_dataloader() to inject CountryBalancedBatchSampler for training.
    2. CAL: Overrides set_model_attributes() to inject ClassWeightedBCEWithLogitsLoss into criterion.
    3. RWC: Wraps model.loss() to compute dual weak/strong forward passes and consistency regularizer.
    """

    _ACTIVE_CCARC_CONFIG: Optional[CCARCTrainingConfig] = None

    def __init__(self, overrides: Optional[Dict[str, Any]] = None, _callbacks: Optional[Any] = None) -> None:
        """Initialize trainer with overrides, separating CCARC-specific arguments from YOLO config."""
        custom_cfg: Dict[str, Any] = {}
        if overrides is not None:
            overrides = dict(overrides)
            # Pop custom CCARC keys so only valid YOLO arguments reach get_cfg()
            for key in ("ccarc_config", "cbs_enabled", "cal_enabled", "rwc_enabled", "target_country", "trainer"):
                if key in overrides:
                    custom_cfg[key] = overrides.pop(key)

        super().__init__(overrides=overrides, _callbacks=_callbacks)

        # Retrieve CCARC configuration from custom_cfg, or fall back to class-level context
        self.ccarc_cfg: Optional[CCARCTrainingConfig] = custom_cfg.get(
            "ccarc_config", CCARCDetectionTrainer._ACTIVE_CCARC_CONFIG
        )
        if self.ccarc_cfg is not None:
            CCARCDetectionTrainer._ACTIVE_CCARC_CONFIG = self.ccarc_cfg
            self.cbs_enabled: bool = bool(custom_cfg.get("cbs_enabled", self.ccarc_cfg.cbs_enabled))
            self.cal_enabled: bool = bool(custom_cfg.get("cal_enabled", self.ccarc_cfg.cal_enabled))
            self.rwc_enabled: bool = bool(custom_cfg.get("rwc_enabled", self.ccarc_cfg.rwc_enabled))
            self.target_country: Optional[str] = custom_cfg.get("target_country", self.ccarc_cfg.target_country)
        else:
            self.cbs_enabled: bool = bool(custom_cfg.get("cbs_enabled", False))
            self.cal_enabled: bool = bool(custom_cfg.get("cal_enabled", False))
            self.rwc_enabled: bool = bool(custom_cfg.get("rwc_enabled", False))
            self.target_country: Optional[str] = custom_cfg.get("target_country", None)

        # Note: CCARC configuration must NEVER be attached to self.args.
        # Ultralytics passes copy(self.args) to DetectionValidator (triggering get_cfg check_dict_alignment),
        # serializes vars(self.args) into checkpoints ('train_args'), and saves args.yaml.
        # All CCARC attributes are maintained strictly on self (self.ccarc_cfg, self.cbs_enabled, etc.).

        self.cbs_sampler: Optional[CountryBalancedBatchSampler] = None
        self.class_weights: Optional[torch.Tensor] = None
        self.perturbation: Optional[PhotometricPerturbation] = None
        self.rwc_loss: Optional[ReliabilityConsistencyLoss] = None

        if self.rwc_enabled:
            perturb_cfg = getattr(self.ccarc_cfg, "perturb_config", None) if self.ccarc_cfg else None
            rwc_cfg = getattr(self.ccarc_cfg, "rwc_config", None) if self.ccarc_cfg else None
            self.perturbation = PhotometricPerturbation(perturb_cfg)
            self.rwc_loss = ReliabilityConsistencyLoss(rwc_cfg)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train") -> DataLoader:
        """Construct dataloader, injecting CountryBalancedBatchSampler during training mode."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."

        # In DDP multi-GPU environments, warn about single-node sampler distribution
        if rank not in (-1, 0) and self.cbs_enabled and mode == "train":
            LOGGER.warning(
                "WARNING: CCARC Country-Balanced Sampling is currently optimized for single-GPU "
                "or main rank execution. Ensure per-rank dataset distribution aligns with country pools."
            )

        dataset = self.build_dataset(dataset_path, mode, batch_size)

        if mode == "train" and self.cbs_enabled:
            im_files = getattr(dataset, "im_files", None)
            if im_files is None or len(im_files) == 0:
                LOGGER.warning("Dataset does not expose im_files; falling back to default YOLO dataloader.")
                return super().get_dataloader(dataset_path, batch_size, rank, mode)

            LOGGER.info(
                f"[CCARC CBS] Activating CountryBalancedBatchSampler for {len(im_files)} training images. "
                f"Target country excluded: {self.target_country}"
            )

            batch_sampler = CountryBalancedBatchSampler(
                samples=im_files,
                batch_size=batch_size,
                target_country=self.target_country,
                seed=self.args.seed,
            )
            self.cbs_sampler = batch_sampler
            workers = getattr(self.args, "workers", 4)
            collate_fn = getattr(dataset, "collate_fn", None)

            # Ultralytics worker count heuristic and seed generator
            nd = torch.cuda.device_count()
            nw = min(os.cpu_count() // max(nd, 1), workers) if os.cpu_count() else workers
            generator = torch.Generator()
            generator.manual_seed(6148914691236517205 + (RANK if RANK != -1 else 0))

            return CCARCInfiniteDataLoader(
                dataset,
                batch_sampler=batch_sampler,
                batch_size=batch_size,
                cbs_sampler=batch_sampler,
                num_workers=nw,
                pin_memory=PIN_MEMORY,
                collate_fn=collate_fn,
                worker_init_fn=seed_worker,
                generator=generator,
            )

        return super().get_dataloader(dataset_path, batch_size, rank, mode)

    def get_validator(self) -> DetectionValidator:
        """Returns a DetectionValidator for YOLO model validation with sanitized arguments.

        Ensures that any custom CCARC parameters are stripped from validator overrides so
        that Ultralytics get_cfg() / check_dict_alignment() succeeds cleanly.
        """
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        val_args = copy(self.args)
        for key in ("ccarc_config", "cbs_enabled", "cal_enabled", "rwc_enabled", "target_country", "trainer"):
            if hasattr(val_args, key):
                delattr(val_args, key)
            elif isinstance(val_args, dict) and key in val_args:
                val_args.pop(key, None)
        return DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=val_args, _callbacks=self.callbacks
        )

    def set_model_attributes(self) -> None:
        """Attach hyperparameters, class weights (CAL), and RWC consistency wrapper."""
        super().set_model_attributes()

        # 1. Attach CAL Class-Weighted Loss
        if self.cal_enabled:
            if self.class_weights is None:
                # Precompute class weights from source training split
                data_yaml = getattr(self.args, "data", "")
                summary_path = Path(data_yaml).parent / "fold_summary.json"
                manifest_path = Path(data_yaml).parent / "fold_manifest.csv"
                source_input: Any = None

                if summary_path.is_file():
                    source_input = summary_path
                elif manifest_path.is_file():
                    source_input = manifest_path
                else:
                    source_input = Path(data_yaml).parent / "labels" / "train"

                try:
                    cal_cfg = getattr(self.ccarc_cfg, "cal_config", None)
                    counts = extract_source_train_class_counts(source_input)
                    self.class_weights = compute_class_weights(counts, config=cal_cfg)
                    LOGGER.info(
                        f"[CCARC CAL] Computed normalized class weights from source training: "
                        f"{dict(zip(DEFAULT_TARGET_CLASSES, [round(w.item(), 3) for w in self.class_weights]))}"
                    )
                except Exception as e:
                    LOGGER.warning(f"[CCARC CAL WARNING] Failed calculating class weights: {e}. Using uniform weights.")
                    self.class_weights = torch.ones(len(DEFAULT_TARGET_CLASSES))

            # Inject into model criterion
            if self.class_weights is not None:
                attach_class_aware_loss(self.model, self.class_weights)

        # 2. Wrap model.loss for RWC Consistency Regularization
        if self.rwc_enabled and self.rwc_loss is not None and self.perturbation is not None:
            self._wrap_model_with_rwc()

    def _wrap_model_with_rwc(self) -> None:
        """Wrap model.loss() to compute dual weak/strong forward passes and consistency loss."""
        raw_model = resolve_detection_model(self.model)
        orig_loss_fn = getattr(raw_model, "loss", None)

        if orig_loss_fn is None or getattr(raw_model, "_ccarc_rwc_wrapped", False):
            return

        raw_model.loss = CCARCCompoundLoss(
            raw_model=raw_model,
            perturbation=self.perturbation,
            rwc_loss=self.rwc_loss,
            epoch=getattr(self, "epoch", 0),
        )
        raw_model._ccarc_rwc_wrapped = True
        LOGGER.info("[CCARC RWC] Successfully wrapped model with Reliability-Weighted Consistency loss.")

    def save_model(self) -> None:
        """Save model training checkpoints, ensuring EMA model and train_args remain clean."""
        # Ensure EMA model does not serialize custom instance-level loss wrappers
        if hasattr(self, "ema") and self.ema is not None and hasattr(self.ema, "ema"):
            raw_ema = resolve_detection_model(self.ema.ema)
            if "loss" in raw_ema.__dict__:
                del raw_ema.loss
            if "_ccarc_rwc_wrapped" in raw_ema.__dict__:
                del raw_ema._ccarc_rwc_wrapped

        super().save_model()


class CCARCLOCOTrainer:
    """High-level orchestrator for CCARC training, source validation, and held-out evaluation."""

    def __init__(self, config: CCARCTrainingConfig) -> None:
        """Initialize CCARC LOCO trainer."""
        self.config = config
        self.persistent_experiment_dir = (Path(config.project_dir) / config.experiment_name).resolve()
        self.local_experiment_dir = (
            (Path(config.local_run_dir) / config.experiment_name).resolve()
            if config.local_run_dir
            else self.persistent_experiment_dir
        )

        self.persistent_weights_dir = self.persistent_experiment_dir / "weights"
        self.local_weights_dir = self.local_experiment_dir / "weights"

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
        """Synchronize checkpoints and metrics from local fast SSD to persistent Google Drive."""
        if self.local_experiment_dir == self.persistent_experiment_dir:
            return

        self.persistent_weights_dir.mkdir(parents=True, exist_ok=True)
        files = [
            ("results.csv", self.local_experiment_dir / "results.csv", self.persistent_experiment_dir / "results.csv"),
            ("args.yaml", self.local_experiment_dir / "args.yaml", self.persistent_experiment_dir / "args.yaml"),
            ("last.pt", self.local_weights_dir / "last.pt", self.persistent_weights_dir / "last.pt"),
            ("best.pt", self.local_weights_dir / "best.pt", self.persistent_weights_dir / "best.pt"),
        ]
        for name, src, dst in files:
            if src.is_file():
                try:
                    shutil.copy2(src, dst)
                except OSError as e:
                    print(f"[CCARC SYNC WARNING] Failed copying {name}: {e}")

    def sync_from_persistent(self) -> None:
        """Restore checkpoints from persistent Google Drive to local SSD for safe resumption."""
        if self.local_experiment_dir == self.persistent_experiment_dir:
            return

        self.local_weights_dir.mkdir(parents=True, exist_ok=True)
        files = [
            ("results.csv", self.persistent_experiment_dir / "results.csv", self.local_experiment_dir / "results.csv"),
            ("args.yaml", self.persistent_experiment_dir / "args.yaml", self.local_experiment_dir / "args.yaml"),
            ("last.pt", self.persistent_weights_dir / "last.pt", self.local_weights_dir / "last.pt"),
            ("best.pt", self.persistent_weights_dir / "best.pt", self.local_weights_dir / "best.pt"),
        ]
        for name, src, dst in files:
            if src.is_file() and not dst.is_file():
                try:
                    shutil.copy2(src, dst)
                except OSError as e:
                    print(f"[CCARC SYNC WARNING] Failed restoring {name}: {e}")

    def dry_run(self) -> Dict[str, Any]:
        """Validate CCARC configuration, dataset paths, and module availability without training."""
        checks: Dict[str, Any] = {}
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Check data.yaml exists and parses
        yaml_path = Path(self.config.data_yaml_path).resolve()
        if not yaml_path.is_file():
            errors.append(f"data.yaml not found: {yaml_path}")
            checks["data_yaml"] = False
        else:
            checks["data_yaml"] = True
            checks["data_yaml_path"] = str(yaml_path)
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    data_cfg = yaml.safe_load(f)

                base_path = Path(data_cfg.get("path", yaml_path.parent))
                train_dir = base_path / data_cfg.get("train", "images/train")
                val_dir = base_path / data_cfg.get("val", "images/val")
                test_dir = base_path / data_cfg.get("test", "images/test")

                checks["train_dir_exists"] = train_dir.is_dir()
                checks["val_dir_exists"] = val_dir.is_dir()
                checks["test_dir_exists"] = test_dir.is_dir()

                if not train_dir.is_dir():
                    errors.append(f"Train directory not found: {train_dir}")
                if not val_dir.is_dir():
                    errors.append(f"Val directory not found: {val_dir}")
                if not test_dir.is_dir():
                    errors.append(f"Test directory not found: {test_dir}")

            except Exception as e:
                errors.append(f"Error parsing data.yaml: {e}")
                checks["data_yaml"] = False

        # 2. Verify write access
        try:
            self.persistent_experiment_dir.mkdir(parents=True, exist_ok=True)
            tf = self.persistent_experiment_dir / ".write_test"
            tf.write_text("ok", encoding="utf-8")
            tf.unlink()
            checks["persistent_output_dir_writable"] = True
            checks["output_dir_writable"] = True
        except Exception as e:
            errors.append(f"Persistent output dir not writable: {e}")
            checks["persistent_output_dir_writable"] = False
            checks["output_dir_writable"] = False

        # 3. Check hardware environment
        hw = check_gpu_environment()
        checks["hardware"] = hw
        checks["resolved_device"] = self.config.resolve_device()

        # 4. Check component configs
        checks["variant"] = self.config.experiment_variant
        checks["cbs_enabled"] = self.config.cbs_enabled
        checks["cal_enabled"] = self.config.cal_enabled
        checks["rwc_enabled"] = self.config.rwc_enabled

        # 5. Check CAL source class counts if available
        if self.config.cal_enabled and yaml_path.is_file():
            try:
                counts = extract_source_train_class_counts(yaml_path.parent)
                w = compute_class_weights(counts, self.config.cal_config)
                checks["cal_class_weights"] = {
                    cls: round(float(w[i]), 3) for i, cls in enumerate(DEFAULT_TARGET_CLASSES)
                }
            except Exception as e:
                warnings.append(f"CAL dry run check: could not extract class counts from {yaml_path.parent}: {e}")

        # Check resume status
        has_last = (self.persistent_weights_dir / "last.pt").is_file()
        has_best = (self.persistent_weights_dir / "best.pt").is_file()
        checks["last_checkpoint_exists"] = has_last
        checks["best_checkpoint_exists"] = has_best
        checks["is_already_completed"] = has_best and (self.persistent_experiment_dir / "results.csv").is_file()

        passed = len(errors) == 0
        report = {
            "status": "PASS" if passed else "FAIL",
            "protocol": f"CCARC_{self.config.experiment_variant}",
            "fold_idx": self.config.fold_idx,
            "target_country": self.config.target_country,
            "checks": checks,
            "errors": errors,
            "warnings": warnings,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        self.persistent_experiment_dir.mkdir(parents=True, exist_ok=True)
        with open(self.persistent_experiment_dir / "dry_run_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    def save_reproducibility_metadata(self) -> Path:
        """Record exact runtime parameters, software versions, and hardware info."""
        self.persistent_experiment_dir.mkdir(parents=True, exist_ok=True)
        hw = check_gpu_environment()

        meta = {
            "protocol": f"CCARC_LOCO_{self.config.experiment_variant}",
            "experiment_variant": self.config.experiment_variant,
            "fold_idx": self.config.fold_idx,
            "target_country": self.config.target_country,
            "components": {
                "cbs_enabled": self.config.cbs_enabled,
                "cal_enabled": self.config.cal_enabled,
                "rwc_enabled": self.config.rwc_enabled,
            },
            "cal_hyperparameters": {
                "alpha": self.config.cal_config.alpha,
                "epsilon": self.config.cal_config.epsilon,
                "max_weight": self.config.cal_config.max_weight,
                "min_weight": self.config.cal_config.min_weight,
            },
            "rwc_hyperparameters": {
                "consistency_weight": self.config.rwc_config.consistency_weight,
                "confidence_threshold": self.config.rwc_config.confidence_threshold,
                "temperature": self.config.rwc_config.temperature,
                "warmup_epochs": self.config.rwc_config.warmup_epochs,
            },
            "optimization": {
                "optimizer": self.config.optimizer,
                "lr0": self.config.lr0,
                "lrf": self.config.lrf,
                "momentum": self.config.momentum,
                "weight_decay": self.config.weight_decay,
                "batch_size": self.config.batch_size,
                "epochs": self.config.epochs,
                "seed": self.config.seed,
            },
            "environment": hw,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        meta_path = self.persistent_experiment_dir / "reproducibility_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return meta_path

    def train_stage(self, force_restart: bool = False) -> Dict[str, Any]:
        """Execute Stage 1: Train YOLOv8s under CCARC proposed components."""
        self.persistent_experiment_dir.mkdir(parents=True, exist_ok=True)
        self.local_experiment_dir.mkdir(parents=True, exist_ok=True)

        config_snapshot_path = self.persistent_experiment_dir / "config_snapshot.yaml"
        self.config.save_yaml(config_snapshot_path)
        self.save_reproducibility_metadata()

        results_csv = self.persistent_experiment_dir / "results.csv"
        best_pt = self.persistent_weights_dir / "best.pt"

        if not force_restart and best_pt.is_file() and results_csv.is_file():
            print(f"[STAGE 1] Experiment {self.config.experiment_name} already completed. Skipping training.")
            return {
                "status": "COMPLETED",
                "best_checkpoint": str(self.best_checkpoint),
                "last_checkpoint": str(self.last_checkpoint),
            }

        resume_training = self.config.resume and (
            (self.local_weights_dir / "last.pt").is_file() or (self.persistent_weights_dir / "last.pt").is_file()
        )
        if resume_training:
            self.sync_from_persistent()
            resume_ckpt = self.local_weights_dir / "last.pt"
            print(f"[STAGE 1] Resuming CCARC training from: {resume_ckpt}")
            model = YOLO(str(resume_ckpt))
        else:
            print(f"[STAGE 1] Initializing fresh weights: {self.config.model_name}")
            model = YOLO(self.config.model_name)

        train_args = self.config.to_yolo_train_args()
        if resume_training:
            train_args["resume"] = True

        # Pass CCARC parameters to trainer via train_args and class context
        CCARCDetectionTrainer._ACTIVE_CCARC_CONFIG = self.config
        train_args["trainer"] = CCARCDetectionTrainer
        train_args["ccarc_config"] = self.config
        train_args["cbs_enabled"] = self.config.cbs_enabled
        train_args["cal_enabled"] = self.config.cal_enabled
        train_args["rwc_enabled"] = self.config.rwc_enabled
        train_args["target_country"] = self.config.target_country

        # Epoch synchronization callback
        def on_fit_epoch_end_callback(trainer_obj: Any) -> None:
            self.sync_to_persistent()

        def on_train_epoch_start_callback(trainer_obj: Any) -> None:
            if hasattr(trainer_obj, "cbs_sampler") and trainer_obj.cbs_sampler is not None:
                trainer_obj.cbs_sampler.set_epoch(trainer_obj.epoch)
            raw_m = resolve_detection_model(getattr(trainer_obj, "model", None))
            loss_fn = getattr(raw_m, "loss", None)
            if hasattr(loss_fn, "set_epoch"):
                loss_fn.set_epoch(trainer_obj.epoch)

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end_callback)
        model.add_callback("on_train_epoch_start", on_train_epoch_start_callback)

        start_time = time.perf_counter()
        print(f"[STAGE 1] Starting CCARC ({self.config.experiment_variant}) for Fold {self.config.fold_idx} ({self.config.target_country})...")
        print(f"         CBS: {self.config.cbs_enabled} | CAL: {self.config.cal_enabled} | RWC: {self.config.rwc_enabled}")

        train_results = model.train(**train_args)
        elapsed_sec = time.perf_counter() - start_time

        self.sync_to_persistent()

        summary = {
            "stage": f"STAGE_1_TRAINING_{self.config.experiment_variant}",
            "fold_idx": self.config.fold_idx,
            "target_country": self.config.target_country,
            "elapsed_seconds": round(elapsed_sec, 2),
            "best_checkpoint": str(self.persistent_weights_dir / "best.pt"),
            "last_checkpoint": str(self.persistent_weights_dir / "last.pt"),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(self.persistent_experiment_dir / "train_stage_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def _extract_metrics(self, val_results: Any) -> Dict[str, Any]:
        """Extract standardized detection metrics from Ultralytics validation results.

        Compatible with both modern Ultralytics (>=8.4: box.mp, box.mr, array p/r)
        and legacy Ultralytics (<=8.3: box.p, box.r scalars).
        """
        box = getattr(val_results, "box", None)
        res_dict = getattr(val_results, "results_dict", {})

        p_val = getattr(box, "mp", None)
        if p_val is None:
            p_val = getattr(box, "p", None)
        if p_val is None:
            p_val = res_dict.get("metrics/precision(B)", 0.0)
        elif isinstance(p_val, (list, tuple, np.ndarray)):
            p_val = float(np.mean(p_val)) if len(p_val) > 0 else 0.0
        p = float(p_val)

        r_val = getattr(box, "mr", None)
        if r_val is None:
            r_val = getattr(box, "r", None)
        if r_val is None:
            r_val = res_dict.get("metrics/recall(B)", 0.0)
        elif isinstance(r_val, (list, tuple, np.ndarray)):
            r_val = float(np.mean(r_val)) if len(r_val) > 0 else 0.0
        r = float(r_val)

        map50_val = getattr(box, "map50", None)
        if map50_val is None:
            map50_val = res_dict.get("metrics/mAP50(B)", 0.0)
        map50 = float(map50_val)

        map_val = getattr(box, "map", None)
        if map_val is None:
            map_val = getattr(box, "map50_95", None)
        if map_val is None:
            map_val = res_dict.get("metrics/mAP50-95(B)", 0.0)
        map50_95 = float(map_val)

        fitness = float(getattr(val_results, "fitness", 0.1 * map50 + 0.9 * map50_95))

        per_class: Dict[str, Dict[str, float]] = {}
        maps_arr = getattr(box, "maps", None)
        ap50_arr = getattr(box, "ap50", None)

        for idx, cls_name in enumerate(self.config.target_classes):
            cls_ap50 = float(ap50_arr[idx]) if (ap50_arr is not None and len(ap50_arr) > idx) else 0.0
            cls_map = float(maps_arr[idx]) if (maps_arr is not None and len(maps_arr) > idx) else 0.0
            per_class[cls_name] = {"ap50": round(cls_ap50, 4), "map50_95": round(cls_map, 4)}

        return {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
            "fitness": round(fitness, 4),
            "per_class": per_class,
        }

    def evaluate_source_val_stage(self, checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
        """Stage 2: Evaluate model checkpoint on source validation data."""
        ckpt = Path(checkpoint_path).resolve() if checkpoint_path else (self.persistent_weights_dir / "best.pt")
        if not ckpt.is_file():
            ckpt = self.local_weights_dir / "best.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found for validation: {ckpt}")

        print(f"[STAGE 2] Evaluating on SOURCE VALIDATION: {ckpt}")
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
            "variant": self.config.experiment_variant,
            "fold_idx": self.config.fold_idx,
            "target_country": self.config.target_country,
            "split": "val",
            "checkpoint": str(ckpt),
            "metrics": metrics,
            "elapsed_seconds": round(elapsed_sec, 2),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(self.persistent_experiment_dir / "source_val_metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def evaluate_held_out_target_stage(self, checkpoint_path: Optional[Path] = None) -> Dict[str, Any]:
        """Stage 3: Explicitly evaluate model on unseen held-out target country."""
        ckpt = Path(checkpoint_path).resolve() if checkpoint_path else (self.persistent_weights_dir / "best.pt")
        if not ckpt.is_file():
            ckpt = self.local_weights_dir / "best.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found for held-out evaluation: {ckpt}")

        print(f"[STAGE 3] Explicitly evaluating on HELD-OUT TARGET ({self.config.target_country}): {ckpt}")
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
            "variant": self.config.experiment_variant,
            "fold_idx": self.config.fold_idx,
            "target_country": self.config.target_country,
            "split": "test",
            "checkpoint": str(ckpt),
            "metrics": metrics,
            "elapsed_seconds": round(elapsed_sec, 2),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(self.persistent_experiment_dir / "held_out_test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def run_fold(
        self,
        dry_run: bool = False,
        eval_target: bool = False,
        eval_only: bool = False,
        force_restart: bool = False,
    ) -> Dict[str, Any]:
        """Run requested stages for a single LOCO fold under CCARC."""
        dry_run_report = self.dry_run()
        if dry_run or dry_run_report["status"] != "PASS":
            return {"dry_run_report": dry_run_report}

        if not eval_only:
            train_summary = self.train_stage(force_restart=force_restart)
        else:
            train_summary = {"status": "SKIPPED_EVAL_ONLY"}

        val_summary = self.evaluate_source_val_stage()

        if eval_target:
            test_summary = self.evaluate_held_out_target_stage()
        else:
            test_summary = {
                "status": "SKIPPED_NOT_REQUESTED",
                "message": "Held-out target evaluation not requested. Target data remained isolated.",
            }

        master_summary = {
            "protocol": f"CCARC_{self.config.experiment_variant}",
            "variant": self.config.experiment_variant,
            "fold_idx": self.config.fold_idx,
            "target_country": self.config.target_country,
            "dry_run": dry_run_report,
            "stage_1_train": train_summary,
            "stage_2_source_val": val_summary,
            "stage_3_held_out_target": test_summary,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(self.persistent_experiment_dir / "fold_master_summary.json", "w", encoding="utf-8") as f:
            json.dump(master_summary, f, indent=2)

        return master_summary

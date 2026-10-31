"""CCARC Experiment Configuration for YOLOv8s Cross-Country Training.

Defines unified configurations for:
- E0: Baseline (Standard YOLOv8s)
- E1: Country-Balanced Sampling (CBS only)
- E2: Class-Aware Learning (CAL only)
- E3: Reliability-Weighted Consistency (RWC only)
- E4: Full CCARC (CBS + CAL + RWC)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import ultralytics
import yaml

from src.data.loco_splitter import DEFAULT_TARGET_CLASSES, LOCO_COUNTRIES
from src.models.ccarc_consistency import ReliabilityConsistencyConfig
from src.models.ccarc_loss import ClassAwareLearningConfig
from src.models.ccarc_perturbations import PhotometricPerturbationConfig
from src.training.baseline_config import (
    BaselineTrainingConfig,
    check_gpu_environment,
    resolve_fold_data_yaml,
)

VALID_VARIANTS: Tuple[str, ...] = ("E0", "E1", "E2", "E3", "E4")


@dataclass
class CCARCTrainingConfig:
    """Experiment configuration for CCARC proposed methods and ablations."""

    # 1. Experiment Variant and Component Toggles
    experiment_variant: str = "E4"  # "E0", "E1", "E2", "E3", "E4"
    cbs_enabled: bool = True  # Country-Balanced Sampling
    cal_enabled: bool = True  # Class-Aware Learning
    rwc_enabled: bool = True  # Reliability-Weighted Consistency

    # 2. Sub-Configurations for Proposed Components
    cal_config: ClassAwareLearningConfig = field(default_factory=ClassAwareLearningConfig)
    rwc_config: ReliabilityConsistencyConfig = field(default_factory=ReliabilityConsistencyConfig)
    perturb_config: PhotometricPerturbationConfig = field(default_factory=PhotometricPerturbationConfig)

    # 3. Model Architecture & Base Hyperparameters (Strictly aligned with E0 baseline)
    model_name: str = "yolov8s.pt"
    img_size: int = 640
    pretrained: bool = True
    num_classes: int = 4
    target_classes: Tuple[str, ...] = DEFAULT_TARGET_CLASSES

    # 4. Optimization Schedule (Strictly matches Baseline SGD)
    epochs: int = 100
    batch_size: int = 16
    patience: int = 20
    optimizer: str = "SGD"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    seed: int = 42
    deterministic: bool = True

    # 5. Baseline Augmentations (Preserved from Baseline)
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0

    # 6. Hardware & Environment
    device: str = "auto"
    workers: int = 4
    amp: bool = True
    exist_ok: bool = True

    # 7. Experiment & Storage Paths (Never overwrites E0 baseline directory)
    fold_idx: int = 0
    target_country: str = "China"
    data_yaml_path: str = ""
    project_dir: str = "experiments/ccarc_experiments"  # Dedicated CCARC directory
    local_run_dir: str = ""  # Local fast SSD (e.g. /content/runs/ccarc on Colab)
    experiment_name: str = ""
    resume: bool = False
    save: bool = True
    save_period: int = -1

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ultralytics_version: str = field(default_factory=lambda: str(ultralytics.__version__))
    torch_version: str = field(default_factory=lambda: str(torch.__version__))

    def __post_init__(self) -> None:
        """Enforce component toggles and naming based on experiment variant."""
        self.apply_variant(self.experiment_variant)
        if not self.experiment_name:
            self.experiment_name = f"fold_{self.fold_idx}_{self.target_country}_{self.experiment_variant}"

    def apply_variant(self, variant: str) -> None:
        """Apply component toggle settings corresponding to ablation variant."""
        variant_upper = variant.upper().strip()
        if variant_upper not in VALID_VARIANTS:
            raise ValueError(f"Invalid experiment_variant '{variant}'. Must be one of {VALID_VARIANTS}.")

        self.experiment_variant = variant_upper
        if variant_upper == "E0":
            # Baseline equivalent
            self.cbs_enabled = False
            self.cal_enabled = False
            self.rwc_enabled = False
        elif variant_upper == "E1":
            # CBS only
            self.cbs_enabled = True
            self.cal_enabled = False
            self.rwc_enabled = False
        elif variant_upper == "E2":
            # CAL only
            self.cbs_enabled = False
            self.cal_enabled = True
            self.rwc_enabled = False
        elif variant_upper == "E3":
            # RWC only
            self.cbs_enabled = False
            self.cal_enabled = False
            self.rwc_enabled = True
        elif variant_upper == "E4":
            # Full CCARC
            self.cbs_enabled = True
            self.cal_enabled = True
            self.rwc_enabled = True

        # Synchronize sub-config enabled flags
        self.cal_config.enabled = self.cal_enabled
        self.rwc_config.enabled = self.rwc_enabled

    def resolve_device(self) -> str:
        """Resolve device string to torch / ultralytics device identifier."""
        if self.device == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        if self.device in ("cuda", "cuda:0") and not torch.cuda.is_available():
            return "cpu"
        return self.device

    def to_yolo_train_args(self) -> Dict[str, Any]:
        """Convert configuration into kwargs expected by Ultralytics model.train()."""
        active_project = self.local_run_dir if self.local_run_dir else self.project_dir
        return {
            "data": str(Path(self.data_yaml_path).resolve()) if self.data_yaml_path else "",
            "epochs": self.epochs,
            "batch": self.batch_size,
            "imgsz": self.img_size,
            "device": self.resolve_device(),
            "workers": self.workers,
            "patience": self.patience,
            "save": self.save,
            "save_period": self.save_period,
            "project": str(Path(active_project).resolve()),
            "name": self.experiment_name,
            "exist_ok": self.exist_ok,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "warmup_momentum": self.warmup_momentum,
            "warmup_bias_lr": self.warmup_bias_lr,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "amp": self.amp,
            "val": True,
            "plots": True,
            "hsv_h": self.hsv_h,
            "hsv_s": self.hsv_s,
            "hsv_v": self.hsv_v,
            "degrees": self.degrees,
            "translate": self.translate,
            "scale": self.scale,
            "shear": self.shear,
            "perspective": self.perspective,
            "flipud": self.flipud,
            "fliplr": self.fliplr,
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "copy_paste": self.copy_paste,
        }

    def save_yaml(self, path: Union[str, Path]) -> None:
        """Serialize configuration snapshot to YAML file."""
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        raw_dict = asdict(self)
        raw_dict["torch_version"] = str(self.torch_version)
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(raw_dict, f, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> CCARCTrainingConfig:
        """Load configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        cal_cfg = ClassAwareLearningConfig(**data.pop("cal_config", {}))
        rwc_cfg = ReliabilityConsistencyConfig(**data.pop("rwc_config", {}))
        perturb_cfg = PhotometricPerturbationConfig(**data.pop("perturb_config", {}))

        return cls(
            cal_config=cal_cfg,
            rwc_config=rwc_cfg,
            perturb_config=perturb_cfg,
            **data,
        )

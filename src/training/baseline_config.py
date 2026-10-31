"""Reproducible YOLOv8s Baseline Training Configuration for CCARC LOCO Experiments.

Defines standardized hyperparameters, hardware environment inspection,
dataclass configuration, and serialization for six-fold cross-country experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
from typing import Any, Dict, List, Optional, Tuple

import torch
import ultralytics
import yaml

from src.data.loco_splitter import DEFAULT_TARGET_CLASSES, LOCO_COUNTRIES


@dataclass
class BaselineTrainingConfig:
    """Standardized configuration for YOLOv8s source-only cross-country baseline.

    Scientific Methodology Notes:
    - Optimizer: Explicitly configured as 'SGD' (momentum=0.937, weight_decay=0.0005, lr0=0.01,
      lrf=0.01) to ensure deterministic optimization dynamics across all folds.
    - Frozen Baseline Augmentations: Uses standard Ultralytics augmentations (flips, scale,
      translate, mosaic) without proposed CCARC road-specific consistency perturbations
      (e.g., weather shifts, asphalt texture perturbations, shadow overlays), which are
      reserved for Phase 3 method experiments.
    - Limitation: Quantitative LOCO test sets currently consist strictly of verified
      target-positive images (D00, D10, D20, D40); background negative images (empty_xml_candidate)
      are excluded by default to avoid false-negative label noise. Full unconstrained
      false-positive rates on undamaged pavement are not measured under this protocol.
    """

    # 1. Model architecture and weights
    model_name: str = "yolov8s.pt"
    img_size: int = 640
    pretrained: bool = True
    num_classes: int = 4
    target_classes: Tuple[str, ...] = DEFAULT_TARGET_CLASSES

    # 2. Optimization & Training Schedule (Explicit SGD baseline)
    epochs: int = 100
    batch_size: int = 16
    patience: int = 20  # Early stopping epochs on source validation fitness
    optimizer: str = "SGD"  # Explicitly 'SGD' to guarantee strict lr0, momentum and weight decay
    lr0: float = 0.01  # Initial learning rate
    lrf: float = 0.01  # Final learning rate fraction (lr0 * lrf)
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    seed: int = 42
    deterministic: bool = True

    # 3. Frozen Standard Augmentations (Preserves canonical camera pitch)
    hsv_h: float = 0.015  # Hue augmentation
    hsv_s: float = 0.7    # Saturation augmentation
    hsv_v: float = 0.4    # Value (brightness) augmentation
    degrees: float = 0.0  # Rotation (+/- deg)
    translate: float = 0.1  # Translation (+/- fraction)
    scale: float = 0.5    # Image scale (+/- gain)
    shear: float = 0.0    # Shear (+/- deg)
    perspective: float = 0.0
    flipud: float = 0.0   # Vertical flip disabled (preserves road driving perspective)
    fliplr: float = 0.5   # Horizontal flip
    mosaic: float = 1.0   # Standard Mosaic augmentation
    mixup: float = 0.0    # Mixup disabled in standard baseline
    copy_paste: float = 0.0

    # 4. Hardware and Runtime Settings
    device: str = "auto"  # 'auto', '0', 'cuda', 'cpu'
    workers: int = 4
    amp: bool = True      # Automatic Mixed Precision
    exist_ok: bool = True

    # 5. Experiment & LOCO Specifics
    fold_idx: int = 0
    target_country: str = "China"
    data_yaml_path: str = ""
    project_dir: str = "experiments/yolov8s_baseline"  # Persistent Google Drive storage
    local_run_dir: str = ""  # Optional fast local storage (e.g. /content/runs on Colab)
    experiment_name: str = "fold_0_China"
    resume: bool = False
    save: bool = True
    save_period: int = -1  # Only keep best.pt and last.pt

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ultralytics_version: str = field(default_factory=lambda: str(ultralytics.__version__))
    torch_version: str = field(default_factory=lambda: str(torch.__version__))

    def resolve_device(self) -> str:
        """Resolve device string to valid torch/ultralytics device."""
        if self.device == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        if self.device in ("cuda", "cuda:0") and not torch.cuda.is_available():
            return "cpu"
        return self.device

    def to_yolo_train_args(self) -> Dict[str, Any]:
        """Convert configuration into kwargs expected by Ultralytics model.train()."""
        active_project = self.local_run_dir if self.local_run_dir else self.project_dir
        return {
            "data": str(Path(self.data_yaml_path).resolve()),
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
            "pretrained": self.pretrained,
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
            "amp": self.amp,
            "plots": True,
            "val": True,  # Evaluates on source val split at each epoch
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        d = asdict(self)
        d["target_classes"] = list(self.target_classes)
        d["torch_version"] = str(self.torch_version)
        d["ultralytics_version"] = str(self.ultralytics_version)
        return d

    def save_yaml(self, path: Path) -> Path:
        """Save configuration snapshot to YAML file."""
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)
        return path

    @classmethod
    def from_yaml(cls, path: Path) -> BaselineTrainingConfig:
        """Load configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if "target_classes" in data:
            data["target_classes"] = tuple(data["target_classes"])
        return cls(**data)


def resolve_fold_data_yaml(views_dir: Path, fold_idx: int, target_country: str) -> Optional[Path]:
    """Resolve data.yaml for a given fold index and country across common directory conventions."""
    candidates = [
        views_dir / f"fold_{fold_idx}_{target_country}" / "data.yaml",
        views_dir / f"fold_{target_country}" / "data.yaml",
        views_dir / f"fold_{fold_idx}" / "data.yaml",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def check_gpu_environment() -> Dict[str, Any]:
    """Inspect and report the hardware execution environment (e.g. Colab NVIDIA L4)."""
    cuda_available = torch.cuda.is_available()
    gpu_info: List[Dict[str, Any]] = []

    if cuda_available:
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            vram_gb = round(props.total_memory / (1024**3), 2)
            gpu_info.append({
                "device_index": idx,
                "name": props.name,
                "total_memory_gb": vram_gb,
                "major_capability": props.major,
                "minor_capability": props.minor,
                "multi_processor_count": props.multi_processor_count,
            })

    return {
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "gpus": gpu_info,
        "pytorch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "os_platform": platform.platform(),
        "python_version": platform.python_version(),
    }

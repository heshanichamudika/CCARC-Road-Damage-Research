"""Class-Aware Learning (CAL) Loss Components for CCARC.

Implements class-frequency-aware classification loss reweighting for YOLOv8s
across the four target RDD2022 damage classes (D00, D10, D20, D40).
Calculates class frequencies strictly from source-country training data (zero
leakage from validation or test splits), normalizes weights to unit mean,
and integrates cleanly with Ultralytics v8DetectionLoss.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.loco_splitter import DEFAULT_TARGET_CLASSES
from src.data.yolo_converter import TARGET_CLASS_TO_ID

try:
    from ultralytics.utils.loss import v8DetectionLoss
except ImportError:
    # Graceful fallback if ultralytics loss is not available in isolated test environments
    v8DetectionLoss = object  # type: ignore[misc,assignment]


@dataclass
class ClassAwareLearningConfig:
    """Configuration for Class-Aware Learning (CAL) loss reweighting.

    Formula:
        raw_weight[c] = (frequency[c] + epsilon) ** (-alpha)
        weight[c] = raw_weight[c] / mean(raw_weight)
    """

    enabled: bool = True
    alpha: float = 0.5  # Power factor in [0.0, 1.0]. 0.0 = uniform (unweighted), 0.5 = sqrt inverse, 1.0 = inverse
    epsilon: float = 1.0  # Numerical stability constant (default 1.0 for integer counts)
    max_weight: Optional[float] = 10.0  # Upper bound on class weights to prevent gradient explosions
    min_weight: Optional[float] = 0.1  # Lower bound on class weights
    frequency_type: str = "count"  # "count" (raw instances) or "proportion" (fraction of total)
    zero_freq_policy: str = "max"  # Policy for classes with 0 instances: 'max', 'neutral', 'epsilon'
    normalize_mean: bool = True  # Normalize weights such that mean(weights) == 1.0
    target_classes: Tuple[str, ...] = DEFAULT_TARGET_CLASSES


def compute_class_weights(
    class_counts: Union[Dict[str, int], Dict[int, int], Sequence[int], np.ndarray, torch.Tensor],
    config: Optional[ClassAwareLearningConfig] = None,
    target_classes: Sequence[str] = DEFAULT_TARGET_CLASSES,
) -> torch.Tensor:
    """Compute normalized class weights from class frequencies.

    Args:
        class_counts: Raw counts per class as a dictionary (name or ID) or sequence.
        config: ClassAwareLearningConfig instance. If None, default config is used.
        target_classes: Ordered list of target class names.

    Returns:
        1D torch.FloatTensor of shape (num_classes,) with mean == 1.0 (if normalized).
    """
    if config is None:
        config = ClassAwareLearningConfig()

    num_classes = len(target_classes)

    # Convert class_counts to ordered numpy array [count_0, count_1, ..., count_{C-1}]
    counts_arr = np.zeros(num_classes, dtype=np.float64)

    if isinstance(class_counts, dict):
        for k, v in class_counts.items():
            if isinstance(k, str) and k in target_classes:
                cls_id = TARGET_CLASS_TO_ID.get(k, target_classes.index(k))
                counts_arr[cls_id] = float(v)
            elif isinstance(k, (int, np.integer)) and 0 <= k < num_classes:
                counts_arr[int(k)] = float(v)
    elif isinstance(class_counts, (list, tuple, np.ndarray, torch.Tensor)):
        if isinstance(class_counts, torch.Tensor):
            vals = class_counts.detach().cpu().numpy().tolist()
        else:
            vals = list(class_counts)
        for i, v in enumerate(vals[:num_classes]):
            counts_arr[i] = float(v)
    else:
        raise TypeError(f"Unsupported class_counts type: {type(class_counts)}")

    # If CAL is disabled or alpha == 0.0, return uniform unit weights
    if not config.enabled or config.alpha == 0.0:
        return torch.ones(num_classes, dtype=torch.float32)

    # Determine base frequencies
    if config.frequency_type == "proportion":
        total_instances = counts_arr.sum()
        if total_instances > 0:
            freqs = counts_arr / total_instances
            eps = config.epsilon / total_instances if config.epsilon >= 1.0 else config.epsilon
        else:
            freqs = np.zeros_like(counts_arr)
            eps = config.epsilon
    else:
        freqs = counts_arr.copy()
        eps = config.epsilon

    # Compute raw inverse-frequency weights: (freq + eps) ** (-alpha)
    raw_weights = np.zeros(num_classes, dtype=np.float64)
    zero_mask = freqs <= 0.0
    non_zero_mask = ~zero_mask

    if non_zero_mask.any():
        raw_weights[non_zero_mask] = (freqs[non_zero_mask] + eps) ** (-config.alpha)

    # Handle zero-frequency classes safely
    if zero_mask.any():
        if config.zero_freq_policy == "max":
            max_observed = raw_weights[non_zero_mask].max() if non_zero_mask.any() else 1.0
            raw_weights[zero_mask] = max_observed
        elif config.zero_freq_policy == "neutral":
            raw_weights[zero_mask] = 1.0
        elif config.zero_freq_policy == "epsilon":
            raw_weights[zero_mask] = (eps) ** (-config.alpha)
        else:
            raise ValueError(f"Unknown zero_freq_policy: '{config.zero_freq_policy}'")

    # Normalize weights so that mean(weights) == 1.0
    if config.normalize_mean:
        mean_val = raw_weights.mean()
        if mean_val > 0:
            weights = raw_weights / mean_val
        else:
            weights = np.ones(num_classes, dtype=np.float64)
    else:
        weights = raw_weights

    # Apply clipping bounds to relative normalized weights (1.0 = neutral)
    if config.max_weight is not None:
        weights = np.minimum(weights, config.max_weight)
    if config.min_weight is not None:
        weights = np.maximum(weights, config.min_weight)

    # Re-normalize so that mean(weights) strictly equals 1.0 after clipping
    if config.normalize_mean and (config.max_weight is not None or config.min_weight is not None):
        mean_val = weights.mean()
        if mean_val > 0:
            weights = weights / mean_val

    return torch.tensor(weights, dtype=torch.float32)


def extract_source_train_class_counts(
    source_input: Union[str, Path, pd.DataFrame, Dict[str, Any]],
    target_classes: Sequence[str] = DEFAULT_TARGET_CLASSES,
) -> Dict[str, int]:
    """Extract class frequencies strictly from source training annotations.

    Guarantees zero validation-set or test-set leakage by only extracting counts
    associated with the 'train' split.

    Args:
        source_input: Path to fold_summary.json, fold_manifest.csv, labels/train directory,
            or loaded DataFrame / dictionary.
        target_classes: Expected class names.

    Returns:
        Dictionary mapping class name (e.g. 'D00') to integer instance count in train split.
    """
    counts = {cls_name: 0 for cls_name in target_classes}

    # 1. From fold_summary.json dict or path
    if isinstance(source_input, (str, Path)) and str(source_input).endswith(".json"):
        with open(source_input, "r", encoding="utf-8") as f:
            data = json.load(f)
        source_input = data

    if isinstance(source_input, dict):
        if "class_counts" in source_input and "train" in source_input["class_counts"]:
            train_counts = source_input["class_counts"]["train"]
            for k, v in train_counts.items():
                if k in counts:
                    counts[k] = int(v)
            return counts
        if "counts" in source_input and "train" in source_input.get("class_counts", {}):
            train_counts = source_input["class_counts"]["train"]
            for k, v in train_counts.items():
                if k in counts:
                    counts[k] = int(v)
            return counts
        # Direct class count dict
        if any(cls_name in source_input for cls_name in target_classes):
            for k, v in source_input.items():
                if k in counts:
                    counts[k] = int(v)
            return counts

    # 2. From fold_manifest.csv
    if isinstance(source_input, (str, Path)) and str(source_input).endswith(".csv"):
        df = pd.read_csv(source_input)
    elif isinstance(source_input, pd.DataFrame):
        df = source_input
    else:
        df = None

    if df is not None:
        if "loco_split" not in df.columns:
            raise ValueError("Manifest DataFrame must contain 'loco_split' column to isolate train data.")

        train_df = df[df["loco_split"] == "train"]
        for _, row in train_df.iterrows():
            cc = row.get("class_counts")
            if pd.notna(cc):
                if isinstance(cc, str):
                    try:
                        cc = json.loads(cc.replace("'", '"'))
                    except Exception:
                        cc = {}
                if isinstance(cc, dict):
                    for k, v in cc.items():
                        if k in counts:
                            counts[k] += int(v)
        return counts

    # 3. From labels/train directory containing *.txt YOLO label files
    if isinstance(source_input, (str, Path)):
        p = Path(source_input)
        if p.is_dir():
            # If a base view dir is passed, look for 'labels/train'
            lbl_dir = p / "labels" / "train" if (p / "labels" / "train").is_dir() else p
            for txt_file in lbl_dir.glob("*.txt"):
                try:
                    content = txt_file.read_text(encoding="utf-8").strip()
                    if not content:
                        continue
                    for line in content.splitlines():
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            if 0 <= cls_id < len(target_classes):
                                cls_name = target_classes[cls_id]
                                counts[cls_name] += 1
                except (OSError, ValueError):
                    continue
            return counts

    raise ValueError(f"Unable to extract source training class counts from: {source_input}")


class ClassWeightedBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss with class-specific weight scaling.

    Scales each class binary cross-entropy loss by its corresponding class weight:
        loss[..., c] = weight[c] * BCEWithLogits(pred[..., c], target[..., c])
    """

    def __init__(self, class_weights: Optional[torch.Tensor] = None) -> None:
        """Initialize loss with optional class weights."""
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def set_class_weights(self, class_weights: Optional[torch.Tensor]) -> None:
        """Update or register class weights tensor."""
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute class-weighted BCE loss.

        Args:
            pred: Predicted logits of shape (batch, anchors, num_classes) or (N, num_classes).
            target: Ground truth scores in [0, 1] matching pred shape.

        Returns:
            Unreduced element-wise loss tensor matching pred shape.
        """
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

        if self.class_weights is not None:
            w = self.class_weights.to(device=pred.device, dtype=pred.dtype)
            # Reshape for broadcasting
            if pred.ndim == 3 and w.ndim == 1:
                w = w.view(1, 1, -1)
            elif pred.ndim == 2 and w.ndim == 1:
                w = w.view(1, -1)
            loss = loss * w

        return loss


class CCARCDetectionLoss(v8DetectionLoss):
    """Custom YOLOv8 Detection Loss with Class-Aware Learning (CAL).

    Inherits from Ultralytics v8DetectionLoss, preserving standard bounding-box
    (CIoU) and distribution focal loss (DFL) mechanics while injecting class-frequency
    reweighting into the classification loss branch.
    """

    def __init__(
        self,
        model: Any,
        class_weights: Optional[torch.Tensor] = None,
        tal_topk: int = 10,
    ) -> None:
        """Initialize CCARC Detection Loss.

        Args:
            model: Ultralytics DetectionModel instance.
            class_weights: Precomputed class weight tensor of shape (num_classes,).
            tal_topk: Top-K candidates for TaskAlignedAssigner.
        """
        super().__init__(model, tal_topk=tal_topk)
        self.class_weights = class_weights.float() if class_weights is not None else None
        # Replace unweighted BCE with ClassWeightedBCEWithLogitsLoss
        self.bce = ClassWeightedBCEWithLogitsLoss(self.class_weights)

    def set_class_weights(self, class_weights: Optional[torch.Tensor]) -> None:
        """Dynamically update class weights."""
        self.class_weights = class_weights.float() if class_weights is not None else None
        self.bce.set_class_weights(self.class_weights)


def resolve_detection_model(model: Any) -> Any:
    """Resolve genuine Ultralytics DetectionModel from YOLO wrapper, trainer, or DDP module.

    Ensures that an internal torch.nn.Sequential (e.g. DetectionModel.model) is NEVER
    returned as the detection model.

    Args:
        model: A YOLO object, BaseTrainer object, DetectionModel, DDP/DP wrapper, or module.

    Returns:
        The underlying DetectionModel instance.
    """
    # 1. If trainer object passed (has .model and .validator or .args)
    if hasattr(model, "model") and hasattr(model, "validator") and hasattr(model, "args"):
        model = model.model

    # 2. Unwrap DistributedDataParallel / DataParallel
    if hasattr(model, "module"):
        model = model.module

    # 3. If YOLO wrapper (has .model, but model.model is NOT Sequential)
    # Note: DetectionModel also has .model, but its .model is torch.nn.Sequential!
    if hasattr(model, "model") and not isinstance(model.model, torch.nn.Sequential):
        model = model.model
        if hasattr(model, "module"):
            model = model.module

    # 4. Final safety check: if model is torch.nn.Sequential, raise an informative TypeError
    if isinstance(model, torch.nn.Sequential):
        raise TypeError(
            f"Expected Ultralytics DetectionModel, but received torch.nn.Sequential. "
            f"Cannot operate on raw layer container without DetectionModel attributes (args, criterion, loss)."
        )

    return model


def attach_class_aware_loss(
    model: Any,
    class_weights: Optional[torch.Tensor],
    tal_topk: int = 10,
) -> Any:
    """Safely attach Class-Aware Learning loss to an Ultralytics model.

    Works across Ultralytics versions by replacing the criterion's BCE module
    or initializing CCARCDetectionLoss.

    Args:
        model: Ultralytics YOLO, DetectionTrainer, or DetectionModel instance.
        class_weights: Tensor of class weights of shape (num_classes,).
        tal_topk: TaskAlignedAssigner top-k parameter.

    Returns:
        The updated loss criterion attached to model.criterion.
    """
    det_model = resolve_detection_model(model)

    # Ensure model has args attribute for v8DetectionLoss if missing
    if getattr(det_model, "args", None) is None:
        det_model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    # 1. If model already has a criterion with .bce attribute, inject ClassWeightedBCEWithLogitsLoss
    if hasattr(det_model, "criterion") and getattr(det_model, "criterion", None) is not None:
        if hasattr(det_model.criterion, "bce"):
            det_model.criterion.bce = ClassWeightedBCEWithLogitsLoss(class_weights)
            return det_model.criterion

    # 2. Otherwise instantiate CCARCDetectionLoss
    criterion = CCARCDetectionLoss(det_model, class_weights=class_weights, tal_topk=tal_topk)
    det_model.criterion = criterion
    return criterion

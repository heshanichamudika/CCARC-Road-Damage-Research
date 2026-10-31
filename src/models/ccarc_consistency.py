"""Reliability-Aware Consistency Learning (RWC) for CCARC Cross-Country Training.

Implements a differentiable teacher/student consistency regularizer for YOLOv8s.
Enforces invariant detection predictions between weakly and strongly augmented
photometric views (x_w, x_s) of the same road surface scene.

Key Features:
- Spatially Aligned Anchors: Compares dense grid predictions directly across identical
  spatial anchor locations (1:1 correspondence) before NMS, avoiding ordering ambiguity.
- Teacher Detachment: Teacher predictions (from weak view) are strictly detached from
  autograd gradients (.detach()), acting as pseudo-targets.
- Calibrated Reliability Filtering: Confidence is measured from calibrated probabilities
  (via sigmoid and temperature scaling), never raw logits. Predictions below
  tau_rel are excluded to eliminate spurious pseudo-targets.
- Differentiable Zero: Returns 0.0 * student.sum() when no prediction passes the
  reliability threshold, ensuring numerical stability and valid autograd graphs.
- Normalized Loss: Scaled by the effective reliability weight sum.
- Epoch Warm-Up: Supports gradual ramp-up of consistency weight over initial epochs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.loco_splitter import DEFAULT_TARGET_CLASSES


@dataclass
class ReliabilityConsistencyConfig:
    """Hyperparameter configuration for Reliability-Aware Consistency Learning."""

    enabled: bool = True
    consistency_weight: float = 1.0  # Overall scaling weight (lambda_con)
    confidence_threshold: float = 0.25  # Reliability cutoff threshold (tau_rel)
    temperature: float = 1.0  # Temperature scaling factor for teacher probabilities
    loss_type: str = "bce"  # 'bce' (soft binary cross-entropy), 'mse', or 'kl'
    weighting_mode: str = "confidence"  # 'confidence', 'binary', or 'power'
    power_gamma: float = 2.0  # Exponent when weighting_mode == 'power'
    warmup_epochs: int = 5  # Number of initial epochs to linearly ramp up consistency weight
    epsilon: float = 1e-7  # Numerical stability constant
    num_classes: int = 4  # Target damage classes (D00, D10, D20, D40)
    reg_max: int = 16  # YOLOv8 DFL coordinate bins per edge


class ReliabilityConsistencyLoss(nn.Module):
    """Reliability-Weighted Prediction Consistency Loss for YOLOv8s."""

    def __init__(self, config: Optional[ReliabilityConsistencyConfig] = None) -> None:
        """Initialize consistency loss regularizer."""
        super().__init__()
        self.config = config or ReliabilityConsistencyConfig()
        self.num_classes = self.config.num_classes
        self.reg_max = self.config.reg_max
        self.no = self.reg_max * 4 + self.num_classes

    def get_effective_weight(self, current_epoch: int) -> float:
        """Calculate effective consistency weight considering warmup schedule."""
        if not self.config.enabled:
            return 0.0

        if self.config.warmup_epochs <= 0:
            return self.config.consistency_weight

        # Linear ramp-up from epoch 0 to warmup_epochs
        warmup_factor = min(1.0, (current_epoch + 1) / float(self.config.warmup_epochs))
        return self.config.consistency_weight * warmup_factor

    def unpack_predictions(
        self,
        preds: Union[torch.Tensor, List[torch.Tensor], Tuple[Any, ...]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract classification logits and optional bbox distribution from YOLO output.

        Handles:
        1. List of multi-stride feature tensors from YOLOv8 detect head:
           [(B, 68, 80, 80), (B, 68, 40, 40), (B, 68, 20, 20)]
        2. Permuted tensor of shape (B, num_anchors, num_classes) or (B, num_anchors, no).

        Returns:
            Tuple of:
                - pred_scores: (B, num_anchors, num_classes) classification logits.
                - pred_distri: Optional (B, num_anchors, 64) DFL bbox distribution.
        """
        feats = preds[1] if (isinstance(preds, tuple) and len(preds) > 1 and isinstance(preds[1], (list, tuple))) else preds

        if isinstance(feats, (list, tuple)):
            # Multiscale feature maps from YOLOv8 detection head
            batch_size = feats[0].shape[0]
            # Flatten spatial dimensions and concatenate across strides: (B, no, total_anchors)
            concat_feats = torch.cat([xi.view(batch_size, self.no, -1) for xi in feats], dim=2)
            # Split channels into bbox DFL (reg_max * 4) and classification logits (nc)
            pred_distri, pred_scores = concat_feats.split((self.reg_max * 4, self.num_classes), dim=1)
            pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (B, total_anchors, nc)
            pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # (B, total_anchors, 64)
            return pred_scores, pred_distri

        if isinstance(preds, torch.Tensor):
            if preds.shape[-1] == self.num_classes:
                return preds, None
            if preds.shape[-1] == self.no:
                pred_distri, pred_scores = preds.split((self.reg_max * 4, self.num_classes), dim=-1)
                return pred_scores, pred_distri
            if preds.shape[1] == self.no and preds.ndim == 3:
                # Shape (B, no, num_anchors)
                pred_distri, pred_scores = preds.split((self.reg_max * 4, self.num_classes), dim=1)
                return pred_scores.permute(0, 2, 1).contiguous(), pred_distri.permute(0, 2, 1).contiguous()

        raise ValueError(
            f"Unable to unpack predictions of type {type(preds)} and shape "
            f"{getattr(preds, 'shape', [getattr(p, 'shape', None) for p in preds] if isinstance(preds, (list, tuple)) else None)}"
        )

    def forward(
        self,
        teacher_preds: Any,
        student_preds: Any,
        current_epoch: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """Compute reliability-weighted consistency loss between teacher and student.

        Args:
            teacher_preds: Weak-view predictions (will be detached).
            student_preds: Strong-view predictions (maintains gradients).
            current_epoch: Active training epoch for warmup scaling.

        Returns:
            Dictionary containing:
                - 'loss': Weighted consistency scalar loss tensor.
                - 'raw_loss': Unscaled consistency loss.
                - 'reliable_anchor_count': Number of anchors exceeding confidence threshold.
                - 'effective_weight': Applied consistency weight lambda.
        """
        # Unpack predictions
        teacher_scores, _ = self.unpack_predictions(teacher_preds)
        student_scores, _ = self.unpack_predictions(student_preds)

        # 1. Strictly detach teacher branch to act as pure pseudo-targets
        teacher_scores = teacher_scores.detach()

        # Check alignment
        if teacher_scores.shape != student_scores.shape:
            raise ValueError(
                f"Teacher and student prediction shapes must match for 1:1 spatial alignment. "
                f"Got teacher: {teacher_scores.shape}, student: {student_scores.shape}"
            )

        # Calculate effective weight for this epoch
        eff_weight = self.get_effective_weight(current_epoch)
        if not self.config.enabled or eff_weight <= 0.0:
            zero_loss = 0.0 * student_scores.sum()
            return {
                "loss": zero_loss,
                "raw_loss": zero_loss.detach(),
                "reliable_anchor_count": torch.tensor(0, device=student_scores.device),
                "effective_weight": torch.tensor(eff_weight, device=student_scores.device),
            }

        cfg = self.config

        # 2. Calibrate teacher probabilities via sigmoid and temperature scaling
        # Avoid applying thresholds directly to uncalibrated logits
        temp = max(cfg.temperature, 1e-4)
        teacher_probs = torch.sigmoid(teacher_scores / temp)

        # 3. Derive anchor reliability confidence: max class probability per anchor
        # Shape: (B, num_anchors)
        teacher_conf, _ = teacher_probs.max(dim=-1)

        # 4. Binary reliability mask: filter out low-confidence / background anchors
        reliable_mask = teacher_conf >= cfg.confidence_threshold  # Shape: (B, num_anchors)
        reliable_count = reliable_mask.sum()

        # 5. Differentiable Zero when no reliable predictions exist
        if reliable_count == 0:
            zero_loss = 0.0 * student_scores.sum()
            return {
                "loss": zero_loss,
                "raw_loss": zero_loss.detach(),
                "reliable_anchor_count": reliable_count,
                "effective_weight": torch.tensor(eff_weight, device=student_scores.device),
            }

        # 6. Calculate continuous reliability weights r_i
        if cfg.weighting_mode == "confidence":
            reliability_weights = teacher_conf  # r_i in [tau_rel, 1.0]
        elif cfg.weighting_mode == "power":
            reliability_weights = teacher_conf ** cfg.power_gamma
        elif cfg.weighting_mode == "binary":
            reliability_weights = torch.ones_like(teacher_conf)
        else:
            raise ValueError(f"Unknown weighting_mode: '{cfg.weighting_mode}'")

        # Zero out weights of unreliable anchors
        masked_weights = reliability_weights * reliable_mask.float()  # (B, num_anchors)

        # 7. Compute element-wise consistency loss between student logits and teacher targets
        if cfg.loss_type == "bce":
            # Continuous soft-target BCE: student logits vs teacher soft probabilities
            # Broadcast over num_classes: F.binary_cross_entropy_with_logits reduction='none'
            pointwise_loss = F.binary_cross_entropy_with_logits(
                student_scores, teacher_probs, reduction="none"
            ).mean(dim=-1)  # Average over classes -> (B, num_anchors)
        elif cfg.loss_type == "mse":
            student_probs = torch.sigmoid(student_scores)
            pointwise_loss = F.mse_loss(
                student_probs, teacher_probs, reduction="none"
            ).mean(dim=-1)  # (B, num_anchors)
        elif cfg.loss_type == "kl":
            # KL divergence on normalized distributions across classes
            p_s = F.log_softmax(student_scores / temp, dim=-1)
            p_t = F.softmax(teacher_scores / temp, dim=-1)
            pointwise_loss = F.kl_div(p_s, p_t, reduction="none").sum(dim=-1)  # (B, num_anchors)
        else:
            raise ValueError(f"Unknown consistency loss_type: '{cfg.loss_type}'")

        # 8. Weight and normalize loss by effective reliability weight sum
        weighted_loss_sum = (pointwise_loss * masked_weights).sum()
        normalizer = torch.clamp(masked_weights.sum(), min=cfg.epsilon)
        raw_loss = weighted_loss_sum / normalizer

        # Final scaled loss
        final_loss = eff_weight * raw_loss

        return {
            "loss": final_loss,
            "raw_loss": raw_loss.detach(),
            "reliable_anchor_count": reliable_count,
            "effective_weight": torch.tensor(eff_weight, device=student_scores.device),
        }

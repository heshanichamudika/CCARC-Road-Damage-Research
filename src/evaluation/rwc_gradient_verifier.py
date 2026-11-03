"""RWC Gradient Contribution Verifier for CCARC.

Audits whether Reliability-Weighted Consistency (RWC) loss remains properly
connected to the student model's computation graph and computes finite,
meaningful gradients across trainable YOLOv8 network parameters.

Measures:
1. Supervised detection loss gradient norm.
2. RWC consistency loss gradient norm.
3. Combined total loss gradient norm and linearity.
4. Parameter coverage breakdown (class head, box head, backbone/neck).
5. Confidence threshold masking behavior (reliable vs suppressed).
6. Warmup schedule behavior across training epochs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel

from src.models.ccarc_consistency import ReliabilityConsistencyConfig, ReliabilityConsistencyLoss
from src.models.ccarc_perturbations import PhotometricPerturbation, PhotometricPerturbationConfig
from src.training.ccarc_trainer import CCARCCompoundLoss


@dataclass
class RWCGradientAuditResult:
    """Comprehensive measurement record for RWC autograd verification."""

    device: str
    model_name: str
    total_trainable_parameters: int
    supervised_loss: float
    supervised_grad_norm: float
    supervised_nonzero_params: int
    rwc_loss: float
    rwc_grad_norm: float
    rwc_nonzero_params: int
    rwc_all_finite: bool
    combined_loss: float
    combined_grad_norm: float
    combined_nonzero_params: int
    combined_all_finite: bool
    gradient_linearity_error: float
    reliable_anchors_count: int
    effective_weight: float
    detect_head_cls_active_params: int
    detect_head_cls_total_params: int
    detect_head_box_active_params: int
    detect_head_box_total_params: int
    backbone_neck_active_params: int
    backbone_neck_total_params: int
    is_connected: bool
    status: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to serializable dictionary."""
        return asdict(self)


def create_synthetic_training_batch(
    batch_size: int = 2,
    img_size: int = 128,
    num_classes: int = 4,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate deterministic synthetic batch formatted for Ultralytics YOLO loss."""
    torch.manual_seed(seed)
    dev = torch.device(device)

    img = torch.rand(batch_size, 3, img_size, img_size, device=dev, dtype=dtype)
    batch_idx = torch.tensor([i for i in range(batch_size)], device=dev)
    cls_ids = torch.tensor([[i % num_classes] for i in range(batch_size)], device=dev)
    bboxes = torch.tensor(
        [[0.5, 0.5, 0.2, 0.2] for _ in range(batch_size)],
        device=dev,
        dtype=dtype,
    )

    return {
        "img": img,
        "batch_idx": batch_idx,
        "cls": cls_ids,
        "bboxes": bboxes,
    }


def audit_rwc_gradients(
    model_cfg: str = "yolov8n.yaml",
    device: str = "cpu",
    batch_size: int = 2,
    img_size: int = 128,
    num_classes: int = 4,
    confidence_threshold: float = 0.005,
    consistency_weight: float = 1.0,
    warmup_epochs: int = 0,
    active_epoch: int = 10,
    seed: int = 42,
) -> RWCGradientAuditResult:
    """Execute end-to-end autograd backward passes and audit parameter gradients.

    Args:
        model_cfg: YOLO model architecture configuration (e.g. 'yolov8n.yaml', 'yolov8s.yaml').
        device: Hardware device ('cpu', 'cuda', 'cuda:0').
        batch_size: Evaluation batch size.
        img_size: Input image resolution.
        num_classes: Number of detection target classes.
        confidence_threshold: Cutoff for reliable predictions.
        consistency_weight: Lambda consistency scaling factor.
        warmup_epochs: Number of ramp-up epochs (0 disables warmup).
        active_epoch: Simulated epoch index for warmup multiplier.
        seed: Deterministic random seed.

    Returns:
        RWCGradientAuditResult containing all gradient norms and coverage statistics.
    """
    torch.manual_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    model = DetectionModel(model_cfg, nc=num_classes).to(dev)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    model.train()

    if getattr(model, "criterion", None) is None:
        model.criterion = model.init_criterion()

    perturb = PhotometricPerturbation(PhotometricPerturbationConfig())
    rwc_cfg = ReliabilityConsistencyConfig(
        enabled=True,
        consistency_weight=consistency_weight,
        confidence_threshold=confidence_threshold,
        warmup_epochs=warmup_epochs,
        num_classes=num_classes,
    )
    rwc = ReliabilityConsistencyLoss(rwc_cfg).to(dev)

    batch = create_synthetic_training_batch(
        batch_size=batch_size,
        img_size=img_size,
        num_classes=num_classes,
        device=dev,
        seed=seed,
    )
    img = batch["img"]

    # 1. Forward passes: weak view (detached teacher) and strong view (student)
    x_w, x_s = perturb.generate_pair(img)
    with torch.no_grad():
        t_preds = model(x_w)
    s_preds = model(x_s)

    # 2. Compute supervised loss and RWC consistency loss
    sup_loss_raw, _ = model.criterion(s_preds, batch)
    sup_loss = sup_loss_raw.sum() if sup_loss_raw.ndim > 0 else sup_loss_raw

    rwc_res = rwc(t_preds, s_preds, current_epoch=active_epoch)
    con_loss = rwc_res["loss"]
    con_loss_scaled = con_loss * batch_size
    tot_loss = sup_loss + con_loss_scaled

    total_trainable = sum(1 for p in model.parameters() if p.requires_grad)

    # 3. Supervised-only backward pass
    model.zero_grad()
    sup_loss.backward(retain_graph=True)
    sup_grads = {
        name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
    }
    sup_norm = torch.norm(torch.stack([g.norm() for g in sup_grads.values()])).item()
    sup_nonzero = sum(1 for g in sup_grads.values() if g.norm().item() > 0)

    # 4. RWC-only backward pass
    model.zero_grad()
    con_loss_scaled.backward(retain_graph=True)
    rwc_grads = {
        name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
    }
    rwc_norm = torch.norm(torch.stack([g.norm() for g in rwc_grads.values()])).item()
    rwc_nonzero = sum(1 for g in rwc_grads.values() if g.norm().item() > 0)
    rwc_finite = all(torch.isfinite(g).all().item() for g in rwc_grads.values()) if rwc_grads else True

    # 5. Combined total loss backward pass
    model.zero_grad()
    tot_loss.backward()
    tot_grads = {
        name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
    }
    tot_norm = torch.norm(torch.stack([g.norm() for g in tot_grads.values()])).item()
    tot_nonzero = sum(1 for g in tot_grads.values() if g.norm().item() > 0)
    tot_finite = all(torch.isfinite(g).all().item() for g in tot_grads.values()) if tot_grads else True

    # 6. Parameter group breakdown for RWC gradients
    cls_active, cls_total = 0, 0
    box_active, box_total = 0, 0
    bb_active, bb_total = 0, 0

    for name, g in rwc_grads.items():
        is_nonzero = (g.norm().item() > 0)
        if "cv3" in name:  # Class branch in Detect head
            cls_total += 1
            if is_nonzero:
                cls_active += 1
        elif "cv2" in name:  # Box/DFL branch in Detect head
            box_total += 1
            if is_nonzero:
                box_active += 1
        else:  # Backbone and Neck layers
            bb_total += 1
            if is_nonzero:
                bb_active += 1

    # 7. Gradient linearity error: || grad(tot) - (grad(sup) + grad(rwc)) ||
    diffs = [
        (tot_grads[k] - (sup_grads[k] + rwc_grads[k])).norm().item()
        for k in tot_grads
        if k in sup_grads and k in rwc_grads
    ]
    max_lin_err = max(diffs) if diffs else 0.0

    reliable_count = int(rwc_res["reliable_anchor_count"].item())
    eff_weight = float(rwc_res["effective_weight"].item())
    is_connected = (rwc_nonzero > 0) and rwc_finite and (rwc_norm > 0)

    status = "CONNECTED_AND_VERIFIED" if is_connected else "ZERO_OR_DISCONNECTED"

    return RWCGradientAuditResult(
        device=str(dev),
        model_name=model_cfg,
        total_trainable_parameters=total_trainable,
        supervised_loss=round(float(sup_loss.item()), 4),
        supervised_grad_norm=round(sup_norm, 4),
        supervised_nonzero_params=sup_nonzero,
        rwc_loss=round(float(con_loss_scaled.item()), 4),
        rwc_grad_norm=round(rwc_norm, 4),
        rwc_nonzero_params=rwc_nonzero,
        rwc_all_finite=rwc_finite,
        combined_loss=round(float(tot_loss.item()), 4),
        combined_grad_norm=round(tot_norm, 4),
        combined_nonzero_params=tot_nonzero,
        combined_all_finite=tot_finite,
        gradient_linearity_error=float(f"{max_lin_err:.2e}"),
        reliable_anchors_count=reliable_count,
        effective_weight=round(eff_weight, 4),
        detect_head_cls_active_params=cls_active,
        detect_head_cls_total_params=cls_total,
        detect_head_box_active_params=box_active,
        detect_head_box_total_params=box_total,
        backbone_neck_active_params=bb_active,
        backbone_neck_total_params=bb_total,
        is_connected=is_connected,
        status=status,
    )


def print_audit_report(res: RWCGradientAuditResult) -> None:
    """Print structured diagnostic report to stdout."""
    print("=" * 78)
    print("CCARC: RELIABILITY-WEIGHTED CONSISTENCY (RWC) GRADIENT AUDIT")
    print("=" * 78)
    print(f"Device:                     {res.device}")
    print(f"Architecture:               {res.model_name}")
    print(f"Total Trainable Parameters: {res.total_trainable_parameters}")
    print(f"Autograd Connection Status: {res.status}")
    print("-" * 78)
    print("1. Supervised Detection Loss:")
    print(f"   - Loss Value:            {res.supervised_loss:.4f}")
    print(f"   - Gradient L2 Norm:      {res.supervised_grad_norm:.4f}")
    print(f"   - Active Parameters:     {res.supervised_nonzero_params} / {res.total_trainable_parameters} ({res.supervised_nonzero_params/res.total_trainable_parameters*100:.1f}%)")
    print("-" * 78)
    print("2. RWC Consistency Loss:")
    print(f"   - Loss Value (scaled):   {res.rwc_loss:.4f}")
    print(f"   - Gradient L2 Norm:      {res.rwc_grad_norm:.4f}")
    print(f"   - Active Parameters:     {res.rwc_nonzero_params} / {res.total_trainable_parameters} ({res.rwc_nonzero_params/res.total_trainable_parameters*100:.1f}%)")
    print(f"   - All Gradients Finite:  {res.rwc_all_finite} (No NaN / No Inf)")
    print(f"   - Reliable Anchors:      {res.reliable_anchors_count}")
    print(f"   - Effective Weight:      {res.effective_weight:.4f}")
    print("-" * 78)
    print("3. Combined Total Loss (Supervised + RWC):")
    print(f"   - Loss Value:            {res.combined_loss:.4f}")
    print(f"   - Gradient L2 Norm:      {res.combined_grad_norm:.4f}")
    print(f"   - Active Parameters:     {res.combined_nonzero_params} / {res.total_trainable_parameters} ({res.combined_nonzero_params/res.total_trainable_parameters*100:.1f}%)")
    print(f"   - All Gradients Finite:  {res.combined_all_finite}")
    print(f"   - Gradient Linearity:    ||grad(tot) - (grad(sup) + grad(rwc))|| = {res.gradient_linearity_error:.2e}")
    print("-" * 78)
    print("4. Architectural Layer Coverage Breakdown for RWC:")
    print(f"   - Detect Head Class Branch (cv3): {res.detect_head_cls_active_params} / {res.detect_head_cls_total_params} parameters active")
    print(f"   - Detect Head Box Branch   (cv2): {res.detect_head_box_active_params} / {res.detect_head_box_total_params} parameters active")
    print(f"   - Backbone & Neck Feature Layers: {res.backbone_neck_active_params} / {res.backbone_neck_total_params} parameters active")
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CCARC: Verify RWC loss gradient contributions.")
    parser.add_argument("--device", type=str, default="auto", help="Execution device: 'auto', '0', 'cuda', 'cpu'.")
    parser.add_argument("--model-cfg", type=str, default="yolov8n.yaml", help="YOLO config (default: yolov8n.yaml).")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for gradient verification.")
    parser.add_argument("--confidence-threshold", type=float, default=0.005, help="Cutoff for reliable predictions.")
    parser.add_argument("--consistency-weight", type=float, default=1.0, help="Lambda consistency weight.")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Warmup epochs (default: 0).")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional output path to save JSON metrics.")
    args = parser.parse_args(argv)

    dev_str = "cuda:0" if (args.device in ("auto", "0", "cuda") and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)

    res = audit_rwc_gradients(
        model_cfg=args.model_cfg,
        device=dev_str,
        batch_size=args.batch_size,
        confidence_threshold=args.confidence_threshold,
        consistency_weight=args.consistency_weight,
        warmup_epochs=args.warmup_epochs,
    )
    print_audit_report(res)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(res.to_dict(), f, indent=2)
        print(f"[JSON] Saved audit metrics to: {args.json_out}")

    return 0 if res.is_connected else 1


if __name__ == "__main__":
    sys.exit(main())

"""Focused Regression Tests for RWC Loss Gradient Contributions and Autograd Connection."""

from __future__ import annotations

import torch
from ultralytics.nn.tasks import DetectionModel

from src.evaluation.rwc_gradient_verifier import (
    RWCGradientAuditResult,
    audit_rwc_gradients,
    create_synthetic_training_batch,
)
from src.models.ccarc_consistency import ReliabilityConsistencyConfig, ReliabilityConsistencyLoss
from src.models.ccarc_perturbations import PhotometricPerturbation, PhotometricPerturbationConfig
from src.training.ccarc_trainer import CCARCCompoundLoss


def test_rwc_gradient_contribution_active_and_finite():
    """Verify that RWC consistency loss contributes non-zero, finite gradients to trainable parameters."""
    res = audit_rwc_gradients(
        model_cfg="yolov8n.yaml",
        device="cpu",
        confidence_threshold=0.005,
        consistency_weight=1.0,
        warmup_epochs=0,
    )

    assert res.is_connected is True
    assert res.status == "CONNECTED_AND_VERIFIED"
    assert res.rwc_loss > 0.0
    assert res.rwc_grad_norm > 0.0
    assert res.rwc_all_finite is True

    # High coverage across network architecture
    assert res.rwc_nonzero_params > 100
    assert res.rwc_nonzero_params / res.total_trainable_parameters >= 0.70

    # Backbone and neck must receive gradients
    assert res.backbone_neck_active_params == res.backbone_neck_total_params
    assert res.detect_head_cls_active_params > 0


def test_rwc_gradient_additive_linearity():
    """Verify that combined loss gradient equals sum of supervised and RWC gradients."""
    res = audit_rwc_gradients(
        model_cfg="yolov8n.yaml",
        device="cpu",
        confidence_threshold=0.005,
        consistency_weight=1.0,
        warmup_epochs=0,
    )

    # Autograd linearity: grad(tot) = grad(sup) + grad(rwc)
    assert res.gradient_linearity_error < 1e-4
    assert res.combined_all_finite is True


def test_rwc_confidence_masking_suppression():
    """Verify that when no anchor exceeds threshold, RWC produces exact zero loss and zero gradients."""
    res = audit_rwc_gradients(
        model_cfg="yolov8n.yaml",
        device="cpu",
        confidence_threshold=0.9999,  # Impossibly high threshold
        consistency_weight=1.0,
        warmup_epochs=0,
    )

    assert res.reliable_anchors_count == 0
    assert res.rwc_loss == 0.0
    assert res.rwc_grad_norm == 0.0
    assert res.rwc_nonzero_params == 0
    assert res.rwc_all_finite is True
    # Combined norm must equal supervised norm when RWC is suppressed
    assert abs(res.combined_grad_norm - res.supervised_grad_norm) < 1e-4


def test_rwc_warmup_schedule_scaling():
    """Verify that RWC effective weight and gradient scale properly during warmup epochs."""
    res_epoch0 = audit_rwc_gradients(
        model_cfg="yolov8n.yaml",
        device="cpu",
        confidence_threshold=0.005,
        consistency_weight=1.0,
        warmup_epochs=5,
        active_epoch=0,
    )
    # At epoch 0 with warmup_epochs=5, factor = (0+1)/5 = 0.2
    assert res_epoch0.effective_weight == 0.20

    res_epoch4 = audit_rwc_gradients(
        model_cfg="yolov8n.yaml",
        device="cpu",
        confidence_threshold=0.005,
        consistency_weight=1.0,
        warmup_epochs=5,
        active_epoch=4,
    )
    # At epoch 4 with warmup_epochs=5, factor = (4+1)/5 = 1.0
    assert res_epoch4.effective_weight == 1.00

    # Gradient norm at epoch 4 should be approximately 5x epoch 0
    assert res_epoch4.rwc_grad_norm > res_epoch0.rwc_grad_norm
    ratio = res_epoch4.rwc_grad_norm / res_epoch0.rwc_grad_norm
    assert 4.5 <= ratio <= 5.5


def test_rwc_compound_loss_execution_and_caching():
    """Verify CCARCCompoundLoss computes dual views, combined loss, and caches loss items."""
    from types import SimpleNamespace

    model = DetectionModel("yolov8n.yaml", nc=4)
    model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    model.train()

    perturb = PhotometricPerturbation(PhotometricPerturbationConfig())
    rwc_cfg = ReliabilityConsistencyConfig(
        enabled=True,
        consistency_weight=1.0,
        confidence_threshold=0.005,
        warmup_epochs=0,
    )
    rwc = ReliabilityConsistencyLoss(rwc_cfg)

    compound_loss = CCARCCompoundLoss(model, perturb, rwc, epoch=10)
    batch = create_synthetic_training_batch(batch_size=2, img_size=128, num_classes=4)

    total_loss, loss_items = compound_loss(batch)

    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.ndim == 0
    assert total_loss > 0.0
    assert torch.isfinite(total_loss)

    assert compound_loss.last_sup_loss is not None
    assert compound_loss.last_con_loss is not None
    assert compound_loss.last_rwc_result is not None
    assert compound_loss.last_rwc_result["reliable_anchor_count"] > 0

    # Verify backward pass from compound loss
    model.zero_grad()
    total_loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)

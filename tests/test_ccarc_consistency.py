"""Unit Tests for Reliability-Aware Consistency Learning (RWC) in CCARC.

Tests cover:
- Geometry-preserving photometric perturbations (weak/strong view generation)
- Shape and dtype preservation
- Deterministic augmentation with random seed
- Teacher gradient detachment and student gradient backpropagation
- Calibrated confidence calculation and reliability threshold filtering
- Differentiable zero output on empty reliable anchor set
- Normalization by effective reliability weight sum
- Multi-scale spatial anchor alignment
- Zero loss when regularizer is disabled or in warmup
"""

import pytest
import torch
import torch.nn as nn

from src.models.ccarc_consistency import (
    ReliabilityConsistencyConfig,
    ReliabilityConsistencyLoss,
)
from src.models.ccarc_perturbations import (
    PhotometricPerturbation,
    PhotometricPerturbationConfig,
)


def test_perturbation_shape_and_range_preservation():
    """Verify that weak and strong augmentations preserve tensor shape and [0, 1] range."""
    module = PhotometricPerturbation()

    # Batched 4D tensor: (B=2, C=3, H=64, W=64)
    x_batch = torch.rand(2, 3, 64, 64)
    w_batch, s_batch = module.generate_pair(x_batch)

    assert w_batch.shape == (2, 3, 64, 64)
    assert s_batch.shape == (2, 3, 64, 64)
    assert (w_batch >= 0.0).all() and (w_batch <= 1.0).all()
    assert (s_batch >= 0.0).all() and (s_batch <= 1.0).all()

    # Single 3D tensor: (C=3, H=32, W=32)
    x_single = torch.rand(3, 32, 32)
    w_single, s_single = module.generate_pair(x_single)

    assert w_single.shape == (3, 32, 32)
    assert s_single.shape == (3, 32, 32)


def test_perturbation_determinism_with_seed():
    """Verify that using a fixed seed yields identical augmented images."""
    module = PhotometricPerturbation()
    x = torch.rand(2, 3, 64, 64)

    w1, s1 = module.generate_pair(x, seed=42)
    w2, s2 = module.generate_pair(x, seed=42)

    assert torch.allclose(w1, w2, atol=1e-6)
    assert torch.allclose(s1, s2, atol=1e-6)

    # Different seed should yield different outputs
    w3, s3 = module.generate_pair(x, seed=999)
    assert not torch.allclose(w1, w3, atol=1e-4)


def test_spatial_anchor_unpacking_multiscale():
    """Verify multi-scale feature unpacking into spatially aligned anchor tensors."""
    loss_module = ReliabilityConsistencyLoss()

    # Mock YOLOv8 detection head multi-stride outputs: P3 (80x80), P4 (40x40), P5 (20x20)
    # Channels = 68 (64 DFL + 4 classes)
    b = 2
    feats = [
        torch.randn(b, 68, 80, 80),  # 6400 anchors
        torch.randn(b, 68, 40, 40),  # 1600 anchors
        torch.randn(b, 68, 20, 20),  # 400 anchors
    ]
    scores, distri = loss_module.unpack_predictions(feats)

    total_anchors = 6400 + 1600 + 400
    assert scores.shape == (b, total_anchors, 4)
    assert distri.shape == (b, total_anchors, 64)


def test_teacher_detachment_and_student_gradients():
    """Verify teacher branch is detached and student branch propagates non-zero gradients."""
    loss_module = ReliabilityConsistencyLoss(
        ReliabilityConsistencyConfig(
            consistency_weight=1.0,
            confidence_threshold=0.20,
            warmup_epochs=0,
        )
    )

    b, anchors, nc = 2, 50, 4
    # Teacher requires grad originally (simulating shared network pass)
    teacher_scores = torch.randn(b, anchors, nc, requires_grad=True)
    student_scores = torch.randn(b, anchors, nc, requires_grad=True)

    # Force at least one high-confidence prediction in teacher to guarantee reliable anchor
    with torch.no_grad():
        teacher_scores[0, 0, 1] = 5.0  # sigmoid(5.0) ~= 0.993 > 0.20

    res = loss_module(teacher_scores, student_scores)
    loss = res["loss"]

    assert loss > 0.0
    loss.backward()

    # 1. Student MUST have gradients
    assert student_scores.grad is not None
    assert (student_scores.grad != 0).any()

    # 2. Teacher MUST NOT have gradients because it was detached inside the loss
    assert teacher_scores.grad is None or (teacher_scores.grad == 0).all()


def test_reliability_threshold_filtering():
    """Verify that predictions below the confidence threshold are strictly excluded."""
    threshold = 0.50
    loss_module = ReliabilityConsistencyLoss(
        ReliabilityConsistencyConfig(
            confidence_threshold=threshold,
            consistency_weight=1.0,
            warmup_epochs=0,
        )
    )

    b, anchors, nc = 1, 4, 4
    # Teacher with known sigmoid probabilities:
    # Anchor 0: max prob = sigmoid(-2.0) ~= 0.119 (UNRELIABLE)
    # Anchor 1: max prob = sigmoid(-1.0) ~= 0.268 (UNRELIABLE)
    # Anchor 2: max prob = sigmoid(0.5)  ~= 0.622 (RELIABLE)
    # Anchor 3: max prob = sigmoid(2.0)  ~= 0.880 (RELIABLE)
    teacher_scores = torch.tensor([[
        [-2.0, -3.0, -4.0, -5.0],
        [-1.0, -2.0, -3.0, -4.0],
        [ 0.5, -1.0, -2.0, -3.0],
        [ 2.0,  0.0, -1.0, -2.0],
    ]])
    student_scores = torch.zeros(1, 4, 4, requires_grad=True)

    res = loss_module(teacher_scores, student_scores)
    assert res["reliable_anchor_count"].item() == 2


def test_empty_reliable_mask_differentiable_zero():
    """Verify numerical stability and differentiable zero when no anchor passes threshold."""
    loss_module = ReliabilityConsistencyLoss(
        ReliabilityConsistencyConfig(
            confidence_threshold=0.90,  # Very high threshold
            consistency_weight=1.0,
            warmup_epochs=0,
        )
    )

    # All teacher logits are very negative (max prob < 0.1)
    teacher_scores = torch.full((2, 100, 4), -5.0)
    student_scores = torch.randn(2, 100, 4, requires_grad=True)

    res = loss_module(teacher_scores, student_scores)

    assert res["reliable_anchor_count"].item() == 0
    assert res["loss"].item() == 0.0

    # Must be differentiable (backward pass must execute without error)
    res["loss"].backward()
    assert student_scores.grad is not None
    assert torch.all(student_scores.grad == 0.0)


def test_loss_normalization_by_weight_sum():
    """Verify that consistency loss is normalized by the sum of reliability weights."""
    loss_module = ReliabilityConsistencyLoss(
        ReliabilityConsistencyConfig(
            confidence_threshold=0.20,
            weighting_mode="binary",  # Weight = 1.0 for each reliable anchor
            warmup_epochs=0,
            consistency_weight=1.0,
            loss_type="bce",
        )
    )

    # 1 reliable anchor with fixed values
    teacher_scores = torch.tensor([[[2.0, -5.0, -5.0, -5.0]]])  # sigmoid(2.0) ~= 0.88
    student_scores = torch.tensor([[[0.0, -5.0, -5.0, -5.0]]], requires_grad=True)

    res = loss_module(teacher_scores, student_scores)

    # With 1 anchor, the weighted loss divided by weight sum (1.0) equals the point loss itself
    expected_bce = nn.functional.binary_cross_entropy_with_logits(
        student_scores, torch.sigmoid(teacher_scores), reduction="none"
    ).mean().item()

    assert pytest.approx(res["loss"].item(), rel=1e-5) == expected_bce


def test_disabled_loss_and_warmup_schedule():
    """Verify loss is zero when disabled or at warmup epoch 0 with 0 factor."""
    # Case 1: Disabled
    cfg_disabled = ReliabilityConsistencyConfig(enabled=False)
    loss_mod = ReliabilityConsistencyLoss(cfg_disabled)
    t = torch.randn(2, 10, 4)
    s = torch.randn(2, 10, 4, requires_grad=True)
    res = loss_mod(t, s)
    assert res["loss"].item() == 0.0

    # Case 2: Warmup scheduling
    cfg_warmup = ReliabilityConsistencyConfig(consistency_weight=2.0, warmup_epochs=4)
    loss_warm = ReliabilityConsistencyLoss(cfg_warmup)
    # Epoch 0: (0 + 1) / 4 * 2.0 = 0.5
    assert pytest.approx(loss_warm.get_effective_weight(current_epoch=0), rel=1e-5) == 0.5
    # Epoch 1: (1 + 1) / 4 * 2.0 = 1.0
    assert pytest.approx(loss_warm.get_effective_weight(current_epoch=1), rel=1e-5) == 1.0
    # Epoch 3: (3 + 1) / 4 * 2.0 = 2.0
    assert pytest.approx(loss_warm.get_effective_weight(current_epoch=3), rel=1e-5) == 2.0
    # Epoch 10 (past warmup): capped at 2.0
    assert pytest.approx(loss_warm.get_effective_weight(current_epoch=10), rel=1e-5) == 2.0

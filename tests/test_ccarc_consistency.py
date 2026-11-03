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


def test_dict_prediction_unpacking_standard_8_4():
    """Verify prediction dictionary unpacking for Ultralytics 8.4 format (scores: B, nc, A; boxes: B, 64, A)."""
    loss_module = ReliabilityConsistencyLoss()
    b, anchors, nc = 2, 300, 4
    pred_dict = {
        "boxes": torch.randn(b, 64, anchors),
        "scores": torch.randn(b, nc, anchors),
        "feats": [torch.randn(b, 68, 20, 20)],
    }

    scores, distri = loss_module.unpack_predictions(pred_dict)
    assert scores.shape == (b, anchors, nc)
    assert distri is not None
    assert distri.shape == (b, anchors, 64)


def test_dict_prediction_with_decoded_boxes():
    """Verify decoded boxes (4 coordinates) are never treated as raw DFL distribution."""
    loss_module = ReliabilityConsistencyLoss()
    b, anchors, nc = 2, 150, 4
    # Shape with 4 box coordinates (xyxy / xywh)
    pred_dict = {
        "boxes": torch.randn(b, 4, anchors),
        "scores": torch.randn(b, nc, anchors),
    }

    scores, distri = loss_module.unpack_predictions(pred_dict)
    assert scores.shape == (b, anchors, nc)
    assert distri is None  # Must NOT be treated as 64-channel DFL!


def test_dict_prediction_one2many_and_tuple_wrapper():
    """Verify one2many dictionary key and tuple (y, pred_dict) wrappers are handled seamlessly."""
    loss_module = ReliabilityConsistencyLoss()
    b, anchors, nc = 2, 200, 4
    inner_dict = {
        "boxes": torch.randn(b, 64, anchors),
        "scores": torch.randn(b, nc, anchors),
    }

    # Case A: one2many dictionary
    one2many_dict = {"one2many": inner_dict}
    scores_a, distri_a = loss_module.unpack_predictions(one2many_dict)
    assert scores_a.shape == (b, anchors, nc)
    assert distri_a.shape == (b, anchors, 64)

    # Case B: tuple wrapper (inference_tensor, pred_dict)
    tuple_wrapper = (torch.randn(b, 84, anchors), inner_dict)
    scores_b, distri_b = loss_module.unpack_predictions(tuple_wrapper)
    assert scores_b.shape == (b, anchors, nc)
    assert distri_b.shape == (b, anchors, 64)


def test_dict_prediction_teacher_detachment_and_student_gradients():
    """Verify teacher detachment and student gradient backpropagation with dictionary predictions."""
    loss_module = ReliabilityConsistencyLoss(
        ReliabilityConsistencyConfig(
            consistency_weight=1.5,
            confidence_threshold=0.20,
            warmup_epochs=0,
        )
    )
    b, anchors, nc = 2, 100, 4

    teacher_scores = torch.randn(b, nc, anchors, requires_grad=True)
    student_scores = torch.randn(b, nc, anchors, requires_grad=True)

    # Force reliable anchor in teacher
    with torch.no_grad():
        teacher_scores[0, 1, 0] = 5.0  # sigmoid(5.0) ~= 0.993 > 0.20

    teacher_dict = {"scores": teacher_scores, "boxes": torch.randn(b, 64, anchors)}
    student_dict = {"scores": student_scores, "boxes": torch.randn(b, 64, anchors)}

    res = loss_module(teacher_dict, student_dict, current_epoch=0)
    loss = res["loss"]

    assert loss > 0.0
    loss.backward()

    # Student must have gradients
    assert student_scores.grad is not None
    assert (student_scores.grad != 0).any()

    # Teacher must NOT have gradients
    assert teacher_scores.grad is None or (teacher_scores.grad == 0).all()


def test_rwc_blur_assignment_amp_fp16_regression():
    """Reproduce and verify fix for exact RuntimeError: expected scalar type Half but found Float.

    Tests out[blur_mask] = blurred[blur_mask] when strong_blur_prob=1.0 forces blur application
    on every sample under FP16 and simulated AMP autocast.
    """
    cfg = PhotometricPerturbationConfig(strong_blur_prob=1.0)
    module = PhotometricPerturbation(cfg)

    # 1. Direct FP16 input
    x_fp16 = torch.rand(4, 3, 64, 64, dtype=torch.float16)
    out_strong = module.generate_strong_view(x_fp16)
    assert out_strong.dtype == torch.float16
    assert out_strong.shape == (4, 3, 64, 64)
    assert torch.isfinite(out_strong).all()

    # 2. Paired view generation in FP16
    x_w, x_s = module.generate_pair(x_fp16)
    assert x_w.dtype == torch.float16
    assert x_s.dtype == torch.float16
    assert torch.isfinite(x_w).all()
    assert torch.isfinite(x_s).all()

    # 3. Under CPU AMP autocast (bfloat16)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        x_bf16 = torch.rand(2, 3, 32, 32, dtype=torch.bfloat16)
        out_bf16 = module.generate_strong_view(x_bf16)
        assert out_bf16.dtype == torch.bfloat16

    # 4. If CUDA is available, test real CUDA AMP
    if torch.cuda.is_available():
        x_cuda_fp16 = torch.rand(2, 3, 64, 64, device="cuda:0", dtype=torch.float16)
        with torch.cuda.amp.autocast():
            out_cuda = module.generate_strong_view(x_cuda_fp16)
            assert out_cuda.device.type == "cuda"
            assert out_cuda.dtype == torch.float16


def test_rwc_perturbations_multiple_consecutive_batches():
    """Verify perturbation module processes multiple consecutive batches without state leakage or dtype drift."""
    cfg = PhotometricPerturbationConfig(strong_blur_prob=0.5, strong_noise_std=0.02)
    module = PhotometricPerturbation(cfg)

    for i in range(10):
        dtype = torch.float16 if i % 2 == 0 else torch.float32
        x = torch.rand(3, 3, 48, 48, dtype=dtype)
        x_w, x_s = module.generate_pair(x)

        assert x_w.dtype == dtype
        assert x_s.dtype == dtype
        assert x_w.shape == (3, 3, 48, 48)
        assert x_s.shape == (3, 3, 48, 48)
        assert (x_w >= 0.0).all() and (x_w <= 1.0).all()
        assert (x_s >= 0.0).all() and (x_s <= 1.0).all()


def test_rwc_consistency_loss_amp_fp16_backward():
    """Verify ReliabilityConsistencyLoss computes valid losses and gradients under FP16."""
    loss_module = ReliabilityConsistencyLoss(
        ReliabilityConsistencyConfig(
            consistency_weight=1.0,
            confidence_threshold=0.30,
            warmup_epochs=0,
        )
    )

    b, anchors, nc = 2, 80, 4
    teacher_scores = torch.randn(b, anchors, nc, dtype=torch.float16, requires_grad=True)
    student_scores = torch.randn(b, anchors, nc, dtype=torch.float16, requires_grad=True)

    # Force at least one reliable anchor
    with torch.no_grad():
        teacher_scores[0, 5, 2] = 4.0  # sigmoid(4.0) > 0.30

    res = loss_module(teacher_scores, student_scores, current_epoch=0)
    loss = res["loss"]

    assert loss.dtype == torch.float16
    assert torch.isfinite(loss)
    assert loss > 0.0

    loss.backward()
    assert student_scores.grad is not None
    assert student_scores.grad.dtype == torch.float16
    assert torch.isfinite(student_scores.grad).all()
    assert (student_scores.grad != 0).any()
    assert teacher_scores.grad is None or (teacher_scores.grad == 0).all()


def test_rwc_consistency_empty_mask_fp16_differentiable_zero():
    """Verify empty mask returns differentiable zero matching input dtype."""
    loss_module = ReliabilityConsistencyLoss(
        ReliabilityConsistencyConfig(
            confidence_threshold=0.9999,  # Impossibly high threshold
            warmup_epochs=0,
        )
    )
    b, anchors, nc = 2, 40, 4
    teacher_scores = torch.zeros(b, anchors, nc, dtype=torch.float16)
    student_scores = torch.randn(b, anchors, nc, dtype=torch.float16, requires_grad=True)

    res = loss_module(teacher_scores, student_scores, current_epoch=0)
    loss = res["loss"]

    assert loss.dtype == torch.float16
    assert loss.item() == 0.0
    assert res["reliable_anchor_count"].item() == 0

    loss.backward()
    assert student_scores.grad is not None
    assert (student_scores.grad == 0.0).all()

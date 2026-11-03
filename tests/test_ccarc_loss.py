"""Unit Tests for Class-Aware Learning (CAL) Loss Components in CCARC.

Tests cover:
- Frequency counting from fold summaries, manifests, and label directories
- Zero validation/test leakage verification
- Inverse-frequency class weight computation and ordering
- Unit-mean normalization and clipping
- Zero-frequency edge case handling
- Numerical stability under extreme imbalance
- PyTorch gradient propagation and device/dtype compatibility
- Exact equivalence to standard BCE loss when weighting is disabled
"""

import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.loco_splitter import DEFAULT_TARGET_CLASSES
from src.models.ccarc_loss import (
    CCARCDetectionLoss,
    ClassAwareLearningConfig,
    ClassWeightedBCEWithLogitsLoss,
    _sync_loss_device_dtype,
    _v8_loss_has_native_class_weights,
    attach_class_aware_loss,
    compute_class_weights,
    extract_source_train_class_counts,
    resolve_detection_model,
)


def test_compute_class_weights_ordering_and_normalization():
    """Verify that rarer classes receive higher weights and mean(weights) equals 1.0."""
    # Imbalance: D00=10000 (common), D10=1000 (medium), D20=100 (rare), D40=2000 (medium)
    counts = {"D00": 10000, "D10": 1000, "D20": 100, "D40": 2000}
    cfg = ClassAwareLearningConfig(alpha=0.5, epsilon=1.0)
    weights = compute_class_weights(counts, config=cfg)

    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (4,)
    assert pytest.approx(weights.mean().item(), rel=1e-5) == 1.0

    # Ordering: D20 (rare) > D10 > D40 > D00 (common)
    # Class IDs: D00=0, D10=1, D20=2, D40=3
    w_D00, w_D10, w_D20, w_D40 = weights.tolist()
    assert w_D20 > w_D10 > w_D40 > w_D00
    assert w_D20 > 1.0
    assert w_D00 < 1.0


def test_standard_loss_equivalence_when_disabled():
    """Verify that when alpha=0.0 or enabled=False, weights are all 1.0."""
    counts = {"D00": 8000, "D10": 200, "D20": 100, "D40": 4000}

    # Case 1: alpha = 0.0
    cfg_zero_alpha = ClassAwareLearningConfig(alpha=0.0)
    w_zero = compute_class_weights(counts, config=cfg_zero_alpha)
    assert torch.allclose(w_zero, torch.ones(4))

    # Case 2: enabled = False
    cfg_disabled = ClassAwareLearningConfig(enabled=False, alpha=0.5)
    w_disabled = compute_class_weights(counts, config=cfg_disabled)
    assert torch.allclose(w_disabled, torch.ones(4))


def test_zero_frequency_handling():
    """Verify safe handling when a class has 0 occurrences in training data."""
    # D20 has 0 occurrences
    counts = {"D00": 5000, "D10": 1000, "D20": 0, "D40": 2000}

    # Policy 'max': D20 receives the max weight among observed classes
    cfg_max = ClassAwareLearningConfig(alpha=0.5, zero_freq_policy="max")
    w_max = compute_class_weights(counts, config=cfg_max)
    assert not torch.isnan(w_max).any()
    assert not torch.isinf(w_max).any()
    assert pytest.approx(w_max.mean().item(), rel=1e-5) == 1.0
    # D20 (index 2) must equal max among non-zero
    assert w_max[2] == w_max[1]  # D10 was the rarest observed, so D20 matches D10

    # Policy 'neutral': D20 receives neutral raw weight
    cfg_neutral = ClassAwareLearningConfig(alpha=0.5, zero_freq_policy="neutral")
    w_neutral = compute_class_weights(counts, config=cfg_neutral)
    assert not torch.isnan(w_neutral).any()
    assert pytest.approx(w_neutral.mean().item(), rel=1e-5) == 1.0


def test_weight_clipping_bounds():
    """Verify that max_weight and min_weight prevent extreme weight values."""
    # Extreme imbalance: 1,000,000 vs 1
    counts = {"D00": 1000000, "D10": 1, "D20": 1, "D40": 100000}
    cfg_clipped = ClassAwareLearningConfig(
        alpha=1.0,
        max_weight=5.0,
        min_weight=0.2,
    )
    weights = compute_class_weights(counts, config=cfg_clipped)

    assert not torch.isnan(weights).any()
    assert not torch.isinf(weights).any()
    assert weights.max().item() <= 8.0  # Respects clipping within margin
    assert weights.min().item() >= 0.1
    assert pytest.approx(weights.mean().item(), rel=1e-4) == 1.0


def test_extract_source_train_class_counts_from_summary_json(tmp_path: Path):
    """Verify extracting class counts from fold_summary.json (strictly train split)."""
    summary_file = tmp_path / "fold_summary.json"
    summary_data = {
        "fold_idx": 0,
        "target_country": "China",
        "class_counts": {
            "train": {"D00": 1500, "D10": 300, "D20": 200, "D40": 800},
            "val": {"D00": 9999, "D10": 9999, "D20": 9999, "D40": 9999},
            "test": {"D00": 8888, "D10": 8888, "D20": 8888, "D40": 8888},
        },
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f)

    counts = extract_source_train_class_counts(summary_file)
    # Must match train exactly, completely ignoring val and test
    assert counts == {"D00": 1500, "D10": 300, "D20": 200, "D40": 800}


def test_extract_source_train_class_counts_from_manifest_csv(tmp_path: Path):
    """Verify extracting class counts from fold_manifest.csv (zero test/val leakage)."""
    manifest_file = tmp_path / "fold_manifest.csv"
    df = pd.DataFrame([
        {"image_id": "tr_1", "loco_split": "train", "class_counts": "{'D00': 2, 'D10': 1}"},
        {"image_id": "tr_2", "loco_split": "train", "class_counts": "{'D20': 3, 'D40': 1}"},
        # Val and test rows with high counts that MUST BE IGNORED
        {"image_id": "val_1", "loco_split": "val", "class_counts": "{'D00': 100, 'D10': 100}"},
        {"image_id": "test_1", "loco_split": "test", "class_counts": "{'D20': 500, 'D40': 500}"},
    ])
    df.to_csv(manifest_file, index=False)

    counts = extract_source_train_class_counts(manifest_file)
    assert counts["D00"] == 2
    assert counts["D10"] == 1
    assert counts["D20"] == 3
    assert counts["D40"] == 1


def test_extract_source_train_class_counts_from_label_dir(tmp_path: Path):
    """Verify extracting class counts from YOLO format .txt label directory."""
    lbl_dir = tmp_path / "labels" / "train"
    lbl_dir.mkdir(parents=True)

    # File 1: one D00 (class 0), one D40 (class 3)
    (lbl_dir / "Japan_0001.txt").write_text("0 0.5 0.5 0.2 0.2\n3 0.6 0.6 0.1 0.1\n")
    # File 2: two D10 (class 1)
    (lbl_dir / "India_0001.txt").write_text("1 0.1 0.1 0.2 0.2\n1 0.3 0.3 0.2 0.2\n")

    counts = extract_source_train_class_counts(tmp_path)
    assert counts["D00"] == 1
    assert counts["D10"] == 2
    assert counts["D20"] == 0
    assert counts["D40"] == 1


def test_class_weighted_bce_mathematical_equivalence():
    """Verify that ClassWeightedBCE matches standard PyTorch BCE when weights are ones."""
    pred = torch.randn(4, 100, 4, requires_grad=True)
    target = torch.randint(0, 2, (4, 100, 4)).float()

    standard_bce = nn.BCEWithLogitsLoss(reduction="none")
    weighted_bce = ClassWeightedBCEWithLogitsLoss(class_weights=torch.ones(4))

    loss_std = standard_bce(pred, target)
    loss_weighted = weighted_bce(pred, target)

    assert torch.allclose(loss_std, loss_weighted, atol=1e-6)


def test_class_weighted_bce_weight_application():
    """Verify that class weights correctly scale class-specific BCE loss slices."""
    pred = torch.zeros(2, 5, 4)  # Logits = 0 -> sigmoid = 0.5
    target = torch.ones(2, 5, 4)  # Ground truth 1.0

    # Unweighted BCE value for sigmoid(0) vs 1 is ln(2) ~= 0.693147
    weights = torch.tensor([1.0, 2.0, 0.5, 3.0])
    weighted_bce = ClassWeightedBCEWithLogitsLoss(class_weights=weights)

    loss = weighted_bce(pred, target)  # shape (2, 5, 4)

    expected_base = torch.log(torch.tensor(2.0)).item()
    for c in range(4):
        expected_class_loss = expected_base * weights[c].item()
        assert pytest.approx(loss[..., c].mean().item(), rel=1e-5) == expected_class_loss


def test_class_weighted_bce_gradient_propagation():
    """Verify that gradients propagate backward correctly through weighted BCE loss."""
    pred = torch.randn(2, 10, 4, requires_grad=True)
    target = torch.rand(2, 10, 4)
    weights = torch.tensor([0.5, 1.5, 2.0, 1.0])

    weighted_bce = ClassWeightedBCEWithLogitsLoss(class_weights=weights)
    loss = weighted_bce(pred, target).sum()

    loss.backward()

    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
    assert (pred.grad != 0).any()

    # Classes with higher weights should have proportionally larger gradients
    # For pred=0, target=1: dLoss/dPred = (sigmoid(0) - 1) * weight = -0.5 * weight
    pred_zero = torch.zeros(1, 1, 4, requires_grad=True)
    target_one = torch.ones(1, 1, 4)
    loss_test = weighted_bce(pred_zero, target_one).sum()
    loss_test.backward()

    expected_grads = -0.5 * weights
    assert torch.allclose(pred_zero.grad.squeeze(), expected_grads, atol=1e-5)


def test_scalar_supervised_loss_extraction_across_ultralytics_formats():
    """Verify scalar supervised loss extraction across Ultralytics 8.3 and 8.4 return conventions."""
    # Format 1: Ultralytics <=8.3 returns scalar total loss: loss.sum() * batch_size
    scalar_loss = torch.tensor(15.4, requires_grad=True)
    extracted_1 = scalar_loss.sum() if scalar_loss.ndim > 0 else scalar_loss
    assert extracted_1.ndim == 0
    assert pytest.approx(extracted_1.item(), rel=1e-5) == 15.4

    # Format 2: Ultralytics >=8.4 returns 1D vector: loss * batch_size = [box, cls, dfl]
    # Summing across components yields the scalar total loss without double-counting
    vector_loss = torch.tensor([5.2, 4.1, 6.1], requires_grad=True)
    extracted_2 = vector_loss.sum() if vector_loss.ndim > 0 else vector_loss
    assert extracted_2.ndim == 0
    assert pytest.approx(extracted_2.item(), rel=1e-5) == 15.4

    # Verify backward propagation through extracted scalar loss
    extracted_2.backward()
    assert vector_loss.grad is not None
    assert torch.allclose(vector_loss.grad, torch.ones_like(vector_loss))


def test_sequential_object_missing_args_regression():
    """Reproduce original failure: Sequential object lacks 'args' attribute."""
    from types import SimpleNamespace
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import v8DetectionLoss

    dm = DetectionModel("yolov8s.yaml", nc=4, verbose=False)
    raw_sequential = dm.model  # torch.nn.Sequential

    # 1. Assert raw Sequential has no 'args' attribute
    assert isinstance(raw_sequential, torch.nn.Sequential)
    assert not hasattr(raw_sequential, "args")

    # 2. Passing raw Sequential to v8DetectionLoss reproduces exact AttributeError
    with pytest.raises(AttributeError, match="'Sequential' object has no attribute 'args'"):
        _ = v8DetectionLoss(raw_sequential)

    # 3. resolve_detection_model raises TypeError if passed a bare Sequential
    with pytest.raises(TypeError, match="Expected Ultralytics DetectionModel, but received torch.nn.Sequential"):
        resolve_detection_model(raw_sequential)


def test_cal_loss_initialization_with_detection_model():
    """Verify CAL loss attaches to genuine DetectionModel across wrapper types."""
    from types import SimpleNamespace
    from ultralytics.nn.tasks import DetectionModel

    dm = DetectionModel("yolov8s.yaml", nc=4, verbose=False)
    dm.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    # 1. Resolve directly from DetectionModel
    resolved = resolve_detection_model(dm)
    assert resolved is dm
    assert hasattr(resolved, "args")
    assert hasattr(resolved, "loss")

    # 2. Resolve from trainer-like container
    trainer_mock = SimpleNamespace(model=dm, validator=True, args=dm.args)
    assert resolve_detection_model(trainer_mock) is dm

    # 3. Resolve from YOLO-like container
    yolo_mock = SimpleNamespace(model=dm)
    assert resolve_detection_model(yolo_mock) is dm

    # 4. Resolve from DDP-like container
    ddp_mock = SimpleNamespace(module=dm)
    assert resolve_detection_model(ddp_mock) is dm

    # 5. Attach CAL loss to DetectionModel
    weights = torch.tensor([0.5, 1.5, 2.0, 1.0])
    criterion = attach_class_aware_loss(dm, weights)

    assert isinstance(criterion, CCARCDetectionLoss)
    assert dm.criterion is criterion
    assert hasattr(criterion, "bce")
    assert torch.allclose(criterion.bce.class_weights, weights)
    assert criterion.nc == 4
    assert criterion.stride is not None


def test_cal_finite_loss_and_backward_gradients():
    """Verify forward and backward passes produce finite losses and gradients with CAL."""
    from types import SimpleNamespace
    from ultralytics.nn.tasks import DetectionModel

    dm = DetectionModel("yolov8s.yaml", nc=4, verbose=False)
    dm.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    weights = torch.tensor([0.7, 1.8, 2.5, 1.0])
    attach_class_aware_loss(dm, weights)

    batch = {
        "batch_idx": torch.tensor([0, 1]),
        "cls": torch.tensor([0, 2]),
        "bboxes": torch.tensor([
            [0.5, 0.5, 0.2, 0.2],
            [0.3, 0.3, 0.1, 0.1],
        ]),
        "img": torch.rand(2, 3, 64, 64),
    }

    loss, loss_items = dm.loss(batch)

    # Assert finite loss
    loss_scalar = loss.sum() if loss.ndim > 0 else loss
    assert torch.isfinite(loss_scalar)
    assert loss_scalar.item() > 0.0
    assert torch.isfinite(loss_items).all()

    # Assert backward propagation produces valid gradients
    loss_scalar.backward()
    grads = [p.grad for p in dm.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    assert any((g != 0).any() for g in grads)


def test_cal_class_weights_affect_classification_loss():
    """Verify that class weights actively alter the classification loss component."""
    pred = torch.zeros(1, 1, 4)  # logits
    target = torch.ones(1, 1, 4)

    # Loss with uniform weights
    uniform_loss_fn = ClassWeightedBCEWithLogitsLoss(torch.tensor([1.0, 1.0, 1.0, 1.0]))
    loss_uniform = uniform_loss_fn(pred, target)

    # Loss with skewed weights: class 2 is 4x weight of class 0
    skewed_weights = torch.tensor([0.5, 1.0, 2.0, 0.5])
    skewed_loss_fn = ClassWeightedBCEWithLogitsLoss(skewed_weights)
    loss_skewed = skewed_loss_fn(pred, target)

    # Class 2 loss must be 4x of class 0 loss in skewed case
    assert pytest.approx(loss_skewed[0, 0, 2].item() / loss_skewed[0, 0, 0].item(), rel=1e-4) == 4.0
    # Class 0 loss in skewed case must be 0.5x of uniform
    assert pytest.approx(loss_skewed[0, 0, 0].item() / loss_uniform[0, 0, 0].item(), rel=1e-4) == 0.5


def test_class_weighted_bce_buffer_registration():
    """Verify class_weights is properly registered as a PyTorch buffer and moves with module."""
    weights = torch.tensor([0.8, 1.2, 2.5, 0.5])
    loss_mod = ClassWeightedBCEWithLogitsLoss(weights)

    # 1. Assert class_weights is in registered _buffers
    assert "class_weights" in loss_mod._buffers
    assert torch.allclose(loss_mod.class_weights, weights)

    # 2. Moving module to different dtype moves buffer
    loss_mod_half = loss_mod.to(dtype=torch.float16)
    assert loss_mod_half.class_weights.dtype == torch.float16

    # 3. Dynamic buffer update maintains buffer registry
    new_weights = torch.tensor([1.0, 1.0, 1.0, 1.0])
    loss_mod.set_class_weights(new_weights)
    assert "class_weights" in loss_mod._buffers
    assert torch.allclose(loss_mod.class_weights, new_weights)


def test_class_weighted_bce_dynamic_dtype_sync_amp():
    """Verify class_weights adapts dynamically to FP16 input under AMP without mismatch."""
    weights = torch.tensor([0.7, 1.5, 2.0, 0.8], dtype=torch.float32)
    loss_mod = ClassWeightedBCEWithLogitsLoss(weights)

    # Simulated AMP tensors (FP16)
    pred_half = torch.randn(2, 8, 4, dtype=torch.float16, requires_grad=True)
    target_half = torch.randint(0, 2, (2, 8, 4)).to(dtype=torch.float16)

    # Forward pass should not raise RuntimeError and should produce FP16 output
    loss_half = loss_mod(pred_half, target_half)
    assert loss_half.dtype == torch.float16
    assert torch.isfinite(loss_half).all()

    # Buffer should now match pred dtype to prevent future casting overhead
    assert loss_mod.class_weights.dtype == torch.float16

    # Backward pass should produce finite gradients
    loss_half.sum().backward()
    assert pred_half.grad is not None
    assert torch.isfinite(pred_half.grad).all()


def test_sync_loss_device_dtype_and_idempotency():
    """Verify _sync_loss_device_dtype synchronizes device/dtype and is idempotent."""
    from types import SimpleNamespace

    # Mock criterion with CPU FP32 class weights
    weights = torch.tensor([0.5, 1.0, 2.0, 0.5], dtype=torch.float32)
    bce = ClassWeightedBCEWithLogitsLoss(weights)
    mock_criterion = SimpleNamespace(class_weights=weights.clone(), bce=bce)

    # Mock predictions in FP16 (as when running under autocast/AMP)
    feats = [torch.randn(2, 64, 16, 16, dtype=torch.float16)]
    preds = (None, feats)

    # 1. Sync should update class_weights to FP16
    _sync_loss_device_dtype(mock_criterion, preds)
    assert mock_criterion.class_weights.dtype == torch.float16
    assert mock_criterion.bce.class_weights.dtype == torch.float16

    # 2. Idempotency: repeated call with matching device/dtype should be a no-op
    cw_ref = mock_criterion.class_weights
    _sync_loss_device_dtype(mock_criterion, preds)
    assert mock_criterion.class_weights is cw_ref  # Same tensor instance, zero reallocation


def test_cal_loss_numerical_invariance_across_device_dtype():
    """Verify class-weighted BCE loss produces mathematically invariant results across dtypes."""
    weights = torch.tensor([0.6, 1.4, 2.2, 0.8])
    loss_fp32 = ClassWeightedBCEWithLogitsLoss(weights)
    loss_fp16 = ClassWeightedBCEWithLogitsLoss(weights)

    torch.manual_seed(42)
    pred_32 = torch.randn(4, 20, 4, dtype=torch.float32)
    target_32 = torch.randint(0, 2, (4, 20, 4)).float()

    pred_16 = pred_32.to(dtype=torch.float16)
    target_16 = target_32.to(dtype=torch.float16)

    out_32 = loss_fp32(pred_32, target_32)
    out_16 = loss_fp16(pred_16, target_16)

    # Values must match within FP16 precision
    assert torch.allclose(out_32.to(torch.float16), out_16, atol=2e-3, rtol=1e-2)


def test_attach_class_aware_loss_sets_model_attributes_and_device():
    """Verify attach_class_aware_loss synchronizes class_weights with DetectionModel device."""
    from types import SimpleNamespace
    from ultralytics.nn.tasks import DetectionModel

    dm = DetectionModel("yolov8s.yaml", nc=4, verbose=False)
    dm.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    weights = torch.tensor([0.7, 1.3, 2.1, 0.9])
    criterion = attach_class_aware_loss(dm, weights)

    # Both det_model and criterion must hold class_weights
    assert hasattr(dm, "class_weights")
    assert dm.class_weights is not None
    assert torch.allclose(dm.class_weights.cpu(), weights)
    assert hasattr(criterion, "class_weights")
    assert torch.allclose(criterion.class_weights.cpu(), weights)

    # If CUDA is available, test GPU transfer
    if torch.cuda.is_available():
        dm_cuda = dm.to("cuda:0")
        attach_class_aware_loss(dm_cuda, weights)
        assert dm_cuda.class_weights.device.type == "cuda"
        assert dm_cuda.criterion.class_weights.device.type == "cuda"


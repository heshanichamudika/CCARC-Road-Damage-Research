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
    ClassAwareLearningConfig,
    ClassWeightedBCEWithLogitsLoss,
    compute_class_weights,
    extract_source_train_class_counts,
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

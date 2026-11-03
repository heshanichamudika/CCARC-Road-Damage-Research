"""Standardized Metrics Extraction and Computation for CCARC LOCO Evaluation.

Provides consistent extraction of detection metrics across Ultralytics versions (<=8.3 and >=8.4),
supporting global Precision, Recall, F1-score, mAP@0.5, mAP@0.5:0.95, Fitness, and per-class
metrics (AP50, mAP50-95, Precision, Recall, F1-score).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from src.data.loco_splitter import DEFAULT_TARGET_CLASSES


def compute_f1(precision: float, recall: float, eps: float = 1e-7) -> float:
    """Compute harmonic mean F1-score from precision and recall.

    Args:
        precision: Precision value in [0.0, 1.0].
        recall: Recall value in [0.0, 1.0].
        eps: Small epsilon to prevent division by zero.

    Returns:
        F1-score rounded to 4 decimal places, bounded in [0.0, 1.0].
    """
    p = max(0.0, min(1.0, float(precision)))
    r = max(0.0, min(1.0, float(recall)))
    denom = p + r
    if denom <= eps:
        return 0.0
    return round(float(2.0 * p * r / denom), 4)


def extract_detection_metrics(
    val_results: Any,
    target_classes: Sequence[str] = DEFAULT_TARGET_CLASSES,
) -> Dict[str, Any]:
    """Extract standardized detection metrics from Ultralytics validation results.

    Compatible with both modern Ultralytics (>=8.4: box.mp, box.mr, array p/r/f1)
    and legacy Ultralytics (<=8.3: box.p, box.r scalars or arrays).

    Args:
        val_results: Return value from Ultralytics model.val().
        target_classes: Sequence of class names in index order.

    Returns:
        Standardized metrics dictionary containing:
            - precision: Macro-average precision across classes.
            - recall: Macro-average recall across classes.
            - f1: Overall F1-score.
            - map50: mAP at IoU threshold 0.50.
            - map50_95: mAP across IoU thresholds 0.50:0.95.
            - fitness: Model selection fitness score (0.1*mAP50 + 0.9*mAP50-95).
            - per_class: Per-class breakdown of AP50, mAP50-95, Precision, Recall, F1.
    """
    box = getattr(val_results, "box", None)
    res_dict = getattr(val_results, "results_dict", {})

    # 1. Precision (mp in >=8.4, p in <=8.3, or dictionary fallback)
    p_val = getattr(box, "mp", None)
    if p_val is None:
        p_val = getattr(box, "p", None)
    if p_val is None:
        p_val = res_dict.get("metrics/precision(B)", 0.0)
    elif isinstance(p_val, (list, tuple, np.ndarray, torch.Tensor)):
        p_val = float(np.mean(p_val)) if len(p_val) > 0 else 0.0
    precision = float(p_val)

    # 2. Recall (mr in >=8.4, r in <=8.3, or dictionary fallback)
    r_val = getattr(box, "mr", None)
    if r_val is None:
        r_val = getattr(box, "r", None)
    if r_val is None:
        r_val = res_dict.get("metrics/recall(B)", 0.0)
    elif isinstance(r_val, (list, tuple, np.ndarray, torch.Tensor)):
        r_val = float(np.mean(r_val)) if len(r_val) > 0 else 0.0
    recall = float(r_val)

    # 3. mAP@0.5
    map50_val = getattr(box, "map50", None)
    if map50_val is None:
        map50_val = res_dict.get("metrics/mAP50(B)", 0.0)
    map50 = float(map50_val)

    # 4. mAP@0.5:0.95
    map_val = getattr(box, "map", None)
    if map_val is None:
        map_val = getattr(box, "map50_95", None)
    if map_val is None:
        map_val = res_dict.get("metrics/mAP50-95(B)", 0.0)
    map50_95 = float(map_val)

    # 5. Fitness = 0.1 * mAP@0.5 + 0.9 * mAP@0.5:0.95
    fitness = float(getattr(val_results, "fitness", 0.1 * map50 + 0.9 * map50_95))

    # 6. Overall F1-score
    # If box exposes f1 array, use its mean; otherwise compute harmonic mean of overall P and R
    f1_arr = getattr(box, "f1", None)
    if f1_arr is not None and isinstance(f1_arr, (list, tuple, np.ndarray, torch.Tensor)) and len(f1_arr) > 0:
        f1 = float(np.mean(f1_arr))
    else:
        f1 = compute_f1(precision, recall)

    # 7. Per-Class Arrays
    maps_arr = getattr(box, "maps", None)
    ap50_arr = getattr(box, "ap50", None)
    p_arr = getattr(box, "p", None)
    r_arr = getattr(box, "r", None)

    per_class: Dict[str, Dict[str, float]] = {}

    for idx, cls_name in enumerate(target_classes):
        cls_ap50 = float(ap50_arr[idx]) if (ap50_arr is not None and len(ap50_arr) > idx) else 0.0
        cls_map = float(maps_arr[idx]) if (maps_arr is not None and len(maps_arr) > idx) else 0.0

        cls_p = float(p_arr[idx]) if (p_arr is not None and isinstance(p_arr, (list, tuple, np.ndarray, torch.Tensor)) and len(p_arr) > idx) else precision
        cls_r = float(r_arr[idx]) if (r_arr is not None and isinstance(r_arr, (list, tuple, np.ndarray, torch.Tensor)) and len(r_arr) > idx) else recall
        cls_f1 = float(f1_arr[idx]) if (f1_arr is not None and isinstance(f1_arr, (list, tuple, np.ndarray, torch.Tensor)) and len(f1_arr) > idx) else compute_f1(cls_p, cls_r)

        per_class[cls_name] = {
            "precision": round(cls_p, 4),
            "recall": round(cls_r, 4),
            "f1": round(cls_f1, 4),
            "ap50": round(cls_ap50, 4),
            "map50_95": round(cls_map, 4),
        }

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "fitness": round(fitness, 4),
        "per_class": per_class,
    }


def normalize_metrics_dict(raw_metrics: Dict[str, Any], target_classes: Sequence[str] = DEFAULT_TARGET_CLASSES) -> Dict[str, Any]:
    """Ensure a loaded metrics dictionary has all required standard fields including F1.

    Args:
        raw_metrics: Pre-existing metrics dictionary (e.g. from held_out_test_metrics.json).
        target_classes: Canonical target class names.

    Returns:
        Standardized metrics dictionary with all fields present.
    """
    p = float(raw_metrics.get("precision", 0.0))
    r = float(raw_metrics.get("recall", 0.0))
    f1 = float(raw_metrics.get("f1", compute_f1(p, r)))
    map50 = float(raw_metrics.get("map50", 0.0))
    map50_95 = float(raw_metrics.get("map50_95", 0.0))
    fitness = float(raw_metrics.get("fitness", round(0.1 * map50 + 0.9 * map50_95, 4)))

    per_class_raw = raw_metrics.get("per_class", {})
    per_class: Dict[str, Dict[str, float]] = {}

    for cls_name in target_classes:
        cls_data = per_class_raw.get(cls_name, {})
        cls_p = float(cls_data.get("precision", p))
        cls_r = float(cls_data.get("recall", r))
        cls_f1 = float(cls_data.get("f1", compute_f1(cls_p, cls_r)))
        cls_ap50 = float(cls_data.get("ap50", 0.0))
        cls_map = float(cls_data.get("map50_95", 0.0))

        per_class[cls_name] = {
            "precision": round(cls_p, 4),
            "recall": round(cls_r, 4),
            "f1": round(cls_f1, 4),
            "ap50": round(cls_ap50, 4),
            "map50_95": round(cls_map, 4),
        }

    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "fitness": round(fitness, 4),
        "per_class": per_class,
    }

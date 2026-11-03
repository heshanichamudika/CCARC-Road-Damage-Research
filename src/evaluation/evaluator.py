"""Leave-One-Country-Out (LOCO) Benchmark Evaluator for CCARC.

Loads the best saved checkpoint for completed experiments across all variants (E0–E4)
and evaluates them strictly on the unseen held-out target country test set.
Aggregates and verifies results without data leakage, synthetic metrics, or overwriting
active training runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
import yaml

from src.data.loco_splitter import DEFAULT_TARGET_CLASSES, LOCO_COUNTRIES
from src.evaluation.metrics import extract_detection_metrics, normalize_metrics_dict
from src.training.baseline_config import check_gpu_environment, resolve_fold_data_yaml
from src.training.ccarc_config import VALID_VARIANTS


def resolve_data_yaml(views_dir: Path, fold_idx: int, target_country: str) -> Optional[Path]:
    """Resolve data.yaml for a LOCO fold across common directory layouts."""
    p = resolve_fold_data_yaml(views_dir, fold_idx, target_country)
    if p is not None and p.is_file():
        return p
    candidates = [
        views_dir / f"fold_{fold_idx}_{target_country}" / "data.yaml",
        views_dir / f"fold_{target_country}" / "data.yaml",
        views_dir / f"fold_{fold_idx}" / "data.yaml",
        views_dir / target_country / "data.yaml",
        views_dir / f"fold{fold_idx}_{target_country}" / "data.yaml",
        views_dir / f"fold{fold_idx}" / "data.yaml",
        views_dir / f"fold_{fold_idx}_{target_country.lower()}" / "data.yaml",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


@dataclass
class ExperimentTargetResult:
    """Evaluation result record for a single variant and LOCO fold."""

    variant: str
    fold_idx: int
    target_country: str
    status: str  # COMPLETED, IN_PROGRESS, PENDING, MISSING_CHECKPOINT, MISSING_DATA_YAML, EVALUATION_ERROR
    source: str  # cached_test_metrics, evaluated_checkpoint, none
    checkpoint_path: Optional[str] = None
    experiment_dir: Optional[str] = None
    data_yaml_path: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
    status_detail: Optional[str] = None
    evaluated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to serializable dictionary."""
        return {
            "variant": self.variant,
            "fold_idx": self.fold_idx,
            "target_country": self.target_country,
            "status": self.status,
            "source": self.source,
            "checkpoint_path": self.checkpoint_path,
            "experiment_dir": self.experiment_dir,
            "data_yaml_path": self.data_yaml_path,
            "elapsed_seconds": self.elapsed_seconds,
            "metrics": self.metrics,
            "status_detail": self.status_detail,
            "evaluated_at": self.evaluated_at,
        }


class LOCOBenchmarkEvaluator:
    """Orchestrates comprehensive cross-country evaluation across LOCO folds and variants."""

    def __init__(
        self,
        views_dir: Union[str, Path] = Path("experiments/loco_splits"),
        baseline_dir: Union[str, Path] = Path("experiments/yolov8s_baseline_full100"),
        ccarc_dir: Union[str, Path] = Path("experiments/ccarc_full_training"),
        variants: Sequence[str] = VALID_VARIANTS,
        folds: Sequence[int] = (0, 1, 2, 3, 4, 5),
        device: str = "auto",
        batch_size: int = 16,
        img_size: int = 640,
        target_classes: Sequence[str] = DEFAULT_TARGET_CLASSES,
        countries: Sequence[str] = LOCO_COUNTRIES,
    ) -> None:
        """Initialize LOCO benchmark evaluator.

        Args:
            views_dir: Path to runtime LOCO dataset views containing data.yaml files.
            baseline_dir: Root directory of E0 baseline experiments.
            ccarc_dir: Root directory of CCARC proposed/ablation experiments (E1-E4).
            variants: Experiment variants to evaluate.
            folds: Fold indices to evaluate (0 to 5).
            device: Hardware device identifier ('auto', '0', 'cuda', 'cpu').
            batch_size: DataLoader batch size for validation.
            img_size: Image resolution for model input.
            target_classes: Target defect classes.
            countries: Canonical sovereign countries for LOCO.
        """
        self.views_dir = Path(views_dir).resolve()
        self.baseline_dir = Path(baseline_dir).resolve()
        self.ccarc_dir = Path(ccarc_dir).resolve()
        self.variants = list(variants)
        self.folds = list(folds)
        self.device = device
        self.batch_size = batch_size
        self.img_size = img_size
        self.target_classes = list(target_classes)
        self.countries = list(countries)

    def resolve_device(self) -> str:
        """Resolve device string to torch / ultralytics device identifier."""
        if self.device == "auto":
            return "0" if torch.cuda.is_available() else "cpu"
        if self.device in ("cuda", "cuda:0") and not torch.cuda.is_available():
            return "cpu"
        return self.device

    def locate_experiment(self, variant: str, fold_idx: int, target_country: str) -> Tuple[Optional[Path], Optional[Path]]:
        """Locate existing experiment directory and best checkpoint for a condition.

        Returns:
            Tuple of (experiment_dir, checkpoint_path). Either may be None if not found.
        """
        candidate_dirs: List[Path] = []

        if variant == "E0":
            candidate_dirs.extend([
                self.baseline_dir / f"fold_{fold_idx}_{target_country}",
                self.baseline_dir / f"fold_{fold_idx}_{target_country}_E0",
                self.baseline_dir / f"fold_{fold_idx}",
                self.ccarc_dir / f"fold_{fold_idx}_{target_country}_E0",
                self.ccarc_dir / f"fold_{fold_idx}_{target_country}",
            ])
        else:
            candidate_dirs.extend([
                self.ccarc_dir / f"fold_{fold_idx}_{target_country}_{variant}",
                self.ccarc_dir / f"fold_{fold_idx}_{target_country}",
                self.baseline_dir / f"fold_{fold_idx}_{target_country}_{variant}",
                self.ccarc_dir / f"{variant}_fold_{fold_idx}_{target_country}",
            ])

        for exp_dir in candidate_dirs:
            if exp_dir.is_dir():
                best_ckpt = exp_dir / "weights" / "best.pt"
                if best_ckpt.is_file():
                    return exp_dir, best_ckpt
                # Also check root of experiment directory if weights folder missing
                if (exp_dir / "best.pt").is_file():
                    return exp_dir, exp_dir / "best.pt"
                # Experiment dir exists, but best.pt not found
                return exp_dir, None

        # Neither directory nor checkpoint found; return primary expected directory
        primary_dir = (
            self.baseline_dir / f"fold_{fold_idx}_{target_country}"
            if variant == "E0"
            else self.ccarc_dir / f"fold_{fold_idx}_{target_country}_{variant}"
        )
        return primary_dir, None

    def check_training_status(
        self,
        exp_dir: Optional[Path],
        best_ckpt: Optional[Path],
    ) -> Tuple[bool, str, str]:
        """Verify training completion status to prevent evaluating intermediate checkpoints.

        Ultralytics saves best.pt whenever validation metrics improve during training.
        An active training run at epoch 15/100 will have best.pt on disk, but must NOT
        be evaluated as a completed experiment.

        Returns:
            Tuple of (is_complete, status, detail).
        """
        if exp_dir is None or not exp_dir.is_dir():
            return False, "PENDING", "Experiment folder not found; training has not yet started."

        # 1. Check for completion summary files
        for summary_name in ("train_stage_summary.json", "fold_master_summary.json"):
            summary_path = exp_dir / summary_name
            if summary_path.is_file():
                try:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    is_smoke = (
                        data.get("smoke_mode") is True
                        or "SMOKE" in str(data.get("protocol", "")).upper()
                        or "SMOKE" in str(data.get("stage", "")).upper()
                        or data.get("smoke_batches_executed") is not None
                    )
                    if is_smoke:
                        return (
                            False,
                            "SMOKE_TEST_CHECKPOINT",
                            "Checkpoint is from a smoke test/debug run, not a completed experiment.",
                        )
                    if best_ckpt and best_ckpt.is_file():
                        return True, "COMPLETED", f"Training completed successfully (verified by {summary_name})."
                    else:
                        return False, "INCOMPLETE", f"Completion summary {summary_name} exists but best.pt is missing."
                except Exception:
                    pass

        # 2. Check for mini-debugging summary
        if (exp_dir / "smoke_master_summary.json").is_file():
            return (
                False,
                "SMOKE_TEST_CHECKPOINT",
                "Checkpoint is from a smoke test/debug run (smoke_master_summary.json present).",
            )

        # 3. Check results.csv row count vs target epochs from args.yaml
        results_csv = exp_dir / "results.csv"
        if results_csv.is_file():
            target_epochs = 100
            args_yaml = exp_dir / "args.yaml"
            if args_yaml.is_file():
                try:
                    with open(args_yaml, "r", encoding="utf-8") as f:
                        args_data = yaml.safe_load(f)
                        if isinstance(args_data, dict) and "epochs" in args_data:
                            target_epochs = int(args_data["epochs"])
                except Exception:
                    pass

            try:
                with open(results_csv, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                completed_epochs = max(0, len(lines) - 1) if len(lines) > 0 else 0
            except Exception:
                completed_epochs = 0

            if completed_epochs < target_epochs:
                return (
                    False,
                    "IN_PROGRESS",
                    f"Training in progress ({completed_epochs}/{target_epochs} epochs completed; best.pt is intermediate).",
                )
            else:
                if best_ckpt and best_ckpt.is_file():
                    return True, "COMPLETED", f"Training completed all {completed_epochs}/{target_epochs} epochs."
                else:
                    return False, "INCOMPLETE", f"Completed {completed_epochs}/{target_epochs} epochs but best.pt is missing."

        # 4. Check last.pt existence (indicates active or incomplete run)
        last_ckpt = (exp_dir / "weights" / "last.pt") if (exp_dir / "weights" / "last.pt").is_file() else (exp_dir / "last.pt")
        if last_ckpt.is_file() and (best_ckpt is None or not best_ckpt.is_file()):
            return False, "IN_PROGRESS", "Training currently active or interrupted (last.pt present; best.pt pending)."

        if last_ckpt.is_file():
            return False, "IN_PROGRESS", "Training active or unconfirmed (last.pt present without completion summary)."

        # 5. Check if best checkpoint exists as standalone
        if best_ckpt and best_ckpt.is_file():
            return True, "COMPLETED", "Standalone checkpoint available."

        return False, "INITIALIZED_NO_CHECKPOINT", "Experiment folder exists but model checkpoint has not been trained."

    def evaluate_condition(
        self,
        variant: str,
        fold_idx: int,
        target_country: str,
        force_reevaluate: bool = False,
        ingest_only: bool = False,
    ) -> ExperimentTargetResult:
        """Evaluate a single experimental condition on its held-out target test set."""
        exp_dir, best_ckpt = self.locate_experiment(variant, fold_idx, target_country)
        data_yaml = resolve_data_yaml(self.views_dir, fold_idx, target_country)

        # 1. Check if cached evaluation metrics already exist
        if exp_dir is not None and exp_dir.is_dir():
            if (exp_dir / "smoke_master_summary.json").is_file():
                return ExperimentTargetResult(
                    variant=variant,
                    fold_idx=fold_idx,
                    target_country=target_country,
                    status="SMOKE_TEST_CHECKPOINT",
                    source="none",
                    checkpoint_path=str(best_ckpt) if best_ckpt else None,
                    experiment_dir=str(exp_dir),
                    data_yaml_path=str(data_yaml) if data_yaml else None,
                    metrics=None,
                    status_detail="Checkpoint is from a smoke test/debug run (smoke_master_summary.json present).",
                )

            cached_metrics_file = exp_dir / "held_out_test_metrics.json"
            if cached_metrics_file.is_file() and not force_reevaluate:
                try:
                    with open(cached_metrics_file, "r", encoding="utf-8") as f:
                        saved_summary = json.load(f)
                    raw_metrics = saved_summary.get("metrics") or {}
                    normalized_metrics = normalize_metrics_dict(raw_metrics, self.target_classes)
                    return ExperimentTargetResult(
                        variant=variant,
                        fold_idx=fold_idx,
                        target_country=target_country,
                        status="COMPLETED",
                        source="cached_test_metrics",
                        checkpoint_path=saved_summary.get("checkpoint_evaluated") or (str(best_ckpt) if best_ckpt else None),
                        experiment_dir=str(exp_dir),
                        data_yaml_path=str(data_yaml) if data_yaml else None,
                        elapsed_seconds=saved_summary.get("elapsed_seconds"),
                        metrics=normalized_metrics,
                        status_detail="Loaded previously evaluated held-out target test metrics.",
                        evaluated_at=saved_summary.get("evaluated_at"),
                    )
                except Exception:
                    pass

        # 2. Verify training completion status (intermediate best.pt guard)
        is_complete, status_label, detail_msg = self.check_training_status(exp_dir, best_ckpt)
        if not is_complete:
            return ExperimentTargetResult(
                variant=variant,
                fold_idx=fold_idx,
                target_country=target_country,
                status=status_label,
                source="none",
                checkpoint_path=str(best_ckpt) if best_ckpt else None,
                experiment_dir=str(exp_dir) if exp_dir else None,
                data_yaml_path=str(data_yaml) if data_yaml else None,
                metrics=None,
                status_detail=detail_msg,
            )

        # 3. Check data.yaml availability for held-out evaluation
        if data_yaml is None or not Path(data_yaml).is_file():
            return ExperimentTargetResult(
                variant=variant,
                fold_idx=fold_idx,
                target_country=target_country,
                status="MISSING_DATA_YAML",
                source="none",
                checkpoint_path=str(best_ckpt),
                experiment_dir=str(exp_dir) if exp_dir else None,
                data_yaml_path=None,
                metrics=None,
                status_detail=f"LOCO view data.yaml not found under {self.views_dir}.",
            )

        # 4. If ingest_only, do not launch PyTorch/YOLO evaluation
        if ingest_only:
            return ExperimentTargetResult(
                variant=variant,
                fold_idx=fold_idx,
                target_country=target_country,
                status="CHECKPOINT_AVAILABLE_NOT_EVALUATED",
                source="none",
                checkpoint_path=str(best_ckpt),
                experiment_dir=str(exp_dir) if exp_dir else None,
                data_yaml_path=str(data_yaml),
                metrics=None,
                status_detail="Checkpoint best.pt is available, but evaluation was skipped because --ingest-only was set.",
            )

        # 5. Execute held-out target evaluation using YOLOv8
        dev = self.resolve_device()
        print(f"\n[EVALUATION] Variant {variant} | Fold {fold_idx} ({target_country})")
        print(f"             Checkpoint: {best_ckpt}")
        print(f"             Target Test Split: {data_yaml} (split='test')")
        print(f"             Device: {dev} | Batch: {self.batch_size} | ImgSz: {self.img_size}")

        start_time = time.perf_counter()
        try:
            model = YOLO(str(best_ckpt))
            val_results = model.val(
                data=str(data_yaml),
                split="test",
                imgsz=self.img_size,
                batch=self.batch_size,
                device=dev,
                plots=False,
                save_json=False,
                verbose=False,
            )
            elapsed_sec = round(time.perf_counter() - start_time, 2)
            metrics = extract_detection_metrics(val_results, self.target_classes)
            now_iso = datetime.now(timezone.utc).isoformat()

            # Save held_out_test_metrics.json in experiment directory
            if exp_dir is not None:
                exp_dir.mkdir(parents=True, exist_ok=True)
                summary_data = {
                    "stage": "STAGE_3_HELD_OUT_TARGET_EVALUATION",
                    "variant": variant,
                    "fold_idx": fold_idx,
                    "target_country": target_country,
                    "split": "test",
                    "checkpoint_evaluated": str(best_ckpt),
                    "elapsed_seconds": elapsed_sec,
                    "metrics": metrics,
                    "evaluated_at": now_iso,
                }
                metrics_out = exp_dir / "held_out_test_metrics.json"
                with open(metrics_out, "w", encoding="utf-8") as f:
                    json.dump(summary_data, f, indent=2)

            print(f"             -> mAP@0.5: {metrics['map50']:.4f} | mAP@0.5:0.95: {metrics['map50_95']:.4f} | F1: {metrics['f1']:.4f}")

            return ExperimentTargetResult(
                variant=variant,
                fold_idx=fold_idx,
                target_country=target_country,
                status="COMPLETED",
                source="evaluated_checkpoint",
                checkpoint_path=str(best_ckpt),
                experiment_dir=str(exp_dir) if exp_dir else None,
                data_yaml_path=str(data_yaml),
                elapsed_seconds=elapsed_sec,
                metrics=metrics,
                status_detail="Successfully evaluated on held-out target test set.",
                evaluated_at=now_iso,
            )

        except Exception as e:
            elapsed_sec = round(time.perf_counter() - start_time, 2)
            print(f"             [ERROR] Evaluation failed: {e}")
            return ExperimentTargetResult(
                variant=variant,
                fold_idx=fold_idx,
                target_country=target_country,
                status="EVALUATION_ERROR",
                source="none",
                checkpoint_path=str(best_ckpt),
                experiment_dir=str(exp_dir) if exp_dir else None,
                data_yaml_path=str(data_yaml),
                elapsed_seconds=elapsed_sec,
                metrics=None,
                status_detail=f"Evaluation execution crashed: {e}",
            )

    def run_benchmark_evaluation(
        self,
        force_reevaluate: bool = False,
        ingest_only: bool = False,
    ) -> Dict[str, Any]:
        """Execute evaluation across all configured variants and folds."""
        hw = check_gpu_environment()
        now_iso = datetime.now(timezone.utc).isoformat()

        all_results: Dict[str, Dict[str, Any]] = {}
        variant_summaries: Dict[str, Dict[str, Any]] = {}

        for variant in self.variants:
            comp_count = 0
            map50_scores = []
            map50_95_scores = []
            f1_scores = []

            for fold_idx in self.folds:
                country = self.countries[fold_idx]
                key = f"{variant}_fold_{fold_idx}_{country}"

                res = self.evaluate_condition(
                    variant=variant,
                    fold_idx=fold_idx,
                    target_country=country,
                    force_reevaluate=force_reevaluate,
                    ingest_only=ingest_only,
                )
                all_results[key] = res.to_dict()

                if res.status == "COMPLETED" and res.metrics is not None:
                    comp_count += 1
                    map50_scores.append(float(res.metrics["map50"]))
                    map50_95_scores.append(float(res.metrics["map50_95"]))
                    f1_scores.append(float(res.metrics["f1"]))

            variant_summaries[variant] = {
                "variant": variant,
                "completed_folds": comp_count,
                "total_folds": len(self.folds),
                "macro_map50": round(float(np.mean(map50_scores)), 4) if map50_scores else None,
                "macro_map50_95": round(float(np.mean(map50_95_scores)), 4) if map50_95_scores else None,
                "macro_f1": round(float(np.mean(f1_scores)), 4) if f1_scores else None,
            }

        return {
            "benchmark": "CCARC_LOCO_CROSS_COUNTRY_EVALUATION",
            "timestamp": now_iso,
            "environment": hw,
            "config": {
                "views_dir": str(self.views_dir),
                "baseline_dir": str(self.baseline_dir),
                "ccarc_dir": str(self.ccarc_dir),
                "variants": self.variants,
                "folds": self.folds,
                "device": self.device,
                "resolved_device": self.resolve_device(),
                "batch_size": self.batch_size,
                "img_size": self.img_size,
                "target_classes": self.target_classes,
                "countries": self.countries,
            },
            "variant_summaries": variant_summaries,
            "experiments": all_results,
        }

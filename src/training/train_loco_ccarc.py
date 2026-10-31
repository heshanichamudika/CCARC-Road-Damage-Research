#!/usr/bin/env python3
"""CLI Training Launcher for CCARC Proposed Methods and Ablation Experiments.

Usage Examples:
    # 1. Dry run verification of Fold 0 (China) under Full CCARC (E4)
    python -m src.training.train_loco_ccarc \\
        --views-dir "/content/datasets/loco_views" \\
        --output-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/ccarc_experiments" \\
        --fold 0 \\
        --variant E4 \\
        --dry-run

    # 2. Train Single Fold 0 under CBS ablation (E1) on Google Colab Pro
    python -m src.training.train_loco_ccarc \\
        --views-dir "/content/datasets/loco_views" \\
        --output-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/ccarc_experiments" \\
        --local-run-dir "/content/runs/ccarc" \\
        --fold 0 \\
        --variant E1 \\
        --epochs 100 \\
        --batch-size 16 \\
        --device 0

    # 3. Train Full CCARC (E4) on Fold 0 with explicit held-out target evaluation
    python -m src.training.train_loco_ccarc \\
        --views-dir "/content/datasets/loco_views" \\
        --output-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/ccarc_experiments" \\
        --local-run-dir "/content/runs/ccarc" \\
        --fold 0 \\
        --variant E4 \\
        --all-stages \\
        --device 0
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from src.data.loco_splitter import DEFAULT_TARGET_CLASSES, LOCO_COUNTRIES
from src.models.ccarc_consistency import ReliabilityConsistencyConfig
from src.models.ccarc_loss import ClassAwareLearningConfig
from src.models.ccarc_perturbations import PhotometricPerturbationConfig
from src.training.baseline_config import (
    check_gpu_environment,
    resolve_fold_data_yaml,
)
from src.training.ccarc_config import CCARCTrainingConfig, VALID_VARIANTS
from src.training.ccarc_trainer import CCARCLOCOTrainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CCARC: Train and evaluate proposed YOLOv8s cross-country models."
    )
    # Fold Selection
    fold_group = parser.add_mutually_exclusive_group(required=True)
    fold_group.add_argument(
        "--fold",
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        help="Fold index (0=China, 1=Czech, 2=India, 3=Japan, 4=Norway, 5=United_States).",
    )
    fold_group.add_argument(
        "--country",
        type=str,
        choices=list(LOCO_COUNTRIES),
        help="Held-out target country to execute.",
    )
    fold_group.add_argument(
        "--all-folds",
        action="store_true",
        help="Execute all six LOCO folds sequentially.",
    )

    # Experiment Variant / Ablation
    parser.add_argument(
        "--variant",
        type=str,
        default="E4",
        choices=list(VALID_VARIANTS),
        help="Ablation variant: E0 (baseline), E1 (CBS), E2 (CAL), E3 (RWC), E4 (Full CCARC).",
    )

    # Directories
    parser.add_argument(
        "--views-dir",
        type=Path,
        default=Path("experiments/loco_splits"),
        help="Directory containing runtime LOCO views and data.yaml files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/ccarc_experiments"),
        help="Persistent output directory (distinct from baseline).",
    )
    parser.add_argument(
        "--local-run-dir",
        type=Path,
        default=None,
        help="Local high-speed SSD directory (e.g. /content/runs/ccarc on Colab).",
    )

    # Base Hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs (default: 100).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--img-size", type=int, default=640, help="Image resolution (default: 640).")
    parser.add_argument("--device", type=str, default="auto", help="Execution device: 'auto', '0', 'cuda', 'cpu'.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42).")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience (default: 20).")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4).")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate (default: 0.01).")
    parser.add_argument("--optimizer", type=str, default="SGD", help="Explicit optimizer (default: 'SGD').")

    # CAL Hyperparameters
    parser.add_argument("--cal-alpha", type=float, default=0.5, help="CAL power scaling factor (default: 0.5).")
    parser.add_argument("--cal-epsilon", type=float, default=1.0, help="CAL numerical stability offset (default: 1.0).")
    parser.add_argument("--cal-max-weight", type=float, default=10.0, help="CAL maximum weight bound (default: 10.0).")

    # RWC Hyperparameters
    parser.add_argument("--rwc-weight", type=float, default=1.0, help="RWC consistency loss weight lambda (default: 1.0).")
    parser.add_argument("--rwc-threshold", type=float, default=0.25, help="RWC reliability threshold tau_rel (default: 0.25).")
    parser.add_argument("--rwc-temperature", type=float, default=1.0, help="RWC teacher temperature scaling (default: 1.0).")
    parser.add_argument("--rwc-warmup", type=int, default=5, help="RWC warmup epochs (default: 5).")

    # Workflow Flags
    parser.add_argument("--dry-run", action="store_true", help="Validate configurations and paths without training.")
    parser.add_argument("--resume", action="store_true", help="Resume from weights/last.pt if present.")
    parser.add_argument("--eval-target", action="store_true", help="Explicitly evaluate Stage 3 held-out target.")
    parser.add_argument("--all-stages", action="store_true", help="Run Stage 1 (train), Stage 2 (source val), and Stage 3 (target test).")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate existing best.pt.")
    parser.add_argument("--force-restart", action="store_true", help="Force restart training even if completed.")

    return parser.parse_args(argv)


def execute_single_fold(
    fold_idx: int,
    target_country: str,
    args: argparse.Namespace,
    views_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Configure and execute CCARC pipeline for a single LOCO fold."""
    data_yaml = resolve_fold_data_yaml(views_dir, fold_idx, target_country)
    experiment_name = f"fold_{fold_idx}_{target_country}_{args.variant}"

    # Build component configs
    cal_cfg = ClassAwareLearningConfig(
        alpha=args.cal_alpha,
        epsilon=args.cal_epsilon,
        max_weight=args.cal_max_weight,
    )
    rwc_cfg = ReliabilityConsistencyConfig(
        consistency_weight=args.rwc_weight,
        confidence_threshold=args.rwc_threshold,
        temperature=args.rwc_temperature,
        warmup_epochs=args.rwc_warmup,
    )

    config = CCARCTrainingConfig(
        experiment_variant=args.variant,
        cal_config=cal_cfg,
        rwc_config=rwc_cfg,
        model_name="yolov8s.pt",
        img_size=args.img_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        seed=args.seed,
        device=args.device,
        workers=args.workers,
        fold_idx=fold_idx,
        target_country=target_country,
        data_yaml_path=str(data_yaml) if data_yaml else "",
        project_dir=str(output_dir),
        local_run_dir=str(args.local_run_dir) if args.local_run_dir else "",
        experiment_name=experiment_name,
        resume=args.resume,
    )

    trainer = CCARCLOCOTrainer(config)

    if args.dry_run:
        print(f"\n[CCARC DRY RUN] Validating Fold {fold_idx} ({target_country}) - Variant {args.variant}...")
        report = trainer.dry_run()
        print(f"Status: {report['status']} | Checks: {report['checks']}")
        if report["errors"]:
            print(f"Errors: {report['errors']}")
        if report["warnings"]:
            print(f"Warnings: {report['warnings']}")
        return {"dry_run_report": report}

    return trainer.run_fold(
        dry_run=False,
        eval_target=(args.eval_target or args.all_stages),
        eval_only=args.eval_only,
        force_restart=args.force_restart,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    views_dir = args.views_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("CCARC: PROPOSED METHOD & ABLATION TRAINING LAUNCHER")
    print("=" * 75)
    print(f"Ablation Variant:  {args.variant}")
    print(f"Views Directory:   {views_dir}")
    print(f"Output Directory:  {output_dir}")
    print(f"Local Run Dir:     {args.local_run_dir}")
    print(f"Epochs:            {args.epochs} | Batch Size: {args.batch_size} | Img Size: {args.img_size}")
    print(f"Device:            {args.device} | Seed: {args.seed} | Optimizer: {args.optimizer}")
    print(f"Dry Run Mode:      {args.dry_run}")
    print(f"Eval Target Mode:  {args.eval_target or args.all_stages}")

    hw = check_gpu_environment()
    print(f"CUDA Available:    {hw['cuda_available']} (PyTorch: {hw['pytorch_version']}, Ultralytics: {hw['ultralytics_version']})")
    print("=" * 75)

    if args.all_folds:
        folds_to_run = list(enumerate(LOCO_COUNTRIES))
    elif args.country:
        idx = LOCO_COUNTRIES.index(args.country)
        folds_to_run = [(idx, args.country)]
    else:
        folds_to_run = [(args.fold, LOCO_COUNTRIES[args.fold])]

    all_results: Dict[str, Any] = {}
    any_failures = False

    for fold_idx, country in folds_to_run:
        print(f"\n>>> Executing Fold {fold_idx}: Target = {country} | Variant = {args.variant}")
        res = execute_single_fold(fold_idx, country, args, views_dir, output_dir)
        all_results[f"fold_{fold_idx}_{country}_{args.variant}"] = res

        if args.dry_run:
            if res.get("dry_run_report", {}).get("status") != "PASS":
                any_failures = True
        else:
            if args.eval_target or args.all_stages:
                if res.get("stage_3_held_out_target", {}).get("metrics") is None:
                    any_failures = True
            else:
                if res.get("stage_2_source_val", {}).get("metrics") is None:
                    any_failures = True

    if len(folds_to_run) > 1 and not args.dry_run:
        summary_file = output_dir / f"ccarc_{args.variant}_benchmark_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "benchmark": f"CCARC_LOCO_{args.variant}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "results": all_results,
            }, f, indent=2)
        print(f"\n[SUMMARY] Saved CCARC benchmark summary to: {summary_file}")

    print("\n" + "=" * 75)
    status_str = "COMPLETED" if not any_failures else "FAILED / INCOMPLETE"
    print(f"CCARC {args.variant} EXECUTION FINISHED: Status = {status_str}")
    print("=" * 75)

    return 1 if any_failures else 0


if __name__ == "__main__":
    sys.exit(main())

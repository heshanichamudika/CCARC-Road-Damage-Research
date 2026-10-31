#!/usr/bin/env python3
"""CLI Training Launcher for YOLOv8s Six-Country LOCO Baseline Experiments.

Usage Examples:
    # 1. Dry run verification of Fold 0 (China)
    python -m src.training.train_loco_baseline \\
        --views-dir "/content/datasets/loco_views" \\
        --output-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/yolov8s_baseline" \\
        --fold 0 \\
        --dry-run

    # 2. Train Single Fold 0 (China) on Google Colab Pro L4 GPU
    python -m src.training.train_loco_baseline \\
        --views-dir "/content/datasets/loco_views" \\
        --output-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/yolov8s_baseline" \\
        --fold 0 \\
        --epochs 100 \\
        --batch-size 16 \\
        --device 0

    # 3. Sequential Six-Fold Benchmark Execution
    python -m src.training.train_loco_baseline \\
        --views-dir "/content/datasets/loco_views" \\
        --output-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/yolov8s_baseline" \\
        --all-folds \\
        --epochs 100 \\
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
from src.training.baseline_config import (
    BaselineTrainingConfig,
    check_gpu_environment,
    resolve_fold_data_yaml,
)
from src.training.yolov8_trainer import YOLOv8LOCOTrainer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate YOLOv8s baseline across six-country LOCO folds."
    )
    # Fold Selection
    fold_group = parser.add_mutually_exclusive_group(required=True)
    fold_group.add_argument(
        "--fold",
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        help="Fold index to execute (0=China, 1=Czech, 2=India, 3=Japan, 4=Norway, 5=United_States).",
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

    # Directories
    parser.add_argument(
        "--views-dir",
        type=Path,
        default=Path("experiments/loco_splits"),
        help="Directory containing runtime LOCO views and data.yaml files (e.g. /content/datasets/loco_views).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/yolov8s_baseline"),
        help="Persistent output directory on Google Drive where checkpoints, metrics and configs are saved.",
    )
    parser.add_argument(
        "--local-run-dir",
        type=Path,
        default=None,
        help="Local fast directory for training execution before checkpoints sync to persistent output-dir (e.g. /content/runs/yolov8s_baseline on Colab).",
    )

    # Hyperparameter Overrides
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs (default: 100).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--img-size", type=int, default=640, help="Image resolution (default: 640).")
    parser.add_argument("--device", type=str, default="auto", help="Execution device: 'auto', '0', 'cuda', 'cpu'.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42).")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience epochs on source val (default: 20).")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader worker processes (default: 4).")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate (default: 0.01).")
    parser.add_argument("--optimizer", type=str, default="SGD", help="Explicit optimizer ('SGD', 'AdamW', default: 'SGD').")

    # Workflow Flags
    parser.add_argument("--dry-run", action="store_true", help="Validate configurations and paths without starting training.")
    parser.add_argument("--resume", action="store_true", help="Resume training from weights/last.pt if present.")
    parser.add_argument("--eval-target", action="store_true", help="Explicitly execute Stage 3 held-out target country test evaluation.")
    parser.add_argument("--all-stages", action="store_true", help="Execute all three stages: Stage 1 training, Stage 2 source validation, and Stage 3 held-out target evaluation.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and run source val (and held-out test if requested) using existing best.pt.")
    parser.add_argument("--force-restart", action="store_true", help="Force restart training even if experiment is already completed.")

    return parser.parse_args(argv)


def execute_single_fold(
    fold_idx: int,
    target_country: str,
    args: argparse.Namespace,
    views_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Configure and execute pipeline for one fold."""
    data_yaml = resolve_fold_data_yaml(views_dir, fold_idx, target_country)
    experiment_name = f"fold_{fold_idx}_{target_country}"

    config = BaselineTrainingConfig(
        model_name="yolov8s.pt",
        img_size=args.img_size,
        pretrained=True,
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
        local_run_dir=str(args.local_run_dir) if args.local_run_dir else None,
        experiment_name=experiment_name,
        resume=args.resume,
    )

    trainer = YOLOv8LOCOTrainer(config)

    if args.dry_run:
        print(f"\n[DRY RUN] Validating Fold {fold_idx} ({target_country})...")
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
    print("CCARC: YOLOv8s SIX-COUNTRY LOCO BASELINE TRAINING LAUNCHER")
    print("=" * 75)
    print(f"Views Directory:   {views_dir}")
    print(f"Output Directory:  {output_dir}")
    print(f"Local Run Dir:     {args.local_run_dir}")
    print(f"Epochs:            {args.epochs} | Batch Size: {args.batch_size} | Img Size: {args.img_size}")
    print(f"Optimizer:         {args.optimizer} | lr0: {args.lr0} | Seed: {args.seed} | Patience: {args.patience}")
    print(f"Device:            {args.device}")
    print(f"Dry Run Mode:      {args.dry_run}")
    print(f"Resume Mode:       {args.resume}")
    print(f"Eval Target Mode:  {args.eval_target or args.all_stages}")
    print(f"Eval Only Mode:    {args.eval_only}")

    # Inspect hardware environment
    hw = check_gpu_environment()
    print(f"CUDA Available:    {hw['cuda_available']} (PyTorch: {hw['pytorch_version']}, Ultralytics: {hw['ultralytics_version']})")
    if hw["cuda_available"]:
        for g in hw["gpus"]:
            print(f"  GPU [{g['device_index']}]: {g['name']} ({g['total_memory_gb']} GB VRAM)")
    print("=" * 75)

    # Determine folds to execute
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
        print(f"\n>>> Executing Fold {fold_idx}: Target = {country}")
        res = execute_single_fold(fold_idx, country, args, views_dir, output_dir)
        all_results[f"fold_{fold_idx}_{country}"] = res

        if args.dry_run:
            if res.get("dry_run_report", {}).get("status") != "PASS":
                any_failures = True
        else:
            # Check for stage completion
            if (args.eval_target or args.all_stages):
                if res.get("stage_3_held_out_target", {}).get("metrics") is None:
                    any_failures = True
            else:
                if res.get("stage_2_source_val", {}).get("metrics") is None:
                    any_failures = True

    # Compile benchmark summary if multiple folds or full run
    if len(folds_to_run) > 1 and not args.dry_run:
        summary_data = []
        for name, r in all_results.items():
            t_m = r.get("stage_3_held_out_target", {}).get("metrics", {})
            v_m = r.get("stage_2_source_val", {}).get("metrics", {})
            summary_data.append({
                "fold": name,
                "target_country": r.get("target_country"),
                "source_val_mAP50": v_m.get("map50"),
                "source_val_mAP50_95": v_m.get("map50_95"),
                "held_out_test_mAP50": t_m.get("map50"),
                "held_out_test_mAP50_95": t_m.get("map50_95"),
            })

        summary_file = output_dir / "six_fold_baseline_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "benchmark": "YOLOv8s_Source_Only_LOCO_Baseline",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "folds": summary_data,
            }, f, indent=2)
        print(f"\n[SUMMARY] Saved Six-Fold Baseline Summary to: {summary_file}")

    print("\n" + "=" * 75)
    status_str = "COMPLETED" if not any_failures else "FAILED / INCOMPLETE"
    print(f"CCARC LOCO EXECUTION FINISHED: Status = {status_str}")
    print("=" * 75)

    return 1 if any_failures else 0


if __name__ == "__main__":
    sys.exit(main())

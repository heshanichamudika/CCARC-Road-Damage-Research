#!/usr/bin/env python3
"""CLI Evaluation Launcher for CCARC Leave-One-Country-Out (LOCO) Benchmark.

Evaluates best saved checkpoints across all five variants (E0–E4) on their respective
unseen held-out target country test splits, computing mAP@0.5, mAP@0.5:0.95, Precision,
Recall, F1-score, and per-class AP metrics, and generating publication comparison tables.

Usage Examples:
    # 1. Ingest completed E0 baseline and active E4 results without launching inference:
    python -m src.evaluation.evaluate_loco \
        --views-dir "/content/rdd2022_recovery/loco_views" \
        --baseline-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/yolov8s_baseline_full100" \
        --ccarc-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/ccarc_full_training" \
        --output-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/evaluation_results" \
        --ingest-only

    # 2. Evaluate completed E0 baseline and newly finished E4 on Colab GPU:
    python -m src.evaluation.evaluate_loco \
        --views-dir "/content/rdd2022_recovery/loco_views" \
        --baseline-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/yolov8s_baseline_full100" \
        --ccarc-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/ccarc_full_training" \
        --output-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/evaluation_results" \
        --device 0

    # 3. Evaluate specifically Fold 4 (Norway) under E4:
    python -m src.evaluation.evaluate_loco \
        --views-dir "/content/rdd2022_recovery/loco_views" \
        --ccarc-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/ccarc_full_training" \
        --output-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/evaluation_results" \
        --variants E4 \
        --folds 4 \
        --device 0
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

from src.data.loco_splitter import LOCO_COUNTRIES
from src.evaluation.evaluator import LOCOBenchmarkEvaluator
from src.evaluation.report_generator import EvaluationReportGenerator
from src.training.baseline_config import check_gpu_environment
from src.training.ccarc_config import VALID_VARIANTS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CCARC: Evaluate YOLOv8s LOCO benchmarks and generate comparison tables."
    )
    # Directories
    parser.add_argument(
        "--views-dir",
        type=Path,
        default=Path("experiments/loco_splits"),
        help="Directory containing runtime LOCO views and data.yaml files.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("experiments/yolov8s_baseline_full100"),
        help="Directory where E0 baseline checkpoints and metrics are saved.",
    )
    parser.add_argument(
        "--ccarc-dir",
        type=Path,
        default=Path("experiments/ccarc_full_training"),
        help="Directory where CCARC experiments (E1-E4) are saved.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/evaluation_results"),
        help="Directory to save aggregated evaluation JSON, CSV, and report tables.",
    )

    # Variant and Fold Selection
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(VALID_VARIANTS),
        choices=list(VALID_VARIANTS),
        help="Experiment variants to evaluate (default: E0 E1 E2 E3 E4).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
        choices=[0, 1, 2, 3, 4, 5],
        help="Fold indices to evaluate (default: 0 1 2 3 4 5).",
    )

    # Evaluation Runtime Settings
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Hardware execution device: 'auto', '0', 'cuda', 'cpu'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for validation inference (default: 16).",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Evaluation image resolution (default: 640).",
    )

    # Workflow Flags
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation with YOLO even if held_out_test_metrics.json exists.",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only aggregate and report existing metrics without running model inference.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify checkpoint paths and test splits without executing evaluation.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    views_dir = args.views_dir.resolve()
    baseline_dir = args.baseline_dir.resolve()
    ccarc_dir = args.ccarc_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CCARC: LEAVE-ONE-COUNTRY-OUT (LOCO) BENCHMARK EVALUATOR")
    print("=" * 80)
    print(f"Variants:         {args.variants}")
    print(f"Folds:            {args.folds}")
    print(f"Views Directory:  {views_dir}")
    print(f"Baseline Dir:     {baseline_dir}")
    print(f"CCARC Dir:        {ccarc_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Device:           {args.device} | Batch: {args.batch_size} | ImgSz: {args.img_size}")
    print(f"Ingest Only:      {args.ingest_only} | Force Re-evaluate: {args.force}")

    hw = check_gpu_environment()
    print(f"CUDA Available:   {hw['cuda_available']} (PyTorch: {hw['pytorch_version']}, Ultralytics: {hw['ultralytics_version']})")
    print("=" * 80)

    evaluator = LOCOBenchmarkEvaluator(
        views_dir=views_dir,
        baseline_dir=baseline_dir,
        ccarc_dir=ccarc_dir,
        variants=args.variants,
        folds=args.folds,
        device=args.device,
        batch_size=args.batch_size,
        img_size=args.img_size,
    )

    if args.dry_run:
        print("\n[DRY RUN] Scanning checkpoint status across requested conditions...")
        for v in args.variants:
            for f in args.folds:
                country = LOCO_COUNTRIES[f]
                exp_dir, best_ckpt = evaluator.locate_experiment(v, f, country)
                ckpt_str = str(best_ckpt) if best_ckpt else "None"
                cached_metrics = (exp_dir / "held_out_test_metrics.json").is_file() if exp_dir else False
                print(f"  {v} | Fold {f} ({country:14s}) -> Dir: {exp_dir.name if exp_dir else 'None':24s} | Checkpoint: {ckpt_str} | Cached Metrics: {cached_metrics}")
        print("\nDry run completed successfully.")
        return 0

    # Execute benchmark evaluation
    benchmark_results = evaluator.run_benchmark_evaluation(
        force_reevaluate=args.force,
        ingest_only=args.ingest_only,
    )

    # Generate and export all machine-readable reports and comparison tables
    generator = EvaluationReportGenerator(
        results_data=benchmark_results,
        variants=args.variants,
        countries=[LOCO_COUNTRIES[f] for f in args.folds],
    )
    exported_files = generator.export_all(output_dir)

    print("\n" + "=" * 80)
    print("EVALUATION PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    for v, summary in benchmark_results.get("variant_summaries", {}).items():
        comp = summary["completed_folds"]
        tot = summary["total_folds"]
        m50 = f"{summary['macro_map50']:.4f}" if summary["macro_map50"] is not None else "N/A"
        m95 = f"{summary['macro_map50_95']:.4f}" if summary["macro_map50_95"] is not None else "N/A"
        f1 = f"{summary['macro_f1']:.4f}" if summary["macro_f1"] is not None else "N/A"
        print(f"  {v:4s}: {comp}/{tot} Completed | Macro mAP@0.5: {m50} | Macro mAP@0.5:0.95: {m95} | Macro F1: {f1}")

    print("\nExported Artifacts:")
    for k, p in exported_files.items():
        print(f"  - {k:15s}: {p}")
    print("=" * 80)

    # Print markdown table preview to stdout
    print("\n" + generator.generate_markdown_table())

    return 0


if __name__ == "__main__":
    sys.exit(main())

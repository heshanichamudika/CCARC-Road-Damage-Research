#!/usr/bin/env python3
"""CLI Command to Generate Six-Country Leave-One-Country-Out (LOCO) Dataset Splits.

Usage example (Local or Google Colab):
    python -m src.data.create_loco_splits \\
        --manifest-path "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/yolo_conversion/yolo_conversion_manifest.csv" \\
        --output-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/loco_splits" \\
        --val-ratio 0.20 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.data.loco_splitter import (
    DEFAULT_TARGET_CLASSES,
    LOCO_COUNTRIES,
    RDD2022LOCOSplitter,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partition RDD2022 dataset into Six-Country Leave-One-Country-Out (LOCO) folds."
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to yolo_conversion_manifest.csv produced by T02.16.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/RDD2022/loco"),
        help="Destination directory where LOCO folds, persistent manifests, and summary statistics will be generated.",
    )
    parser.add_argument(
        "--views-dir",
        type=Path,
        default=None,
        help="Optional directory for runtime YOLO dataset views (symlinks and data.yaml), e.g. '/content/datasets/loco_views' on Colab fast local SSD. Defaults to --output-dir.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Fraction of source country data allocated to validation (default: 0.20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic, reproducible splitting (default: 42).",
    )
    parser.add_argument(
        "--incomplete-label-policy",
        choices=["include", "exclude"],
        default="exclude",
        help="Handling of images flagged with incomplete_label_risk (default: 'exclude' for conservative baseline; 'include' for ablation studies).",
    )
    parser.add_argument(
        "--include-empty-candidates",
        action="store_true",
        help="Include empty XML images (candidate negatives) in the supervised pool. Default: False.",
    )
    parser.add_argument(
        "--include-quarantined-non-target",
        action="store_true",
        help="Include quarantined non-target images in the pool. Default: False.",
    )
    parser.add_argument(
        "--no-clean-output",
        action="store_true",
        help="Do not clean the output directory before generating folds.",
    )
    return parser.parse_args(args)


def main() -> int:
    args = parse_args()

    manifest_path = args.manifest_path.resolve()
    output_dir = args.output_dir.resolve()

    if not manifest_path.is_file():
        print(f"Error: Manifest file does not exist: {manifest_path}")
        return 1

    views_dir = args.views_dir.resolve() if args.views_dir is not None else None

    print("=" * 70)
    print("CCARC: SIX-COUNTRY LEAVE-ONE-COUNTRY-OUT (LOCO) DATASET PARTITIONING")
    print("=" * 70)
    print(f"Manifest Path:           {manifest_path}")
    print(f"Output Directory:        {output_dir}")
    print(f"Views Directory:         {views_dir if views_dir else output_dir}")
    print(f"Validation Ratio:        {args.val_ratio:.2f}")
    print(f"Random Seed:             {args.seed}")
    print(f"Incomplete Label Policy: {args.incomplete_label_policy}")
    print(f"Include Empty XML:       {args.include_empty_candidates}")
    print(f"Clean Output Dir:        {not args.no_clean_output}")
    print("=" * 70)

    df_manifest = pd.read_csv(manifest_path)

    splitter = RDD2022LOCOSplitter(
        val_ratio=args.val_ratio,
        seed=args.seed,
        incomplete_label_policy=args.incomplete_label_policy,
        include_empty_candidates=args.include_empty_candidates,
        include_quarantined_non_target=args.include_quarantined_non_target,
    )

    master_df, summary = splitter.run_all_folds(
        manifest_df=df_manifest,
        output_dir=output_dir,
        views_dir=views_dir,
        clean_output=not args.no_clean_output,
    )

    print("\n" + "=" * 70)
    print("LOCO FOLDS SUMMARY")
    print("=" * 70)
    for fold in summary["folds"]:
        idx = fold["fold_idx"]
        target = fold["target_country"]
        counts = fold["counts"]
        print(
            f"Fold {idx} (Target: {target:<13}) -> Train: {counts['train']:<6} | Val: {counts['val']:<6} | Test: {counts['test']:<6} | YAML: {fold['data_yaml']}"
        )

    all_passed = summary["totals"]["all_leak_checks_passed"]
    print(f"\nAll Leak Checks Passed: {all_passed}")
    if all_passed:
        print("[SUCCESS] Six LOCO folds successfully created with verified zero leakage.")
        return 0
    else:
        print("[ERROR] Leakage check failed on one or more folds!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""CLI Command to Validate RDD2022 Dataset and Generate Manifest & Statistics.

Usage example (Local or Google Colab):
    python -m src.data.validate_rdd2022 \\
        --data-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/datasets/RDD2022/extracted" \\
        --output-dir "data/processed/RDD2022"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.rdd2022_validator import (
    DEFAULT_TARGET_CLASSES,
    RDD2022Validator,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate RDD2022 dataset and generate research manifest & statistics."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the extracted RDD2022 dataset directory containing subset folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/RDD2022"),
        help="Output directory where manifest.csv and summary.json will be saved.",
    )
    parser.add_argument(
        "--skip-image-decode",
        action="store_true",
        help="Skip image OpenCV decoding verification (useful for rapid dry-runs).",
    )
    parser.add_argument(
        "--target-classes",
        nargs="+",
        default=list(DEFAULT_TARGET_CLASSES),
        help="Target damage classes to track (default: D00 D10 D20 D40).",
    )
    return parser.parse_args(args)


def main() -> int:
    args = parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not data_dir.is_dir():
        print(f"Error: Dataset directory does not exist: {data_dir}")
        return 1

    print("=" * 70)
    print("CCARC: RDD2022 DATASET VALIDATION & MANIFEST GENERATION")
    print("=" * 70)
    print(f"Dataset Root:    {data_dir}")
    print(f"Output Dir:      {output_dir}")
    print(f"Verify Images:   {not args.skip_image_decode}")
    print(f"Target Classes:  {args.target_classes}")
    print("=" * 70)

    validator = RDD2022Validator(
        target_classes=args.target_classes,
        verify_images=not args.skip_image_decode,
    )

    df, summary = validator.run(data_root=data_dir, output_dir=output_dir)

    totals = summary.get("totals", {})
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total Images:           {totals.get('total_images', 0):,}")
    print(f"Train Split:            {totals.get('train_images', 0):,}")
    print(f"Test Split:             {totals.get('test_images', 0):,}")
    print(f"LOCO-Eligible:          {totals.get('loco_eligible_images', 0):,}")
    print(f"Status Breakdown:       {totals.get('annotation_status_counts', {})}")
    print(f"Target Class Boxes:     {summary.get('by_class', {}).get('target', {})}")
    print(f"Non-Target Boxes:       {totals.get('total_non_target_boxes', 0):,}")
    print(f"Invalid Bounding Boxes: {totals.get('total_invalid_boxes', 0)}")

    if summary.get("invalid_boxes"):
        print("\nRecorded Invalid Bounding Boxes:")
        for inv in summary["invalid_boxes"]:
            print(f"  - {inv['image_id']} ({inv['country']}/{inv['subset']}): {inv['class_name']} -> {inv['error']}")

    print("\n[SUCCESS] Manifest and dataset statistics generated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

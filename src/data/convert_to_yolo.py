#!/usr/bin/env python3
"""CLI Command to Clean RDD2022 Pascal VOC Annotations and Convert to YOLO Format.

Usage example (Local or Google Colab):
    python -m src.data.convert_to_yolo \\
        --data-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/datasets/RDD2022/extracted" \\
        --output-dir "data/processed/RDD2022/yolo"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.yolo_converter import (
    DEFAULT_TARGET_CLASSES,
    RDD2022YOLOConverter,
)


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean RDD2022 Pascal VOC annotations and convert to YOLO format."
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
        default=Path("data/processed/RDD2022/yolo"),
        help="Output directory where labels/, conversion manifest, and summary JSON will be stored.",
    )
    parser.add_argument(
        "--include-empty-as-negatives",
        action="store_true",
        help="Generate 0-byte label files for empty XML images (candidate negatives). Default: False.",
    )
    parser.add_argument(
        "--clamp-out-of-bounds",
        action="store_true",
        help="Clamp minor out-of-bounds coordinates to image dimensions instead of strictly rejecting them.",
    )
    parser.add_argument(
        "--no-clean-output",
        action="store_true",
        help="Do not clean the output labels directory before running conversion.",
    )
    parser.add_argument(
        "--target-classes",
        nargs="+",
        default=list(DEFAULT_TARGET_CLASSES),
        help="Target damage classes to convert (default: D00 D10 D20 D40).",
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
    print("CCARC: RDD2022 CLEANING & YOLO LABEL CONVERSION")
    print("=" * 70)
    print(f"Dataset Root:                {data_dir}")
    print(f"Output Directory:            {output_dir}")
    print(f"Clean Existing Labels:       {not args.no_clean_output}")
    print(f"Include Empty as Negatives:  {args.include_empty_as_negatives}")
    print(f"Clamp Out-of-Bounds:         {args.clamp_out_of_bounds}")
    print(f"Target Classes:              {args.target_classes}")
    print("=" * 70)

    converter = RDD2022YOLOConverter(
        target_classes=args.target_classes,
        include_empty_as_negatives=args.include_empty_as_negatives,
        quarantine_non_target=True,
        clamp_out_of_bounds=args.clamp_out_of_bounds,
    )

    df, summary = converter.convert_dataset(
        data_root=data_dir,
        output_dir=output_dir,
        clean_output=not args.no_clean_output,
    )

    totals = summary.get("totals", {})
    print("\n" + "=" * 70)
    print("CONVERSION AUDIT SUMMARY")
    print("=" * 70)
    print(f"Total Images Scanned:        {totals.get('total_images_scanned', 0):,}")
    print(f"YOLO Label Files Written:    {totals.get('total_label_files_written', 0):,}")
    print(f"Total YOLO Boxes Written:    {totals.get('total_yolo_boxes', 0):,}")
    print(f"Status Breakdown:            {totals.get('conversion_status_counts', {})}")
    print(f"YOLO Box Counts:             {totals.get('yolo_class_counts', {})}")
    print(f"Invalid Boxes Excluded:      {totals.get('total_invalid_boxes_excluded', 0)}")
    print(f"Incomplete-Label Risk Images:{totals.get('images_with_incomplete_label_risk', 0):,}")

    if summary.get("excluded_boxes_log"):
        print("\nExcluded Bounding Boxes Audit Log:")
        for exc in summary["excluded_boxes_log"][:10]:
            print(f"  - {exc['image_id']} ({exc['country']}/{exc['subset']}): {exc['class_name']} -> {exc['reason']}")
        if len(summary["excluded_boxes_log"]) > 10:
            print(f"  ... and {len(summary['excluded_boxes_log']) - 10} more (see yolo_conversion_summary.json)")

    print("\n[SUCCESS] YOLO conversion and audit report completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

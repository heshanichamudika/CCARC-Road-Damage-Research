#!/usr/bin/env python3
"""CLI Command to Run Comprehensive RDD2022 LOCO Dataset Integrity & Leakage Audit.

Usage Example (Local or Google Colab):
    python -m src.data.audit_loco_splits \\
        --conversion-manifest "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/yolo_conversion/yolo_conversion_manifest.csv" \\
        --loco-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/loco_splits" \\
        --views-dir "/content/datasets/loco_views" \\
        --output-dir "/content/drive/MyDrive/CCARC-Road-Damage-Research/experiments/loco_splits/audit"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from src.data.loco_auditor import CheckStatus, RDD2022LOCOAuditor


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive Integrity & Leakage Audit for RDD2022 LOCO Dataset Partitions."
    )
    parser.add_argument(
        "--conversion-manifest",
        type=Path,
        required=True,
        help="Path to yolo_conversion_manifest.csv produced by T02.16.",
    )
    parser.add_argument(
        "--loco-dir",
        type=Path,
        required=True,
        help="Directory containing loco_master_manifest.csv, loco_summary.json, and fold dirs.",
    )
    parser.add_argument(
        "--views-dir",
        type=Path,
        default=None,
        help="Optional directory containing runtime YOLO dataset symlink views (defaults to --loco-dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for audit reports (defaults to <loco-dir>/audit).",
    )
    parser.add_argument(
        "--check-hashes",
        action="store_true",
        help="Compute cryptographic MD5 hashes of images to identify bitwise duplicates (optional).",
    )
    parser.add_argument(
        "--sample-images",
        type=int,
        default=None,
        help="Limit number of images decoded for readability check (default: None, checks all).",
    )
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    conversion_manifest = args.conversion_manifest.resolve()
    loco_dir = args.loco_dir.resolve()
    views_dir = args.views_dir.resolve() if args.views_dir is not None else loco_dir
    output_dir = args.output_dir.resolve() if args.output_dir is not None else (loco_dir / "audit")

    if not conversion_manifest.is_file():
        print(f"Error: Conversion manifest not found: {conversion_manifest}")
        return 1
    if not loco_dir.is_dir():
        print(f"Error: LOCO directory not found: {loco_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("CCARC: RDD2022 LOCO DATASET INTEGRITY & LEAKAGE AUDIT")
    print("=" * 75)
    print(f"Conversion Manifest: {conversion_manifest}")
    print(f"LOCO Directory:      {loco_dir}")
    print(f"Views Directory:     {views_dir}")
    print(f"Audit Output Dir:    {output_dir}")
    print(f"Check Image Hashes:  {args.check_hashes}")
    print("=" * 75)

    auditor = RDD2022LOCOAuditor(
        conversion_manifest_path=conversion_manifest,
        loco_dir=loco_dir,
        views_dir=views_dir,
        check_hashes=args.check_hashes,
        sample_image_limit=args.sample_images,
    )

    report = auditor.run_full_audit()

    # Save JSON report
    json_path = output_dir / "loco_audit_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    # Save Markdown report
    md_path = output_dir / "loco_audit_report.md"
    md_path.write_text(report.to_markdown(), encoding="utf-8")

    print("\n" + "=" * 75)
    print(f"AUDIT SUMMARY: OVERALL STATUS = {report.overall_status.value}")
    print("=" * 75)
    for check_id, check in report.checks.items():
        symbol = "✅" if check.status == CheckStatus.PASS else ("⚠️" if check.status == CheckStatus.WARN else "❌")
        print(f"{symbol} [{check.status.value:<4}] {check_id:<28} : {check.message}")

    print("-" * 75)
    reconcil = report.reconciliation
    print(f"Target Reconciliation: {reconcil.get('total_selected_loco'):,} selected + {reconcil.get('total_excluded'):,} excluded = {reconcil.get('total_converted_target'):,} converted target labels.")
    print(f"Exclusion Breakdown:   {reconcil.get('exclusion_breakdown')}")
    print(f"Audit Reports Written: {json_path} and {md_path}")
    print("=" * 75)

    if report.overall_status == CheckStatus.FAIL:
        print("[AUDIT FAILED] One or more critical integrity checks failed.")
        return 1
    else:
        print("[AUDIT PASSED] Dataset integrity, zero leakage and reconciliation verified.")
        return 0


if __name__ == "__main__":
    sys.exit(main())

"""RDD2022 Cleaning Policy and Pascal VOC to YOLO Label Conversion Module.

Implements rigorous bounding box cleaning, coordinate normalization,
quarantining of non-target-only images, isolation of empty XML negative candidates,
and generation of conversion manifests and audit logs for CCARC research.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd

from src.data.rdd2022_validator import (
    DEFAULT_TARGET_CLASSES,
    AnnotationStatus,
    ParsedXML,
    RDD2022Validator,
    map_subset_to_country,
    parse_pascal_voc_xml,
)

# Standardized YOLO class ID mapping for CCARC
TARGET_CLASS_TO_ID: Dict[str, int] = {
    "D00": 0,  # Longitudinal Crack
    "D10": 1,  # Transverse Crack
    "D20": 2,  # Alligator Crack
    "D40": 3,  # Pothole
}

ID_TO_TARGET_CLASS: Dict[int, str] = {v: k for k, v in TARGET_CLASS_TO_ID.items()}


class ConversionStatus:
    """Categorical classification of YOLO label conversion outcome."""

    CONVERTED_TARGET = "converted_target"  # Valid target annotations written to YOLO format
    QUARANTINED_NON_TARGET = (
        "quarantined_non_target"  # Quarantined: only non-target objects present
    )
    EMPTY_XML_CANDIDATE = (
        "empty_xml_candidate"  # Empty XML: candidate negative, not verified background
    )
    UNLABELED_TEST = "unlabeled_test"  # Official unlabeled test split (no ground truth created)
    FAILED_CONVERSION = (
        "failed_conversion"  # Parse error or all boxes were invalid
    )
    MISSING_ANNOTATION = (
        "missing_annotation"  # Train image missing XML file
    )


@dataclass
class YOLOBox:
    """Validated, normalized YOLO bounding box."""

    class_id: int
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float

    def to_yolo_line(self) -> str:
        """Format box as standard YOLO format line: <class_id> <x_center> <y_center> <width> <height>."""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}\n"


@dataclass
class BoxConversionResult:
    """Outcome of attempting to convert a single bounding box."""

    is_valid: bool
    yolo_box: Optional[YOLOBox] = None
    exclusion_reason: Optional[str] = None
    original_coords: Optional[Tuple[float, float, float, float]] = None
    clamped_coords: Optional[Tuple[float, float, float, float]] = None
    was_clamped: bool = False
    class_name: str = ""


@dataclass
class ImageConversionRecord:
    """Comprehensive diagnostic record for an image conversion operation."""

    image_id: str
    image_path: str
    xml_path: Optional[str]
    label_path: Optional[str]
    country: str
    subset: str
    split: str
    img_width: Optional[int]
    img_height: Optional[int]
    conversion_status: str
    yolo_boxes_count: int
    count_D00: int
    count_D10: int
    count_D20: int
    count_D40: int
    excluded_boxes_count: int
    clamped_boxes_count: int
    has_non_target_objects: bool
    non_target_classes: str
    incomplete_label_risk: bool
    validation_flags: str
    loco_eligible: bool
    exclusion_details: List[Dict[str, Any]] = field(default_factory=list)
    clamped_details: List[Dict[str, Any]] = field(default_factory=list)

    def to_manifest_dict(self) -> Dict[str, Any]:
        """Convert record to a flat dictionary for CSV export."""
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "xml_path": self.xml_path or "",
            "label_path": self.label_path or "",
            "country": self.country,
            "subset": self.subset,
            "split": self.split,
            "width": self.img_width if self.img_width is not None else "",
            "height": self.img_height if self.img_height is not None else "",
            "conversion_status": self.conversion_status,
            "yolo_boxes_count": self.yolo_boxes_count,
            "count_D00": self.count_D00,
            "count_D10": self.count_D10,
            "count_D20": self.count_D20,
            "count_D40": self.count_D40,
            "excluded_boxes_count": self.excluded_boxes_count,
            "clamped_boxes_count": self.clamped_boxes_count,
            "has_non_target_objects": self.has_non_target_objects,
            "non_target_classes": self.non_target_classes,
            "incomplete_label_risk": self.incomplete_label_risk,
            "validation_flags": self.validation_flags,
            "loco_eligible": self.loco_eligible,
        }


def convert_pascal_to_yolo_box(
    class_name: str,
    xmin_raw: Any,
    ymin_raw: Any,
    xmax_raw: Any,
    ymax_raw: Any,
    img_width: int,
    img_height: int,
    class_mapping: Dict[str, int] = TARGET_CLASS_TO_ID,
    clamp_out_of_bounds: bool = False,
) -> BoxConversionResult:
    """Validate and convert Pascal VOC pixel coordinates to normalized YOLO format.

    By default (clamp_out_of_bounds=False), strictly rejects out-of-bounds coordinates.
    If clamp_out_of_bounds=True, clamps overflow coordinates to [0, W] and [0, H],
    explicitly recording both original and corrected coordinates.
    """
    if class_name not in class_mapping:
        return BoxConversionResult(
            is_valid=False,
            exclusion_reason=f"non_target_class: {class_name}",
            class_name=class_name,
        )

    class_id = class_mapping[class_name]

    if img_width <= 0 or img_height <= 0:
        return BoxConversionResult(
            is_valid=False,
            exclusion_reason=f"invalid_image_dimensions: {img_width}x{img_height}",
            class_name=class_name,
        )

    # 1. Parse numeric coordinates
    try:
        xmin = float(xmin_raw)
        ymin = float(ymin_raw)
        xmax = float(xmax_raw)
        ymax = float(ymax_raw)
    except (ValueError, TypeError) as err:
        return BoxConversionResult(
            is_valid=False,
            exclusion_reason=f"non_numeric_coordinates: {err}",
            class_name=class_name,
        )

    original_coords = (xmin, ymin, xmax, ymax)

    # 2. Check for non-finite values (NaN/Inf)
    if not (
        math.isfinite(xmin)
        and math.isfinite(ymin)
        and math.isfinite(xmax)
        and math.isfinite(ymax)
    ):
        return BoxConversionResult(
            is_valid=False,
            exclusion_reason="non_finite_coordinates",
            original_coords=original_coords,
            class_name=class_name,
        )

    # 3. Check for zero-width / zero-height before clamping (e.g. Japan_001265)
    if xmin == xmax:
        return BoxConversionResult(
            is_valid=False,
            exclusion_reason=f"zero_width: xmin={xmin} == xmax={xmax}",
            original_coords=original_coords,
            class_name=class_name,
        )

    if ymin == ymax:
        return BoxConversionResult(
            is_valid=False,
            exclusion_reason=f"zero_height: ymin={ymin} == ymax={ymax}",
            original_coords=original_coords,
            class_name=class_name,
        )

    if xmin > xmax:
        return BoxConversionResult(
            is_valid=False,
            exclusion_reason=f"inverted_x: xmin={xmin} > xmax={xmax}",
            original_coords=original_coords,
            class_name=class_name,
        )

    if ymin > ymax:
        return BoxConversionResult(
            is_valid=False,
            exclusion_reason=f"inverted_y: ymin={ymin} > ymax={ymax}",
            original_coords=original_coords,
            class_name=class_name,
        )

    # 4. Strict out-of-bounds rejection vs optional clamping
    was_clamped = False
    clamped_coords = None

    if not clamp_out_of_bounds:
        # Strict mode: reject any box exceeding image boundary
        if xmin < 0.0 or ymin < 0.0:
            return BoxConversionResult(
                is_valid=False,
                exclusion_reason=f"out_of_bounds_negative: xmin={xmin}, ymin={ymin}",
                original_coords=original_coords,
                class_name=class_name,
            )
        if xmax > float(img_width):
            return BoxConversionResult(
                is_valid=False,
                exclusion_reason=f"out_of_bounds_x: xmax={xmax} > width={img_width}",
                original_coords=original_coords,
                class_name=class_name,
            )
        if ymax > float(img_height):
            return BoxConversionResult(
                is_valid=False,
                exclusion_reason=f"out_of_bounds_y: ymax={ymax} > height={img_height}",
                original_coords=original_coords,
                class_name=class_name,
            )
        final_xmin, final_ymin, final_xmax, final_ymax = xmin, ymin, xmax, ymax
    else:
        # Clamping mode: clamp to image boundaries and log transformation
        final_xmin = max(0.0, min(xmin, float(img_width)))
        final_ymin = max(0.0, min(ymin, float(img_height)))
        final_xmax = max(0.0, min(xmax, float(img_width)))
        final_ymax = max(0.0, min(ymax, float(img_height)))

        was_clamped = (
            final_xmin != xmin
            or final_ymin != ymin
            or final_xmax != xmax
            or final_ymax != ymax
        )
        if was_clamped:
            clamped_coords = (final_xmin, final_ymin, final_xmax, final_ymax)

    box_w_px = final_xmax - final_xmin
    box_h_px = final_ymax - final_ymin

    # Check if clamping collapsed the box (e.g. box was completely outside bounds)
    if box_w_px <= 0.0:
        return BoxConversionResult(
            is_valid=False,
            exclusion_reason=f"collapsed_width_after_bounds_check: {box_w_px}",
            original_coords=original_coords,
            class_name=class_name,
        )

    if box_h_px <= 0.0:
        return BoxConversionResult(
            is_valid=False,
            exclusion_reason=f"collapsed_height_after_bounds_check: {box_h_px}",
            original_coords=original_coords,
            class_name=class_name,
        )

    # 5. Compute normalized YOLO parameters
    x_center = (final_xmin + final_xmax) / (2.0 * float(img_width))
    y_center = (final_ymin + final_ymax) / (2.0 * float(img_height))
    norm_w = box_w_px / float(img_width)
    norm_h = box_h_px / float(img_height)

    # Floating point precision edge-case guard: clamp to [0.0, 1.0]
    EPSILON = 1e-6
    if not (0.0 < x_center < 1.0 and 0.0 < y_center < 1.0):
        return BoxConversionResult(
            is_valid=False,
            exclusion_reason=f"normalized_center_out_of_bounds: ({x_center}, {y_center})",
            original_coords=original_coords,
            class_name=class_name,
        )

    if not (0.0 < norm_w <= 1.0 + EPSILON and 0.0 < norm_h <= 1.0 + EPSILON):
        return BoxConversionResult(
            is_valid=False,
            exclusion_reason=f"normalized_dimensions_out_of_bounds: w={norm_w}, h={norm_h}",
            original_coords=original_coords,
            class_name=class_name,
        )

    norm_w = min(norm_w, 1.0)
    norm_h = min(norm_h, 1.0)

    yolo_box = YOLOBox(
        class_id=class_id,
        class_name=class_name,
        x_center=x_center,
        y_center=y_center,
        width=norm_w,
        height=norm_h,
    )

    return BoxConversionResult(
        is_valid=True,
        yolo_box=yolo_box,
        exclusion_reason=None,
        original_coords=original_coords,
        clamped_coords=clamped_coords,
        was_clamped=was_clamped,
        class_name=class_name,
    )
    norm_h = box_h_px / float(img_height)

    # 6. Validate normalized ranges strictly (finite, >0, <=1.0)
    if not (0.0 < x_center < 1.0 and 0.0 < y_center < 1.0):
        return BoxConversionResult(
            is_valid=False,
            exclusion_reason=f"normalized_center_out_of_bounds: ({x_center}, {y_center})",
            original_coords=original_coords,
            class_name=class_name,
        )

    if not (0.0 < norm_w <= 1.0 and 0.0 < norm_h <= 1.0):
        return BoxConversionResult(
            is_valid=False,
            exclusion_reason=f"normalized_dimensions_out_of_bounds: w={norm_w}, h={norm_h}",
            original_coords=original_coords,
            class_name=class_name,
        )

    yolo_box = YOLOBox(
        class_id=class_id,
        class_name=class_name,
        x_center=x_center,
        y_center=y_center,
        width=norm_w,
        height=norm_h,
    )

    return BoxConversionResult(
        is_valid=True,
        yolo_box=yolo_box,
        exclusion_reason=None,
        original_coords=original_coords,
        class_name=class_name,
    )


class RDD2022YOLOConverter:
    """Converts RDD2022 Pascal VOC annotations to YOLO format with strict research cleaning policies."""

    def __init__(
        self,
        target_classes: Sequence[str] = DEFAULT_TARGET_CLASSES,
        include_empty_as_negatives: bool = False,
        quarantine_non_target: bool = True,
        clamp_out_of_bounds: bool = False,
    ) -> None:
        self.target_classes = tuple(target_classes)
        self.class_mapping = {
            cls_name: idx for idx, cls_name in enumerate(self.target_classes)
        }
        self.include_empty_as_negatives = include_empty_as_negatives
        self.quarantine_non_target = quarantine_non_target
        self.clamp_out_of_bounds = clamp_out_of_bounds
        self.validator = RDD2022Validator(target_classes=self.target_classes, verify_images=False)

    def convert_image_entry(
        self,
        img_path: Path,
        xml_path: Optional[Path],
        country: str,
        subset: str,
        split: str,
        labels_output_root: Path,
    ) -> Tuple[ImageConversionRecord, Optional[str]]:
        """Process a single image entry according to cleaning and conversion policies.

        Returns:
            (ImageConversionRecord, optional label text to write).
        """
        image_id = img_path.stem
        validation_flags: List[str] = []
        exclusion_details: List[Dict[str, Any]] = []
        clamped_details: List[Dict[str, Any]] = []

        # 1. Official test images: Never generate ground truth labels
        if split == "test":
            record = ImageConversionRecord(
                image_id=image_id,
                image_path=str(img_path),
                xml_path=str(xml_path) if xml_path else None,
                label_path=None,
                country=country,
                subset=subset,
                split=split,
                img_width=None,
                img_height=None,
                conversion_status=ConversionStatus.UNLABELED_TEST,
                yolo_boxes_count=0,
                count_D00=0,
                count_D10=0,
                count_D20=0,
                count_D40=0,
                excluded_boxes_count=0,
                clamped_boxes_count=0,
                has_non_target_objects=False,
                non_target_classes="",
                incomplete_label_risk=False,
                validation_flags="",
                loco_eligible=False,
                exclusion_details=[],
                clamped_details=[],
            )
            return record, None

        # 2. Missing XML in train split
        if xml_path is None or not xml_path.is_file():
            record = ImageConversionRecord(
                image_id=image_id,
                image_path=str(img_path),
                xml_path=None,
                label_path=None,
                country=country,
                subset=subset,
                split=split,
                img_width=None,
                img_height=None,
                conversion_status=ConversionStatus.MISSING_ANNOTATION,
                yolo_boxes_count=0,
                count_D00=0,
                count_D10=0,
                count_D20=0,
                count_D40=0,
                excluded_boxes_count=0,
                clamped_boxes_count=0,
                has_non_target_objects=False,
                non_target_classes="",
                incomplete_label_risk=False,
                validation_flags="missing_xml",
                loco_eligible=False,
                exclusion_details=[],
                clamped_details=[],
            )
            return record, None

        # 3. Parse XML
        parsed = parse_pascal_voc_xml(xml_path, target_classes=self.target_classes)
        img_width = parsed.width
        img_height = parsed.height

        if parsed.parse_error:
            record = ImageConversionRecord(
                image_id=image_id,
                image_path=str(img_path),
                xml_path=str(xml_path),
                label_path=None,
                country=country,
                subset=subset,
                split=split,
                img_width=img_width,
                img_height=img_height,
                conversion_status=ConversionStatus.FAILED_CONVERSION,
                yolo_boxes_count=0,
                count_D00=0,
                count_D10=0,
                count_D20=0,
                count_D40=0,
                excluded_boxes_count=0,
                clamped_boxes_count=0,
                has_non_target_objects=False,
                non_target_classes="",
                incomplete_label_risk=False,
                validation_flags=";".join(parsed.validation_flags),
                loco_eligible=True,
                exclusion_details=[],
                clamped_details=[],
            )
            return record, None

        # Format non-target summary
        non_target_items = [f"{k}:{v}" for k, v in sorted(parsed.non_target_counts.items())]
        non_target_str = ";".join(non_target_items)
        has_non_target = len(parsed.non_target_counts) > 0

        # Construct destination label path mirroring subset/split layout
        label_file_path = labels_output_root / subset / split / f"{image_id}.txt"

        # 4. Handle Empty XML (0 objects)
        if parsed.status == AnnotationStatus.EMPTY_XML:
            label_text = None
            assigned_label_path = None
            if self.include_empty_as_negatives:
                label_text = ""  # Zero-byte label file for negative background
                assigned_label_path = str(label_file_path)

            record = ImageConversionRecord(
                image_id=image_id,
                image_path=str(img_path),
                xml_path=str(xml_path),
                label_path=assigned_label_path,
                country=country,
                subset=subset,
                split=split,
                img_width=img_width,
                img_height=img_height,
                conversion_status=ConversionStatus.EMPTY_XML_CANDIDATE,
                yolo_boxes_count=0,
                count_D00=0,
                count_D10=0,
                count_D20=0,
                count_D40=0,
                excluded_boxes_count=0,
                clamped_boxes_count=0,
                has_non_target_objects=False,
                non_target_classes="",
                incomplete_label_risk=False,
                validation_flags="empty_xml",
                loco_eligible=True,
                exclusion_details=[],
                clamped_details=[],
            )
            return record, label_text

        # 5. Handle Non-Target-Only Images (Quarantined by default)
        if parsed.status == AnnotationStatus.NON_TARGET_ONLY:
            record = ImageConversionRecord(
                image_id=image_id,
                image_path=str(img_path),
                xml_path=str(xml_path),
                label_path=None,
                country=country,
                subset=subset,
                split=split,
                img_width=img_width,
                img_height=img_height,
                conversion_status=ConversionStatus.QUARANTINED_NON_TARGET,
                yolo_boxes_count=0,
                count_D00=0,
                count_D10=0,
                count_D20=0,
                count_D40=0,
                excluded_boxes_count=0,
                clamped_boxes_count=0,
                has_non_target_objects=True,
                non_target_classes=non_target_str,
                incomplete_label_risk=False,
                validation_flags="quarantined_non_target_only",
                loco_eligible=True,
                exclusion_details=[],
                clamped_details=[],
            )
            return record, None

        # 6. Handle Target-Labeled Images (may contain valid target, invalid target, and non-target)
        if img_width is None or img_height is None or img_width <= 0 or img_height <= 0:
            validation_flags.append("missing_dimensions_cannot_normalize")
            record = ImageConversionRecord(
                image_id=image_id,
                image_path=str(img_path),
                xml_path=str(xml_path),
                label_path=None,
                country=country,
                subset=subset,
                split=split,
                img_width=img_width,
                img_height=img_height,
                conversion_status=ConversionStatus.FAILED_CONVERSION,
                yolo_boxes_count=0,
                count_D00=0,
                count_D10=0,
                count_D20=0,
                count_D40=0,
                excluded_boxes_count=0,
                clamped_boxes_count=0,
                has_non_target_objects=has_non_target,
                non_target_classes=non_target_str,
                incomplete_label_risk=has_non_target,
                validation_flags=";".join(validation_flags),
                loco_eligible=True,
                exclusion_details=[],
                clamped_details=[],
            )
            return record, None

        valid_yolo_boxes: List[YOLOBox] = []
        target_counts: Dict[str, int] = {cls_name: 0 for cls_name in self.target_classes}

        for box in parsed.boxes:
            if not box.is_target:
                # Non-target boxes are excluded from YOLO labels by definition
                continue

            result = convert_pascal_to_yolo_box(
                class_name=box.class_name,
                xmin_raw=box.xmin,
                ymin_raw=box.ymin,
                xmax_raw=box.xmax,
                ymax_raw=box.ymax,
                img_width=img_width,
                img_height=img_height,
                class_mapping=self.class_mapping,
                clamp_out_of_bounds=self.clamp_out_of_bounds,
            )

            if result.is_valid and result.yolo_box is not None:
                valid_yolo_boxes.append(result.yolo_box)
                target_counts[result.yolo_box.class_name] = (
                    target_counts.get(result.yolo_box.class_name, 0) + 1
                )
                if result.was_clamped and result.clamped_coords is not None:
                    clamped_details.append(
                        {
                            "image_id": image_id,
                            "xml_path": str(xml_path),
                            "country": country,
                            "subset": subset,
                            "class_name": box.class_name,
                            "original_coords": result.original_coords,
                            "clamped_coords": result.clamped_coords,
                        }
                    )
                    validation_flags.append("box_clamped_to_boundaries")
            else:
                # Log invalid object exclusion without discarding entire image
                exclusion_details.append(
                    {
                        "image_id": image_id,
                        "xml_path": str(xml_path),
                        "country": country,
                        "subset": subset,
                        "class_name": box.class_name,
                        "reason": result.exclusion_reason,
                        "original_coords": result.original_coords,
                    }
                )
                validation_flags.append(f"excluded_box:{result.exclusion_reason}")

        # Incomplete label risk: image contains both target damage and unannotated non-target damage
        incomplete_label_risk = has_non_target and (len(valid_yolo_boxes) > 0)
        if incomplete_label_risk:
            validation_flags.append("contains_unlabeled_non_target_objects")

        if len(valid_yolo_boxes) > 0:
            conversion_status = ConversionStatus.CONVERTED_TARGET
            label_text = "".join(b.to_yolo_line() for b in valid_yolo_boxes)
            assigned_label_path = str(label_file_path)
        else:
            conversion_status = ConversionStatus.FAILED_CONVERSION
            label_text = None
            assigned_label_path = None
            validation_flags.append("no_valid_target_boxes_after_cleaning")

        record = ImageConversionRecord(
            image_id=image_id,
            image_path=str(img_path),
            xml_path=str(xml_path),
            label_path=assigned_label_path,
            country=country,
            subset=subset,
            split=split,
            img_width=img_width,
            img_height=img_height,
            conversion_status=conversion_status,
            yolo_boxes_count=len(valid_yolo_boxes),
            count_D00=target_counts.get("D00", 0),
            count_D10=target_counts.get("D10", 0),
            count_D20=target_counts.get("D20", 0),
            count_D40=target_counts.get("D40", 0),
            excluded_boxes_count=len(exclusion_details),
            clamped_boxes_count=len(clamped_details),
            has_non_target_objects=has_non_target,
            non_target_classes=non_target_str,
            incomplete_label_risk=incomplete_label_risk,
            validation_flags=";".join(sorted(set(validation_flags))),
            loco_eligible=True,
            exclusion_details=exclusion_details,
            clamped_details=clamped_details,
        )

        return record, label_text

    def convert_dataset(
        self,
        data_root: Path,
        output_dir: Path,
        clean_output: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute full dataset cleaning and YOLO label conversion.

        Args:
            data_root: Root directory containing extracted RDD2022 subsets.
            output_dir: Destination directory for labels, conversion manifest, and summary JSON.
            clean_output: If True, safely cleans the destination labels directory before writing.
        """
        data_root = data_root.resolve()
        output_dir = output_dir.resolve()
        labels_dir = output_dir / "labels"

        if clean_output and labels_dir.exists():
            print(f"Cleaning existing label directory to prevent stale mixing: {labels_dir}")
            shutil.rmtree(labels_dir)

        labels_dir.mkdir(parents=True, exist_ok=True)

        print(f"Discovering RDD2022 image entries in: {data_root}")
        entries = self.validator.discover_image_entries(data_root)
        print(f"Discovered {len(entries)} total image entries across subsets.")

        records: List[ImageConversionRecord] = []
        all_exclusions: List[Dict[str, Any]] = []
        all_clamped: List[Dict[str, Any]] = []
        labels_written_count = 0

        for img_path, country, subset, split in entries:
            xml_path = self.validator.locate_matching_xml(img_path, data_root, split)
            record, label_text = self.convert_image_entry(
                img_path=img_path,
                xml_path=xml_path,
                country=country,
                subset=subset,
                split=split,
                labels_output_root=labels_dir,
            )

            if label_text is not None and record.label_path is not None:
                lbl_path = Path(record.label_path)
                lbl_path.parent.mkdir(parents=True, exist_ok=True)
                lbl_path.write_text(label_text, encoding="utf-8")
                labels_written_count += 1

            records.append(record)
            if record.exclusion_details:
                all_exclusions.extend(record.exclusion_details)
            if record.clamped_details:
                all_clamped.extend(record.clamped_details)

        manifest_path = output_dir / "yolo_conversion_manifest.csv"
        summary_path = output_dir / "yolo_conversion_summary.json"

        # Generate manifest CSV
        df = pd.DataFrame([r.to_manifest_dict() for r in records])
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(manifest_path, index=False)
        print(f"Generated conversion manifest: {manifest_path} ({len(df)} rows)")

        # Generate summary JSON
        summary = self._generate_conversion_summary(
            records=records,
            all_exclusions=all_exclusions,
            all_clamped=all_clamped,
            data_root=data_root,
            output_dir=output_dir,
            labels_written_count=labels_written_count,
        )

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Generated conversion summary: {summary_path}")

        return df, summary

    def _generate_conversion_summary(
        self,
        records: List[ImageConversionRecord],
        all_exclusions: List[Dict[str, Any]],
        all_clamped: List[Dict[str, Any]],
        data_root: Path,
        output_dir: Path,
        labels_written_count: int,
    ) -> Dict[str, Any]:
        """Aggregate country-wise and class-wise conversion audit statistics."""
        status_counts: Dict[str, int] = {}
        for r in records:
            status_counts[r.conversion_status] = (
                status_counts.get(r.conversion_status, 0) + 1
            )

        class_totals: Dict[str, int] = {cls_name: 0 for cls_name in self.target_classes}
        for r in records:
            class_totals["D00"] += r.count_D00
            class_totals["D10"] += r.count_D10
            class_totals["D20"] += r.count_D20
            class_totals["D40"] += r.count_D40

        by_country: Dict[str, Any] = {}
        countries = sorted(list({r.country for r in records}))

        for country in countries:
            c_records = [r for r in records if r.country == country]
            c_status: Dict[str, int] = {}
            for r in c_records:
                c_status[r.conversion_status] = c_status.get(r.conversion_status, 0) + 1

            c_classes = {cls_name: 0 for cls_name in self.target_classes}
            for r in c_records:
                c_classes["D00"] += r.count_D00
                c_classes["D10"] += r.count_D10
                c_classes["D20"] += r.count_D20
                c_classes["D40"] += r.count_D40

            by_country[country] = {
                "total_images": len(c_records),
                "labels_written": sum(1 for r in c_records if r.label_path is not None),
                "status_breakdown": c_status,
                "yolo_target_boxes": c_classes,
                "total_yolo_boxes": sum(c_classes.values()),
                "excluded_invalid_boxes": sum(r.excluded_boxes_count for r in c_records),
                "clamped_boxes": sum(r.clamped_boxes_count for r in c_records),
                "incomplete_label_risk_images": sum(
                    1 for r in c_records if r.incomplete_label_risk
                ),
            }

        return {
            "dataset_name": "RDD2022",
            "pipeline": "Pascal_VOC_to_YOLO_Conversion",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_root": str(data_root),
            "output_dir": str(output_dir),
            "policies": {
                "target_classes": list(self.target_classes),
                "class_mapping": self.class_mapping,
                "include_empty_as_negatives": self.include_empty_as_negatives,
                "quarantine_non_target": self.quarantine_non_target,
                "clamp_out_of_bounds": self.clamp_out_of_bounds,
                "exclude_invalid_boxes_individually": True,
                "exclude_official_test_labels": True,
            },
            "totals": {
                "total_images_scanned": len(records),
                "total_label_files_written": labels_written_count,
                "conversion_status_counts": status_counts,
                "yolo_class_counts": class_totals,
                "total_yolo_boxes": sum(class_totals.values()),
                "total_invalid_boxes_excluded": len(all_exclusions),
                "total_boxes_clamped": len(all_clamped),
                "images_with_incomplete_label_risk": sum(
                    1 for r in records if r.incomplete_label_risk
                ),
            },
            "by_class": {
                str(self.class_mapping[cls_name]): {
                    "class_name": cls_name,
                    "count": class_totals[cls_name],
                }
                for cls_name in self.target_classes
            },
            "by_country": by_country,
            "excluded_boxes_log": all_exclusions,
            "clamped_boxes_log": all_clamped,
        }

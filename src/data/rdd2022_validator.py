"""RDD2022 Dataset Validation and Manifest Generation Module.

Implements Pascal VOC XML validation, bounding-box geometry integrity checks,
image decoding verification, subset-to-country mapping, manifest generation,
and comprehensive country/class-level statistics for cross-country road damage research.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import xml.etree.ElementTree as ET

import cv2
import pandas as pd

# Core target classes standardized across RDD2022 for cross-country research
DEFAULT_TARGET_CLASSES: Tuple[str, ...] = ("D00", "D10", "D20", "D40")

# 7 RDD2022 archives/subsets mapped to 6 sovereign countries
SUBSET_TO_COUNTRY_MAP: Dict[str, Tuple[str, str]] = {
    "china_drone": ("China", "China_Drone"),
    "china_motorbike": ("China", "China_MotorBike"),
    "czech": ("Czech", "Czech"),
    "india": ("India", "India"),
    "japan": ("Japan", "Japan"),
    "norway": ("Norway", "Norway"),
    "united_states": ("United_States", "United_States"),
}

IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp"}


class AnnotationStatus:
    """Categorical classification of image annotation state."""

    LABELED_TARGET = "labeled_target"  # At least one target class (D00, D10, D20, D40)
    NON_TARGET_ONLY = (
        "non_target_only"  # Objects present, but only non-target (e.g. D01, D11)
    )
    EMPTY_XML = "empty_xml"  # Valid XML file but 0 objects
    UNLABELED_TEST = "unlabeled_test"  # Official unlabeled test split
    MISSING_XML = "missing_xml"  # Train split image missing matching XML file


@dataclass
class BoundingBox:
    """Represents a validated Pascal VOC bounding box."""

    class_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    is_target: bool
    is_valid: bool
    validation_error: Optional[str] = None

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin


@dataclass
class ParsedXML:
    """Result of parsing and validating an XML annotation file."""

    xml_path: Optional[str]
    filename: Optional[str]
    width: Optional[int]
    height: Optional[int]
    depth: Optional[int]
    boxes: List[BoundingBox] = field(default_factory=list)
    target_counts: Dict[str, int] = field(default_factory=dict)
    non_target_counts: Dict[str, int] = field(default_factory=dict)
    status: str = AnnotationStatus.EMPTY_XML
    validation_flags: List[str] = field(default_factory=list)
    parse_error: Optional[str] = None


@dataclass
class ImageRecord:
    """Complete diagnostic record for an RDD2022 image."""

    image_id: str
    image_path: str
    xml_path: Optional[str]
    country: str
    subset: str
    split: str
    actual_width: Optional[int]
    actual_height: Optional[int]
    actual_channels: Optional[int]
    xml_width: Optional[int]
    xml_height: Optional[int]
    xml_depth: Optional[int]
    count_D00: int
    count_D10: int
    count_D20: int
    count_D40: int
    total_target_boxes: int
    total_non_target_boxes: int
    non_target_classes: str
    annotation_status: str
    total_boxes: int
    total_invalid_boxes: int
    is_valid: bool
    validation_flags: str
    loco_eligible: bool
    invalid_box_details: List[Dict[str, Any]] = field(default_factory=list)

    def to_manifest_dict(self) -> Dict[str, Any]:
        """Convert record to a flat dictionary for CSV export."""
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "xml_path": self.xml_path or "",
            "country": self.country,
            "subset": self.subset,
            "split": self.split,
            "actual_width": self.actual_width
            if self.actual_width is not None
            else "",
            "actual_height": self.actual_height
            if self.actual_height is not None
            else "",
            "actual_channels": self.actual_channels
            if self.actual_channels is not None
            else "",
            "xml_width": self.xml_width if self.xml_width is not None else "",
            "xml_height": self.xml_height
            if self.xml_height is not None
            else "",
            "xml_depth": self.xml_depth if self.xml_depth is not None else "",
            "count_D00": self.count_D00,
            "count_D10": self.count_D10,
            "count_D20": self.count_D20,
            "count_D40": self.count_D40,
            "total_target_boxes": self.total_target_boxes,
            "total_non_target_boxes": self.total_non_target_boxes,
            "non_target_classes": self.non_target_classes,
            "annotation_status": self.annotation_status,
            "total_boxes": self.total_boxes,
            "total_invalid_boxes": self.total_invalid_boxes,
            "is_valid": self.is_valid,
            "validation_flags": self.validation_flags,
            "loco_eligible": self.loco_eligible,
        }


def map_subset_to_country(subset_dir_name: str) -> Tuple[str, str]:
    """Map an archive directory name to standard (Country, Subset).

    Normalizes China_Drone and China_MotorBike to Country: 'China' while
    preserving their specific subset identity.
    """
    normalized = subset_dir_name.strip().lower().replace("-", "_")
    if normalized in SUBSET_TO_COUNTRY_MAP:
        return SUBSET_TO_COUNTRY_MAP[normalized]

    # Flexible prefix matching for subfolder names
    for key, (country, subset) in SUBSET_TO_COUNTRY_MAP.items():
        if key in normalized:
            return country, subset

    # Fallback: capitalize clean directory name
    clean_name = subset_dir_name.strip()
    return clean_name, clean_name


def validate_bounding_box(
    class_name: str,
    xmin_raw: Any,
    ymin_raw: Any,
    xmax_raw: Any,
    ymax_raw: Any,
    target_classes: Sequence[str] = DEFAULT_TARGET_CLASSES,
    img_width: Optional[int] = None,
    img_height: Optional[int] = None,
) -> BoundingBox:
    """Validate a single bounding box geometry and classify target membership.

    Enforces finite coordinates, positive dimensions (xmin < xmax, ymin < ymax),
    non-negativity, and boundary conditions.
    """
    is_target = class_name in target_classes

    # Parse numeric coordinates
    try:
        xmin = float(xmin_raw)
        ymin = float(ymin_raw)
        xmax = float(xmax_raw)
        ymax = float(ymax_raw)
    except (ValueError, TypeError) as err:
        return BoundingBox(
            class_name=class_name,
            xmin=0.0,
            ymin=0.0,
            xmax=0.0,
            ymax=0.0,
            is_target=is_target,
            is_valid=False,
            validation_error=f"non_numeric_coordinates: {err}",
        )

    # Check finite numbers (no NaN or Inf)
    if not (
        math.isfinite(xmin)
        and math.isfinite(ymin)
        and math.isfinite(xmax)
        and math.isfinite(ymax)
    ):
        return BoundingBox(
            class_name=class_name,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            is_target=is_target,
            is_valid=False,
            validation_error="non_finite_coordinates (NaN/Inf)",
        )

    # Check dimension positivity
    if xmin == xmax:
        return BoundingBox(
            class_name=class_name,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            is_target=is_target,
            is_valid=False,
            validation_error=f"zero_width (xmin={xmin}, xmax={xmax})",
        )

    if ymin == ymax:
        return BoundingBox(
            class_name=class_name,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            is_target=is_target,
            is_valid=False,
            validation_error=f"zero_height (ymin={ymin}, ymax={ymax})",
        )

    if xmin > xmax:
        return BoundingBox(
            class_name=class_name,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            is_target=is_target,
            is_valid=False,
            validation_error=f"inverted_x (xmin={xmin} > xmax={xmax})",
        )

    if ymin > ymax:
        return BoundingBox(
            class_name=class_name,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            is_target=is_target,
            is_valid=False,
            validation_error=f"inverted_y (ymin={ymin} > ymax={ymax})",
        )

    # Check non-negative coordinates
    if xmin < 0 or ymin < 0:
        return BoundingBox(
            class_name=class_name,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            is_target=is_target,
            is_valid=False,
            validation_error=f"negative_coordinates (xmin={xmin}, ymin={ymin})",
        )

    # Check image bounds if dimensions provided
    if img_width is not None and xmax > img_width:
        return BoundingBox(
            class_name=class_name,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            is_target=is_target,
            is_valid=False,
            validation_error=f"out_of_bounds_x (xmax={xmax} > width={img_width})",
        )

    if img_height is not None and ymax > img_height:
        return BoundingBox(
            class_name=class_name,
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            is_target=is_target,
            is_valid=False,
            validation_error=f"out_of_bounds_y (ymax={ymax} > height={img_height})",
        )

    return BoundingBox(
        class_name=class_name,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        is_target=is_target,
        is_valid=True,
        validation_error=None,
    )


def parse_pascal_voc_xml(
    xml_path: Path, target_classes: Sequence[str] = DEFAULT_TARGET_CLASSES
) -> ParsedXML:
    """Parse a Pascal VOC annotation file and validate its objects."""
    if not xml_path.is_file():
        return ParsedXML(
            xml_path=str(xml_path),
            filename=None,
            width=None,
            height=None,
            depth=None,
            status=AnnotationStatus.MISSING_XML,
            parse_error="file_not_found",
        )

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as err:
        return ParsedXML(
            xml_path=str(xml_path),
            filename=None,
            width=None,
            height=None,
            depth=None,
            status="corrupted_xml",
            validation_flags=["xml_parse_error"],
            parse_error=f"xml_parse_error: {err}",
        )

    filename_elem = root.find("filename")
    filename = filename_elem.text.strip() if filename_elem is not None else None

    # Parse XML declared dimensions
    width, height, depth = None, None, None
    size_elem = root.find("size")
    if size_elem is not None:
        try:
            w_text = size_elem.findtext("width")
            h_text = size_elem.findtext("height")
            d_text = size_elem.findtext("depth")
            if w_text:
                width = int(float(w_text))
            if h_text:
                height = int(float(h_text))
            if d_text:
                depth = int(float(d_text))
        except (ValueError, TypeError):
            pass

    boxes: List[BoundingBox] = []
    target_counts: Dict[str, int] = {cls_name: 0 for cls_name in target_classes}
    non_target_counts: Dict[str, int] = {}
    validation_flags: List[str] = []

    objects = root.findall("object")
    for obj in objects:
        name_elem = obj.find("name")
        class_name = (
            name_elem.text.strip()
            if name_elem is not None and name_elem.text
            else "UNKNOWN"
        )

        bndbox = obj.find("bndbox")
        if bndbox is None:
            boxes.append(
                BoundingBox(
                    class_name=class_name,
                    xmin=0,
                    ymin=0,
                    xmax=0,
                    ymax=0,
                    is_target=class_name in target_classes,
                    is_valid=False,
                    validation_error="missing_bndbox_element",
                )
            )
            validation_flags.append("missing_bndbox")
            continue

        xmin_raw = bndbox.findtext("xmin")
        ymin_raw = bndbox.findtext("ymin")
        xmax_raw = bndbox.findtext("xmax")
        ymax_raw = bndbox.findtext("ymax")

        box = validate_bounding_box(
            class_name=class_name,
            xmin_raw=xmin_raw,
            ymin_raw=ymin_raw,
            xmax_raw=xmax_raw,
            ymax_raw=ymax_raw,
            target_classes=target_classes,
            img_width=width,
            img_height=height,
        )
        boxes.append(box)

        if not box.is_valid:
            validation_flags.append(f"invalid_box:{box.validation_error}")

        if box.is_target:
            target_counts[class_name] = target_counts.get(class_name, 0) + 1
        else:
            non_target_counts[class_name] = (
                non_target_counts.get(class_name, 0) + 1
            )

    total_target = sum(target_counts.values())
    total_non_target = sum(non_target_counts.values())

    # Determine annotation status
    if len(boxes) == 0:
        status = AnnotationStatus.EMPTY_XML
    elif total_target > 0:
        status = AnnotationStatus.LABELED_TARGET
    else:
        status = AnnotationStatus.NON_TARGET_ONLY

    return ParsedXML(
        xml_path=str(xml_path),
        filename=filename,
        width=width,
        height=height,
        depth=depth,
        boxes=boxes,
        target_counts=target_counts,
        non_target_counts=non_target_counts,
        status=status,
        validation_flags=validation_flags,
        parse_error=None,
    )


def verify_image_file(
    image_path: Path,
    expected_width: Optional[int] = None,
    expected_height: Optional[int] = None,
) -> Tuple[Optional[Tuple[int, int, int]], List[str]]:
    """Decode an image file to verify readability and compare with XML metadata.

    Returns:
        (actual_dimensions, validation_flags)
        actual_dimensions: (width, height, channels) or None if unreadable.
    """
    flags: List[str] = []
    if not image_path.is_file():
        return None, ["image_file_not_found"]

    img = cv2.imread(str(image_path))
    if img is None:
        return None, ["corrupt_image_decode_failed"]

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape) > 2 else 1

    if expected_width is not None and expected_width != w:
        flags.append(f"dimension_mismatch_width (xml={expected_width}, img={w})")
    if expected_height is not None and expected_height != h:
        flags.append(
            f"dimension_mismatch_height (xml={expected_height}, img={h})"
        )

    return (w, h, c), flags


class RDD2022Validator:
    """Main validator and manifest generator for RDD2022."""

    def __init__(
        self,
        target_classes: Sequence[str] = DEFAULT_TARGET_CLASSES,
        verify_images: bool = True,
    ) -> None:
        self.target_classes = tuple(target_classes)
        self.verify_images = verify_images

    def discover_image_entries(
        self, data_root: Path
    ) -> List[Tuple[Path, str, str, str]]:
        """Discover image files across all subsets and splits.

        Handles both direct (e.g. extracted/China_Drone/train/...) and nested
        archive directory structures (e.g. extracted/China_Drone/China_Drone/train/...).

        Returns list of (image_path, country, subset, split).
        """
        data_root = data_root.resolve()
        entries: List[Tuple[Path, str, str, str]] = []

        # Find all image files
        for img_path in data_root.rglob("*"):
            if (
                img_path.is_file()
                and img_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                # Identify subset, country, and split based on path relative to data_root
                rel_parts = img_path.relative_to(data_root).parts
                if not rel_parts:
                    continue

                # Match against known subset names across all path components
                country = None
                subset = None
                for part in rel_parts:
                    part_clean = part.strip().lower().replace("-", "_")
                    if part_clean in SUBSET_TO_COUNTRY_MAP:
                        country, subset = SUBSET_TO_COUNTRY_MAP[part_clean]
                        break

                if country is None or subset is None:
                    # Fallback to the first relative directory component
                    country, subset = map_subset_to_country(rel_parts[0])

                # Determine split (train vs test)
                split = "train"
                for part in rel_parts:
                    part_lower = part.lower()
                    if "test" in part_lower:
                        split = "test"
                        break
                    elif "train" in part_lower:
                        split = "train"
                        break

                entries.append((img_path, country, subset, split))

        return sorted(entries, key=lambda x: str(x[0]))

    def locate_matching_xml(
        self, img_path: Path, data_root: Path, split: str
    ) -> Optional[Path]:
        """Locate the corresponding XML annotation file for an image.

        Handles standard and nested RDD2022 structures:
        - .../train/images/{stem}.jpg -> .../train/annotations/xmls/{stem}.xml
        - .../train/images/{stem}.jpg -> .../train/annotations/{stem}.xml
        - Same-directory XML
        """
        if split == "test":
            # Test split images are officially unlabeled in RDD2022
            return None

        stem = img_path.stem
        parts = list(img_path.parts)

        # 1. Search backwards for 'images' directory
        for idx in range(len(parts) - 1, -1, -1):
            if parts[idx].lower() == "images":
                candidate_xml_dir = Path(*parts[:idx]) / "annotations" / "xmls"
                candidate = candidate_xml_dir / f"{stem}.xml"
                if candidate.is_file():
                    return candidate
                # Secondary candidate: .../annotations/{stem}.xml
                candidate_2 = (
                    Path(*parts[:idx]) / "annotations" / f"{stem}.xml"
                )
                if candidate_2.is_file():
                    return candidate_2

        # 2. Same directory as image
        sibling_xml = img_path.with_suffix(".xml")
        if sibling_xml.is_file():
            return sibling_xml

        # 3. Check within the parent split directory of images
        parent_dir = img_path.parent
        if parent_dir.name.lower() == "images":
            split_dir = parent_dir.parent
            candidate = split_dir / "annotations" / "xmls" / f"{stem}.xml"
            if candidate.is_file():
                return candidate
            candidate_sub = list(split_dir.rglob(f"{stem}.xml"))
            if candidate_sub:
                return candidate_sub[0]

        return None

    def validate_dataset(self, data_root: Path) -> List[ImageRecord]:
        """Scan and validate all images and annotations in the dataset root."""
        data_root = data_root.resolve()
        image_entries = self.discover_image_entries(data_root)

        records: List[ImageRecord] = []
        for img_path, country, subset, split in image_entries:
            xml_path = self.locate_matching_xml(img_path, data_root, split)
            record = self._process_single_entry(
                img_path=img_path,
                xml_path=xml_path,
                country=country,
                subset=subset,
                split=split,
            )
            records.append(record)

        return records

    def _process_single_entry(
        self,
        img_path: Path,
        xml_path: Optional[Path],
        country: str,
        subset: str,
        split: str,
    ) -> ImageRecord:
        """Process and validate a single image and optional XML pair."""
        image_id = img_path.stem
        validation_flags: List[str] = []
        invalid_boxes_details: List[Dict[str, Any]] = []

        parsed_xml: Optional[ParsedXML] = None
        if split == "test":
            annotation_status = AnnotationStatus.UNLABELED_TEST
            loco_eligible = False  # Official test images excluded from quantitative LOCO evaluation
            xml_width, xml_height, xml_depth = None, None, None
            count_d00 = count_d10 = count_d20 = count_d40 = 0
            total_target = 0
            total_non_target = 0
            non_target_str = ""
            total_boxes = 0
            total_invalid = 0
        elif xml_path is not None and xml_path.is_file():
            parsed_xml = parse_pascal_voc_xml(
                xml_path, target_classes=self.target_classes
            )
            annotation_status = parsed_xml.status
            loco_eligible = True
            xml_width, xml_height, xml_depth = (
                parsed_xml.width,
                parsed_xml.height,
                parsed_xml.depth,
            )

            count_d00 = parsed_xml.target_counts.get("D00", 0)
            count_d10 = parsed_xml.target_counts.get("D10", 0)
            count_d20 = parsed_xml.target_counts.get("D20", 0)
            count_d40 = parsed_xml.target_counts.get("D40", 0)
            total_target = sum(parsed_xml.target_counts.values())
            total_non_target = sum(parsed_xml.non_target_counts.values())

            # Format non-target classes as 'D01:2;D11:1'
            non_target_items = [
                f"{k}:{v}"
                for k, v in sorted(parsed_xml.non_target_counts.items())
            ]
            non_target_str = ";".join(non_target_items)

            total_boxes = len(parsed_xml.boxes)
            invalid_boxes = [b for b in parsed_xml.boxes if not b.is_valid]
            total_invalid = len(invalid_boxes)

            for b in invalid_boxes:
                invalid_boxes_details.append(
                    {
                        "image_id": image_id,
                        "xml_path": str(xml_path),
                        "country": country,
                        "subset": subset,
                        "class_name": b.class_name,
                        "is_target": b.is_target,
                        "box": [b.xmin, b.ymin, b.xmax, b.ymax],
                        "error": b.validation_error,
                    }
                )

            validation_flags.extend(parsed_xml.validation_flags)
        else:
            # Train split without matching XML
            annotation_status = AnnotationStatus.MISSING_XML
            loco_eligible = False
            validation_flags.append("missing_xml")
            xml_width, xml_height, xml_depth = None, None, None
            count_d00 = count_d10 = count_d20 = count_d40 = 0
            total_target = 0
            total_non_target = 0
            non_target_str = ""
            total_boxes = 0
            total_invalid = 0

        # Verify image file decoding and actual dimensions
        actual_dims = None
        if self.verify_images and img_path.is_file():
            actual_dims, img_flags = verify_image_file(
                img_path,
                expected_width=xml_width,
                expected_height=xml_height,
            )
            validation_flags.extend(img_flags)

        actual_w = actual_dims[0] if actual_dims else None
        actual_h = actual_dims[1] if actual_dims else None
        actual_c = actual_dims[2] if actual_dims else None

        # Overall validity check: valid if no critical validation flags
        # Note: Non-target only or empty XML are valid images, but invalid boxes make an image invalid
        is_valid = True
        critical_flags = [
            f
            for f in validation_flags
            if any(
                crit in f
                for crit in (
                    "invalid_box",
                    "corrupt_image",
                    "xml_parse_error",
                    "dimension_mismatch",
                )
            )
        ]
        if critical_flags:
            is_valid = False

        return ImageRecord(
            image_id=image_id,
            image_path=str(img_path),
            xml_path=str(xml_path) if xml_path else None,
            country=country,
            subset=subset,
            split=split,
            actual_width=actual_w,
            actual_height=actual_h,
            actual_channels=actual_c,
            xml_width=xml_width,
            xml_height=xml_height,
            xml_depth=xml_depth,
            count_D00=count_d00,
            count_D10=count_d10,
            count_D20=count_d20,
            count_D40=count_d40,
            total_target_boxes=total_target,
            total_non_target_boxes=total_non_target,
            non_target_classes=non_target_str,
            annotation_status=annotation_status,
            total_boxes=total_boxes,
            total_invalid_boxes=total_invalid,
            is_valid=is_valid,
            validation_flags=";".join(sorted(set(validation_flags))),
            loco_eligible=loco_eligible,
            invalid_box_details=invalid_boxes_details,
        )

    def generate_manifest(
        self, records: List[ImageRecord], output_csv: Path
    ) -> pd.DataFrame:
        """Export image validation records into a standardized CSV manifest."""
        rows = [r.to_manifest_dict() for r in records]
        df = pd.DataFrame(rows)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        return df

    def generate_summary(
        self, records: List[ImageRecord], data_root: Path, output_json: Path
    ) -> Dict[str, Any]:
        """Aggregate country-wise and class-wise statistics into a structured JSON summary."""
        total_images = len(records)
        train_images = sum(1 for r in records if r.split == "train")
        test_images = sum(1 for r in records if r.split == "test")
        loco_eligible = sum(1 for r in records if r.loco_eligible)

        status_counts: Dict[str, int] = {}
        for r in records:
            status_counts[r.annotation_status] = (
                status_counts.get(r.annotation_status, 0) + 1
            )

        target_class_totals: Dict[str, int] = {
            cls_name: 0 for cls_name in self.target_classes
        }
        for r in records:
            target_class_totals["D00"] += r.count_D00
            target_class_totals["D10"] += r.count_D10
            target_class_totals["D20"] += r.count_D20
            target_class_totals["D40"] += r.count_D40

        non_target_totals: Dict[str, int] = {}
        for r in records:
            if r.non_target_classes:
                for item in r.non_target_classes.split(";"):
                    if ":" in item:
                        cls_name, count_str = item.split(":", 1)
                        try:
                            non_target_totals[cls_name] = non_target_totals.get(
                                cls_name, 0
                            ) + int(count_str)
                        except ValueError:
                            pass

        all_invalid_details: List[Dict[str, Any]] = []
        for r in records:
            all_invalid_details.extend(r.invalid_box_details)

        # Country-wise and subset-wise aggregation
        by_country: Dict[str, Any] = {}
        countries = sorted(list({r.country for r in records}))

        for country in countries:
            c_records = [r for r in records if r.country == country]
            subsets = sorted(list({r.subset for r in c_records}))

            c_subsets_stats: Dict[str, Any] = {}
            for sub in subsets:
                sub_records = [r for r in c_records if r.subset == sub]
                c_subsets_stats[sub] = {
                    "total_images": len(sub_records),
                    "train_images": sum(
                        1 for r in sub_records if r.split == "train"
                    ),
                    "test_images": sum(
                        1 for r in sub_records if r.split == "test"
                    ),
                    "target_boxes": sum(
                        r.total_target_boxes for r in sub_records
                    ),
                    "non_target_boxes": sum(
                        r.total_non_target_boxes for r in sub_records
                    ),
                    "empty_xml_images": sum(
                        1
                        for r in sub_records
                        if r.annotation_status == AnnotationStatus.EMPTY_XML
                    ),
                    "non_target_only_images": sum(
                        1
                        for r in sub_records
                        if r.annotation_status
                        == AnnotationStatus.NON_TARGET_ONLY
                    ),
                    "invalid_boxes": sum(
                        r.total_invalid_boxes for r in sub_records
                    ),
                }

            c_target_totals = {cls_name: 0 for cls_name in self.target_classes}
            for r in c_records:
                c_target_totals["D00"] += r.count_D00
                c_target_totals["D10"] += r.count_D10
                c_target_totals["D20"] += r.count_D20
                c_target_totals["D40"] += r.count_D40

            c_status_counts: Dict[str, int] = {}
            for r in c_records:
                c_status_counts[r.annotation_status] = (
                    c_status_counts.get(r.annotation_status, 0) + 1
                )

            by_country[country] = {
                "subsets": c_subsets_stats,
                "total_images": len(c_records),
                "train_images": sum(1 for r in c_records if r.split == "train"),
                "test_images": sum(1 for r in c_records if r.split == "test"),
                "loco_eligible_images": sum(1 for r in c_records if r.loco_eligible),
                "annotation_status_counts": c_status_counts,
                "target_class_counts": c_target_totals,
                "total_target_boxes": sum(
                    r.total_target_boxes for r in c_records
                ),
                "total_non_target_boxes": sum(
                    r.total_non_target_boxes for r in c_records
                ),
                "invalid_boxes_count": sum(
                    r.total_invalid_boxes for r in c_records
                ),
            }

        summary: Dict[str, Any] = {
            "dataset_name": "RDD2022",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_root": str(data_root.resolve()),
            "totals": {
                "total_images": total_images,
                "train_images": train_images,
                "test_images": test_images,
                "loco_eligible_images": loco_eligible,
                "annotation_status_counts": status_counts,
                "total_target_boxes": sum(target_class_totals.values()),
                "total_non_target_boxes": sum(non_target_totals.values()),
                "total_invalid_boxes": len(all_invalid_details),
            },
            "by_class": {
                "target": target_class_totals,
                "non_target": non_target_totals,
            },
            "by_country": by_country,
            "invalid_boxes": all_invalid_details,
        }

        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary

    def run(
        self, data_root: Path, output_dir: Path
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute full validation pipeline: scan, validate, write CSV and JSON."""
        print(f"Scanning RDD2022 dataset at: {data_root}")
        records = self.validate_dataset(data_root)
        print(f"Total discovered images: {len(records)}")

        manifest_path = output_dir / "rdd2022_manifest.csv"
        summary_path = output_dir / "rdd2022_summary.json"

        df = self.generate_manifest(records, manifest_path)
        print(f"Generated manifest: {manifest_path} ({len(df)} rows)")

        summary = self.generate_summary(records, data_root, summary_path)
        print(f"Generated statistics summary: {summary_path}")

        return df, summary

"""Unit and Integration Tests for RDD2022 Dataset Validation and Manifest Generation."""

import json
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import pytest

from src.data.rdd2022_validator import (
    AnnotationStatus,
    BoundingBox,
    DEFAULT_TARGET_CLASSES,
    RDD2022Validator,
    map_subset_to_country,
    parse_pascal_voc_xml,
    validate_bounding_box,
    verify_image_file,
)


def create_synthetic_image(path: Path, width: int = 100, height: int = 80, channels: int = 3) -> None:
    """Helper to create a valid synthetic image file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img_data = np.zeros((height, width, channels), dtype=np.uint8)
    # Put a subtle gradient so it's a realistic non-empty image
    img_data[:, :, 0] = np.arange(width, dtype=np.uint8) % 255
    cv2.imwrite(str(path), img_data)


def create_voc_xml(
    path: Path,
    filename: str,
    width: int,
    height: int,
    depth: int = 3,
    objects: list[dict] | None = None,
) -> None:
    """Helper to generate a Pascal VOC XML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    obj_str = ""
    if objects:
        for obj in objects:
            obj_str += f"""
    <object>
        <name>{obj['name']}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>{obj['xmin']}</xmin>
            <ymin>{obj['ymin']}</ymin>
            <xmax>{obj['xmax']}</xmax>
            <ymax>{obj['ymax']}</ymax>
        </bndbox>
    </object>"""

    xml_content = f"""<annotation>
    <folder>images</folder>
    <filename>{filename}</filename>
    <size>
        <width>{width}</width>
        <height>{height}</height>
        <depth>{depth}</depth>
    </size>{obj_str}
</annotation>"""
    path.write_text(xml_content.strip(), encoding="utf-8")


def test_subset_to_country_mapping():
    """Verify that all 7 subsets map to the correct 6 sovereign countries."""
    assert map_subset_to_country("China_Drone") == ("China", "China_Drone")
    assert map_subset_to_country("China_MotorBike") == ("China", "China_MotorBike")
    assert map_subset_to_country("Czech") == ("Czech", "Czech")
    assert map_subset_to_country("India") == ("India", "India")
    assert map_subset_to_country("Japan") == ("Japan", "Japan")
    assert map_subset_to_country("Norway") == ("Norway", "Norway")
    assert map_subset_to_country("United_States") == ("United_States", "United_States")

    # Case-insensitive & formatting robustness
    assert map_subset_to_country("china_drone")[0] == "China"
    assert map_subset_to_country("china-motorbike")[0] == "China"


def test_bounding_box_valid():
    """Verify valid bounding boxes for target and non-target classes."""
    box_target = validate_bounding_box(
        class_name="D00",
        xmin_raw=10,
        ymin_raw=20,
        xmax_raw=50,
        ymax_raw=80,
        img_width=100,
        img_height=100,
    )
    assert box_target.is_valid
    assert box_target.is_target
    assert box_target.width == 40
    assert box_target.height == 60
    assert box_target.validation_error is None

    box_non_target = validate_bounding_box(
        class_name="D01",
        xmin_raw="15.5",
        ymin_raw="25.5",
        xmax_raw="60.0",
        ymax_raw="90.0",
    )
    assert box_non_target.is_valid
    assert not box_non_target.is_target


def test_bounding_box_invalid_zero_width():
    """Verify detection of zero-width bounding box (the known Japan_001265 anomaly)."""
    box = validate_bounding_box(
        class_name="D20",
        xmin_raw=450,
        ymin_raw=100,
        xmax_raw=450,  # xmin == xmax
        ymax_raw=200,
    )
    assert not box.is_valid
    assert "zero_width" in box.validation_error
    assert box.width == 0


def test_bounding_box_invalid_geometries():
    """Verify detection of various invalid geometry anomalies."""
    # Zero height
    box_zh = validate_bounding_box("D10", 10, 50, 40, 50)
    assert not box_zh.is_valid
    assert "zero_height" in box_zh.validation_error

    # Inverted X
    box_inv_x = validate_bounding_box("D00", 50, 10, 20, 80)
    assert not box_inv_x.is_valid
    assert "inverted_x" in box_inv_x.validation_error

    # Inverted Y
    box_inv_y = validate_bounding_box("D00", 10, 80, 50, 20)
    assert not box_inv_y.is_valid
    assert "inverted_y" in box_inv_y.validation_error

    # Negative coordinates
    box_neg = validate_bounding_box("D40", -5, 10, 50, 80)
    assert not box_neg.is_valid
    assert "negative_coordinates" in box_neg.validation_error

    # Non-finite coordinates
    box_nan = validate_bounding_box("D40", float("nan"), 10, 50, 80)
    assert not box_nan.is_valid
    assert "non_finite_coordinates" in box_nan.validation_error

    # Out of bounds
    box_oob = validate_bounding_box("D40", 10, 10, 150, 80, img_width=100, img_height=100)
    assert not box_oob.is_valid
    assert "out_of_bounds_x" in box_oob.validation_error


def test_parse_voc_xml_empty(tmp_path: Path):
    """Verify handling of XML files without objects (empty XML)."""
    xml_path = tmp_path / "empty.xml"
    create_voc_xml(xml_path, "empty.jpg", width=640, height=640, objects=[])

    parsed = parse_pascal_voc_xml(xml_path)
    assert parsed.status == AnnotationStatus.EMPTY_XML
    assert len(parsed.boxes) == 0
    assert sum(parsed.target_counts.values()) == 0
    assert len(parsed.non_target_counts) == 0
    assert parsed.parse_error is None


def test_parse_voc_xml_non_target_only(tmp_path: Path):
    """Verify handling of XML files containing only non-target classes (e.g. D01, D11)."""
    xml_path = tmp_path / "non_target.xml"
    create_voc_xml(
        xml_path,
        "non_target.jpg",
        width=640,
        height=640,
        objects=[
            {"name": "D01", "xmin": 10, "ymin": 20, "xmax": 100, "ymax": 150},
            {"name": "D11", "xmin": 30, "ymin": 40, "xmax": 200, "ymax": 250},
        ],
    )

    parsed = parse_pascal_voc_xml(xml_path)
    assert parsed.status == AnnotationStatus.NON_TARGET_ONLY
    assert len(parsed.boxes) == 2
    assert sum(parsed.target_counts.values()) == 0
    assert parsed.non_target_counts.get("D01") == 1
    assert parsed.non_target_counts.get("D11") == 1


def test_parse_voc_xml_labeled_target(tmp_path: Path):
    """Verify handling of XML files containing target classes and mixed classes."""
    xml_path = tmp_path / "mixed.xml"
    create_voc_xml(
        xml_path,
        "mixed.jpg",
        width=640,
        height=640,
        objects=[
            {"name": "D00", "xmin": 10, "ymin": 20, "xmax": 50, "ymax": 80},
            {"name": "D40", "xmin": 100, "ymin": 120, "xmax": 200, "ymax": 250},
            {"name": "D01", "xmin": 300, "ymin": 320, "xmax": 400, "ymax": 450},
        ],
    )

    parsed = parse_pascal_voc_xml(xml_path)
    assert parsed.status == AnnotationStatus.LABELED_TARGET
    assert len(parsed.boxes) == 3
    assert parsed.target_counts["D00"] == 1
    assert parsed.target_counts["D40"] == 1
    assert parsed.target_counts["D10"] == 0
    assert parsed.target_counts["D20"] == 0
    assert parsed.non_target_counts["D01"] == 1


def test_parse_voc_xml_malformed(tmp_path: Path):
    """Verify graceful error capture on malformed XML syntax."""
    xml_path = tmp_path / "corrupt.xml"
    xml_path.write_text("<annotation><unclosed_tag>", encoding="utf-8")

    parsed = parse_pascal_voc_xml(xml_path)
    assert parsed.status == "corrupted_xml"
    assert "xml_parse_error" in parsed.validation_flags
    assert parsed.parse_error is not None


def test_image_verification_and_dimension_mismatch(tmp_path: Path):
    """Verify image decoding and XML dimension mismatch detection."""
    img_path = tmp_path / "test.jpg"
    create_synthetic_image(img_path, width=120, height=90, channels=3)

    # 1. Matching dimensions
    dims, flags = verify_image_file(img_path, expected_width=120, expected_height=90)
    assert dims == (120, 90, 3)
    assert len(flags) == 0

    # 2. Mismatched width and height
    dims_mismatch, flags_mismatch = verify_image_file(img_path, expected_width=200, expected_height=150)
    assert dims_mismatch == (120, 90, 3)
    assert any("dimension_mismatch_width" in f for f in flags_mismatch)
    assert any("dimension_mismatch_height" in f for f in flags_mismatch)


def test_full_dataset_scan_and_manifest_generation(tmp_path: Path):
    """Integration test simulating all 7 subsets, train/test splits, and known anomalies."""
    data_root = tmp_path / "rdd_extracted"

    # Define test entries: (subset, split, filename, objects, img_w, img_h, xml_w, xml_h)
    entries = [
        # China_Drone: train, target D00
        ("China_Drone", "train", "China_Drone_000001", [{"name": "D00", "xmin": 10, "ymin": 10, "xmax": 50, "ymax": 50}], 100, 100, 100, 100),
        # China_MotorBike: train, empty XML
        ("China_MotorBike", "train", "China_MotorBike_000001", [], 100, 100, 100, 100),
        # Czech: train, non-target only (D01)
        ("Czech", "train", "Czech_000001", [{"name": "D01", "xmin": 15, "ymin": 20, "xmax": 60, "ymax": 70}], 100, 100, 100, 100),
        # India: train, target D40
        ("India", "train", "India_000001", [{"name": "D40", "xmin": 20, "ymin": 25, "xmax": 70, "ymax": 85}], 100, 100, 100, 100),
        # Japan: train, zero-width D20 box (reproducing Japan_001265)
        ("Japan", "train", "Japan_001265", [{"name": "D20", "xmin": 45, "ymin": 20, "xmax": 45, "ymax": 80}], 100, 100, 100, 100),
        # Norway: train, multiple target classes (D10, D20)
        ("Norway", "train", "Norway_000001", [{"name": "D10", "xmin": 10, "ymin": 10, "xmax": 40, "ymax": 40}, {"name": "D20", "xmin": 50, "ymin": 50, "xmax": 90, "ymax": 90}], 100, 100, 100, 100),
        # United_States: test split, no XML
        ("United_States", "test", "United_States_000001", None, 100, 100, None, None),
    ]

    for subset, split, stem, objs, img_w, img_h, xml_w, xml_h in entries:
        img_file = data_root / subset / split / "images" / f"{stem}.jpg"
        create_synthetic_image(img_file, width=img_w, height=img_h)

        if objs is not None:
            xml_file = data_root / subset / split / "annotations" / "xmls" / f"{stem}.xml"
            create_voc_xml(xml_file, f"{stem}.jpg", width=xml_w, height=xml_h, objects=objs)

    output_dir = tmp_path / "processed"
    validator = RDD2022Validator(verify_images=True)
    df, summary = validator.run(data_root=data_root, output_dir=output_dir)

    # 1. Manifest CSV assertions
    assert len(df) == 7
    manifest_csv = output_dir / "rdd2022_manifest.csv"
    assert manifest_csv.is_file()

    # Verify China grouping
    china_records = df[df["country"] == "China"]
    assert len(china_records) == 2
    assert set(china_records["subset"]) == {"China_Drone", "China_MotorBike"}

    # Verify LOCO eligibility
    train_records = df[df["split"] == "train"]
    assert len(train_records) == 6
    assert all(train_records["loco_eligible"] == True)

    test_records = df[df["split"] == "test"]
    assert len(test_records) == 1
    assert test_records.iloc[0]["loco_eligible"] == False
    assert test_records.iloc[0]["annotation_status"] == AnnotationStatus.UNLABELED_TEST

    # Verify Japan_001265 invalid box detection
    japan_rec = df[df["image_id"] == "Japan_001265"].iloc[0]
    assert japan_rec["total_invalid_boxes"] == 1
    assert japan_rec["is_valid"] == False
    assert "zero_width" in japan_rec["validation_flags"]

    # Verify Empty XML classification
    china_mb = df[df["image_id"] == "China_MotorBike_000001"].iloc[0]
    assert china_mb["annotation_status"] == AnnotationStatus.EMPTY_XML

    # Verify Non-Target Only classification
    czech_rec = df[df["image_id"] == "Czech_000001"].iloc[0]
    assert czech_rec["annotation_status"] == AnnotationStatus.NON_TARGET_ONLY
    assert czech_rec["count_D00"] == 0
    assert czech_rec["total_non_target_boxes"] == 1

    # 2. Summary JSON assertions
    summary_json = output_dir / "rdd2022_summary.json"
    assert summary_json.is_file()
    with open(summary_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    totals = data["totals"]
    assert totals["total_images"] == 7
    assert totals["train_images"] == 6
    assert totals["test_images"] == 1
    assert totals["loco_eligible_images"] == 6
    assert totals["total_invalid_boxes"] == 1

    # Check invalid box recorded in summary
    assert len(data["invalid_boxes"]) == 1
    assert data["invalid_boxes"][0]["image_id"] == "Japan_001265"
    assert data["invalid_boxes"][0]["class_name"] == "D20"
    assert "zero_width" in data["invalid_boxes"][0]["error"]

    # Check country-level stats
    assert "China" in data["by_country"]
    assert "Japan" in data["by_country"]
    assert data["by_country"]["Japan"]["invalid_boxes_count"] == 1


def test_nested_extracted_structure(tmp_path: Path):
    """Verify discovery when archives extract into nested folders like China_Drone/China_Drone/train/..."""
    data_root = tmp_path / "extracted"

    # Create nested China_Drone: extracted/China_Drone/China_Drone/train/images/...
    cd_img = data_root / "China_Drone" / "China_Drone" / "train" / "images" / "China_Drone_000042.jpg"
    cd_xml = data_root / "China_Drone" / "China_Drone" / "train" / "annotations" / "xmls" / "China_Drone_000042.xml"
    create_synthetic_image(cd_img, width=640, height=480)
    create_voc_xml(cd_xml, "China_Drone_000042.jpg", width=640, height=480, objects=[{"name": "D00", "xmin": 100, "ymin": 100, "xmax": 200, "ymax": 200}])

    # Create nested Japan: extracted/Japan/Japan/train/images/...
    jp_img = data_root / "Japan" / "Japan" / "train" / "images" / "Japan_000099.jpg"
    jp_xml = data_root / "Japan" / "Japan" / "train" / "annotations" / "xmls" / "Japan_000099.xml"
    create_synthetic_image(jp_img, width=640, height=480)
    create_voc_xml(jp_xml, "Japan_000099.jpg", width=640, height=480, objects=[{"name": "D40", "xmin": 50, "ymin": 50, "xmax": 150, "ymax": 150}])

    # Create nested United_States test: extracted/United_States/United_States/test/images/...
    us_img = data_root / "United_States" / "United_States" / "test" / "images" / "United_States_000077.jpg"
    create_synthetic_image(us_img, width=640, height=480)

    output_dir = tmp_path / "nested_out"
    validator = RDD2022Validator(verify_images=True)
    df, summary = validator.run(data_root=data_root, output_dir=output_dir)

    assert len(df) == 3
    # Check China attribution
    cd_row = df[df["image_id"] == "China_Drone_000042"].iloc[0]
    assert cd_row["country"] == "China"
    assert cd_row["subset"] == "China_Drone"
    assert cd_row["split"] == "train"
    assert cd_row["annotation_status"] == AnnotationStatus.LABELED_TARGET
    assert cd_row["count_D00"] == 1
    assert cd_row["xml_path"] != ""

    # Check Japan attribution
    jp_row = df[df["image_id"] == "Japan_000099"].iloc[0]
    assert jp_row["country"] == "Japan"
    assert jp_row["subset"] == "Japan"
    assert jp_row["split"] == "train"
    assert jp_row["count_D40"] == 1

    # Check US test attribution
    us_row = df[df["image_id"] == "United_States_000077"].iloc[0]
    assert us_row["country"] == "United_States"
    assert us_row["subset"] == "United_States"
    assert us_row["split"] == "test"
    assert us_row["loco_eligible"] == False

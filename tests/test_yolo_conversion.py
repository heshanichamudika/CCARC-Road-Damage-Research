"""Unit and Integration Tests for RDD2022 Cleaning Policy and YOLO Label Conversion."""

import json
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
import pytest

from src.data.yolo_converter import (
    ID_TO_TARGET_CLASS,
    TARGET_CLASS_TO_ID,
    ConversionStatus,
    RDD2022YOLOConverter,
    convert_pascal_to_yolo_box,
)
from tests.test_dataset_validation import create_synthetic_image, create_voc_xml


def test_class_mapping():
    """Verify target class mapping and ID assignments."""
    assert TARGET_CLASS_TO_ID["D00"] == 0
    assert TARGET_CLASS_TO_ID["D10"] == 1
    assert TARGET_CLASS_TO_ID["D20"] == 2
    assert TARGET_CLASS_TO_ID["D40"] == 3

    assert ID_TO_TARGET_CLASS[0] == "D00"
    assert ID_TO_TARGET_CLASS[1] == "D10"
    assert ID_TO_TARGET_CLASS[2] == "D20"
    assert ID_TO_TARGET_CLASS[3] == "D40"

    assert "D01" not in TARGET_CLASS_TO_ID
    assert "D11" not in TARGET_CLASS_TO_ID


def test_coordinate_conversion_valid():
    """Verify exact mathematical normalization of Pascal VOC bounding box to YOLO."""
    img_w, img_h = 640, 640
    # Box: xmin=100, ymin=200, xmax=300, ymax=400
    res = convert_pascal_to_yolo_box(
        class_name="D00",
        xmin_raw=100,
        ymin_raw=200,
        xmax_raw=300,
        ymax_raw=400,
        img_width=img_w,
        img_height=img_h,
    )

    assert res.is_valid
    assert res.yolo_box is not None
    box = res.yolo_box
    assert box.class_id == 0
    assert box.class_name == "D00"

    # Math: x_center = (100+300)/(2*640) = 400/1280 = 0.3125
    assert pytest.approx(box.x_center, abs=1e-5) == 0.3125
    # Math: y_center = (200+400)/(2*640) = 600/1280 = 0.46875
    assert pytest.approx(box.y_center, abs=1e-5) == 0.46875
    # Math: w = (300-100)/640 = 200/640 = 0.3125
    assert pytest.approx(box.width, abs=1e-5) == 0.3125
    # Math: h = (400-200)/640 = 200/640 = 0.3125
    assert pytest.approx(box.height, abs=1e-5) == 0.3125

    # Check YOLO text line output
    line = box.to_yolo_line()
    tokens = line.strip().split()
    assert tokens[0] == "0"
    assert tokens[1] == "0.312500"
    assert tokens[2] == "0.468750"
    assert tokens[3] == "0.312500"
    assert tokens[4] == "0.312500"


def test_strict_out_of_bounds_rejection_by_default():
    """Verify that by default, out-of-bounds boxes are strictly rejected rather than clamped."""
    img_w, img_h = 640, 480
    # xmax exceeds width (642 > 640)
    res_x = convert_pascal_to_yolo_box(
        class_name="D10",
        xmin_raw=500,
        ymin_raw=400,
        xmax_raw=642,
        ymax_raw=450,
        img_width=img_w,
        img_height=img_h,
        clamp_out_of_bounds=False,
    )
    assert not res_x.is_valid
    assert "out_of_bounds_x" in res_x.exclusion_reason

    # ymax exceeds height (485 > 480)
    res_y = convert_pascal_to_yolo_box(
        class_name="D10",
        xmin_raw=500,
        ymin_raw=400,
        xmax_raw=600,
        ymax_raw=485,
        img_width=img_w,
        img_height=img_h,
        clamp_out_of_bounds=False,
    )
    assert not res_y.is_valid
    assert "out_of_bounds_y" in res_y.exclusion_reason

    # negative coordinates
    res_neg = convert_pascal_to_yolo_box(
        class_name="D10",
        xmin_raw=-5,
        ymin_raw=100,
        xmax_raw=200,
        ymax_raw=200,
        img_width=img_w,
        img_height=img_h,
        clamp_out_of_bounds=False,
    )
    assert not res_neg.is_valid
    assert "out_of_bounds_negative" in res_neg.exclusion_reason


def test_coordinate_conversion_clamping_with_explicit_logging():
    """Verify that when clamp_out_of_bounds=True, overflow coordinates are safely clamped and logged."""
    img_w, img_h = 640, 480
    # xmax exceeds width (642 > 640), ymax exceeds height (485 > 480)
    res = convert_pascal_to_yolo_box(
        class_name="D10",
        xmin_raw=500,
        ymin_raw=400,
        xmax_raw=642,
        ymax_raw=485,
        img_width=img_w,
        img_height=img_h,
        clamp_out_of_bounds=True,
    )

    assert res.is_valid
    assert res.was_clamped is True
    assert res.original_coords == (500.0, 400.0, 642.0, 485.0)
    assert res.clamped_coords == (500.0, 400.0, 640.0, 480.0)

    box = res.yolo_box
    assert box is not None
    assert box.class_id == 1
    # Clamped xmax = 640 -> w_px = 140 -> norm_w = 140 / 640 = 0.21875
    # Clamped xcenter = (500 + 640) / (2 * 640) = 1140 / 1280 = 0.890625
    assert pytest.approx(box.x_center, abs=1e-5) == 0.890625
    assert pytest.approx(box.width, abs=1e-5) == 0.21875
    assert 0.0 < box.x_center < 1.0
    assert 0.0 < box.y_center < 1.0
    assert 0.0 < box.width <= 1.0
    assert 0.0 < box.height <= 1.0


def test_edge_case_precision_full_image_box():
    """Verify edge case: box spanning the entire image (0, 0, W, H)."""
    res = convert_pascal_to_yolo_box(
        class_name="D00",
        xmin_raw=0,
        ymin_raw=0,
        xmax_raw=640,
        ymax_raw=640,
        img_width=640,
        img_height=640,
    )
    assert res.is_valid
    box = res.yolo_box
    assert box is not None
    assert pytest.approx(box.x_center, abs=1e-6) == 0.5
    assert pytest.approx(box.y_center, abs=1e-6) == 0.5
    assert pytest.approx(box.width, abs=1e-6) == 1.0
    assert pytest.approx(box.height, abs=1e-6) == 1.0


def test_coordinate_conversion_invalid_boxes():
    """Verify exclusion of zero-width (Japan_001265), zero-height, inverted and non-finite boxes."""
    # 1. Zero width (reproducing Japan_001265 anomaly)
    res_zw = convert_pascal_to_yolo_box("D20", 450, 100, 450, 200, 640, 640)
    assert not res_zw.is_valid
    assert "zero_width" in res_zw.exclusion_reason

    # 2. Zero height
    res_zh = convert_pascal_to_yolo_box("D40", 100, 300, 200, 300, 640, 640)
    assert not res_zh.is_valid
    assert "zero_height" in res_zh.exclusion_reason

    # 3. Inverted coordinates
    res_inv = convert_pascal_to_yolo_box("D00", 300, 100, 100, 200, 640, 640)
    assert not res_inv.is_valid
    assert "inverted_x" in res_inv.exclusion_reason

    # 4. Non-finite values (NaN / Inf)
    res_nan = convert_pascal_to_yolo_box("D00", float("nan"), 100, 200, 200, 640, 640)
    assert not res_nan.is_valid
    assert "non_finite" in res_nan.exclusion_reason

    # 5. Box entirely outside boundaries
    res_oob = convert_pascal_to_yolo_box("D00", 700, 800, 900, 1000, 640, 640)
    assert not res_oob.is_valid
    assert "out_of_bounds" in res_oob.exclusion_reason

    # In clamping mode, completely outside box collapses to zero width/height
    res_collapsed = convert_pascal_to_yolo_box("D00", 700, 800, 900, 1000, 640, 640, clamp_out_of_bounds=True)
    assert not res_collapsed.is_valid
    assert "collapsed" in res_collapsed.exclusion_reason


def test_mixed_class_and_invalid_box_isolation(tmp_path: Path):
    """Verify policy #4 & #7: exclude invalid box without discarding valid ones, and record non-target risk."""
    converter = RDD2022YOLOConverter()

    img_file = tmp_path / "Japan_001265.jpg"
    create_synthetic_image(img_file, width=640, height=640)

    # XML contains:
    # 1 valid D00 box
    # 1 zero-width D20 box (invalid)
    # 1 non-target D01 box
    xml_file = tmp_path / "Japan_001265.xml"
    create_voc_xml(
        xml_file,
        "Japan_001265.jpg",
        width=640,
        height=640,
        objects=[
            {"name": "D00", "xmin": 50, "ymin": 50, "xmax": 150, "ymax": 150},
            {"name": "D20", "xmin": 450, "ymin": 100, "xmax": 450, "ymax": 200},  # zero width!
            {"name": "D01", "xmin": 200, "ymin": 200, "xmax": 300, "ymax": 300},  # non-target!
        ],
    )

    labels_dir = tmp_path / "labels"
    record, label_text = converter.convert_image_entry(
        img_path=img_file,
        xml_path=xml_file,
        country="Japan",
        subset="Japan",
        split="train",
        labels_output_root=labels_dir,
    )

    # Status must be converted_target because D00 is valid
    assert record.conversion_status == ConversionStatus.CONVERTED_TARGET
    assert record.yolo_boxes_count == 1
    assert record.count_D00 == 1
    assert record.count_D20 == 0  # invalid box excluded

    # Exclusion audit
    assert record.excluded_boxes_count == 1
    assert "zero_width" in record.exclusion_details[0]["reason"]

    # Incomplete label risk because D01 is present
    assert record.has_non_target_objects == True
    assert record.incomplete_label_risk == True
    assert "D01:1" in record.non_target_classes

    # Label text must contain ONLY the valid D00 box (class 0)
    assert label_text is not None
    lines = label_text.strip().split("\n")
    assert len(lines) == 1
    assert lines[0].startswith("0 ")


def test_empty_xml_policy(tmp_path: Path):
    """Verify policy #5: empty XMLs are isolated as candidate negatives, not verified background."""
    img_file = tmp_path / "empty.jpg"
    xml_file = tmp_path / "empty.xml"
    create_synthetic_image(img_file, width=640, height=640)
    create_voc_xml(xml_file, "empty.jpg", width=640, height=640, objects=[])

    labels_dir = tmp_path / "labels"

    # Default policy: include_empty_as_negatives = False -> no label file written
    converter_default = RDD2022YOLOConverter(include_empty_as_negatives=False)
    rec_default, text_default = converter_default.convert_image_entry(
        img_path=img_file,
        xml_path=xml_file,
        country="India",
        subset="India",
        split="train",
        labels_output_root=labels_dir,
    )
    assert rec_default.conversion_status == ConversionStatus.EMPTY_XML_CANDIDATE
    assert text_default is None
    assert rec_default.label_path is None

    # Opt-in policy: include_empty_as_negatives = True -> 0-byte file generated
    converter_optin = RDD2022YOLOConverter(include_empty_as_negatives=True)
    rec_optin, text_optin = converter_optin.convert_image_entry(
        img_path=img_file,
        xml_path=xml_file,
        country="India",
        subset="India",
        split="train",
        labels_output_root=labels_dir,
    )
    assert rec_optin.conversion_status == ConversionStatus.EMPTY_XML_CANDIDATE
    assert text_optin == ""
    assert rec_optin.label_path is not None


def test_non_target_only_quarantine_policy(tmp_path: Path):
    """Verify policy #6: non-target-only images are quarantined, NOT converted to empty labels."""
    img_file = tmp_path / "non_target.jpg"
    xml_file = tmp_path / "non_target.xml"
    create_synthetic_image(img_file, width=640, height=640)
    create_voc_xml(
        xml_file,
        "non_target.jpg",
        width=640,
        height=640,
        objects=[
            {"name": "D01", "xmin": 100, "ymin": 100, "xmax": 200, "ymax": 200},
            {"name": "D11", "xmin": 300, "ymin": 300, "xmax": 400, "ymax": 400},
        ],
    )

    labels_dir = tmp_path / "labels"
    converter = RDD2022YOLOConverter()
    rec, label_text = converter.convert_image_entry(
        img_path=img_file,
        xml_path=xml_file,
        country="Czech",
        subset="Czech",
        split="train",
        labels_output_root=labels_dir,
    )

    assert rec.conversion_status == ConversionStatus.QUARANTINED_NON_TARGET
    assert label_text is None
    assert rec.label_path is None
    assert rec.has_non_target_objects == True
    assert "D01:1" in rec.non_target_classes
    assert "D11:1" in rec.non_target_classes


def test_official_test_split_skipped(tmp_path: Path):
    """Verify policy #8: official test split images are never given ground-truth labels."""
    img_file = tmp_path / "test_img.jpg"
    create_synthetic_image(img_file, width=640, height=640)

    labels_dir = tmp_path / "labels"
    converter = RDD2022YOLOConverter()
    rec, label_text = converter.convert_image_entry(
        img_path=img_file,
        xml_path=None,
        country="Norway",
        subset="Norway",
        split="test",
        labels_output_root=labels_dir,
    )

    assert rec.conversion_status == ConversionStatus.UNLABELED_TEST
    assert rec.loco_eligible == False
    assert label_text is None
    assert rec.label_path is None


def test_full_yolo_conversion_pipeline_and_rerun_safety(tmp_path: Path):
    """Integration test simulating full conversion, nested directories, and rerun safety."""
    data_root = tmp_path / "extracted"

    # 1. China_Drone nested: train, target D00
    cd_img = data_root / "China_Drone" / "China_Drone" / "train" / "images" / "China_Drone_000001.jpg"
    cd_xml = data_root / "China_Drone" / "China_Drone" / "train" / "annotations" / "xmls" / "China_Drone_000001.xml"
    create_synthetic_image(cd_img, width=640, height=480)
    create_voc_xml(cd_xml, "China_Drone_000001.jpg", 640, 480, objects=[{"name": "D00", "xmin": 50, "ymin": 50, "xmax": 200, "ymax": 200}])

    # 2. Czech: train, non-target only (quarantined)
    cz_img = data_root / "Czech" / "train" / "images" / "Czech_000001.jpg"
    cz_xml = data_root / "Czech" / "train" / "annotations" / "xmls" / "Czech_000001.xml"
    create_synthetic_image(cz_img, width=640, height=480)
    create_voc_xml(cz_xml, "Czech_000001.jpg", 640, 480, objects=[{"name": "D01", "xmin": 100, "ymin": 100, "xmax": 300, "ymax": 300}])

    # 3. Japan: train, Japan_001265 with zero width D20 and valid D40
    jp_img = data_root / "Japan" / "train" / "images" / "Japan_001265.jpg"
    jp_xml = data_root / "Japan" / "train" / "annotations" / "xmls" / "Japan_001265.xml"
    create_synthetic_image(jp_img, width=640, height=480)
    create_voc_xml(
        jp_xml,
        "Japan_001265.jpg",
        640,
        480,
        objects=[
            {"name": "D20", "xmin": 450, "ymin": 100, "xmax": 450, "ymax": 200},  # zero width
            {"name": "D40", "xmin": 50, "ymin": 50, "xmax": 150, "ymax": 150},   # valid
        ],
    )

    # 4. US: test, unlabeled
    us_img = data_root / "United_States" / "test" / "images" / "United_States_000001.jpg"
    create_synthetic_image(us_img, width=640, height=480)

    output_dir = tmp_path / "yolo_out"

    converter = RDD2022YOLOConverter(include_empty_as_negatives=False)
    df, summary = converter.convert_dataset(data_root=data_root, output_dir=output_dir, clean_output=True)

    # Assertions on manifest
    assert len(df) == 4
    manifest_csv = output_dir / "yolo_conversion_manifest.csv"
    assert manifest_csv.is_file()

    # China_Drone converted
    cd_row = df[df["image_id"] == "China_Drone_000001"].iloc[0]
    assert cd_row["conversion_status"] == ConversionStatus.CONVERTED_TARGET
    assert cd_row["yolo_boxes_count"] == 1
    assert Path(cd_row["label_path"]).is_file()

    # Czech quarantined
    cz_row = df[df["image_id"] == "Czech_000001"].iloc[0]
    assert cz_row["conversion_status"] == ConversionStatus.QUARANTINED_NON_TARGET
    assert cz_row["label_path"] == ""

    # Japan_001265 converted (valid D40 converted, invalid D20 excluded)
    jp_row = df[df["image_id"] == "Japan_001265"].iloc[0]
    assert jp_row["conversion_status"] == ConversionStatus.CONVERTED_TARGET
    assert jp_row["count_D40"] == 1
    assert jp_row["count_D20"] == 0
    assert jp_row["excluded_boxes_count"] == 1
    lbl_file = Path(jp_row["label_path"])
    assert lbl_file.is_file()
    lines = lbl_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    assert lines[0].startswith("3 ")  # Class 3 is D40

    # Summary JSON assertions
    summary_json = output_dir / "yolo_conversion_summary.json"
    assert summary_json.is_file()
    with open(summary_json, "r", encoding="utf-8") as f:
        sdata = json.load(f)

    totals = sdata["totals"]
    assert totals["total_images_scanned"] == 4
    assert totals["total_label_files_written"] == 2
    assert totals["total_yolo_boxes"] == 2
    assert totals["total_invalid_boxes_excluded"] == 1

    # RERUN SAFETY CHECK:
    # Running again with clean_output=True should cleanly overwrite without duplicate or corrupted states
    df_rerun, summary_rerun = converter.convert_dataset(data_root=data_root, output_dir=output_dir, clean_output=True)
    assert len(df_rerun) == 4
    assert summary_rerun["totals"]["total_label_files_written"] == 2


def test_subset_path_isolation_china_drone_and_motorbike(tmp_path: Path):
    """Verify that China_Drone and China_MotorBike labels are strictly separated and cannot collide."""
    data_root = tmp_path / "extracted"

    # Create China_Drone image + XML
    cd_img = data_root / "China_Drone" / "train" / "images" / "same_stem.jpg"
    cd_xml = data_root / "China_Drone" / "train" / "annotations" / "xmls" / "same_stem.xml"
    create_synthetic_image(cd_img, 640, 480)
    create_voc_xml(cd_xml, "same_stem.jpg", 640, 480, objects=[{"name": "D00", "xmin": 10, "ymin": 10, "xmax": 100, "ymax": 100}])

    # Create China_MotorBike image + XML with the same stem name to test isolation
    cm_img = data_root / "China_MotorBike" / "train" / "images" / "same_stem.jpg"
    cm_xml = data_root / "China_MotorBike" / "train" / "annotations" / "xmls" / "same_stem.xml"
    create_synthetic_image(cm_img, 640, 480)
    create_voc_xml(cm_xml, "same_stem.jpg", 640, 480, objects=[{"name": "D10", "xmin": 20, "ymin": 20, "xmax": 200, "ymax": 200}])

    output_dir = tmp_path / "yolo_isolated"
    converter = RDD2022YOLOConverter()
    df, summary = converter.convert_dataset(data_root=data_root, output_dir=output_dir, clean_output=True)

    assert len(df) == 2
    cd_rec = df[df["subset"] == "China_Drone"].iloc[0]
    cm_rec = df[df["subset"] == "China_MotorBike"].iloc[0]

    # Verify separate paths
    assert cd_rec["label_path"] != cm_rec["label_path"]
    assert "China_Drone" in cd_rec["label_path"]
    assert "China_MotorBike" in cm_rec["label_path"]

    # Verify both label files exist independently
    assert Path(cd_rec["label_path"]).is_file()
    assert Path(cm_rec["label_path"]).is_file()

    # Verify contents are distinct
    cd_content = Path(cd_rec["label_path"]).read_text(encoding="utf-8").strip()
    cm_content = Path(cm_rec["label_path"]).read_text(encoding="utf-8").strip()
    assert cd_content.startswith("0 ")  # D00
    assert cm_content.startswith("1 ")  # D10

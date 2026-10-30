"""Data handling, RDD2022 ingestion, preprocessing and dataset definitions."""

from src.data.rdd2022_validator import (
    AnnotationStatus,
    BoundingBox,
    DEFAULT_TARGET_CLASSES,
    ImageRecord,
    ParsedXML,
    RDD2022Validator,
    map_subset_to_country,
    parse_pascal_voc_xml,
    validate_bounding_box,
    verify_image_file,
)

__all__ = [
    "AnnotationStatus",
    "BoundingBox",
    "DEFAULT_TARGET_CLASSES",
    "ImageRecord",
    "ParsedXML",
    "RDD2022Validator",
    "map_subset_to_country",
    "parse_pascal_voc_xml",
    "validate_bounding_box",
    "verify_image_file",
]

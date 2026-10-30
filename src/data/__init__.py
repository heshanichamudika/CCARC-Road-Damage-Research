"""Data handling, RDD2022 ingestion, preprocessing and dataset definitions."""

from src.data.loco_auditor import (
    AuditCheckResult,
    CheckStatus,
    LOCOAuditReport,
    RDD2022LOCOAuditor,
)
from src.data.loco_splitter import (
    CHINA_SUBSETS,
    LOCO_COUNTRIES,
    FoldSplitResult,
    RDD2022LOCOSplitter,
    create_link_or_copy,
)
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
from src.data.yolo_converter import (
    ID_TO_TARGET_CLASS,
    TARGET_CLASS_TO_ID,
    BoxConversionResult,
    ConversionStatus,
    ImageConversionRecord,
    RDD2022YOLOConverter,
    YOLOBox,
    convert_pascal_to_yolo_box,
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
    "ID_TO_TARGET_CLASS",
    "TARGET_CLASS_TO_ID",
    "BoxConversionResult",
    "ConversionStatus",
    "ImageConversionRecord",
    "RDD2022YOLOConverter",
    "YOLOBox",
    "convert_pascal_to_yolo_box",
    "CHINA_SUBSETS",
    "LOCO_COUNTRIES",
    "FoldSplitResult",
    "RDD2022LOCOSplitter",
    "create_link_or_copy",
    "AuditCheckResult",
    "CheckStatus",
    "LOCOAuditReport",
    "RDD2022LOCOAuditor",
]

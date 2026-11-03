"""Model architecture definitions, YOLOv8 wrappers, and CCARC consistency modules."""

from src.models.ccarc_consistency import (
    ReliabilityConsistencyConfig,
    ReliabilityConsistencyLoss,
)
from src.models.ccarc_loss import (
    CCARCDetectionLoss,
    ClassAwareLearningConfig,
    ClassWeightedBCEWithLogitsLoss,
    attach_class_aware_loss,
    compute_class_weights,
    extract_source_train_class_counts,
    resolve_detection_model,
)
from src.models.ccarc_perturbations import (
    PhotometricPerturbation,
    PhotometricPerturbationConfig,
)

__all__ = [
    "ClassAwareLearningConfig",
    "compute_class_weights",
    "extract_source_train_class_counts",
    "ClassWeightedBCEWithLogitsLoss",
    "CCARCDetectionLoss",
    "attach_class_aware_loss",
    "resolve_detection_model",
    "PhotometricPerturbation",
    "PhotometricPerturbationConfig",
    "ReliabilityConsistencyConfig",
    "ReliabilityConsistencyLoss",
]

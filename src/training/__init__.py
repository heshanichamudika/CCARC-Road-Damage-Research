"""Training pipelines, loss computation, and LOCO training loops."""

from src.training.baseline_config import (
    BaselineTrainingConfig,
    check_gpu_environment,
    resolve_fold_data_yaml,
)
from src.training.ccarc_sampler import (
    CountryBalancedBatchSampler,
    CountryBalancedSampler,
    build_country_index_map,
    create_country_balanced_batch_sampler,
    load_manifest_country_lookup,
    resolve_sample_country,
)
from src.training.ccarc_config import (
    CCARCTrainingConfig,
    VALID_VARIANTS,
)
from src.training.ccarc_trainer import (
    CCARCDetectionTrainer,
    CCARCLOCOTrainer,
)
from src.training.yolov8_trainer import YOLOv8LOCOTrainer


def __getattr__(name: str):
    if name in ("AblationOrchestrator", "ExperimentStatus", "VARIANT_SPECIFICATIONS"):
        from src.training.ablation_orchestrator import (
            AblationOrchestrator,
            ExperimentStatus,
            VARIANT_SPECIFICATIONS,
        )
        mapping = {
            "AblationOrchestrator": AblationOrchestrator,
            "ExperimentStatus": ExperimentStatus,
            "VARIANT_SPECIFICATIONS": VARIANT_SPECIFICATIONS,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaselineTrainingConfig",
    "check_gpu_environment",
    "YOLOv8LOCOTrainer",
    "resolve_fold_data_yaml",
    "CountryBalancedBatchSampler",
    "CountryBalancedSampler",
    "build_country_index_map",
    "create_country_balanced_batch_sampler",
    "load_manifest_country_lookup",
    "resolve_sample_country",
    "CCARCTrainingConfig",
    "VALID_VARIANTS",
    "CCARCDetectionTrainer",
    "CCARCLOCOTrainer",
    "AblationOrchestrator",
    "ExperimentStatus",
    "VARIANT_SPECIFICATIONS",
]

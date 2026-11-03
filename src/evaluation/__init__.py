"""Evaluation protocols, Leave-One-Country-Out (LOCO) testing, and metrics reporting."""

from src.evaluation.evaluator import (
    ExperimentTargetResult,
    LOCOBenchmarkEvaluator,
)
from src.evaluation.metrics import (
    compute_f1,
    extract_detection_metrics,
    normalize_metrics_dict,
)
from src.evaluation.report_generator import (
    EvaluationReportGenerator,
    VARIANT_DESCRIPTIONS,
)
from src.evaluation.rwc_gradient_verifier import (
    RWCGradientAuditResult,
    audit_rwc_gradients,
)

__all__ = [
    "compute_f1",
    "extract_detection_metrics",
    "normalize_metrics_dict",
    "ExperimentTargetResult",
    "LOCOBenchmarkEvaluator",
    "EvaluationReportGenerator",
    "VARIANT_DESCRIPTIONS",
    "RWCGradientAuditResult",
    "audit_rwc_gradients",
]

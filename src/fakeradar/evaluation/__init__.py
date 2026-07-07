from .metrics import (
    accuracy_at,
    average_precision,
    balanced_accuracy,
    expected_calibration_error,
    roc_auc,
    summarize,
    tpr_at_fpr,
)

__all__ = [
    "roc_auc",
    "average_precision",
    "accuracy_at",
    "balanced_accuracy",
    "tpr_at_fpr",
    "expected_calibration_error",
    "summarize",
]

# calibrate.py is imported lazily (from .calibrate import ...) by its users so
# that importing fakeradar.evaluation never pulls in torch DataLoader machinery.

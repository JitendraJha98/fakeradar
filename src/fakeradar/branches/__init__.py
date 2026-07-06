"""Forensic branches. Importing this package registers all built-in branches."""

from .base import BRANCH_REGISTRY, DetectionBranch, build_branch, register_branch
from .frequency import FrequencyBranch
from .gradient import GradientFieldBranch
from .semantic import SemanticBranch

__all__ = [
    "BRANCH_REGISTRY",
    "DetectionBranch",
    "build_branch",
    "register_branch",
    "GradientFieldBranch",
    "FrequencyBranch",
    "SemanticBranch",
]

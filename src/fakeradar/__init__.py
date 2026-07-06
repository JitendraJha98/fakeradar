"""fakeradar — local-first AI-generated image detection.

Multi-branch forensic pipeline: gradient-field analysis (signature branch),
frequency-spectrum analysis, and optional frozen foundation-model semantics,
fused into a single calibrated score.
"""

__version__ = "0.1.0"

__all__ = ["Detector", "DetectionResult", "__version__"]


def __getattr__(name):  # lazy imports keep `import fakeradar` torch-free
    if name in ("Detector", "DetectionResult"):
        from .detector import DetectionResult, Detector

        return {"Detector": Detector, "DetectionResult": DetectionResult}[name]
    raise AttributeError(f"module 'fakeradar' has no attribute {name!r}")

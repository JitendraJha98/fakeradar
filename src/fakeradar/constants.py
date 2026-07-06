"""Shared constants."""

# ImageNet / CLIP normalization (semantic branch fallback)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_CROP_SIZE = 256
DEFAULT_SEMANTIC_SIZE = 224

VERDICT_REAL = "likely_real"
VERDICT_FAKE = "likely_ai"
VERDICT_UNCERTAIN = "uncertain"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

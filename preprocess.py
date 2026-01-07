"""
MedAI — Enterprise-Grade Medical Image Preprocessing Framework
Enhanced version with better handling for various image sizes
"""

import io
import warnings
from typing import Tuple, Dict, Union, Any, List
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torch
import torchvision.transforms as transforms
from loguru import logger

warnings.filterwarnings("ignore")  # suppress non-critical PIL / NumPy warnings

# =============================================================================
# ENUMS & CONFIGURATION
# =============================================================================

class ImageModality(Enum):
    XRAY = "xray"
    CT = "ct"
    MRI = "mri"
    ULTRASOUND = "ultrasound"

@dataclass(frozen=True)
class PreprocessingConfig:
    NORMALIZATION_MEAN: List[float] = (0.485, 0.456, 0.406)
    NORMALIZATION_STD: List[float] = (0.229, 0.224, 0.225)
    MIN_DIMENSION: int = 100  # Reduced from 224 to handle smaller images
    MAX_DIMENSION: int = 4096

# =============================================================================
# IMAGE QUALITY ANALYSIS
# =============================================================================

class ImageQualityAnalyzer:
    @staticmethod
    def analyze_contrast(image: Image.Image) -> Tuple[float, str]:
        gray = image.convert("L")
        contrast = float(np.std(np.array(gray)))
        if contrast < 30: label = "Low contrast"
        elif contrast < 60: label = "Moderate contrast"
        elif contrast < 100: label = "Good contrast"
        else: label = "High contrast"
        return contrast, label

    @staticmethod
    def analyze_histogram(image: Image.Image) -> Dict[str, Any]:
        gray = image.convert("L")
        arr = np.array(gray)
        return {
            "mean_intensity": float(np.mean(arr)),
            "median_intensity": float(np.median(arr)),
            "std_intensity": float(np.std(arr)),
            "dynamic_range": float(np.max(arr) - np.min(arr)),
            "min_intensity": float(np.min(arr)),
            "max_intensity": float(np.max(arr)),
        }

    @staticmethod
    def get_quality_score(image: Image.Image) -> Dict[str, Any]:
        contrast_score, contrast_label = ImageQualityAnalyzer.analyze_contrast(image)
        hist = ImageQualityAnalyzer.analyze_histogram(image)

        contrast_factor = min(contrast_score / 150, 1.0) * 0.5
        hist_factor = 0.7 if hist["dynamic_range"] > 100 else 0.6 if hist["dynamic_range"] > 50 else 0.5
        overall = ((contrast_factor + hist_factor) / 2) * 100

        return {
            "overall_score": round(overall, 1),
            "contrast": {"score": round(contrast_score, 2), "message": contrast_label},
            "histogram": hist,
            "quality_category": ImageQualityAnalyzer._category(overall),
            "recommendations": ImageQualityAnalyzer._recommendations(contrast_score, hist),
        }

    @staticmethod
    def _category(score: float) -> str:
        if score >= 80: return "Excellent"
        if score >= 60: return "Good"
        if score >= 40: return "Fair"
        if score >= 20: return "Poor"
        return "Unacceptable"

    @staticmethod
    def _recommendations(contrast: float, hist: Dict[str, Any]) -> List[str]:
        recs = []
        if contrast < 40: recs.append("Low contrast detected — adjust exposure settings.")
        if hist["dynamic_range"] < 50: recs.append("Limited dynamic range — review acquisition parameters.")
        if not recs: recs.append("Image quality is acceptable for AI inference.")
        return recs

# =============================================================================
# IMAGE ENHANCEMENT
# =============================================================================

class MedicalImageEnhancer:
    @staticmethod
    def enhance_contrast(image: Image.Image) -> Image.Image:
        return ImageEnhance.Contrast(image).enhance(1.2)

    @staticmethod
    def normalize_intensity(image: Image.Image) -> Image.Image:
        gray = image.convert("L")
        arr = np.array(gray, dtype=np.float32)
        p1, p99 = np.percentile(arr, (1, 99))
        arr = np.clip(arr, p1, p99)
        arr = ((arr - p1) / (p99 - p1) * 255).astype(np.uint8)
        norm = Image.fromarray(arr)
        return Image.merge("RGB", (norm, norm, norm)) if image.mode == "RGB" else norm

    @staticmethod
    def equalize_histogram(image: Image.Image) -> Image.Image:
        gray = image.convert("L")
        eq = ImageOps.equalize(gray)
        return Image.merge("RGB", (eq, eq, eq)) if image.mode == "RGB" else eq

# =============================================================================
# PREPROCESSOR
# =============================================================================

class MedicalImagePreprocessor:
    def __init__(self, modality: Union[str, ImageModality] = ImageModality.XRAY):
        self.modality = ImageModality(modality.lower()) if isinstance(modality, str) else modality
        self.config = PreprocessingConfig()
        self.analyzer = ImageQualityAnalyzer()
        self.enhancer = MedicalImageEnhancer()
        logger.info(f"MedAI preprocessor initialized for {self.modality.value.upper()}")

    def load_image(self, image_bytes: bytes) -> Tuple[Image.Image, Dict[str, Any]]:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert("RGB")
        
        # Log image size for debugging
        logger.info(f"Image loaded: size={image.size}, mode={image.mode}")
        
        # More lenient size checking - warn but don't fail
        if image.width < self.config.MIN_DIMENSION or image.height < self.config.MIN_DIMENSION:
            logger.warning(f"Image size {image.size} is below minimum {self.config.MIN_DIMENSION}px. Processing anyway.")
        
        if image.width > self.config.MAX_DIMENSION or image.height > self.config.MAX_DIMENSION:
            logger.warning(f"Image size {image.size} exceeds maximum {self.config.MAX_DIMENSION}px. Will be resized.")
        
        quality = self.analyzer.get_quality_score(image)
        return image, quality

    def preprocess(self, image: Image.Image, quality: Dict[str, Any]) -> Image.Image:
        if quality["overall_score"] < 60:
            if quality["contrast"]["score"] < 40:
                image = self.enhancer.enhance_contrast(image)
            image = self.enhancer.normalize_intensity(image)
        return image.convert("RGB")

    def transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(256),  # Resize to at least 256px on smaller dimension
            transforms.CenterCrop(224),  # Crop to 224x224 for model input
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.NORMALIZATION_MEAN,
                                 std=self.config.NORMALIZATION_STD),
        ])
    
    def adaptive_transform(self, image: Image.Image) -> transforms.Compose:
        """Create adaptive transform based on image size"""
        width, height = image.size
        
        # For very small images, use different resize strategy
        if width < 224 or height < 224:
            # For images smaller than 224, resize to 224 first
            return transforms.Compose([
                transforms.Resize(224),  # Upscale small images
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.NORMALIZATION_MEAN,
                                     std=self.config.NORMALIZATION_STD),
            ])
        else:
            # For normal sized images, use standard transform
            return self.transform()

    def process(self, image_bytes: bytes) -> Tuple[torch.Tensor, Dict[str, Any]]:
        image, quality = self.load_image(image_bytes)
        image = self.preprocess(image, quality)
        
        # Use adaptive transform based on image size
        transform = self.adaptive_transform(image)
        tensor = transform(image).unsqueeze(0)
        
        if tensor.shape[1] != 3:
            raise RuntimeError("RGB channel enforcement failed")
        
        metadata = {
            "modality": self.modality.value,
            "original_size": image.size,
            "tensor_shape": list(tensor.shape),
            "quality_report": quality,
            "enhanced": quality["overall_score"] < 60,
            "transform_used": "adaptive" if image.width < 224 or image.height < 224 else "standard"
        }
        return tensor, metadata

# =============================================================================
# MULTIPLE-FIELD BACKEND SOLUTION (MULTER SAFE)
# =============================================================================

def get_uploaded_image(req_files: dict) -> Union[bytes, None]:
    """
    Accept multiple possible upload field names to prevent Multer errors.
    Example accepted fields: 'image', 'file', 'scan'
    """
    for key in ["image", "file", "scan"]:
        if key in req_files and len(req_files[key]) > 0:
            return req_files[key][0].buffer  # Multer memoryStorage buffer
    return None

# =============================================================================
# BACKWARD-COMPATIBLE API
# =============================================================================

def preprocess_image(image_bytes: bytes,
                     modality: str = "xray",
                     return_metadata: bool = False,
                     enable_enhancement: bool = True
                     ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
    preprocessor = MedicalImagePreprocessor(modality)
    if enable_enhancement:
        tensor, meta = preprocessor.process(image_bytes)
        return (tensor, meta) if return_metadata else tensor
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = preprocessor.transform()(image).unsqueeze(0)
    return tensor

# =============================================================================
# UTILITIES
# =============================================================================

def ensure_rgb_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[1] == 1:
        tensor = tensor.repeat(1, 3, 1, 1)
    elif tensor.shape[1] == 4:
        tensor = tensor[:, :3]
    elif tensor.shape[1] != 3:
        raise ValueError(f"Invalid channel count: {tensor.shape[1]}")
    return tensor

def debug_image_info(image_bytes: bytes) -> Dict[str, Any]:
    image = Image.open(io.BytesIO(image_bytes))
    return {
        "format": image.format,
        "mode": image.mode,
        "size": image.size,
        "bands": image.getbands(),
        "info": image.info
    }

# =============================================================================
# SIMPLE PREPROCESSING FOR TESTING
# =============================================================================

def simple_preprocess(image_bytes: bytes) -> torch.Tensor:
    """Simple preprocessing for testing when full preprocessor fails"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Adaptive resizing for small images
        width, height = image.size
        if width < 224 or height < 224:
            # Upscale small images
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225]),
            ])
        else:
            # Standard transform for normal images
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225]),
            ])
        
        tensor = transform(image).unsqueeze(0)
        return tensor
    except Exception as e:
        logger.error(f"Simple preprocessing failed: {e}")
        raise

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "preprocess_image",
    "simple_preprocess",
    "MedicalImagePreprocessor",
    "ImageQualityAnalyzer",
    "MedicalImageEnhancer",
    "ImageModality",
    "PreprocessingConfig",
    "ensure_rgb_tensor",
    "debug_image_info",
    "get_uploaded_image",
]
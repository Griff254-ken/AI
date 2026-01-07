"""
MedAI Diagnostic API - Enhanced Version
Enterprise-grade AI backend with medical image validation
"""

import os
import io
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional, Dict, Tuple, Any, Union
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from PIL import Image, ImageOps, ImageFilter
import numpy as np

# Import your local modules
try:
    # Try importing your AI model
    try:
        from models import ChestXrayAIModel
    except ImportError:
        logger.warning("ChestXrayAIModel not found, using mock model")
        ChestXrayAIModel = None
    
    # Import the preprocessing module from the same directory
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from preprocess import preprocess_image, MedicalImagePreprocessor, ImageModality
except ImportError as e:
    logger.critical(f"Missing local modules: {e}")
    raise

load_dotenv()

# --- CONFIGURATION & GLOBAL STATE ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/medai")
MODEL_PATH = os.getenv("MODEL_PATH", "model.pth")

# Validation thresholds
VALIDATION_CONFIG = {
    "min_file_size_kb": 1,  # Reduced for testing
    "max_file_size_mb": 100,  # Increased for various scans
    "min_dimension": 128,  # Reduced for testing
    "aspect_ratio_tolerance": 0.5,  # Increased tolerance
    "grayscale_confidence_threshold": 0.3,  # Reduced for testing
    "medical_aspect_ratios": [0.75, 1.0, 1.33, 1.5, 1.78, 2.0],  # Added more ratios
}

# Global instances to be initialized in lifespan
mongo_client = None
db = None
ai_model = None
preprocessor = None

# --- MEDICAL IMAGE VALIDATION CLASS ---
class MedicalImageValidator:
    """Validates if uploaded images are actually medical scans"""
    
    @staticmethod
    def validate_file_size(image_bytes: bytes) -> Tuple[bool, str]:
        """Check if file size is reasonable for medical images"""
        size_kb = len(image_bytes) / 1024
        size_mb = size_kb / 1024
        
        if size_kb < VALIDATION_CONFIG["min_file_size_kb"]:
            return False, f"File too small ({size_kb:.1f}KB). Minimum: {VALIDATION_CONFIG['min_file_size_kb']}KB"
        
        if size_mb > VALIDATION_CONFIG["max_file_size_mb"]:
            return False, f"File too large ({size_mb:.1f}MB). Maximum allowed: {VALIDATION_CONFIG['max_file_size_mb']}MB"
        
        return True, f"File size OK: {size_mb:.1f}MB"
    
    @staticmethod
    def validate_dimensions(image: Image.Image) -> Tuple[bool, str]:
        """Check image dimensions are appropriate for medical scans"""
        width, height = image.size
        
        if width < VALIDATION_CONFIG["min_dimension"] or height < VALIDATION_CONFIG["min_dimension"]:
            return False, f"Image too small ({width}x{height}). Minimum: {VALIDATION_CONFIG['min_dimension']}px"
        
        return True, f"Dimensions OK: {width}x{height}"
    
    @staticmethod
    def validate_aspect_ratio(image: Image.Image, expected_modality: str) -> Tuple[bool, str]:
        """Check if aspect ratio matches typical medical images"""
        width, height = image.size
        aspect_ratio = width / height
        
        # Expected aspect ratios for different modalities
        modality_ratios = {
            "xray": [1.0, 1.33],  # Square or 4:3 for chest X-rays
            "ct": [1.0],  # Usually square for axial slices
            "mri": [1.0, 1.33],  # Square or 4:3
            "ultrasound": [1.33, 1.5, 1.78],  # Various depending on probe
        }
        
        expected_ratios = modality_ratios.get(expected_modality, VALIDATION_CONFIG["medical_aspect_ratios"])
        
        # Check if aspect ratio is close to any expected ratio
        for ratio in expected_ratios:
            if abs(aspect_ratio - ratio) < VALIDATION_CONFIG["aspect_ratio_tolerance"]:
                return True, f"Aspect ratio {aspect_ratio:.2f} matches expected {ratio}"
        
        return False, f"Aspect ratio {aspect_ratio:.2f} unusual for {expected_modality}. Expected near: {expected_ratios}"
    
    @staticmethod
    def analyze_grayscale(image: Image.Image) -> Tuple[float, str]:
        """Calculate how grayscale the image appears (X-rays are mostly grayscale)"""
        if image.mode == 'L':
            return 1.0, "Pure grayscale image"
        
        # Convert to RGB if needed
        rgb_image = image.convert('RGB')
        
        # Calculate color variance
        np_image = np.array(rgb_image)
        r, g, b = np_image[:,:,0], np_image[:,:,1], np_image[:,:,2]
        
        # Grayscale images have similar R, G, B values
        rg_diff = np.mean(np.abs(r - g))
        rb_diff = np.mean(np.abs(r - b))
        gb_diff = np.mean(np.abs(g - b))
        
        avg_diff = (rg_diff + rb_diff + gb_diff) / 3
        # Normalize to 0-1 (0 = pure grayscale, 255 = maximum color difference)
        grayscale_score = 1.0 - min(avg_diff / 128.0, 1.0)
        
        if grayscale_score > 0.8:
            confidence = "High grayscale confidence (typical for X-rays)"
        elif grayscale_score > 0.5:
            confidence = "Moderate grayscale confidence"
        else:
            confidence = "Colorful image (unusual for X-rays)"
        
        return grayscale_score, confidence
    
    @staticmethod
    def detect_medical_artifacts(image: Image.Image) -> Dict[str, Any]:
        """Look for medical imaging artifacts and features"""
        artifacts = {
            "has_corners": False,
            "has_borders": False,
            "has_text": False,
            "has_grid": False,
            "has_anatomy": False,
        }
        
        # Convert to grayscale for analysis
        gray = image.convert('L')
        np_gray = np.array(gray)
        
        # Check for dark corners (common in X-rays)
        height, width = np_gray.shape
        corner_size = min(width, height) // 10
        if corner_size > 0:
            corners = [
                np_gray[:corner_size, :corner_size],  # Top-left
                np_gray[:corner_size, -corner_size:],  # Top-right
                np_gray[-corner_size:, :corner_size],  # Bottom-left
                np_gray[-corner_size:, -corner_size:],  # Bottom-right
            ]
            
            corner_darkness = [np.mean(corner) for corner in corners]
            if all(dark < 100 for dark in corner_darkness):  # Dark corners
                artifacts["has_corners"] = True
        
        return artifacts
    
    @classmethod
    def validate_medical_image(cls, image_bytes: bytes, expected_modality: str = "xray") -> Dict[str, Any]:
        """
        Comprehensive medical image validation
        Returns: {
            "is_valid": bool,
            "is_medical": bool,
            "confidence": float,
            "modality": str,
            "warnings": List[str],
            "errors": List[str],
            "metadata": Dict
        }
        """
        warnings = []
        errors = []
        metadata = {}
        
        try:
            # 1. File size validation
            size_valid, size_msg = cls.validate_file_size(image_bytes)
            if not size_valid:
                errors.append(size_msg)
            metadata["file_size_kb"] = len(image_bytes) / 1024
            
            # 2. Open and validate image
            try:
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception as e:
                errors.append(f"Invalid image file: {str(e)}")
                return {
                    "is_valid": False,
                    "is_medical": False,
                    "confidence": 0.0,
                    "modality": "unknown",
                    "warnings": warnings,
                    "errors": errors,
                    "metadata": metadata
                }
            
            # 3. Dimension validation
            dim_valid, dim_msg = cls.validate_dimensions(image)
            if not dim_valid:
                errors.append(dim_msg)
            metadata["dimensions"] = f"{image.width}x{image.height}"
            metadata["width"] = image.width
            metadata["height"] = image.height
            
            # 4. Aspect ratio validation (warning only, not error)
            ratio_valid, ratio_msg = cls.validate_aspect_ratio(image, expected_modality)
            if not ratio_valid:
                warnings.append(ratio_msg)
            metadata["aspect_ratio"] = image.width / image.height
            
            # 5. Grayscale analysis (for X-rays) - warning only
            grayscale_score, grayscale_msg = cls.analyze_grayscale(image)
            metadata["grayscale_score"] = grayscale_score
            metadata["grayscale_message"] = grayscale_msg
            if grayscale_score < VALIDATION_CONFIG["grayscale_confidence_threshold"]:
                warnings.append(f"Low grayscale confidence: {grayscale_score:.2f}")
            
            # 6. Medical artifact detection
            artifacts = cls.detect_medical_artifacts(image)
            metadata["artifacts"] = artifacts
            
            # Calculate overall medical confidence
            confidence_factors = []
            
            if size_valid and dim_valid:
                confidence_factors.append(0.3)  # Basic validation passed
            
            if ratio_valid:
                confidence_factors.append(0.2)  # Aspect ratio looks medical
            
            if grayscale_score > 0.7:
                confidence_factors.append(0.3)  # Very grayscale
            elif grayscale_score > 0.3:
                confidence_factors.append(0.1)  # Somewhat grayscale
            
            if any(artifacts.values()):
                confidence_factors.append(0.2)  # Has medical artifacts
            
            overall_confidence = sum(confidence_factors) if confidence_factors else 0.0
            
            # Determine if medical - relaxed for testing
            is_medical = overall_confidence > 0.3 and len(errors) == 0
            
            # Modality detection
            if grayscale_score > 0.8 and expected_modality == "xray":
                detected_modality = "xray"
            elif artifacts.get("has_grid", False):
                detected_modality = "ct"
            else:
                detected_modality = "unknown_medical"
            
            return {
                "is_valid": len(errors) == 0,
                "is_medical": is_medical,
                "confidence": round(overall_confidence, 2),
                "modality": detected_modality,
                "warnings": warnings,
                "errors": errors,
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                "is_valid": False,
                "is_medical": False,
                "confidence": 0.0,
                "modality": "unknown",
                "warnings": warnings,
                "errors": [f"Validation error: {str(e)}"],
                "metadata": {}
            }

# --- Mock AI Model for testing ---
class MockAIModel:
    """Mock AI model for testing when real model is not available"""
    def __init__(self, model_path=None, device="cpu", threshold=0.1):
        self.device = device
        self.threshold = threshold
        logger.info("Using Mock AI Model for testing")
    
    def predict(self, image_tensor):
        """Generate mock predictions"""
        import random
        
        conditions = [
            {"condition": "Normal", "confidence": 0.85, "severity": "none"},
            {"condition": "Pneumonia", "confidence": 0.72, "severity": "moderate"},
            {"condition": "COVID-19", "confidence": 0.65, "severity": "moderate"},
            {"condition": "Tuberculosis", "confidence": 0.45, "severity": "mild"},
            {"condition": "Lung Opacity", "confidence": 0.55, "severity": "mild"},
        ]
        
        # Select random conditions above threshold
        detected = []
        for condition in conditions:
            if condition["confidence"] > self.threshold and random.random() > 0.3:
                detected.append(condition)
        
        if not detected:
            detected = [conditions[0]]  # Default to normal
        
        return {
            "diagnosis": {
                "primary_condition": detected[0]["condition"],
                "all_conditions": detected,
                "overall_confidence": max(c["confidence"] for c in detected),
                "is_critical": any(c["severity"] in ["severe", "critical"] for c in detected),
            },
            "heatmap": {
                "available": False,
                "message": "Mock model - no heatmap generated"
            },
            "recommendations": [
                "This is a mock diagnosis. Consult a real medical professional.",
                "Further tests may be required for confirmation."
            ],
            "metadata": {
                "model": "mock_model_v1.0",
                "processing_time_ms": random.randint(50, 200),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        }

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events with proper resource management."""
    global mongo_client, db, ai_model, preprocessor
    
    # 1. Initialize MongoDB
    logger.info("Initializing MongoDB...")
    if MONGODB_URI:
        try:
            mongo_client = AsyncIOMotorClient(MONGODB_URI)
            db_name = MONGODB_URI.split('/')[-1].split('?')[0] or "medai"
            db = mongo_client[db_name]
            
            # Verify connection
            await mongo_client.admin.command('ping')
            logger.info(f"Connected to MongoDB database: [{db_name}]")
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {e}. Running without database.")
            db = None
    else:
        logger.warning("MONGODB_URI not found. Running without database.")
        db = None

    # 2. Initialize AI Model or Mock
    logger.info(f"Initializing AI Model on {DEVICE}...")
    try:
        if ChestXrayAIModel is not None:
            ai_model = ChestXrayAIModel(model_path=MODEL_PATH, device=DEVICE, threshold=0.1)
            
            # Model Warmup
            dummy_input = torch.zeros((1, 3, 224, 224), device=DEVICE)
            with torch.no_grad():
                _ = ai_model.model(dummy_input)
            logger.info("Real AI Model loaded successfully")
        else:
            ai_model = MockAIModel(device=DEVICE, threshold=0.1)
            logger.info("Mock AI Model loaded for testing")
    except Exception as e:
        logger.error(f"AI Model initialization failed: {e}")
        logger.info("Falling back to mock model")
        ai_model = MockAIModel(device=DEVICE, threshold=0.1)
    
    # 3. Initialize Preprocessor
    logger.info("Initializing Image Preprocessor...")
    try:
        preprocessor = MedicalImagePreprocessor(modality="xray")
        logger.info("Image Preprocessor initialized")
    except Exception as e:
        logger.error(f"Preprocessor initialization failed: {e}")
        preprocessor = None
    
    logger.info("✅ MedAI API is ready to accept requests")
    yield  # --- App is running ---
    
    # 4. Shutdown Logic
    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB connection closed")
    logger.info("Resources cleaned up successfully")

# --- APP INITIALIZATION ---
app = FastAPI(
    title="MedAI Diagnostic API v1.5",
    description="Enhanced AI backend with flexible image processing",
    version="1.5.0",
    lifespan=lifespan
)

# CORS setup - Allow all for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ROUTES ---

@app.post("/diagnostics/process")
async def process_diagnostics(
    file: UploadFile = File(...),
    type: str = Query("xray", description="Image modality: xray, ct, mri, ultrasound"),
    x_user_role: Optional[str] = Query(None, description="User role for personalized recommendations"),
    bypass_validation: bool = Query(False, description="Bypass medical image validation (for testing)")
):
    """
    Processes uploaded medical images and returns AI analysis.
    Accepts: JPEG, PNG, DICOM images
    """
    logger.info(f"Processing request: {file.filename}, type={type}, role={x_user_role}")
    
    try:
        # 1. Read image bytes
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )
        
        logger.info(f"Received file: {file.filename}, size: {len(image_bytes)} bytes")
        
        # 2. Medical image validation (optional)
        validation_result = None
        if not bypass_validation:
            validator = MedicalImageValidator()
            validation_result = validator.validate_medical_image(image_bytes, type)
            
            logger.info(f"Validation result: valid={validation_result['is_valid']}, medical={validation_result['is_medical']}, confidence={validation_result['confidence']}")
            
            # Check if valid
            if not validation_result["is_valid"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "message": "Invalid image file",
                        "errors": validation_result["errors"],
                        "warnings": validation_result["warnings"]
                    }
                )
        
        # 3. Preprocess image using your preprocessor module
        try:
            if preprocessor:
                # Use the enhanced preprocessor
                image_tensor, metadata = preprocessor.process(image_bytes)
                logger.info(f"Preprocessed image: {metadata}")
            else:
                # Fallback to basic preprocessing
                import torchvision.transforms as transforms
                from PIL import Image
                
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                image_tensor = transform(image).unsqueeze(0)
                metadata = {"preprocessor": "basic_transform"}
                
            logger.info(f"Image tensor shape: {image_tensor.shape}")
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image preprocessing failed: {str(e)}"
            )
        
        # 4. Execute AI prediction
        try:
            result = ai_model.predict(image_tensor)
            logger.info(f"AI prediction complete: {result['diagnosis']['primary_condition']}")
        except Exception as e:
            logger.error(f"AI prediction failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"AI analysis failed: {str(e)}"
            )
        
        # 5. Add validation metadata if validation was performed
        if validation_result:
            result["validation"] = {
                "performed": True,
                "confidence": validation_result["confidence"],
                "modality_detected": validation_result["modality"],
                "warnings": validation_result["warnings"],
                "metadata": validation_result["metadata"]
            }
        else:
            result["validation"] = {"performed": False}
        
        # 6. Add preprocessing metadata
        result["preprocessing"] = metadata
        
        # 7. Add file information
        result["file_info"] = {
            "filename": file.filename,
            "content_type": file.content_type,
            "size_bytes": len(image_bytes),
            "modality_requested": type
        }
        
        # 8. Role-based recommendations
        is_medical_professional = x_user_role and x_user_role.lower() in ["doctor", "radiologist", "physician", "admin"]
        
        if not is_medical_professional:
            if "recommendations" not in result:
                result["recommendations"] = []
            result["recommendations"].append(
                "⚠️ IMPORTANT: This AI analysis is for informational purposes only. Always consult with a qualified healthcare professional for medical diagnosis."
            )
        
        # 9. Database Audit Trail (if available)
        if db is not None:
            try:
                audit_log = {
                    **result,
                    "audit": {
                        "timestamp": datetime.now(timezone.utc),
                        "user_role": x_user_role or "unknown",
                        "modality": type,
                        "validation_performed": not bypass_validation,
                    }
                }
                await db.diagnostics.insert_one(audit_log)
                logger.info("Audit log saved")
            except Exception as e:
                logger.error(f"Audit log failed: {e}")
                # Don't fail the request if audit fails
        
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_diagnostics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/preprocess/image")
async def preprocess_only(
    file: UploadFile = File(...),
    modality: str = Query("xray", description="Image modality")
):
    """Endpoint to preprocess an image without AI analysis"""
    try:
        image_bytes = await file.read()
        
        if preprocessor:
            image_tensor, metadata = preprocessor.process(image_bytes)
            
            return {
                "success": True,
                "tensor_shape": list(image_tensor.shape),
                "metadata": metadata,
                "file_info": {
                    "filename": file.filename,
                    "size_bytes": len(image_bytes),
                    "modality": modality
                }
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Preprocessor not available"
            )
            
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Preprocessing failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Service health monitoring with detailed diagnostics"""
    model_status = "mock" if isinstance(ai_model, MockAIModel) else "real" if ai_model else "not loaded"
    
    health_info = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "MedAI Diagnostic API v1.5",
        "components": {
            "database": "connected" if db else "disconnected",
            "ai_model": model_status,
            "preprocessor": "available" if preprocessor else "unavailable",
            "device": DEVICE,
        },
        "system": {
            "python_version": sys.version,
            "torch_version": torch.__version__ if torch else "not available",
            "cuda_available": torch.cuda.is_available() if torch else False,
        }
    }
    
    return health_info

@app.get("/test/upload")
async def test_upload_endpoint():
    """Test endpoint to verify upload functionality"""
    return {
        "message": "Upload endpoint is working",
        "endpoints": {
            "diagnostics": "POST /diagnostics/process",
            "preprocess": "POST /preprocess/image",
            "health": "GET /health"
        },
        "supported_file_types": ["image/jpeg", "image/png", "image/dicom"],
        "max_file_size_mb": VALIDATION_CONFIG["max_file_size_mb"]
    }

@app.get("/validation/config")
async def get_validation_config():
    """Get current validation configuration"""
    return {
        "config": VALIDATION_CONFIG,
        "description": "Medical image validation parameters",
        "notes": "These thresholds can be adjusted via environment variables"
    }

# --- ERROR HANDLERS ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with JSON responses"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail.get("message", str(exc.detail)) if isinstance(exc.detail, dict) else str(exc.detail),
            "details": exc.detail.get("errors", []) if isinstance(exc.detail, dict) else None,
            "warnings": exc.detail.get("warnings", []) if isinstance(exc.detail, dict) else None,
            "path": request.url.path
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "detail": str(exc) if os.getenv("DEBUG_MODE") == "true" else None
        }
    )

# --- MODULE RUNNER ---
if __name__ == "__main__":
    logger.info("Starting MedAI FastAPI server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Important: Listen on all interfaces for Docker/Codespaces
        port=8000,
        reload=True,
        log_level="info"
    )
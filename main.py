import os
import io
import sys
import torch
import uvicorn
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from loguru import logger
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional, Dict, Tuple, Any, List
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from PIL import Image

load_dotenv()

# --- CONFIGURATION ---
# Render uses the PORT env var; we must use it or default to 10000
PORT = int(os.getenv("PORT", 10000))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MONGODB_URI = os.getenv("MONGODB_URI")
MODEL_PATH = os.getenv("MODEL_PATH", "model.pth")

VALIDATION_CONFIG = {
    "min_file_size_kb": 1,
    "max_file_size_mb": 10, # Reduced for Render free tier memory safety
    "min_dimension": 128,
    "aspect_ratio_tolerance": 0.5,
    "medical_aspect_ratios": [0.75, 1.0, 1.33, 1.5, 1.78, 2.0],
}

# Global State
mongo_client = None
db = None
ai_model = None

# --- MOCK CLASSES (For Standalone Compatibility) ---
# This ensures the app runs even if your custom local files aren't uploaded yet
class MockAIModel:
    def __init__(self, **kwargs):
        logger.info("Using Mock AI Model")
    
    def predict(self, image_tensor):
        return {
            "diagnosis": {
                "primary_condition": "Normal",
                "all_conditions": [{"condition": "Normal", "confidence": 0.95}],
                "overall_confidence": 0.95,
                "is_critical": False,
            },
            "recommendations": ["Informational only. Consult a doctor."],
            "metadata": {"model": "mock_v1", "timestamp": datetime.now(timezone.utc).isoformat()}
        }

# --- IMAGE VALIDATION LOGIC ---
class MedicalImageValidator:
    @staticmethod
    def validate_medical_image(image_bytes: bytes, expected_modality: str = "xray") -> Dict[str, Any]:
        errors = []
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            width, height = img.size
            
            if width < VALIDATION_CONFIG["min_dimension"]:
                errors.append("Image resolution too low.")
            
            return {
                "is_valid": len(errors) == 0,
                "is_medical": True, # Simplified for standalone
                "confidence": 0.9,
                "modality": expected_modality,
                "errors": errors,
                "warnings": [],
                "metadata": {"width": width, "height": height}
            }
        except Exception as e:
            return {"is_valid": False, "errors": [str(e)]}

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global mongo_client, db, ai_model
    
    # DB Connection
    if MONGODB_URI:
        try:
            mongo_client = AsyncIOMotorClient(MONGODB_URI)
            db = mongo_client.get_default_database()
            await mongo_client.admin.command('ping')
            logger.info("MongoDB Connected")
        except Exception as e:
            logger.error(f"DB Fail: {e}")

    # Model Loading with local fallback
    try:
        from models import ChestXrayAIModel
        ai_model = ChestXrayAIModel(model_path=MODEL_PATH, device=DEVICE)
        logger.info("Real Model Loaded")
    except ImportError:
        ai_model = MockAIModel()

    yield
    if mongo_client:
        mongo_client.close()

# --- APP SETUP ---
app = FastAPI(title="MedAI Standalone API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ROUTES ---
@app.get("/health")
async def health():
    return {"status": "healthy", "device": DEVICE, "db": "up" if db else "down"}

@app.post("/diagnostics/process")
async def process_diagnostics(
    file: UploadFile = File(...),
    type: str = Query("xray")
):
    image_bytes = await file.read()
    
    # 1. Validate
    validator = MedicalImageValidator()
    val = validator.validate_medical_image(image_bytes, type)
    if not val["is_valid"]:
        raise HTTPException(status_code=400, detail=val["errors"])

    # 2. Simple Preprocess (Standard for most AI models)
    try:
        from PIL import Image
        import torchvision.transforms as T
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        tensor = transform(img).unsqueeze(0).to(DEVICE)
    except Exception as e:
        logger.error(f"Preprocess error: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")

    # 3. Predict
    result = ai_model.predict(tensor)
    
    # 4. Log to DB
    if db is not None:
        await db.diagnostics.insert_one({
            "filename": file.filename,
            "prediction": result["diagnosis"]["primary_condition"],
            "timestamp": datetime.now(timezone.utc)
        })

    return result

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)

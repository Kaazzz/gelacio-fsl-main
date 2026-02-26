from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import config
import fsl_inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting GelacioFSL API — loading model...")
    fsl_inference.load_model()
    logger.info("Model ready. Serving on %s:%d", config.HOST, config.PORT)
    yield
    logger.info("Shutting down.")


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GelacioFSL API",
    description="Real-time Filipino Sign Language recognition via LandmarkTransformerV4",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ──────────────────────────────────────────────────────────────────────────────

class LandmarkRequest(BaseModel):
    landmarks: list[list[float]]   # shape (T, F) — T frames × F features
    mask: list[float]              # shape (T,)   — 1=valid, 0=padding


class SignPrediction(BaseModel):
    sign: str
    probability: float


class PredictResponse(BaseModel):
    top_predictions: list[SignPrediction]
    top_sign: str
    top_probability: float
    num_classes: int
    mock: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "message": "GelacioFSL API — Sign Language Recognition",
        "status": "running",
        "version": "2.0.0",
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": fsl_inference.is_loaded(),
        "num_classes": config.NUM_CLASSES,
        "seq_len": config.SEQ_LEN,
        "device": config.DEVICE,
    }


@app.post("/api/predict-landmarks", response_model=PredictResponse)
async def predict_landmarks(req: LandmarkRequest):
    """
    Run sign language inference on a landmark buffer.

    Body:
        landmarks: (T, F) landmark feature array
        mask:      (T,) validity mask (1=valid frame, 0=padding)
    """
    if not req.landmarks or not req.mask:
        raise HTTPException(status_code=400, detail="landmarks and mask must be non-empty")

    T = len(req.landmarks)
    if len(req.mask) != T:
        raise HTTPException(
            status_code=400,
            detail=f"landmarks has {T} frames but mask has {len(req.mask)} entries",
        )

    try:
        result = fsl_inference.predict(req.landmarks, req.mask)
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return PredictResponse(**result)


if __name__ == "__main__":
    print(f"Starting GelacioFSL API on {config.HOST}:{config.PORT}")
    uvicorn.run(app, host=config.HOST, port=config.PORT)

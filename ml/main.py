"""
FastAPI application for Student Placement Prediction.

Loads the trained Logistic Regression model (v3) and exposes:
  GET  /health   – readiness check
  POST /predict  – placement prediction for a single student
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Resolve model path relative to *this* script, not the working directory.
_SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = _SCRIPT_DIR / "models" / "placement_model_v3.joblib"
THRESHOLD = 0.55
POSITIVE_CLASS = "Placed"

# ---------------------------------------------------------------------------
# Application state (populated during lifespan startup)
# ---------------------------------------------------------------------------
_model = None
_placed_index: int | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup; release on shutdown."""
    global _model, _placed_index

    logger.info("Loading model from %s …", MODEL_PATH)
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Make sure placement_model_v3.joblib is in ml/models/."
        )

    try:
        _model = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model: {exc}") from exc

    # Locate the "Placed" class index dynamically using model.classes_
    classes: List[str] = list(_model.classes_)
    if POSITIVE_CLASS not in classes:
        raise RuntimeError(
            f"Expected class '{POSITIVE_CLASS}' not found in model.classes_: {classes}"
        )
    _placed_index = classes.index(POSITIVE_CLASS)
    logger.info(
        "Model loaded. classes_=%s, placed_index=%d, threshold=%.2f",
        classes, _placed_index, THRESHOLD,
    )

    yield  # ---- application is running ----

    # Cleanup (nothing to release for a joblib model, but kept for clarity)
    _model = None
    _placed_index = None
    logger.info("Model released.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Student Placement Prediction API",
    version="3.0.0",
    description="Predicts student placement probability using a Logistic Regression model (v3).",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class StudentFeatures(BaseModel):
    """All 14 input features expected by the model pipeline."""

    age: int = Field(..., ge=16, le=60, description="Student age")
    gender: str = Field(..., description="Gender (e.g. Male, Female)")
    cgpa: float = Field(..., ge=0.0, le=10.0, description="CGPA on a 10-point scale")
    branch: str = Field(..., description="Branch / department (e.g. CSE, ECE, ME, CE, EE)")
    college_tier: int = Field(..., ge=1, le=3, description="College tier (1, 2, or 3)")
    internships_count: int = Field(..., ge=0, description="Number of internships completed")
    projects_count: int = Field(..., ge=0, description="Number of projects completed")
    certifications_count: int = Field(..., ge=0, description="Number of certifications earned")
    coding_skill_score: int = Field(..., ge=0, le=100, description="Coding skill score (0–100)")
    communication_skill_score: int = Field(..., ge=0, le=100, description="Communication skill score (0–100)")
    aptitude_score: int = Field(..., ge=0, le=100, description="Aptitude test score (0–100)")
    logical_reasoning_score: int = Field(..., ge=0, le=100, description="Logical reasoning score (0–100)")
    mock_interview_score: int = Field(..., ge=0, le=100, description="Mock interview score (0–100)")
    backlogs: int = Field(..., ge=0, description="Number of backlogs")

    model_config = {"json_schema_extra": {
        "examples": [{
            "age": 22,
            "gender": "Male",
            "cgpa": 8.0,
            "branch": "CSE",
            "college_tier": 1,
            "internships_count": 2,
            "projects_count": 3,
            "certifications_count": 3,
            "coding_skill_score": 80,
            "communication_skill_score": 75,
            "aptitude_score": 78,
            "logical_reasoning_score": 76,
            "mock_interview_score": 80,
            "backlogs": 0,
        }]
    }}


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted class: 'Placed' or 'Not Placed'")
    placement_probability: float = Field(..., description="Probability of being Placed (0.0–1.0)")
    threshold: float = Field(..., description="Decision threshold used")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    threshold: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Check that the service and model are ready."""
    return HealthResponse(
        status="ok" if _model is not None else "unavailable",
        model_loaded=_model is not None,
        model_version="v3",
        threshold=THRESHOLD,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(student: StudentFeatures):
    """Return placement prediction for a single student."""
    if _model is None or _placed_index is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check /health.")

    # Build a single-row DataFrame in the exact feature order the pipeline expects.
    row = {
        "age": student.age,
        "gender": student.gender,
        "cgpa": student.cgpa,
        "branch": student.branch,
        "college_tier": student.college_tier,
        "internships_count": student.internships_count,
        "projects_count": student.projects_count,
        "certifications_count": student.certifications_count,
        "coding_skill_score": student.coding_skill_score,
        "communication_skill_score": student.communication_skill_score,
        "aptitude_score": student.aptitude_score,
        "logical_reasoning_score": student.logical_reasoning_score,
        "mock_interview_score": student.mock_interview_score,
        "backlogs": student.backlogs,
    }
    df = pd.DataFrame([row])

    try:
        probabilities = _model.predict_proba(df)[0]
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    placed_prob = float(probabilities[_placed_index])
    prediction = POSITIVE_CLASS if placed_prob >= THRESHOLD else "Not Placed"

    return PredictionResponse(
        prediction=prediction,
        placement_probability=round(placed_prob, 4),
        threshold=THRESHOLD,
    )

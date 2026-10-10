
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from backend.core.config import (
    ALLOWED_ORIGINS,
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    SPACY_MODEL_PRIMARY,
    SPACY_MODEL_SECONDARY,
    SENTENCE_TRANSFORMER_MODEL,
)
from backend.api.routes import router

logger = logging.getLogger("ats_resume_scorer")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ATS Resume Analyzer API...")

    # Load spaCy NLP model
    logger.info(f"Loading spaCy NLP model: {SPACY_MODEL_PRIMARY}")

    import spacy

    try:
        app.state.nlp = spacy.load(SPACY_MODEL_PRIMARY)
        logger.info(f"Loaded {SPACY_MODEL_PRIMARY}")

    except OSError:
        logger.warning(
            f"{SPACY_MODEL_PRIMARY} not found. "
            f"Falling back to {SPACY_MODEL_SECONDARY}"
        )
        app.state.nlp = spacy.load(SPACY_MODEL_SECONDARY)
        logger.info(f"Loaded {SPACY_MODEL_SECONDARY} (fallback)")

    # Download the Hugging Face repository
    logger.info(
        f"Downloading model repository: {SENTENCE_TRANSFORMER_MODEL}"
    )

    model_dir = snapshot_download(
        repo_id=SENTENCE_TRANSFORMER_MODEL,
        allow_patterns=[
            "ats_finetuned_model/**",
        ],
    )

    # Locate the actual SentenceTransformer model
    fine_tuned_model_path = Path(model_dir) / "ats_finetuned_model"

    if not fine_tuned_model_path.is_dir():
        raise FileNotFoundError(
            f"Fine-tuned model folder not found: {fine_tuned_model_path}. "
            "Check the folder name and files in your Hugging Face repository."
        )

    logger.info(
        f"Loading SentenceTransformer from: {fine_tuned_model_path}"
    )

    app.state.embedder = SentenceTransformer(
        str(fine_tuned_model_path)
    )

    logger.info("Fine-tuned SentenceTransformer loaded successfully.")
    logger.info("All models loaded. API is ready to serve requests.")

    yield

    logger.info("Shutting down the API...")


app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "name": "ATS Resume Analyzer API",
        "version": "2.0.0",
        "endpoints": {
            "POST /api/v1/analyze-resume": "Analyze a resume",
            "GET /api/v1/history": "Get user history",
            "DELETE /api/v1/history/:id": "Delete a history entry",
            "GET /api/v1/health": "Health check",
            "POST /api/v1/generate-pdf": "Generate PDF report from data",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

from pathlib import Path
import logging
import joblib
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from sentence_transformers import SentenceTransformer

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("headline_api")

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI()

# -------------------------
# Model configuration
# -------------------------
MODEL_FILE = Path("svm.joblib")
LOCAL_EMBEDDER = Path("/opt/huggingface_models/all-MiniLM-L6-v2")
FALLBACK_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"

clf = None
embedder = None


# -------------------------
# Request/Response models
# -------------------------
class ScoreRequest(BaseModel):
    headlines: conlist(str, min_length=1)


class ScoreResponse(BaseModel):
    labels: List[str]


# -------------------------
# Load models once
# -------------------------
@app.on_event("startup")
def load_models():
    global clf, embedder

    try:
        logger.info("Loading classifier...")
        clf = joblib.load(MODEL_FILE)

        embedder_id = (
            str(LOCAL_EMBEDDER)
            if LOCAL_EMBEDDER.exists()
            else FALLBACK_EMBEDDER
        )

        logger.info("Loading sentence transformer...")
        embedder = SentenceTransformer(embedder_id)

        logger.info("Models loaded successfully.")

    except Exception as e:
        logger.critical("Model loading failed", exc_info=True)
        raise e


# -------------------------
# Endpoints
# -------------------------
@app.get("/status")
def status():
    return {"status": "OK"}


@app.post("/score_headlines", response_model=ScoreResponse)
def score_headlines(req: ScoreRequest):
    if clf is None or embedder is None:
        logger.error("Models not loaded")
        raise HTTPException(status_code=500, detail="Model not ready")

    headlines = req.headlines
    logger.info("Scoring %d headlines", len(headlines))

    try:
        embeddings = embedder.encode(
            headlines,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        labels = clf.predict(embeddings)

        return ScoreResponse(labels=list(labels))

    except Exception:
        logger.error("Scoring failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Scoring failed")

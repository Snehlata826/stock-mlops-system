"""
FastAPI REST inference endpoint — Production v2.1
Run: uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import os, time, uuid, traceback
from typing import List, Optional
from contextlib import asynccontextmanager
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.common.config import SUPPORTED_ASSETS, PROCESSED_DATA_DIR, MODEL_NAME
from src.inference.model_loader import ModelLoader
from src.inference.predict import predict

# ── Logging setup ─────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stderr,
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | <level>{message}</level>"
)
Path("logs").mkdir(exist_ok=True)
logger.add("logs/api.log", rotation="50 MB", retention="14 days", level="INFO")

# ── Model loader (initialised at startup) ─────────────────────
_loader: Optional[ModelLoader] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _loader
    logger.info("🚀 Stock MLOps API starting...")
    _loader = ModelLoader()
    logger.info(f"✅ Ready — supported assets: {SUPPORTED_ASSETS}")
    yield
    logger.info("🛑 API shutting down.")

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Stock MLOps Prediction API",
    description="XGBoost stock direction prediction with MLflow model registry",
    version="2.1.0",
    lifespan=lifespan,
)

# ── CORS — reads from env, defaults cover localhost + Docker ──
_origins = [
    o.strip() for o in
    os.getenv("CORS_ORIGINS", "http://localhost:8501,http://frontend:8501,http://127.0.0.1:8501")
    .split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request timing ────────────────────────────────────────────
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    rid = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()
    response: Response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = rid
    response.headers["X-Process-Time-Ms"] = f"{ms:.1f}"
    logger.info(f"[{rid}] {request.method} {request.url.path} → {response.status_code} ({ms:.0f}ms)")
    return response

# ── Global error handler ──────────────────────────────────────
@app.exception_handler(Exception)
async def global_error(request: Request, exc: Exception):
    logger.error(f"Unhandled: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__}
    )

# ── Schemas ───────────────────────────────────────────────────
class PredictRequest(BaseModel):
    ticker: str
    top_n: Optional[int] = 10

    @validator("ticker")
    def to_upper(cls, v):
        return v.strip().upper()

class PredictionRow(BaseModel):
    date: str
    direction: str
    prob_up: float
    prob_down: float
    confidence: float

class PredictResponse(BaseModel):
    ticker: str
    model: str
    n_predictions: int
    predictions: List[PredictionRow]
    summary: dict

class HealthResponse(BaseModel):
    status: str
    supported_assets: List[str]
    version: str

# ── Routes ────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse)
def root():
    return HealthResponse(status="ok", supported_assets=SUPPORTED_ASSETS, version="2.1.0")

@app.get("/health")
def health():
    """Liveness probe — Docker healthcheck hits this."""
    return {"status": "ok"}

@app.get("/readyz")
def readiness():
    """Readiness probe — only 200 when model loader is initialised."""
    if _loader is None:
        raise HTTPException(status_code=503, detail="Model loader not ready")
    return {"status": "ready"}

@app.get("/assets")
def list_assets():
    return {"supported_assets": SUPPORTED_ASSETS}

@app.get("/assets/{ticker}/status")
def asset_status(ticker: str):
    """Used by Streamlit to check if inference data exists before calling /predict."""
    ticker = ticker.upper()
    if ticker not in SUPPORTED_ASSETS:
        raise HTTPException(status_code=400, detail=f"Unsupported ticker '{ticker}'")
    return {
        "ticker": ticker,
        "inference_ready":      (PROCESSED_DATA_DIR / f"features_inference_{ticker}.csv").exists(),
        "training_data_exists": (PROCESSED_DATA_DIR / f"features_train_{ticker}.csv").exists(),
    }

@app.get("/model/{ticker}/info")
def model_info(ticker: str):
    ticker = ticker.upper()
    if ticker not in SUPPORTED_ASSETS:
        raise HTTPException(status_code=400, detail=f"Unsupported ticker '{ticker}'")
    try:
        name    = f"{MODEL_NAME}_{ticker}"
        model   = _loader.get_model(name)
        version = _loader.model_versions.get(name, "unknown")
        return {"ticker": ticker, "model_name": name, "version": version, "n_features": model.n_features_in_}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Model not found: {exc}")

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    ticker = req.ticker
    if ticker not in SUPPORTED_ASSETS:
        raise HTTPException(status_code=400,
            detail=f"Unsupported ticker '{ticker}'. Supported: {SUPPORTED_ASSETS}")

    if not (PROCESSED_DATA_DIR / f"features_inference_{ticker}.csv").exists():
        raise HTTPException(status_code=404,
            detail=f"No inference features for '{ticker}'. Click 'Run Market Analysis' first.")

    try:
        df = predict(ticker)
    except Exception as exc:
        logger.error(f"Prediction failed for {ticker}: {exc}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(exc))

    rows = [
        PredictionRow(
            date=str(r["Date"]), direction=r["direction"],
            prob_up=float(r["prob_up"]), prob_down=float(r["prob_down"]),
            confidence=float(r["confidence"])
        )
        for _, r in df.tail(req.top_n).iterrows()
    ]

    return PredictResponse(
        ticker=ticker,
        model=f"{MODEL_NAME}_{ticker}",
        n_predictions=len(df),
        predictions=rows,
        summary={
            "bias":           "Bullish" if df["prob_up"].mean() > 0.5 else "Bearish",
            "avg_prob_up":    float(df["prob_up"].mean()),
            "avg_confidence": float(df["confidence"].mean()),
            "up_signals":     int((df["direction"] == "UP").sum()),
            "down_signals":   int((df["direction"] == "DOWN").sum()),
        },
    )
    
@app.get("/assets/{ticker}/price")
def get_price_data(ticker: str, interval: str = "15min"):
    """Return OHLCV price data for candlestick chart."""
    ticker = ticker.upper()
    if ticker not in SUPPORTED_ASSETS:
        raise HTTPException(status_code=400, detail=f"Unsupported ticker '{ticker}'")
    try:
        from src.ingestion.fetch_realtime import fetch_realtime_data
        df = fetch_realtime_data(ticker=ticker, interval=interval, force_refresh=False)
        if df.empty:
            # Fall back to historical data
            raw_path = Path(f"data/raw/historical_{ticker}.csv")
            if raw_path.exists():
                import pandas as pd
                df = pd.read_csv(raw_path).tail(100)
            else:
                raise HTTPException(status_code=404, detail=f"No price data for {ticker}")
        # Convert to JSON-safe format
        df["Date"] = df["Date"].astype(str)
        return {
            "ticker": ticker,
            "interval": interval,
            "n_rows": len(df),
            "data": df[["Date","Open","High","Low","Close","Volume"]].to_dict(orient="records")
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/assets/{ticker}/run_pipeline")
def run_pipeline(ticker: str):
    """Fetch realtime data + engineer features. Call before /predict."""
    ticker = ticker.upper()
    if ticker not in SUPPORTED_ASSETS:
        raise HTTPException(status_code=400, detail=f"Unsupported ticker '{ticker}'")
    try:
        from src.ingestion.fetch_realtime import fetch_realtime_data
        from src.features.feature_engineering import engineer_features
        df = fetch_realtime_data(ticker=ticker, interval="15min", force_refresh=True)
        if df.empty:
            import shutil
            src = Path(f"data/raw/historical_{ticker}.csv")
            dst = Path(f"data/raw/realtime_{ticker}.csv")
            if src.exists():
                shutil.copy(src, dst)
        engineer_features(
            Path(f"data/raw/realtime_{ticker}.csv"),
            Path(f"data/processed/features_inference_{ticker}.csv"),
            is_training=False
        )
        return {"status": "ok", "ticker": ticker, "message": "Pipeline complete"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
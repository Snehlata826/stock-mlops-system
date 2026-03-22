"""
FastAPI REST inference endpoint.
Serves model predictions via HTTP — production-grade serving layer.

Run: uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.common.config import SUPPORTED_ASSETS, PROCESSED_DATA_DIR, MODEL_NAME
from src.common.utils import logger
from src.inference.model_loader import ModelLoader
from src.inference.predict import predict

app = FastAPI(
    title="Stock MLOps Prediction API",
    description="XGBoost-based stock direction prediction with MLflow model registry",
    version="2.0.0",
)

_loader = ModelLoader()


class PredictRequest(BaseModel):
    ticker: str
    top_n: Optional[int] = 10


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


@app.get("/", response_model=HealthResponse)
def root():
    return HealthResponse(
        status="ok",
        supported_assets=SUPPORTED_ASSETS,
        version="2.0.0",
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    ticker = req.ticker.upper()

    if ticker not in SUPPORTED_ASSETS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported ticker '{ticker}'. Supported: {SUPPORTED_ASSETS}"
        )

    data_path = PROCESSED_DATA_DIR / f"features_inference_{ticker}.csv"
    if not data_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No inference features found for {ticker}. Run the pipeline first."
        )

    try:
        df = predict(ticker)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    df_tail = df.tail(req.top_n)
    rows = []
    for _, row in df_tail.iterrows():
        rows.append(PredictionRow(
            date=str(row["Date"]),
            direction=row["direction"],
            prob_up=float(row["prob_up"]),
            prob_down=float(row["prob_down"]),
            confidence=float(row["confidence"]),
        ))

    up_count = int((df["direction"] == "UP").sum())
    down_count = int((df["direction"] == "DOWN").sum())
    avg_confidence = float(df["confidence"].mean())
    bias = "Bullish" if df["prob_up"].mean() > 0.5 else "Bearish"

    return PredictResponse(
        ticker=ticker,
        model=f"{MODEL_NAME}_{ticker}",
        n_predictions=len(df),
        predictions=rows,
        summary={
            "bias": bias,
            "avg_prob_up": float(df["prob_up"].mean()),
            "avg_confidence": avg_confidence,
            "up_signals": up_count,
            "down_signals": down_count,
        },
    )


@app.get("/assets")
def list_assets():
    return {"supported_assets": SUPPORTED_ASSETS}


@app.get("/model/{ticker}/info")
def model_info(ticker: str):
    ticker = ticker.upper()
    if ticker not in SUPPORTED_ASSETS:
        raise HTTPException(status_code=400, detail=f"Unsupported ticker '{ticker}'")
    try:
        model_name = f"{MODEL_NAME}_{ticker}"
        model = _loader.get_model(model_name)
        version = _loader.model_versions.get(model_name, "unknown")
        return {
            "ticker": ticker,
            "model_name": model_name,
            "version": version,
            "n_features": model.n_features_in_,
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model not found: {e}")

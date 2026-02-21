import streamlit as st
from pathlib import Path
import sys
import pandas as pd

# -------------------------------------------------------
# Path setup
# -------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

REPORTS_DIR = ROOT_DIR / "reports"

# -------------------------------------------------------
# Backend imports
# -------------------------------------------------------
from src.ingestion.fetch_realtime import fetch_realtime_data
from src.features.feature_engineering import engineer_features
from src.inference.predict import predict
from src.monitoring.drift_monitor import monitor_drift
from src.common.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# -------------------------------------------------------
# UI Components
# -------------------------------------------------------
from components.sidebar import render_sidebar
from components.header import render_header
from components.candles import render_candles
from components.bias_cards import render_bias_cards
from components.probability_chart import render_probability
from components.model_health import render_model_health

# -------------------------------------------------------
# Page config
# -------------------------------------------------------
st.set_page_config(
    page_title="Stock Prediction System",
    page_icon="📈",
    layout="wide",
)

# -------------------------------------------------------
# Session state init
# -------------------------------------------------------
st.session_state.setdefault("price_data", None)
st.session_state.setdefault("predictions", None)
st.session_state.setdefault("drift", None)

# -------------------------------------------------------
# Sidebar (controls)
# -------------------------------------------------------
ticker, interval_label, horizon_label, run_analysis = render_sidebar()

INTERVAL_MAP = {
    "5m": "5min",
    "15m": "15min",
    "1h": "60min"
}

# Safe interval mapping
api_interval = None
if interval_label in INTERVAL_MAP:
    api_interval = INTERVAL_MAP[interval_label]

# -------------------------------------------------------
# Header
# -------------------------------------------------------
render_header(
    ticker=ticker if ticker else "Not Selected",
    interval=interval_label,
    horizon=horizon_label
)

# -------------------------------------------------------
# Instructions
# -------------------------------------------------------
with st.expander("ℹ️ How to use this dashboard", expanded=True):
    st.markdown("""
    **Step 1:** Select Asset, Candle Interval, and Forecast Horizon  
    **Step 2:** Click **Run Market Analysis** from the sidebar  

    This system will:
    - Fetch latest market data
    - Engineer technical indicators
    - Predict directional market bias
    - Enable model health & drift checks

    ⚠️ Predictions are probabilistic — not price targets.
    """)

# -------------------------------------------------------
# Run pipeline (ONLY via sidebar button)
# -------------------------------------------------------
if run_analysis and ticker and api_interval and horizon_label not in ["Select Horizon"]:

    # Reset previous results before new run
    st.session_state.price_data = None
    st.session_state.predictions = None
    st.session_state.drift = None

    with st.spinner("Fetching real-time market data..."):
        st.session_state.price_data = fetch_realtime_data(
            ticker=ticker,
            interval=api_interval,
            force_refresh=True
        )

    with st.spinner("Engineering features..."):
        engineer_features(
            RAW_DATA_DIR / f"realtime_{ticker}.csv",
            PROCESSED_DATA_DIR / f"features_inference_{ticker}.csv",
            is_training=False
        )

    with st.spinner("Running model inference..."):
        st.session_state.predictions = predict(ticker=ticker)

    st.success("✅ Market analysis complete")

elif run_analysis:
    st.sidebar.error("⚠️ Please select Asset, Interval, and Horizon before running analysis.")

# -------------------------------------------------------
# Price Action (Candles)
# -------------------------------------------------------
if st.session_state.price_data is not None:
    st.markdown("---")
    st.markdown("## 📊 Price Action")
    render_candles(
        st.session_state.price_data,
        ticker=ticker
    )

# -------------------------------------------------------
# Helper: build horizon-aware summary
# -------------------------------------------------------
def build_summary(predictions, interval, horizon):
    WINDOWS = {
        "5m": {"Next Interval": 1, "1 Day": 78, "1 Month": 1560, "3 Months": 4680},
        "15m": {"Next Interval": 1, "1 Day": 26, "1 Month": 520, "3 Months": 1560},
        "1h": {"Next Interval": 1, "1 Day": 7, "1 Month": 160, "3 Months": 480},
    }

    window = min(WINDOWS[interval][horizon], len(predictions))
    recent = predictions.tail(window)

    prob_up = recent["prob_up"].mean()
    prob_down = recent["prob_down"].mean()

    return {
        "direction": "UP" if prob_up > prob_down else "DOWN",
        "confidence": max(prob_up, prob_down),
        "window": window,
        "horizon": horizon
    }

# -------------------------------------------------------
# Model Outlook
# -------------------------------------------------------
if st.session_state.predictions is not None:
    st.markdown("---")
    st.markdown("## 🔮 Model Outlook")

    if interval_label in ["5m", "15m", "1h"] and horizon_label not in ["Select Horizon"]:
        summary = build_summary(
            st.session_state.predictions,
            interval_label,
            horizon_label
        )

        render_bias_cards(summary)

        st.caption(
            f"Candle interval ({interval_label}) defines data granularity. "
            f"Forecast horizon ({horizon_label}) aggregates multiple candles "
            f"to estimate directional bias."
        )

        st.markdown("### 📈 Model Conviction Over Time")
        render_probability(st.session_state.predictions)

# -------------------------------------------------------
# Model Health
# -------------------------------------------------------
if st.session_state.predictions is not None:
    st.markdown("---")
    st.markdown("## 🩺 Model Health")

    if st.button("Check Model Health", use_container_width=True):
        with st.spinner("Analyzing data drift..."):
            st.session_state.drift = monitor_drift(ticker=ticker)

    if st.session_state.drift is not None:
        render_model_health(st.session_state.drift)

# -------------------------------------------------------
# Drift Report (HTML – Evidently)
# -------------------------------------------------------
if st.session_state.predictions is not None:
    st.markdown("---")
    st.markdown("## 📊 Data Drift Report")

    drift_report_path = REPORTS_DIR / f"drift_report_{ticker}.html"

    if st.button("📉 View Detailed Drift Report", use_container_width=True):
        if drift_report_path.exists():
            with open(drift_report_path, "r", encoding="utf-8") as f:
                st.components.v1.html(
                    f.read(),
                    height=900,
                    scrolling=True
                )
        else:
            st.warning(
                f"No drift report found for {ticker}. "
                "Run Model Health check first."
            )

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.markdown("---")
st.caption(
    "MLOps Stock Prediction System | Multi-horizon (≤ 3 months) | "
    "XGBoost + MLflow | Educational use only"
)
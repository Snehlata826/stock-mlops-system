import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingestion.fetch_realtime import fetch_realtime_data
from src.features.feature_engineering import engineer_features
from src.inference.predict import predict
from src.monitoring.drift_monitor import monitor_drift
from src.common.config import (
    DEFAULT_TICKER, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    BASE_DIR
)

# -------------------------------------------------------
# Page config
# -------------------------------------------------------
st.set_page_config(
    page_title="Stock Prediction MLOps",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------------------------
# Title
# -------------------------------------------------------
st.title("📈 Stock Price Prediction System")
st.markdown("*Local MLOps system with real-time predictions*")

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------
st.sidebar.header("Configuration")

ticker = st.sidebar.text_input("Stock Ticker", value=DEFAULT_TICKER)

interval_label = st.sidebar.selectbox(
    "Interval",
    ["5m", "15m", "1h"],
    index=0
)

# UI → API interval mapping
INTERVAL_MAP = {
    "5m": "5min",
    "15m": "15min",
    "1h": "60min"
}
api_interval = INTERVAL_MAP[interval_label]

# -------------------------------------------------------
# Detect interval change
# -------------------------------------------------------
if "prev_interval" not in st.session_state:
    st.session_state.prev_interval = api_interval

interval_changed = st.session_state.prev_interval != api_interval
st.session_state.prev_interval = api_interval

# -------------------------------------------------------
# Main actions
# -------------------------------------------------------
col1, col2, col3 = st.columns(3)

# ---------------- Fetch Real-time ----------------
with col1:
    if st.button("🔄 Fetch Real-time Data", use_container_width=True) or interval_changed:
        with st.spinner("Fetching data..."):
            try:
                df = fetch_realtime_data(
                    ticker=ticker,
                    interval=api_interval,
                    force_refresh=interval_changed
                )
                st.success(f"✓ Data ready ({len(df)} records)")
                st.session_state["data_fetched"] = True
            except Exception as e:
                st.error(f"Error: {e}")

# ---------------- Feature Engineering ----------------
with col2:
    if st.button("⚙️ Engineer Features", use_container_width=True):
        with st.spinner("Engineering features..."):
            try:
                input_path = RAW_DATA_DIR / "realtime_prices.csv"
                output_path = PROCESSED_DATA_DIR / "features_inference.csv"

                df = engineer_features(
                    input_path,
                    output_path,
                    is_training=False
                )

                st.success(f"✓ Created {len(df.columns)} features")
                st.session_state["features_ready"] = True
            except Exception as e:
                st.error(f"Error: {e}")

# ---------------- Prediction ----------------
with col3:
    if st.button("🎯 Make Predictions", use_container_width=True):
        with st.spinner("Predicting..."):
            try:
                results = predict()
                st.session_state["predictions"] = results
                st.success(f"✓ Generated {len(results)} predictions")
            except Exception as e:
                st.error(f"Error: {e}")

# -------------------------------------------------------
# Display predictions
# -------------------------------------------------------
if "predictions" in st.session_state:
    st.markdown("---")
    st.header("Prediction Results")

    predictions = st.session_state["predictions"]
    latest = predictions.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Prediction", latest["direction"])
    c2.metric("Confidence", f"{latest['confidence']:.2%}")
    c3.metric("Prob UP", f"{latest['prob_up']:.2%}")
    c4.metric("Prob DOWN", f"{latest['prob_down']:.2%}")

    st.subheader("Prediction Timeline")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=predictions["Date"],
        y=predictions["prob_up"],
        mode="lines+markers",
        name="Probability UP",
        line=dict(color="green")
    ))
    fig.add_trace(go.Scatter(
        x=predictions["Date"],
        y=predictions["prob_down"],
        mode="lines+markers",
        name="Probability DOWN",
        line=dict(color="red")
    ))

    fig.update_layout(
        height=400,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Probability"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detailed Predictions")
    st.dataframe(
        predictions[
            ["Date", "direction", "prob_up", "prob_down", "confidence"]
        ].tail(20),
        use_container_width=True
    )

# -------------------------------------------------------
# Drift monitoring
# -------------------------------------------------------
st.markdown("---")
st.header("Drift Monitoring")

if st.button("📊 Check Data Drift", use_container_width=True):
    with st.spinner("Analyzing drift..."):
        try:
            drift_results = monitor_drift()

            d1, d2, d3 = st.columns(3)
            d1.metric("Drifted Columns", drift_results["n_drifted_columns"])
            d2.metric("Total Columns", drift_results["total_columns"])
            d3.metric("Drift %", f"{drift_results['drift_percentage']:.1f}%")

            report_path = BASE_DIR / "reports" / "drift_report.html"
            if report_path.exists():
                st.success(f"✓ Drift report generated: {report_path}")
                st.info("Open the report in your browser for detailed analysis")

            if drift_results["drift_percentage"] > 30:
                st.warning("⚠️ Significant drift detected! Retraining recommended.")
            else:
                st.success("✓ No significant drift detected")

        except Exception as e:
            st.error(f"Error: {e}")

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.markdown("---")
st.markdown(
    "**MLOps Stock Prediction System** | Powered by XGBoost + MLflow + Evidently"
)

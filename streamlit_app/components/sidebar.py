import streamlit as st
from src.common.companies import COMPANIES

def render_sidebar():
    st.sidebar.header("📊 Market Configuration")

    # Asset selection
    asset = st.sidebar.selectbox(
        "Asset",
        list(COMPANIES.keys())
    )
    ticker = COMPANIES[asset]

    # Candle interval
    interval = st.sidebar.selectbox(
        "Candle Interval",
        ["5m", "15m", "1h"]
    )

    # Forecast horizon
    horizon = st.sidebar.selectbox(
        "Forecast Horizon",
        ["Next Interval", "1 Day", "1 Month", "3 Months"]
    )

    # Divider (SIDEBAR ONLY)
    st.sidebar.markdown("---")

    # Run analysis button (SIDEBAR ONLY)
    run_analysis = st.sidebar.button(
        "🚀 Run Market Analysis",
        use_container_width=True
    )

    # Footer info
    st.sidebar.caption(f"Ticker: **{ticker}**")
    st.sidebar.caption("Predictions are probabilistic, not financial advice.")

    return ticker, interval, horizon, run_analysis

import streamlit as st
from src.common.companies import COMPANIES

def render_sidebar():
    st.sidebar.header("📊 Market Configuration")

    # Asset selection (Default: Not Selected)
    asset_options = ["Select Asset"] + list(COMPANIES.keys())
    asset = st.sidebar.selectbox(
        "Asset",
        asset_options,
        index=0
    )

    ticker = None
    if asset != "Select Asset":
        ticker = COMPANIES[asset]

    # Candle interval (Default: Not Selected)
    interval_options = ["Select Interval", "5m", "15m", "1h"]
    interval = st.sidebar.selectbox(
        "Candle Interval",
        interval_options,
        index=0
    )

    # Forecast horizon (Default: Not Selected)
    horizon_options = ["Select Horizon", "Next Interval", "1 Day", "1 Month", "3 Months"]
    horizon = st.sidebar.selectbox(
        "Forecast Horizon",
        horizon_options,
        index=0
    )

    # Divider
    st.sidebar.markdown("---")

    # Run analysis button
    run_analysis = st.sidebar.button(
        "🚀 Run Market Analysis",
        use_container_width=True
    )

    # Footer info
    if ticker:
        st.sidebar.caption(f"Ticker: **{ticker}**")
    st.sidebar.caption("Predictions are probabilistic, not financial advice.")

    # Validation check
    if run_analysis:
        if asset == "Select Asset" or interval == "Select Interval" or horizon == "Select Horizon":
            st.sidebar.error("⚠️ Please select all options before running analysis.")
            run_analysis = False

    return ticker, interval, horizon, run_analysis
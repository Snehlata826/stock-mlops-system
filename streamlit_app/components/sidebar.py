import streamlit as st
from src.common.companies import COMPANIES


def render_sidebar():
    st.sidebar.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid rgba(255,255,255,0.07);
    }
    .sidebar-section {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 14px;
    }
    .sidebar-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: rgba(255,255,255,0.35);
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("## ⚙️ Configuration")

    st.sidebar.markdown('<div class="sidebar-label">Asset</div>', unsafe_allow_html=True)
    asset = st.sidebar.selectbox("Asset", list(COMPANIES.keys()), label_visibility="collapsed")
    ticker = COMPANIES[asset]

    st.sidebar.markdown('<div class="sidebar-label">Candle Interval</div>', unsafe_allow_html=True)
    interval = st.sidebar.selectbox("Interval", ["5m", "15m", "1h"], label_visibility="collapsed")

    st.sidebar.markdown('<div class="sidebar-label">Forecast Horizon</div>', unsafe_allow_html=True)
    horizon = st.sidebar.selectbox(
        "Horizon",
        ["Next Interval", "1 Day", "1 Month", "3 Months"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    run_analysis = st.sidebar.button("🚀 Run Market Analysis", use_container_width=True, type="primary")

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Ticker:** `{ticker}`")
    st.sidebar.caption("Predictions are probabilistic — not financial advice.")

    return ticker, interval, horizon, run_analysis

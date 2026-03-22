"""
Stock MLOps Dashboard v2.1 — Full cloud support
All features work via backend API on Streamlit Cloud.
"""
import os
import streamlit as st
from pathlib import Path
import sys
import pandas as pd

# ── Detect environment ────────────────────────────────────────
_IS_CLOUD = os.path.exists("/mount/src")

# ── Path setup ────────────────────────────────────────────────
ROOT_DIR = Path("/mount/src/stock-mlops-system") if _IS_CLOUD else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Stock MLOps",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)

# ── API client ────────────────────────────────────────────────
try:
    from api_client import get_client, APIError
except ImportError:
    from streamlit_app.api_client import get_client, APIError

# ── Design system ─────────────────────────────────────────────
try:
    from components.theme import inject_theme
except ImportError:
    from streamlit_app.components.theme import inject_theme
inject_theme()

# ── UI components ─────────────────────────────────────────────
try:
    from components.sidebar import render_sidebar
    from components.header import render_header
    from components.candles import render_candles
    from components.bias_cards import render_bias_cards
    from components.probability_chart import render_probability
    from components.model_health import render_model_health
    from components.walkforward import render_walkforward_results, render_baseline_comparison
    from components.backtest import render_backtest
    from components.ui import section_header, info_panel, empty_state, metric_row
except ImportError:
    from streamlit_app.components.sidebar import render_sidebar
    from streamlit_app.components.header import render_header
    from streamlit_app.components.candles import render_candles
    from streamlit_app.components.bias_cards import render_bias_cards
    from streamlit_app.components.probability_chart import render_probability
    from streamlit_app.components.model_health import render_model_health
    from streamlit_app.components.walkforward import render_walkforward_results, render_baseline_comparison
    from streamlit_app.components.backtest import render_backtest
    from streamlit_app.components.ui import section_header, info_panel, empty_state, metric_row

# ── Local-only imports ────────────────────────────────────────
if not _IS_CLOUD:
    from src.ingestion.fetch_realtime import fetch_realtime_data
    from src.features.feature_engineering import engineer_features
    from src.inference.predict import predict
    from src.monitoring.drift_monitor import monitor_drift
    from src.common.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# ── Session state ─────────────────────────────────────────────
for k in ["price_data", "predictions", "drift", "wf_results", "baselines", "backtest"]:
    st.session_state.setdefault(k, None)

# ── Backend status ────────────────────────────────────────────
@st.cache_data(ttl=30)
def _backend_alive() -> bool:
    return get_client().is_alive()

_backend_ok = _backend_alive()
_api_url = os.getenv("API_BASE_URL", "http://localhost:8000")

# ── Sidebar ───────────────────────────────────────────────────
ticker, api_interval, interval_label, horizon_label, run_analysis = render_sidebar()

# ── Backend badge ─────────────────────────────────────────────
if _backend_ok:
    st.sidebar.markdown(
        "<div style='background:rgba(0,229,160,0.08);border:1px solid rgba(0,229,160,0.22);"
        "border-radius:8px;padding:8px 12px;margin-top:8px;'>"
        "<span style='font-size:9px;color:#3d4d6b;font-family:JetBrains Mono,monospace;"
        "text-transform:uppercase;'>API Backend</span><br>"
        "<span style='font-size:11px;color:#00e5a0;font-family:JetBrains Mono,monospace;"
        "font-weight:700;'>● Connected</span>"
        f"<span style='font-size:9px;color:#3d4d6b;font-family:JetBrains Mono,monospace;"
        f"margin-left:6px;'>{_api_url}</span>"
        "</div>", unsafe_allow_html=True)
else:
    st.sidebar.markdown(
        "<div style='background:rgba(255,77,109,0.08);border:1px solid rgba(255,77,109,0.22);"
        "border-radius:8px;padding:8px 12px;margin-top:8px;'>"
        "<span style='font-size:9px;color:#3d4d6b;font-family:JetBrains Mono,monospace;"
        "text-transform:uppercase;'>API Backend</span><br>"
        "<span style='font-size:11px;color:#ff4d6d;font-family:JetBrains Mono,monospace;"
        "font-weight:700;'>● Offline</span><br>"
        "<span style='font-size:9px;color:#3d4d6b;font-family:JetBrains Mono,monospace;'>"
        "Start Docker + ngrok locally</span>"
        "</div>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
render_header(ticker=ticker, interval=interval_label, horizon=horizon_label)

# ── Constants ─────────────────────────────────────────────────
INTERVAL_MAP = {"5m": "5min", "15m": "15min", "1h": "60min"}
api_interval_str = INTERVAL_MAP[api_interval]

WINDOWS = {
    "5m":  {"Next Interval": 1, "1 Day": 78,  "1 Month": 1560, "3 Months": 4680},
    "15m": {"Next Interval": 1, "1 Day": 26,  "1 Month": 520,  "3 Months": 1560},
    "1h":  {"Next Interval": 1, "1 Day": 7,   "1 Month": 160,  "3 Months": 480},
}

def build_summary(predictions, interval, horizon) -> dict:
    window = min(WINDOWS[interval][horizon], len(predictions))
    recent = predictions.tail(window)
    prob_up = recent["prob_up"].mean()
    prob_down = recent["prob_down"].mean()
    return {
        "direction":  "UP" if prob_up > prob_down else "DOWN",
        "confidence": max(prob_up, prob_down),
        "window":     window,
        "horizon":    horizon,
    }

# ── Pipeline ──────────────────────────────────────────────────
if run_analysis:
    if not _backend_ok:
        st.error("❌ Backend is offline. Start Docker Desktop + ngrok locally first.")
    else:
        # Step 1 — Run pipeline on backend
        with st.spinner("⚙️ Fetching data & engineering features..."):
            try:
                pipe_resp = get_client().run_pipeline(ticker)
                st.toast(f"✅ Pipeline complete for {ticker}")
            except APIError as exc:
                st.warning(f"Pipeline warning: {exc.detail} — trying with existing data...")
            except Exception as exc:
                st.warning(f"Pipeline note: {exc}")

        # Step 2 — Fetch price data
        with st.spinner("📈 Loading price data..."):
            try:
                price_resp = get_client().get_price_data(ticker, interval=api_interval_str)
                price_df = pd.DataFrame(price_resp["data"])
                price_df["Date"] = pd.to_datetime(price_df["Date"])
                st.session_state.price_data = price_df
            except APIError as exc:
                st.warning(f"Price data unavailable: {exc.detail}")
                st.session_state.price_data = None
            except Exception as exc:
                st.warning(f"Price data error: {exc}")
                st.session_state.price_data = None

        # Step 3 — Get predictions
        with st.spinner("🤖 Running XGBoost inference..."):
            try:
                api_resp = get_client().predict(ticker, top_n=500)
                rows = api_resp.get("predictions", [])
                df = pd.DataFrame(rows)
                if not df.empty:
                    df.rename(columns={"date": "Date"}, inplace=True)
                    df["prediction"] = (df["direction"] == "UP").astype(int)
                st.session_state.predictions = df
                for k in ["drift", "wf_results", "baselines", "backtest"]:
                    st.session_state[k] = None
                bias = api_resp.get("summary", {}).get("bias", "Unknown")
                conf = api_resp.get("summary", {}).get("avg_confidence", 0)
                st.success(f"✅ Analysis complete — {bias} bias · {conf:.1%} confidence")
            except APIError as exc:
                st.error(f"**Prediction error {exc.status_code}:** {exc.detail}")
            except Exception as exc:
                st.error(f"**Error:** {exc}")

# ── Tabs ──────────────────────────────────────────────────────
tab_price, tab_outlook, tab_validate, tab_backtest, tab_health = st.tabs([
    "📊  Price Action",
    "🔮  Model Outlook",
    "📐  Validation",
    "💹  Backtest",
    "🩺  Health",
])

# ── TAB 1 — Price Action ──────────────────────────────────────
with tab_price:
    if st.session_state.price_data is not None and not st.session_state.price_data.empty:
        df = st.session_state.price_data
        close_last  = df["Close"].iloc[-1]
        close_first = df["Close"].iloc[0]
        pct_chg     = (close_last - close_first) / close_first * 100
        metric_row([
            {"label": "Last Close",    "value": f"${close_last:.2f}",
             "variant": "bull" if pct_chg >= 0 else "bear"},
            {"label": "Period High",   "value": f"${df['High'].max():.2f}"},
            {"label": "Period Low",    "value": f"${df['Low'].min():.2f}"},
            {"label": "Period Change", "value": f"{pct_chg:+.2f}%",
             "variant": "bull" if pct_chg >= 0 else "bear"},
            {"label": "Candles",       "value": str(len(df)),
             "sub": f"{interval_label} bars"},
        ])
        render_candles(df, ticker=ticker)
    else:
        empty_state("📉", "No price data loaded yet.",
                    "Click ▶ Run Market Analysis in the sidebar.")

# ── TAB 2 — Model Outlook ─────────────────────────────────────
with tab_outlook:
    if st.session_state.predictions is not None and not st.session_state.predictions.empty:
        summary = build_summary(st.session_state.predictions, api_interval, horizon_label)
        render_bias_cards(summary)
        render_probability(st.session_state.predictions)
        st.caption(
            f"Interval ({interval_label}) · Horizon ({horizon_label}) · "
            f"{summary['window']} intervals aggregated."
        )
    else:
        empty_state("🔮", "No predictions yet.", "Click ▶ Run Market Analysis first.")

# ── TAB 3 — Validation ────────────────────────────────────────
with tab_validate:
    section_header("Walk-Forward Validation", badge="No Data Leakage")
    info_panel(
        "Walk-forward validation runs on the <strong>local backend</strong>. "
        "Results are available when running locally with Docker."
    )
    if not _IS_CLOUD:
        col_ctrl, col_run = st.columns([1, 3])
        with col_ctrl:
            n_splits = st.selectbox("Folds", [3, 5, 8, 10], index=1)
        with col_run:
            st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
            wf_btn = st.button("▶  Run Walk-Forward Validation", use_container_width=True)
        if wf_btn:
            with st.spinner(f"Running {n_splits}-fold walk-forward validation..."):
                try:
                    from src.training.walk_forward_validation import run_walk_forward_validation
                    st.session_state.wf_results = run_walk_forward_validation(ticker, n_splits=n_splits)
                    st.success("Walk-forward validation complete.")
                except Exception as exc:
                    st.error(f"Walk-forward failed: {exc}")
        if st.session_state.wf_results:
            render_walkforward_results(st.session_state.wf_results)
        else:
            empty_state("📐", "Run walk-forward validation to see results.")
    else:
        empty_state("📐", "Available in local mode only.",
                    "Run Docker locally and open http://localhost:8501")

# ── TAB 4 — Backtest ──────────────────────────────────────────
with tab_backtest:
    section_header("Strategy Backtest", badge="Out-of-Sample")
    info_panel("Backtesting runs on the <strong>local backend</strong> using held-out training data.")
    if not _IS_CLOUD:
        col_strat, col_run_bt = st.columns([2, 2])
        with col_strat:
            strategy = st.selectbox(
                "Strategy", ["long_only", "long_short", "buy_and_hold"],
                format_func=lambda x: {
                    "long_only":    "Long Only",
                    "long_short":   "Long / Short",
                    "buy_and_hold": "Buy & Hold"
                }[x])
        with col_run_bt:
            st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
            bt_btn = st.button("▶  Run Backtest", use_container_width=True)
        if bt_btn:
            with st.spinner("Running backtest..."):
                try:
                    from src.training.backtesting import run_backtest
                    st.session_state.backtest = run_backtest(ticker, strategy=strategy)
                    st.success("Backtest complete.")
                except Exception as exc:
                    st.error(f"Backtest failed: {exc}")
        if st.session_state.backtest:
            render_backtest(st.session_state.backtest)
        else:
            empty_state("💹", "No backtest results yet.", "Select a strategy and click ▶ Run Backtest.")
    else:
        empty_state("💹", "Available in local mode only.",
                    "Run Docker locally and open http://localhost:8501")

# ── TAB 5 — Health ────────────────────────────────────────────
with tab_health:
    section_header("Model Health & Data Drift")
    info_panel("Drift monitoring runs on the <strong>local backend</strong>.")
    if not _IS_CLOUD:
        health_btn = st.button("▶  Check Model Health", use_container_width=True)
        if health_btn:
            with st.spinner("Analysing feature distributions..."):
                try:
                    st.session_state.drift = monitor_drift(ticker=ticker)
                    st.success("Drift analysis complete.")
                except Exception as exc:
                    st.error(f"Drift monitor failed: {exc}")
        if st.session_state.drift:
            render_model_health(st.session_state.drift)
        else:
            empty_state("🩺", "Health check not run yet.", "Click ▶ Check Model Health above.")
    else:
        empty_state("🩺", "Available in local mode only.",
                    "Run Docker locally and open http://localhost:8501")

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
_mode = "Cloud API" if _IS_CLOUD else ("API" if _backend_ok else "Local")
st.markdown(
    f"<div style='display:flex;justify-content:space-between;align-items:center;"
    f"flex-wrap:wrap;gap:8px;'>"
    f"<span style='font-family:JetBrains Mono,monospace;font-size:10px;color:#3d4d6b;'>"
    f"Stock MLOps v2.1 · XGBoost · MLflow · Mode: {_mode}</span>"
    f"<span style='font-family:JetBrains Mono,monospace;font-size:10px;color:#3d4d6b;'>"
    f"⚠ Educational use only — not financial advice</span></div>",
    unsafe_allow_html=True)
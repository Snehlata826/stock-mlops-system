"""
Stock MLOps Dashboard — Production-grade Streamlit app.
Refactored for clarity, performance, and professional UX.

Run: streamlit run streamlit_app/app.py
"""
import streamlit as st
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# ── Page config (must be FIRST Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Stock MLOps",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)

# ── Design system ─────────────────────────────────────────────────────────────
from components.theme import inject_theme
inject_theme()

# ── Backend imports ───────────────────────────────────────────────────────────
from src.ingestion.fetch_realtime import fetch_realtime_data
from src.features.feature_engineering import engineer_features
from src.inference.predict import predict
from src.monitoring.drift_monitor import monitor_drift
from src.common.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

# ── UI components ─────────────────────────────────────────────────────────────
from components.sidebar import render_sidebar
from components.header import render_header
from components.candles import render_candles
from components.bias_cards import render_bias_cards
from components.probability_chart import render_probability
from components.model_health import render_model_health
from components.walkforward import render_walkforward_results, render_baseline_comparison
from components.backtest import render_backtest
from components.ui import section_header, info_panel, empty_state, metric_row

# ── Session state initialisation ─────────────────────────────────────────────
_STATE_KEYS = ["price_data", "predictions", "drift", "wf_results", "baselines", "backtest"]
for k in _STATE_KEYS:
    st.session_state.setdefault(k, None)

# ── Sidebar ───────────────────────────────────────────────────────────────────
ticker, api_interval, interval_label, horizon_label, run_analysis = render_sidebar()

# ── Header ────────────────────────────────────────────────────────────────────
render_header(ticker=ticker, interval=interval_label, horizon=horizon_label)

# ── Interval → API format map ─────────────────────────────────────────────────
INTERVAL_MAP = {"5m": "5min", "15m": "15min", "1h": "60min"}
api_interval_str = INTERVAL_MAP[api_interval]

# ── Horizon → bar count map ────────────────────────────────────────────────────
WINDOWS = {
    "5m":  {"Next Interval": 1, "1 Day": 78,  "1 Month": 1560, "3 Months": 4680},
    "15m": {"Next Interval": 1, "1 Day": 26,  "1 Month": 520,  "3 Months": 1560},
    "1h":  {"Next Interval": 1, "1 Day": 7,   "1 Month": 160,  "3 Months": 480},
}


def build_summary(predictions, interval, horizon) -> dict:
    window = min(WINDOWS[interval][horizon], len(predictions))
    recent = predictions.tail(window)
    prob_up   = recent["prob_up"].mean()
    prob_down = recent["prob_down"].mean()
    return {
        "direction":  "UP" if prob_up > prob_down else "DOWN",
        "confidence": max(prob_up, prob_down),
        "window":     window,
        "horizon":    horizon,
    }


# ── Pipeline execution ────────────────────────────────────────────────────────
if run_analysis:
    # ① Fetch data
    with st.spinner("Fetching market data…"):
        try:
            st.session_state.price_data = fetch_realtime_data(
                ticker=ticker, interval=api_interval_str, force_refresh=True
            )
        except Exception as e:
            st.error(f"**Data fetch failed:** {e}")
            st.stop()

    # ② Engineer features
    with st.spinner("Engineering 40+ technical features…"):
        try:
            engineer_features(
                RAW_DATA_DIR / f"realtime_{ticker}.csv",
                PROCESSED_DATA_DIR / f"features_inference_{ticker}.csv",
                is_training=False,
            )
        except Exception as e:
            st.error(f"**Feature engineering failed:** {e}")
            st.stop()

    # ③ Inference
    with st.spinner("Running XGBoost inference…"):
        try:
            st.session_state.predictions = predict(ticker=ticker)
            # Clear stale downstream results when re-running
            for k in ["drift", "wf_results", "baselines", "backtest"]:
                st.session_state[k] = None
            st.success("✅  Analysis complete — explore the tabs below.")
        except Exception as e:
            st.warning(
                f"**Inference failed** (model may not be trained yet): {e}\n\n"
                f"Train first with: `bash scripts/run_training.sh {ticker}`"
            )


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_price, tab_outlook, tab_validate, tab_backtest, tab_health = st.tabs([
    "📊  Price Action",
    "🔮  Model Outlook",
    "📐  Validation",
    "💹  Backtest",
    "🩺  Health",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Price Action
# ─────────────────────────────────────────────────────────────────────────────
with tab_price:
    if st.session_state.price_data is not None:
        df = st.session_state.price_data

        # OHLCV summary
        close_last  = df["Close"].iloc[-1]
        close_first = df["Close"].iloc[0]
        pct_chg     = (close_last - close_first) / close_first * 100
        pct_variant = "bull" if pct_chg >= 0 else "bear"

        metric_row([
            {"label": "Last Close",    "value": f"${close_last:.2f}",          "variant": pct_variant},
            {"label": "Period High",   "value": f"${df['High'].max():.2f}"},
            {"label": "Period Low",    "value": f"${df['Low'].min():.2f}"},
            {"label": "Period Change", "value": f"{pct_chg:+.2f}%",            "variant": pct_variant},
            {"label": "Candles",       "value": str(len(df)),                  "sub": f"{interval_label} bars"},
        ])

        render_candles(df, ticker=ticker)

    else:
        empty_state(
            "📉",
            "No price data loaded yet.",
            "Click  ▶  Run Market Analysis  in the sidebar to begin.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Model Outlook
# ─────────────────────────────────────────────────────────────────────────────
with tab_outlook:
    if st.session_state.predictions is not None:
        summary = build_summary(st.session_state.predictions, api_interval, horizon_label)

        render_bias_cards(summary)
        render_probability(st.session_state.predictions)

        st.caption(
            f"Candle interval ({interval_label}) defines data granularity. "
            f"Horizon ({horizon_label}) aggregates the last {summary['window']} intervals. "
            "Confidence = mean probability of predicted direction."
        )

    else:
        empty_state(
            "🔮",
            "No predictions available yet.",
            "Run Market Analysis first, then return to this tab.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: Walk-forward Validation + Baselines
# ─────────────────────────────────────────────────────────────────────────────
with tab_validate:
    section_header("Walk-Forward Validation", badge="No Data Leakage")
    info_panel(
        "Walk-forward validation trains <strong>only on past data</strong> and tests on <strong>future data</strong>, "
        "simulating real deployment. This eliminates the data leakage of random splits on time-series."
    )

    col_ctrl, col_run = st.columns([1, 3])
    with col_ctrl:
        n_splits = st.selectbox("Folds", [3, 5, 8, 10], index=1)
    with col_run:
        st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
        wf_btn = st.button("▶  Run Walk-Forward Validation", use_container_width=True)

    if wf_btn:
        with st.spinner(f"Running {n_splits}-fold walk-forward validation…"):
            try:
                from src.training.walk_forward_validation import run_walk_forward_validation
                st.session_state.wf_results = run_walk_forward_validation(ticker, n_splits=n_splits)
                st.success("Walk-forward validation complete.")
            except Exception as e:
                st.error(f"Walk-forward failed: {e}")

    if st.session_state.wf_results:
        render_walkforward_results(st.session_state.wf_results)
    else:
        empty_state("📐", "Run walk-forward validation to see results.", "Requires a trained model.")

    st.markdown("---")

    section_header("Baseline Comparison", badge="ARIMA + Naive")
    info_panel(
        "Comparing XGBoost against statistical and naive baselines. "
        "If the model can't outperform a coin flip or momentum strategy, it isn't adding value."
    )
    bl_btn = st.button("▶  Run Baseline Models", use_container_width=True)
    if bl_btn:
        with st.spinner("Running ARIMA and naive baselines…"):
            try:
                from src.training.baseline_models import run_all_baselines
                st.session_state.baselines = run_all_baselines(ticker)
                st.success("Baselines complete.")
            except Exception as e:
                st.error(f"Baseline failed: {e}")

    if st.session_state.baselines and st.session_state.wf_results:
        wf_acc = st.session_state.wf_results["aggregate"]["accuracy_mean"]
        render_baseline_comparison(st.session_state.baselines, wf_acc)
    elif st.session_state.baselines:
        st.info("Run Walk-Forward Validation first to show the comparison chart.")
    elif not bl_btn:
        empty_state("📊", "Baseline comparison not run yet.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: Backtest
# ─────────────────────────────────────────────────────────────────────────────
with tab_backtest:
    section_header("Strategy Backtest", badge="Out-of-Sample")
    info_panel(
        "Simulates a trading strategy on the held-out <strong>20%</strong> of training data "
        "(never seen during model fitting). Computes cumulative return, Sharpe ratio, "
        "max drawdown, and alpha vs buy-and-hold."
    )

    col_strat, col_run_bt = st.columns([2, 2])
    with col_strat:
        strategy = st.selectbox(
            "Strategy",
            ["long_only", "long_short", "buy_and_hold"],
            format_func=lambda x: {
                "long_only":     "Long Only  (hold cash when bearish)",
                "long_short":    "Long / Short  (full exposure always)",
                "buy_and_hold":  "Buy & Hold  (passive baseline)",
            }[x],
        )
    with col_run_bt:
        st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
        bt_btn = st.button("▶  Run Backtest", use_container_width=True)

    if bt_btn:
        with st.spinner("Running backtest on held-out data…"):
            try:
                from src.training.backtesting import run_backtest
                st.session_state.backtest = run_backtest(ticker, strategy=strategy)
                st.success("Backtest complete.")
            except Exception as e:
                st.error(f"Backtest failed: {e}")

    if st.session_state.backtest:
        render_backtest(st.session_state.backtest)
    else:
        empty_state("💹", "No backtest results yet.", "Select a strategy and click  ▶  Run Backtest.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5: Model Health
# ─────────────────────────────────────────────────────────────────────────────
with tab_health:
    section_header("Model Health & Data Drift")
    info_panel(
        "Compares current inference feature distributions against <strong>reference distributions</strong> "
        "from training time. High drift means the live market looks different from training — "
        "predictions may degrade and retraining may be needed."
    )

    health_btn = st.button("▶  Check Model Health", use_container_width=True)
    if health_btn:
        with st.spinner("Analysing feature distributions…"):
            try:
                st.session_state.drift = monitor_drift(ticker=ticker)
                st.success("Drift analysis complete.")
            except Exception as e:
                st.error(f"Drift monitor failed: {e}")

    if st.session_state.drift:
        render_model_health(st.session_state.drift)

        report_path = Path(st.session_state.drift.get("report_path", ""))
        if report_path.exists():
            if st.button("📄  View Full Evidently Report"):
                with open(report_path, "r", encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=880, scrolling=True)
        else:
            st.caption("Run drift check first to generate the HTML report.")
    else:
        empty_state("🩺", "Health check not run yet.", "Click  ▶  Check Model Health  above.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
    <span style="font-family:'JetBrains Mono',monospace; font-size:10px; color:#3d4d6b;">
        Stock MLOps System v2.0 &nbsp;·&nbsp; XGBoost + MLflow + Evidently &nbsp;·&nbsp; 
        Walk-forward &nbsp;·&nbsp; FastAPI
    </span>
    <span style="font-family:'JetBrains Mono',monospace; font-size:10px; color:#3d4d6b;">
        ⚠&nbsp; Educational use only — not financial advice
    </span>
</div>
""", unsafe_allow_html=True)
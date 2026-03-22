import streamlit as st
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# ── Page config (must be first) ──────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global dark theme CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #070b14 !important;
}
[data-testid="stAppViewContainer"] > .main {
    background: #070b14;
}
.block-container {
    padding-top: 1.5rem !important;
    max-width: 1200px;
}
h1, h2, h3, h4 { color: #e2e8f0 !important; }
p, li, span { color: #a0aec0; }
.stMetric label { color: rgba(255,255,255,0.5) !important; font-size: 0.72rem !important; }
.stMetric [data-testid="metric-container"] { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 10px; padding: 12px 16px; }
div[data-testid="stExpander"] { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 10px; }
.stButton button {
    background: linear-gradient(135deg, #1a365d 0%, #2a4a7f 100%) !important;
    color: #63b3ed !important;
    border: 1px solid rgba(99,179,237,0.3) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton button:hover {
    border-color: #63b3ed !important;
    box-shadow: 0 0 16px rgba(99,179,237,0.2) !important;
}
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #1a3a5c 0%, #0d47a1 100%) !important;
    border-color: rgba(99,179,237,0.5) !important;
}
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
}
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.2px;
    margin: 0;
}
.section-badge {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    background: rgba(99,179,237,0.1);
    color: #63b3ed;
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 20px;
    padding: 2px 10px;
}
.info-box {
    background: rgba(99,179,237,0.05);
    border: 1px solid rgba(99,179,237,0.15);
    border-left: 3px solid #63b3ed;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 12px 0;
    color: rgba(255,255,255,0.6);
    font-size: 0.85rem;
    line-height: 1.6;
}
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.1) !important;
    color: #e2e8f0 !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.02);
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.07);
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: rgba(255,255,255,0.45) !important;
    border-radius: 8px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,179,237,0.12) !important;
    color: #63b3ed !important;
}
hr { border-color: rgba(255,255,255,0.07) !important; }
</style>
""", unsafe_allow_html=True)

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

# ── Session state ─────────────────────────────────────────────────────────────
for key in ["price_data", "predictions", "drift", "wf_results", "baselines", "backtest"]:
    st.session_state.setdefault(key, None)

# ── Sidebar ───────────────────────────────────────────────────────────────────
ticker, interval_label, horizon_label, run_analysis = render_sidebar()

INTERVAL_MAP = {"5m": "5min", "15m": "15min", "1h": "60min"}
api_interval = INTERVAL_MAP[interval_label]

# ── Header ────────────────────────────────────────────────────────────────────
render_header(ticker=ticker, interval=interval_label, horizon=horizon_label)

# ── How to use ────────────────────────────────────────────────────────────────
with st.expander("ℹ️ How to use this dashboard", expanded=False):
    st.markdown("""
    <div class="info-box">
    <strong>Step 1</strong> — Select Asset, Interval & Horizon from the sidebar.<br>
    <strong>Step 2</strong> — Click <strong>Run Market Analysis</strong>.<br>
    <strong>Step 3</strong> — Explore the tabs: Price Action · Outlook · Validation · Backtest · Health.<br><br>
    The pipeline fetches live data → engineers 40+ technical features → loads your trained XGBoost model → returns probabilistic direction forecasts.<br>
    Walk-forward validation and backtesting tabs require a trained model (<code>bash scripts/run_training.sh {ticker}</code>).
    </div>
    """, unsafe_allow_html=True)

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_analysis:
    with st.spinner("🔄 Fetching market data..."):
        try:
            st.session_state.price_data = fetch_realtime_data(
                ticker=ticker, interval=api_interval, force_refresh=True
            )
        except Exception as e:
            st.error(f"Data fetch failed: {e}")
            st.stop()

    with st.spinner("⚙️ Engineering features..."):
        try:
            engineer_features(
                RAW_DATA_DIR / f"realtime_{ticker}.csv",
                PROCESSED_DATA_DIR / f"features_inference_{ticker}.csv",
                is_training=False,
            )
        except Exception as e:
            st.error(f"Feature engineering failed: {e}")
            st.stop()

    with st.spinner("🧠 Running inference..."):
        try:
            st.session_state.predictions = predict(ticker=ticker)
            st.success("✅ Analysis complete")
        except Exception as e:
            st.warning(f"Inference failed (model may not be trained yet): {e}")

# ── Helper ────────────────────────────────────────────────────────────────────
def build_summary(predictions, interval, horizon):
    WINDOWS = {
        "5m":  {"Next Interval": 1, "1 Day": 78, "1 Month": 1560, "3 Months": 4680},
        "15m": {"Next Interval": 1, "1 Day": 26, "1 Month": 520,  "3 Months": 1560},
        "1h":  {"Next Interval": 1, "1 Day": 7,  "1 Month": 160,  "3 Months": 480},
    }
    window = min(WINDOWS[interval][horizon], len(predictions))
    recent = predictions.tail(window)
    prob_up = recent["prob_up"].mean()
    prob_down = recent["prob_down"].mean()
    return {
        "direction": "UP" if prob_up > prob_down else "DOWN",
        "confidence": max(prob_up, prob_down),
        "window": window,
        "horizon": horizon,
    }

def section_header(icon, title, badge=None):
    badge_html = f'<span class="section-badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="section-header">
        <span style="font-size:1.2rem">{icon}</span>
        <span class="section-title">{title}</span>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Price Action",
    "🔮 Model Outlook",
    "📐 Validation",
    "💹 Backtest",
    "🩺 Health",
])

# ─── Tab 1: Price Action ──────────────────────────────────────────────────────
with tab1:
    if st.session_state.price_data is not None:
        render_candles(st.session_state.price_data, ticker=ticker)
        df = st.session_state.price_data
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Last Close", f"${df['Close'].iloc[-1]:.2f}")
        c2.metric("High", f"${df['High'].max():.2f}")
        c3.metric("Low", f"${df['Low'].min():.2f}")
        pct = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
        c4.metric("Period Chg", f"{pct:+.2f}%")
    else:
        st.markdown('<div class="info-box">Run Market Analysis from the sidebar to load price data.</div>', unsafe_allow_html=True)

# ─── Tab 2: Model Outlook ─────────────────────────────────────────────────────
with tab2:
    if st.session_state.predictions is not None:
        summary = build_summary(st.session_state.predictions, interval_label, horizon_label)
        render_bias_cards(summary)
        st.markdown("---")
        section_header("📈", "Conviction over time")
        render_probability(st.session_state.predictions)
        st.caption(
            f"Candle interval ({interval_label}) defines data granularity. "
            f"Horizon ({horizon_label}) aggregates {summary['window']} intervals."
        )
    else:
        st.markdown('<div class="info-box">Run Market Analysis first to see model predictions.</div>', unsafe_allow_html=True)

# ─── Tab 3: Walk-forward Validation + Baselines ───────────────────────────────
with tab3:
    section_header("📐", "Walk-forward validation", badge="No data leakage")
    st.markdown("""
    <div class="info-box">
    Walk-forward validation trains only on <em>past</em> data and tests on <em>future</em> data — 
    simulating real deployment. This avoids the data leakage of random train/test splits on time-series.
    </div>
    """, unsafe_allow_html=True)

    col_wf, col_bl = st.columns([3, 1])
    with col_bl:
        n_splits = st.selectbox("Folds", [3, 5, 8, 10], index=1)

    if st.button("▶ Run Walk-Forward Validation", use_container_width=True):
        with st.spinner("Running walk-forward validation..."):
            try:
                from src.training.walk_forward_validation import run_walk_forward_validation
                st.session_state.wf_results = run_walk_forward_validation(ticker, n_splits=n_splits)
                st.success("Walk-forward validation complete")
            except Exception as e:
                st.error(f"Walk-forward failed: {e}")

    if st.session_state.wf_results:
        render_walkforward_results(st.session_state.wf_results)

    st.markdown("---")
    section_header("📊", "Baseline comparison", badge="ARIMA + Naive")
    if st.button("▶ Run Baseline Models", use_container_width=True):
        with st.spinner("Running ARIMA and naive baselines..."):
            try:
                from src.training.baseline_models import run_all_baselines
                st.session_state.baselines = run_all_baselines(ticker)
                st.success("Baselines complete")
            except Exception as e:
                st.error(f"Baseline failed: {e}")

    if st.session_state.baselines and st.session_state.wf_results:
        wf_acc = st.session_state.wf_results["aggregate"]["accuracy_mean"]
        render_baseline_comparison(st.session_state.baselines, wf_acc)
    elif st.session_state.baselines:
        st.info("Run Walk-Forward Validation first to see the comparison chart.")

# ─── Tab 4: Backtest ──────────────────────────────────────────────────────────
with tab4:
    section_header("💹", "Strategy backtest", badge="Out-of-sample")
    st.markdown("""
    <div class="info-box">
    Simulates a trading strategy on the held-out 20% of training data (never seen during model fitting).
    Compares cumulative return, Sharpe ratio and max drawdown vs buy-and-hold.
    </div>
    """, unsafe_allow_html=True)

    strategy = st.selectbox(
        "Strategy",
        ["long_only", "long_short", "buy_and_hold"],
        format_func=lambda x: {
            "long_only": "Long Only (cash when bearish)",
            "long_short": "Long / Short",
            "buy_and_hold": "Buy & Hold (baseline)",
        }[x]
    )

    if st.button("▶ Run Backtest", use_container_width=True):
        with st.spinner("Running backtest..."):
            try:
                from src.training.backtesting import run_backtest
                st.session_state.backtest = run_backtest(ticker, strategy=strategy)
                st.success("Backtest complete")
            except Exception as e:
                st.error(f"Backtest failed: {e}")

    if st.session_state.backtest:
        render_backtest(st.session_state.backtest)

# ─── Tab 5: Model Health ──────────────────────────────────────────────────────
with tab5:
    section_header("🩺", "Model health & data drift")
    st.markdown("""
    <div class="info-box">
    Compares the distribution of current inference features against reference features from training.
    High drift means the live market looks different from when the model was trained — predictions may degrade.
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶ Check Model Health", use_container_width=True):
        with st.spinner("Analysing data drift..."):
            try:
                st.session_state.drift = monitor_drift(ticker=ticker)
                st.success("Drift analysis complete")
            except Exception as e:
                st.error(f"Drift monitor failed: {e}")

    if st.session_state.drift:
        render_model_health(st.session_state.drift)

        # Show Evidently HTML report inline
        report_path = Path(st.session_state.drift.get("report_path", ""))
        if st.button("📄 View Full Drift Report"):
            if report_path.exists():
                with open(report_path, "r", encoding="utf-8") as f:
                    st.components.v1.html(f.read(), height=900, scrolling=True)
            else:
                st.warning("No HTML report found. Run drift check first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Stock MLOps System v2.0 · XGBoost + MLflow + Evidently · "
    "Walk-forward validation · Backtesting · FastAPI · Educational use only"
)

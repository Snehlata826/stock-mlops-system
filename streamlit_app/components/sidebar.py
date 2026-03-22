"""
Sidebar: configuration panel + workflow trigger.
Groups controls logically: Asset → Interval → Horizon → Run.
"""
import streamlit as st
from src.common.companies import COMPANIES


INTERVAL_OPTIONS = {
    "5 min":  "5m",
    "15 min": "15m",
    "1 hour": "1h",
}

HORIZON_OPTIONS = [
    "Next Interval",
    "1 Day",
    "1 Month",
    "3 Months",
]


def render_sidebar():
    sb = st.sidebar

    # ── Logo / Brand ─────────────────────────────────────────────────────────
    sb.markdown("""
    <div style="padding: 0.4rem 0 1.2rem; border-bottom: 1px solid rgba(255,255,255,0.06); margin-bottom: 1.2rem;">
        <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:800; 
                    background:linear-gradient(135deg,#00d4ff,#00e5a0); 
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text; letter-spacing:-0.3px;">
            STOCK MLOPS
        </div>
        <div style="font-size:9px; color:#3d4d6b; font-family:'JetBrains Mono',monospace; 
                    letter-spacing:0.2em; text-transform:uppercase; margin-top:2px;">
            Prediction System v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Step 1: Asset ────────────────────────────────────────────────────────
    sb.markdown("""
    <div style="font-size:9px; font-weight:700; letter-spacing:0.18em; 
                text-transform:uppercase; color:#3d4d6b; margin-bottom:6px; 
                font-family:'JetBrains Mono',monospace;">
        ① Asset
    </div>
    """, unsafe_allow_html=True)

    asset_names = list(COMPANIES.keys())
    selected_asset = sb.selectbox(
        "Asset", asset_names, label_visibility="collapsed"
    )
    ticker = COMPANIES[selected_asset]

    sb.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Step 2: Data Settings ────────────────────────────────────────────────
    sb.markdown("""
    <div style="font-size:9px; font-weight:700; letter-spacing:0.18em; 
                text-transform:uppercase; color:#3d4d6b; margin-bottom:6px;
                font-family:'JetBrains Mono',monospace;">
        ② Data Settings
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = sb.columns(2)
    with col1:
        st.markdown('<div style="font-size:9px;color:#3d4d6b;text-transform:uppercase;letter-spacing:0.1em;font-family:\'JetBrains Mono\',monospace;margin-bottom:3px;">Interval</div>', unsafe_allow_html=True)
        interval_label = st.selectbox(
            "Interval", list(INTERVAL_OPTIONS.keys()), label_visibility="collapsed"
        )
    with col2:
        st.markdown('<div style="font-size:9px;color:#3d4d6b;text-transform:uppercase;letter-spacing:0.1em;font-family:\'JetBrains Mono\',monospace;margin-bottom:3px;">Horizon</div>', unsafe_allow_html=True)
        horizon = st.selectbox(
            "Horizon", HORIZON_OPTIONS, label_visibility="collapsed"
        )

    interval_code = INTERVAL_OPTIONS[interval_label]

    sb.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    sb.markdown("<hr>", unsafe_allow_html=True)

    # ── Step 3: Run ──────────────────────────────────────────────────────────
    run_analysis = sb.button(
        "▶  Run Market Analysis",
        use_container_width=True,
        type="primary",
    )

    # ── Ticker Badge ─────────────────────────────────────────────────────────
    sb.markdown(f"""
    <div style="margin-top:1rem; background:rgba(0,212,255,0.06); border:1px solid rgba(0,212,255,0.14);
                border-radius:8px; padding:10px 12px;">
        <div style="font-size:9px; color:#3d4d6b; letter-spacing:0.15em; 
                    text-transform:uppercase; font-family:'JetBrains Mono',monospace; 
                    margin-bottom:4px;">Active Ticker</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:800; color:#00d4ff;">
            {ticker}
        </div>
        <div style="font-size:10px; color:#3d4d6b; font-family:'JetBrains Mono',monospace; margin-top:2px;">
            {interval_label} · {horizon}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Help ─────────────────────────────────────────────────────────────────
    sb.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    with sb.expander("ℹ  Quick Guide"):
        st.markdown("""
        **Step 1** — Pick an asset  
        **Step 2** — Set interval & horizon  
        **Step 3** — Click **Run Market Analysis**  
        
        Then explore tabs:  
        `Price` · `Outlook` · `Validate` · `Backtest` · `Health`
        
        Train a model first:  
        `bash scripts/run_training.sh AAPL`
        """)

    sb.markdown("---")
    sb.caption("⚠ Predictions are probabilistic. Not financial advice.")

    return ticker, interval_code, interval_label, horizon, run_analysis
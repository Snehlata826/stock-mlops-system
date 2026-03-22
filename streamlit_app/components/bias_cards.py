"""
Market bias summary cards: direction, confidence, horizon, window.
"""
import streamlit as st


def render_bias_cards(summary: dict):
    direction = summary["direction"]
    is_bull = direction == "UP"

    bias_label = "Bullish" if is_bull else "Bearish"
    icon        = "▲" if is_bull else "▼"
    bias_cls    = "bull" if is_bull else "bear"

    bull_color  = "#00e5a0"
    bear_color  = "#ff4d6d"
    val_color   = bull_color if is_bull else bear_color

    conf       = summary["confidence"]
    conf_pct   = f"{conf:.1%}"
    horizon    = summary["horizon"]
    window     = summary["window"]

    # Confidence interpretation
    if conf >= 0.65:
        conf_label = "Strong"
        conf_color = bull_color
    elif conf >= 0.55:
        conf_label = "Moderate"
        conf_color = "#ffb547"
    else:
        conf_label = "Weak"
        conf_color = "#8896b3"

    st.markdown(f"""
    <style>
    .bias-grid {{
        display: grid;
        grid-template-columns: 2.2fr 1fr 1fr 1fr;
        gap: 12px;
        margin-bottom: 1rem;
    }}
    @media (max-width: 768px) {{
        .bias-grid {{ grid-template-columns: 1fr 1fr; }}
    }}
    .bias-main {{
        background: {"rgba(0,229,160,0.08)" if is_bull else "rgba(255,77,109,0.08)"};
        border: 1px solid {"rgba(0,229,160,0.22)" if is_bull else "rgba(255,77,109,0.22)"};
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        position: relative;
        overflow: hidden;
    }}
    .bias-main::before {{
        content:'';
        position:absolute; top:0; left:0; right:0; height:2px;
        background: {"linear-gradient(90deg,#00e5a0,transparent)" if is_bull else "linear-gradient(90deg,#ff4d6d,transparent)"};
    }}
    .bias-card {{
        background: #0c1220;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        transition: border-color 0.15s;
    }}
    .bias-card:hover {{ border-color: rgba(255,255,255,0.12); }}
    .bias-lbl {{
        font-size: 9.5px;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #3d4d6b;
        margin-bottom: 7px;
        font-family: 'JetBrains Mono', monospace;
    }}
    .bias-val-main {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.05rem;
        font-weight: 700;
        color: {val_color};
        line-height: 1;
        letter-spacing: -0.02em;
    }}
    .bias-val {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.05rem;
        font-weight: 700;
        color: #f0f4ff;
        line-height: 1;
        letter-spacing: -0.02em;
    }}
    .bias-sub {{
        font-size: 10px;
        color: #3d4d6b;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 4px;
    }}
    </style>

    <div class="bias-grid">
        <div class="bias-main">
            <div class="bias-lbl">Market Bias</div>
            <div class="bias-val-main">{icon} {bias_label}</div>
            <div class="bias-sub" style="color:{val_color};opacity:0.7">{conf_pct} confidence · {conf_label} signal</div>
        </div>
        <div class="bias-card">
            <div class="bias-lbl">Confidence</div>
            <div class="bias-val" style="color:{conf_color}">{conf_pct}</div>
            <div class="bias-sub">{conf_label}</div>
        </div>
        <div class="bias-card">
            <div class="bias-lbl">Horizon</div>
            <div class="bias-val" style="font-size:0.95rem;padding-top:3px;white-space:nowrap;">{horizon}</div>
            <div class="bias-sub">Forecast window</div>
        </div>
        <div class="bias-card">
            <div class="bias-lbl">Intervals</div>
            <div class="bias-val">{window}</div>
            <div class="bias-sub">Aggregated</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Confidence = mean prob of predicted direction across aggregated intervals. "
        "Above ~55% indicates meaningful bias; above ~65% is a strong signal."
    )
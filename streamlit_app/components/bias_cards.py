import streamlit as st


def render_bias_cards(summary: dict):
    bias = "Bullish" if summary["direction"] == "UP" else "Bearish"
    color = "#68d391" if bias == "Bullish" else "#fc8181"
    icon = "▲" if bias == "Bullish" else "▼"
    bg = "rgba(104,211,145,0.08)" if bias == "Bullish" else "rgba(252,129,129,0.08)"
    border = "rgba(104,211,145,0.3)" if bias == "Bullish" else "rgba(252,129,129,0.3)"

    conf = summary["confidence"]
    conf_pct = f"{conf:.1%}"
    horizon = summary["horizon"]
    window = summary["window"]

    st.markdown(f"""
    <style>
    .bias-grid {{
        display: grid;
        grid-template-columns: 2fr 1fr 1fr 1fr;
        gap: 14px;
        margin-bottom: 18px;
    }}
    .bias-card {{
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 18px 20px;
    }}
    .bias-card-main {{
        background: {bg};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 18px 20px;
    }}
    .card-label {{
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.4);
        margin-bottom: 8px;
    }}
    .card-value {{
        font-size: 1.6rem;
        font-weight: 800;
        color: {color};
        line-height: 1;
    }}
    .card-value-neutral {{
        font-size: 1.6rem;
        font-weight: 800;
        color: #e2e8f0;
        line-height: 1;
    }}
    </style>
    <div class="bias-grid">
        <div class="bias-card-main">
            <div class="card-label">Market Bias</div>
            <div class="card-value">{icon} {bias}</div>
        </div>
        <div class="bias-card">
            <div class="card-label">Confidence</div>
            <div class="card-value-neutral">{conf_pct}</div>
        </div>
        <div class="bias-card">
            <div class="card-label">Horizon</div>
            <div class="card-value-neutral" style="font-size:1.1rem;padding-top:4px">{horizon}</div>
        </div>
        <div class="bias-card">
            <div class="card-label">Intervals</div>
            <div class="card-value-neutral">{window}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Confidence = average probability of the predicted direction across aggregated intervals. "
        "Values above ~55% indicate meaningful directional bias."
    )

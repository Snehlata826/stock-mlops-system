import streamlit as st


def render_model_health(drift: dict):
    drifted = drift["n_drifted_columns"]
    total = drift["total_columns"]
    drift_pct = drift["drift_percentage"]

    if drift_pct < 20:
        status_color = "#68d391"
        status_bg = "rgba(104,211,145,0.08)"
        status_border = "rgba(104,211,145,0.3)"
        status_text = "Stable"
        status_icon = "●"
        message = "Model predictions are reliable. No action needed."
    elif drift_pct < 30:
        status_color = "#f6e05e"
        status_bg = "rgba(246,224,94,0.08)"
        status_border = "rgba(246,224,94,0.3)"
        status_text = "Degrading"
        status_icon = "◑"
        message = "Market behavior is shifting. Monitor closely and consider retraining soon."
    else:
        status_color = "#fc8181"
        status_bg = "rgba(252,129,129,0.08)"
        status_border = "rgba(252,129,129,0.3)"
        status_text = "Unstable"
        status_icon = "○"
        message = "Significant drift detected. Predictions may be unreliable — retraining is strongly recommended."

    st.markdown(f"""
    <style>
    .health-grid {{
        display: grid;
        grid-template-columns: 2fr 1fr 1fr;
        gap: 14px;
        margin-bottom: 16px;
    }}
    .health-main {{
        background: {status_bg};
        border: 1px solid {status_border};
        border-radius: 12px;
        padding: 18px 20px;
    }}
    .health-card {{
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 18px 20px;
    }}
    .health-label {{
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.4);
        margin-bottom: 8px;
    }}
    .health-value-main {{
        font-size: 1.5rem;
        font-weight: 800;
        color: {status_color};
    }}
    .health-value {{
        font-size: 1.5rem;
        font-weight: 800;
        color: #e2e8f0;
    }}
    .health-message {{
        font-size: 0.82rem;
        color: rgba(255,255,255,0.5);
        margin-top: 6px;
    }}
    </style>
    <div class="health-grid">
        <div class="health-main">
            <div class="health-label">Model Health</div>
            <div class="health-value-main">{status_icon} {status_text}</div>
            <div class="health-message">{message}</div>
        </div>
        <div class="health-card">
            <div class="health-label">Drifted Features</div>
            <div class="health-value">{drifted} / {total}</div>
        </div>
        <div class="health-card">
            <div class="health-label">Drift %</div>
            <div class="health-value">{drift_pct:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

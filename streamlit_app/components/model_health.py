"""
Model health & drift monitoring display.
"""
import streamlit as st


def render_model_health(drift: dict):
    drifted   = drift["n_drifted_columns"]
    total     = drift["total_columns"]
    drift_pct = drift["drift_percentage"]

    if drift_pct < 20:
        color   = "#00e5a0"
        bg      = "rgba(0,229,160,0.07)"
        border  = "rgba(0,229,160,0.20)"
        dot     = "●"
        status  = "Stable"
        msg     = "Model predictions are reliable. Feature distributions match training data."
    elif drift_pct < 35:
        color   = "#ffb547"
        bg      = "rgba(255,181,71,0.07)"
        border  = "rgba(255,181,71,0.22)"
        dot     = "◑"
        status  = "Degrading"
        msg     = "Market behavior is shifting. Monitor closely — consider retraining soon."
    else:
        color   = "#ff4d6d"
        bg      = "rgba(255,77,109,0.07)"
        border  = "rgba(255,77,109,0.22)"
        dot     = "○"
        status  = "Unstable"
        msg     = "Significant drift detected. Predictions may be unreliable — retraining strongly recommended."

    st.markdown(f"""
    <style>
    .health-grid {{
        display: grid;
        grid-template-columns: 2.2fr 1fr 1fr;
        gap: 12px;
        margin-bottom: 1rem;
    }}
    .health-main {{
        background: {bg};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        position: relative;
        overflow: hidden;
    }}
    .health-main::before {{
        content:''; position:absolute; top:0; left:0; right:0; height:2px;
        background: linear-gradient(90deg,{color},transparent);
    }}
    .health-card {{
        background: #0c1220;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
    }}
    .h-lbl {{
        font-size: 9.5px; font-weight: 700; letter-spacing: 0.14em;
        text-transform: uppercase; color: #3d4d6b;
        margin-bottom: 7px; font-family: 'JetBrains Mono', monospace;
    }}
    .h-val-main {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.05rem; font-weight: 700;
        color: {color}; line-height: 1; letter-spacing: -0.02em;
    }}
    .h-val {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.05rem; font-weight: 700;
        color: #f0f4ff; line-height: 1; letter-spacing: -0.02em;
    }}
    .h-msg {{
        font-size: 11px; color: rgba(255,255,255,0.4); 
        font-family:'JetBrains Mono',monospace; margin-top:6px; line-height:1.5;
    }}
    </style>
    <div class="health-grid">
        <div class="health-main">
            <div class="h-lbl">Model Health</div>
            <div class="h-val-main"><span style="font-size:0.7rem;vertical-align:middle;">{dot}</span> {status}</div>
            <div class="h-msg">{msg}</div>
        </div>
        <div class="health-card">
            <div class="h-lbl">Drifted Features</div>
            <div class="h-val">{drifted}<span style="font-size:1rem;color:#3d4d6b"> /{total}</span></div>
        </div>
        <div class="health-card">
            <div class="h-lbl">Drift Rate</div>
            <div class="h-val" style="color:{color}">{drift_pct:.1f}<span style="font-size:1rem">%</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Progress bar
    bar_color = color
    st.markdown(f"""
    <div style="margin: 0.5rem 0 1rem;">
        <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
            <span style="font-size:9.5px;color:#3d4d6b;font-family:'JetBrains Mono',monospace;
                         text-transform:uppercase;letter-spacing:0.12em;">Feature Drift</span>
            <span style="font-size:9.5px;color:{bar_color};font-family:'JetBrains Mono',monospace;">{drift_pct:.1f}%</span>
        </div>
        <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:5px;overflow:hidden;">
            <div style="width:{min(drift_pct,100):.1f}%;height:100%;
                        background:linear-gradient(90deg,{bar_color},{bar_color}88);
                        border-radius:4px;transition:width 0.4s ease;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:3px;">
            <span style="font-size:9px;color:#3d4d6b;font-family:'JetBrains Mono',monospace;">0%</span>
            <span style="font-size:9px;color:#3d4d6b;font-family:'JetBrains Mono',monospace;">Stable &lt;20%</span>
            <span style="font-size:9px;color:#3d4d6b;font-family:'JetBrains Mono',monospace;">Critical &gt;35%</span>
            <span style="font-size:9px;color:#3d4d6b;font-family:'JetBrains Mono',monospace;">100%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
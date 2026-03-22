"""
Design system & theme injection for Stock MLOps Dashboard.
Single source of truth for all visual tokens.
"""
import streamlit as st


# ─── Design Tokens ────────────────────────────────────────────────────────────
COLORS = {
    "bg_base":      "#04070f",
    "bg_surface":   "#080e1a",
    "bg_card":      "#0c1220",
    "bg_elevated":  "#111827",
    "bg_border":    "rgba(255,255,255,0.06)",
    "bg_border_hov":"rgba(255,255,255,0.12)",

    "accent":       "#00d4ff",
    "accent_dim":   "rgba(0,212,255,0.12)",
    "accent_glow":  "rgba(0,212,255,0.25)",

    "bull":         "#00e5a0",
    "bull_dim":     "rgba(0,229,160,0.10)",
    "bull_border":  "rgba(0,229,160,0.25)",

    "bear":         "#ff4d6d",
    "bear_dim":     "rgba(255,77,109,0.10)",
    "bear_border":  "rgba(255,77,109,0.25)",

    "warn":         "#ffb547",
    "warn_dim":     "rgba(255,181,71,0.10)",

    "text_primary":   "#f0f4ff",
    "text_secondary": "#8896b3",
    "text_muted":     "#3d4d6b",

    "chart_bg":     "#07101e",
}

CHART_THEME = {
    "plot_bgcolor":  "#07101e",
    "paper_bgcolor": "#07101e",
    "font_color":    "#8896b3",
    "grid_color":    "rgba(255,255,255,0.04)",
    "legend_bg":     "rgba(4,7,15,0.8)",
    "legend_border": "rgba(255,255,255,0.08)",
}

def apply_chart_theme(fig, height=360, title=None):
    """Apply consistent dark theme to any plotly figure."""
    updates = dict(
        height=height,
        plot_bgcolor=CHART_THEME["plot_bgcolor"],
        paper_bgcolor=CHART_THEME["paper_bgcolor"],
        font=dict(color=CHART_THEME["font_color"], family="'JetBrains Mono', monospace", size=11),
        xaxis=dict(gridcolor=CHART_THEME["grid_color"], zeroline=False, showline=False),
        yaxis=dict(gridcolor=CHART_THEME["grid_color"], zeroline=False, showline=False),
        legend=dict(
            bgcolor=CHART_THEME["legend_bg"],
            bordercolor=CHART_THEME["legend_border"],
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=12, r=12, t=44 if title else 20, b=12),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="rgba(255,255,255,0.1)",
            font=dict(color="#f0f4ff", size=11),
        ),
    )
    if title:
        updates["title"] = dict(
            text=title,
            font=dict(size=13, color="#f0f4ff", family="'JetBrains Mono', monospace"),
            x=0,
            xanchor="left",
            pad=dict(l=4),
        )
    fig.update_layout(**updates)
    return fig


def inject_theme():
    """Inject the full CSS design system into the Streamlit app."""
    st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap" rel="stylesheet">

<style>
/* ═══════════════ RESET & BASE ═══════════════ */
:root {
    --bg:           #04070f;
    --bg-surface:   #080e1a;
    --bg-card:      #0c1220;
    --bg-elevated:  #111827;
    --border:       rgba(255,255,255,0.06);
    --border-hov:   rgba(255,255,255,0.12);

    --accent:       #00d4ff;
    --accent-dim:   rgba(0,212,255,0.12);
    --accent-glow:  rgba(0,212,255,0.25);

    --bull:         #00e5a0;
    --bull-dim:     rgba(0,229,160,0.10);
    --bull-border:  rgba(0,229,160,0.25);

    --bear:         #ff4d6d;
    --bear-dim:     rgba(255,77,109,0.10);
    --bear-border:  rgba(255,77,109,0.25);

    --warn:         #ffb547;
    --warn-dim:     rgba(255,181,71,0.10);

    --t1: #f0f4ff;
    --t2: #8896b3;
    --t3: #3d4d6b;

    --mono: 'JetBrains Mono', monospace;
    --display: 'Syne', sans-serif;
    --r: 8px;
    --r-lg: 14px;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {
    background: var(--bg) !important;
    color: var(--t1) !important;
    font-family: var(--mono) !important;
}
.block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1280px !important;
}
#MainMenu, footer, [data-testid="stDecoration"],
[data-testid="stToolbar"], [data-testid="stStatusWidget"] { 
    display: none !important; 
}

/* Scrollbar */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--bg-elevated); border-radius: 2px; }

/* ═══════════════ SIDEBAR ═══════════════ */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
    min-width: 260px !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.2rem 1rem !important; }
[data-testid="stSidebar"] * { font-family: var(--mono) !important; }

[data-testid="stSidebar"] label {
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
    color: var(--t3) !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    font-size: 12px !important;
    color: var(--t1) !important;
    transition: border-color 0.15s !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div:hover {
    border-color: var(--border-hov) !important;
}
[data-testid="stSidebar"] hr {
    border-color: var(--border) !important;
    margin: 1rem 0 !important;
}
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] li {
    font-size: 11px !important;
    color: var(--t2) !important;
}
[data-testid="stSidebar"] .stCaption p {
    font-size: 10px !important;
    color: var(--t3) !important;
}

/* ═══════════════ BUTTONS ═══════════════ */
.stButton > button {
    background: var(--accent-dim) !important;
    color: var(--accent) !important;
    border: 1px solid rgba(0,212,255,0.25) !important;
    border-radius: var(--r) !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: rgba(0,212,255,0.18) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 0 20px var(--accent-glow) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(0,212,255,0.2) 0%, rgba(0,229,160,0.15) 100%) !important;
    border-color: rgba(0,212,255,0.4) !important;
    box-shadow: 0 0 24px rgba(0,212,255,0.12) !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 32px rgba(0,212,255,0.25) !important;
}

/* ═══════════════ TABS ═══════════════ */
[data-testid="stTabs"] [role="tablist"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
    padding: 0 !important;
    margin-bottom: 1.5rem !important;
}
[data-testid="stTabs"] button[role="tab"] {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: var(--t3) !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.65rem 1.2rem !important;
    border-radius: 0 !important;
    transition: all 0.15s !important;
    margin-bottom: -1px !important;
}
[data-testid="stTabs"] button[role="tab"]:hover { color: var(--t2) !important; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }

/* ═══════════════ METRICS ═══════════════ */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    padding: 0.85rem 1rem !important;
    transition: border-color 0.15s !important;
}
[data-testid="stMetric"]:hover { border-color: var(--border-hov) !important; }
[data-testid="stMetricLabel"] p {
    font-family: var(--mono) !important;
    font-size: 9.5px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.14em !important;
    color: var(--t3) !important;
    margin-bottom: 4px !important;
    font-weight: 700 !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: var(--t1) !important;
    letter-spacing: -0.02em !important;
}
[data-testid="stMetricDelta"] {
    font-size: 10px !important;
    font-family: var(--mono) !important;
}

/* ═══════════════ EXPANDERS ═══════════════ */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-family: var(--mono) !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    color: var(--t2) !important;
    padding: 0.7rem 1rem !important;
}
[data-testid="stExpander"] summary:hover { color: var(--t1) !important; }

/* ═══════════════ SELECTBOX ═══════════════ */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    border-radius: var(--r) !important;
    color: var(--t1) !important;
    font-family: var(--mono) !important;
    font-size: 12px !important;
}
.stSelectbox label {
    font-size: 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: var(--t3) !important;
    font-weight: 600 !important;
}

/* ═══════════════ ALERTS ═══════════════ */
[data-testid="stAlert"] {
    border-radius: var(--r) !important;
    font-family: var(--mono) !important;
    font-size: 11px !important;
    border-left-width: 3px !important;
}

/* ═══════════════ MARKDOWN ═══════════════ */
.stMarkdown p, .stMarkdown li {
    font-family: var(--mono) !important;
    font-size: 12px !important;
    color: var(--t2) !important;
    line-height: 1.75 !important;
}
.stMarkdown strong { color: var(--t1) !important; font-weight: 600 !important; }
.stMarkdown code {
    background: var(--bg-elevated) !important;
    color: var(--accent) !important;
    padding: 1px 6px !important;
    border-radius: 4px !important;
    font-size: 11px !important;
    font-family: var(--mono) !important;
}
.stMarkdown h3 {
    font-family: var(--display) !important;
    font-size: 0.95rem !important;
    color: var(--t1) !important;
    font-weight: 700 !important;
    margin-top: 1.2rem !important;
}

/* ═══════════════ PLOTLY CHARTS ═══════════════ */
[data-testid="stPlotlyChart"] {
    border: 1px solid var(--border) !important;
    border-radius: var(--r-lg) !important;
    overflow: hidden;
}

/* ═══════════════ SPINNER ═══════════════ */
[data-testid="stSpinner"] > div { border-top-color: var(--accent) !important; }

/* ═══════════════ HR ═══════════════ */
hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* ═══════════════ CAPTION ═══════════════ */
[data-testid="stCaptionContainer"] p {
    font-family: var(--mono) !important;
    font-size: 10px !important;
    color: var(--t3) !important;
    line-height: 1.6 !important;
}

/* ═══════════════ COLUMNS ═══════════════ */
[data-testid="column"] { gap: 0 !important; }

/* ═══════════════ CUSTOM COMPONENTS ═══════════════ */
.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--r-lg);
    padding: 0.85rem 1rem;
    transition: border-color 0.15s, box-shadow 0.15s;
    height: 100%;
}
.stat-card:hover {
    border-color: var(--border-hov);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.stat-card .label {
    font-size: 9.5px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--t3);
    margin-bottom: 7px;
    font-family: var(--mono);
}
.stat-card .value {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--t1);
    font-family: var(--mono);
    line-height: 1;
    margin-bottom: 4px;
    letter-spacing: -0.02em;
}
.stat-card .sub {
    font-size: 10px;
    color: var(--t3);
    font-family: var(--mono);
}
.stat-card.bull { 
    background: var(--bull-dim); 
    border-color: var(--bull-border); 
}
.stat-card.bear { 
    background: var(--bear-dim); 
    border-color: var(--bear-border); 
}
.stat-card.accent { 
    background: var(--accent-dim); 
    border-color: rgba(0,212,255,0.2); 
}

.section-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 2rem 0 1.25rem;
}
.section-divider .title {
    font-family: var(--mono);
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--t1);
    white-space: nowrap;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.section-divider .line {
    flex: 1;
    height: 1px;
    background: var(--border);
}
.section-divider .badge {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    background: var(--accent-dim);
    color: var(--accent);
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 20px;
    padding: 2px 9px;
    white-space: nowrap;
    font-family: var(--mono);
}

.info-panel {
    background: rgba(0,212,255,0.04);
    border: 1px solid rgba(0,212,255,0.12);
    border-left: 3px solid var(--accent);
    border-radius: 0 var(--r) var(--r) 0;
    padding: 12px 16px;
    margin: 10px 0 16px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--t2);
    line-height: 1.7;
}
.info-panel strong { color: var(--t1); }

.empty-state {
    background: var(--bg-card);
    border: 1px dashed var(--border);
    border-radius: var(--r-lg);
    padding: 2.5rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.empty-state .icon { font-size: 2rem; margin-bottom: 0.75rem; opacity: 0.4; }
.empty-state .msg {
    font-family: var(--mono);
    font-size: 12px;
    color: var(--t3);
    line-height: 1.6;
}
.empty-state .hint {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--t3);
    margin-top: 6px;
    opacity: 0.7;
}
</style>
""", unsafe_allow_html=True)
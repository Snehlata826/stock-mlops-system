import streamlit as st


def inject_theme():
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Syne:wght@600;700;800&display=swap" rel="stylesheet">

        <style>
        /* ═══════════════════ VARIABLES ═══════════════════ */
        :root {
            --bg:          #080b10;
            --bg-card:     #0d1117;
            --bg-raised:   #131820;
            --bg-border:   rgba(255,255,255,0.07);

            --teal:        #00c896;
            --teal-glow:   rgba(0,200,150,0.15);
            --red:         #f0424e;
            --red-glow:    rgba(240,66,78,0.12);
            --amber:       #f59e0b;
            --blue:        #3b82f6;

            --t1:  #dde1ea;   /* primary text   */
            --t2:  #7a8099;   /* secondary text  */
            --t3:  #3d4259;   /* muted           */

            --mono: 'JetBrains Mono', monospace;
            --sans: 'Syne', sans-serif;
            --r:    6px;
        }

        /* ═══════════════════ GLOBAL ═══════════════════ */
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"], .main {
            background: var(--bg) !important;
            color: var(--t1) !important;
            font-family: var(--mono) !important;
        }
        #MainMenu, footer, [data-testid="stDecoration"],
        [data-testid="stToolbar"] { display:none !important; }

        /* scrollbar */
        ::-webkit-scrollbar { width:3px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: var(--bg-raised); border-radius:2px; }

        /* ═══════════════════ SIDEBAR ═══════════════════ */
        [data-testid="stSidebar"] {
            background: var(--bg-card) !important;
            border-right: 1px solid var(--bg-border) !important;
        }
        [data-testid="stSidebar"] * { font-family: var(--mono) !important; }

        [data-testid="stSidebar"] label {
            font-size: 10px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.12em !important;
            color: var(--t3) !important;
            font-weight: 500 !important;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: var(--bg-raised) !important;
            border: 1px solid var(--bg-border) !important;
            border-radius: var(--r) !important;
            font-family: var(--mono) !important;
            font-size: 13px !important;
            color: var(--t1) !important;
        }

        /* ═══════════════════ BUTTONS ═══════════════════ */
        .stButton > button {
            background: var(--teal) !important;
            color: #060a0e !important;
            border: none !important;
            border-radius: var(--r) !important;
            font-family: var(--mono) !important;
            font-size: 11px !important;
            font-weight: 700 !important;
            letter-spacing: 0.12em !important;
            text-transform: uppercase !important;
            padding: 0.5rem 1.1rem !important;
            transition: opacity 0.15s, box-shadow 0.15s !important;
            box-shadow: 0 0 18px rgba(0,200,150,0.18) !important;
        }
        .stButton > button:hover {
            opacity: 0.88 !important;
            box-shadow: 0 0 28px rgba(0,200,150,0.35) !important;
        }

        /* ═══════════════════ TABS ═══════════════════ */
        [data-testid="stTabs"] [role="tablist"] {
            background: transparent !important;
            border-bottom: 1px solid var(--bg-border) !important;
            gap: 0 !important;
        }
        [data-testid="stTabs"] button[role="tab"] {
            font-family: var(--mono) !important;
            font-size: 11px !important;
            font-weight: 500 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1em !important;
            color: var(--t3) !important;
            background: transparent !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            padding: 0.6rem 1.1rem !important;
            border-radius: 0 !important;
            transition: color 0.15s, border-color 0.15s !important;
        }
        [data-testid="stTabs"] button[role="tab"]:hover {
            color: var(--t2) !important;
        }
        [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: var(--teal) !important;
            border-bottom: 2px solid var(--teal) !important;
            background: transparent !important;
        }
        /* hide the red ink line Streamlit adds */
        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            display: none !important;
        }
        [data-testid="stTabs"] [data-baseweb="tab-border"] {
            display: none !important;
        }

        /* ═══════════════════ METRICS ═══════════════════ */
        [data-testid="stMetric"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--bg-border) !important;
            border-radius: var(--r) !important;
            padding: 1rem 1.2rem !important;
        }
        [data-testid="stMetricLabel"] p {
            font-family: var(--mono) !important;
            font-size: 10px !important;
            text-transform: uppercase !important;
            letter-spacing: 0.12em !important;
            color: var(--t3) !important;
        }
        [data-testid="stMetricValue"] {
            font-family: var(--sans) !important;
            font-size: 1.5rem !important;
            font-weight: 700 !important;
            color: var(--t1) !important;
        }

        /* ═══════════════════ EXPANDER ═══════════════════ */
        [data-testid="stExpander"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--bg-border) !important;
            border-radius: var(--r) !important;
        }
        [data-testid="stExpander"] summary {
            font-family: var(--mono) !important;
            font-size: 11px !important;
            color: var(--t3) !important;
        }
        [data-testid="stExpander"] summary:hover {
            color: var(--t2) !important;
        }

        /* ═══════════════════ PLOTLY CHARTS ═══════════════════ */
        [data-testid="stPlotlyChart"] {
            border: 1px solid var(--bg-border) !important;
            border-radius: var(--r) !important;
            overflow: hidden;
            background: var(--bg-card) !important;
        }

        /* ═══════════════════ ALERTS / TOASTS ═══════════════════ */
        /* Remove all default streamlit alert styling */
        [data-testid="stAlert"] {
            border-radius: var(--r) !important;
            font-family: var(--mono) !important;
            font-size: 12px !important;
        }

        /* ═══════════════════ MARKDOWN ═══════════════════ */
        .stMarkdown p, .stMarkdown li {
            font-family: var(--mono) !important;
            font-size: 12px !important;
            color: var(--t2) !important;
            line-height: 1.7 !important;
        }
        .stMarkdown strong {
            color: var(--t1) !important;
            font-weight: 600 !important;
        }
        .stMarkdown code {
            background: var(--bg-raised) !important;
            color: var(--teal) !important;
            padding: 1px 5px !important;
            border-radius: 3px !important;
            font-size: 11px !important;
        }

        /* ═══════════════════ SPINNER ═══════════════════ */
        [data-testid="stSpinner"] > div {
            border-top-color: var(--teal) !important;
        }

        /* ═══════════════════ HR ═══════════════════ */
        hr {
            border: none !important;
            border-top: 1px solid var(--bg-border) !important;
            margin: 1.5rem 0 !important;
        }

        /* ═══════════════════ CAPTION ═══════════════════ */
        [data-testid="stCaptionContainer"] p, .stCaption {
            font-family: var(--mono) !important;
            font-size: 10px !important;
            color: var(--t3) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

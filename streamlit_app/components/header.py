import streamlit as st

def render_header(ticker: str, interval: str, horizon: str):
    st.markdown("""
    <style>
    .header-container {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d1117 100%);
        border: 1px solid rgba(99, 179, 237, 0.2);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .header-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #63b3ed, #68d391, #f6e05e, #fc8181);
    }
    .header-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #63b3ed 0%, #68d391 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 8px 0;
        line-height: 1.2;
    }
    .header-sub {
        color: rgba(255,255,255,0.55);
        font-size: 0.9rem;
        letter-spacing: 0.5px;
        margin: 0;
    }
    .header-badge {
        display: inline-block;
        background: rgba(99, 179, 237, 0.12);
        border: 1px solid rgba(99, 179, 237, 0.3);
        color: #63b3ed;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-right: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="header-container">
        <p class="header-title">📈 Stock Prediction System</p>
        <p class="header-sub">
            <span class="header-badge">{ticker}</span>
            <span class="header-badge">{interval} candles</span>
            <span class="header-badge">{horizon} horizon</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

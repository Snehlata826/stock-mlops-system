"""
Reusable UI primitives for the Stock MLOps Dashboard.
All components use the design system defined in theme.py.
"""
import streamlit as st


def stat_card(label: str, value: str, sub: str = "", variant: str = "default"):
    """Render a styled stat card. variant: default | bull | bear | accent"""
    cls = f"stat-card {variant}" if variant != "default" else "stat-card"
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    <div class="{cls}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, badge: str = ""):
    """Render a styled section divider with optional badge."""
    badge_html = f'<span class="badge">{badge}</span>' if badge else ""
    st.markdown(f"""
    <div class="section-divider">
        <span class="title">{title}</span>
        <span class="line"></span>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


def info_panel(content: str):
    """Render a styled info/help panel."""
    st.markdown(f'<div class="info-panel">{content}</div>', unsafe_allow_html=True)


def empty_state(icon: str, message: str, hint: str = ""):
    hint_html = f'<div class="hint">{hint}</div>' if hint else ""
    st.markdown(f"""
    <div class="empty-state">
        <div class="icon">{icon}</div>
        <div class="msg">{message}</div>
        {hint_html}
    </div>
    <style>
    .empty-state .msg {{
        color: #8896b3 !important;
        font-size: 13px !important;
    }}
    .empty-state .hint {{
        color: #6b7a99 !important;
        font-size: 12px !important;
    }}
    .empty-state .icon {{
        opacity: 0.7 !important;
    }}
    </style>
    """, unsafe_allow_html=True)

def metric_row(metrics: list):
    """
    Render a row of stat cards.
    metrics: list of dicts with keys: label, value, sub, variant
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            stat_card(
                label=m.get("label", ""),
                value=m.get("value", "—"),
                sub=m.get("sub", ""),
                variant=m.get("variant", "default"),
            )
    # Spacer
    st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)
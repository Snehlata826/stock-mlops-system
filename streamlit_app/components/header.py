"""
Page header: displays ticker, interval, horizon as a clean banner.
"""
import streamlit as st


def render_header(ticker: str, interval: str, horizon: str):
    badge = (
        "<span style='background:rgba(0,212,255,0.1);border:1px solid rgba(0,212,255,0.22);"
        "color:#00d4ff;border-radius:20px;padding:4px 14px;font-size:11px;font-weight:700;"
        "letter-spacing:0.05em;font-family:JetBrains Mono,monospace;margin-right:6px;'>"
        f"{ticker}</span>"
        "<span style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);"
        "color:#8896b3;border-radius:20px;padding:4px 14px;font-size:11px;font-weight:600;"
        "font-family:JetBrains Mono,monospace;margin-right:6px;'>"
        f"{interval}</span>"
        "<span style='background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);"
        "color:#8896b3;border-radius:20px;padding:4px 14px;font-size:11px;font-weight:600;"
        "font-family:JetBrains Mono,monospace;'>"
        f"{horizon}</span>"
    )

    st.markdown(
        "<div style='background:linear-gradient(135deg,#080e1a 0%,#0c1624 60%,#07101e 100%);"
        "border:1px solid rgba(0,212,255,0.12);border-radius:14px;padding:1.4rem 1.8rem;"
        "margin-bottom:1.4rem;position:relative;overflow:hidden;'>"
        "<div style='position:absolute;top:0;left:0;right:0;height:2px;"
        "background:linear-gradient(90deg,#00d4ff,#00e5a0,transparent);'></div>"
        "<div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;'>"
        "<div>"
        "<div style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;"
        "background:linear-gradient(135deg,#00d4ff,#00e5a0);-webkit-background-clip:text;"
        "-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1;margin-bottom:5px;'>"
        "&#128200; Stock MLOps Dashboard</div>"
        "<div style='font-size:11px;color:#3d4d6b;font-family:JetBrains Mono,monospace;'>"
        "XGBoost &middot; MLflow &middot; Walk-forward &middot; Evidently</div>"
        "</div>"
        f"<div>{badge}</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )
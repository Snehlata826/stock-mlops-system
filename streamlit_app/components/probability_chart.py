"""
Probability conviction chart: bullish vs bearish probability over time.
"""
import plotly.graph_objects as go
import streamlit as st
from components.theme import apply_chart_theme


def render_probability(predictions):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=predictions["Date"],
        y=predictions["prob_up"],
        name="Bullish",
        line=dict(color="#00e5a0", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,229,160,0.07)",
        hovertemplate="%{y:.1%}<extra>Bullish</extra>",
    ))

    fig.add_trace(go.Scatter(
        x=predictions["Date"],
        y=predictions["prob_down"],
        name="Bearish",
        line=dict(color="#ff4d6d", width=2),
        fill="tozeroy",
        fillcolor="rgba(255,77,109,0.07)",
        hovertemplate="%{y:.1%}<extra>Bearish</extra>",
    ))

    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="rgba(255,255,255,0.15)",
        annotation_text="50%",
        annotation_font_color="rgba(255,255,255,0.3)",
        annotation_font_size=10,
    )

    apply_chart_theme(fig, height=320, title="Conviction Over Time")
    fig.update_layout(
        yaxis=dict(
            tickformat=".0%",
            range=[0, 1],
        ),
    )

st.plotly_chart(fig, use_container_width=True, config={
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
    "scrollZoom": True,
})
   
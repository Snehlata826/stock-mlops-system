import plotly.graph_objects as go
import streamlit as st


def render_probability(predictions):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=predictions["Date"],
        y=predictions["prob_up"],
        name="Bullish probability",
        line=dict(color="#68d391", width=2),
        fill="tozeroy",
        fillcolor="rgba(104,211,145,0.08)",
    ))

    fig.add_trace(go.Scatter(
        x=predictions["Date"],
        y=predictions["prob_down"],
        name="Bearish probability",
        line=dict(color="#fc8181", width=2),
        fill="tozeroy",
        fillcolor="rgba(252,129,129,0.08)",
    ))

    fig.add_hline(
        y=0.5,
        line_dash="dot",
        line_color="rgba(255,255,255,0.2)",
        annotation_text="50% threshold",
        annotation_font_color="rgba(255,255,255,0.4)",
    )

    fig.update_layout(
        title=dict(text="Model conviction over time", font=dict(size=15, color="#e2e8f0")),
        height=340,
        hovermode="x unified",
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font=dict(color="#a0aec0"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            range=[0, 1],
            tickformat=".0%",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

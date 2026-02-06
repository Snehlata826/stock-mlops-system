import plotly.graph_objects as go
import streamlit as st

def render_probability(predictions):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=predictions["Date"],
        y=predictions["prob_up"],
        name="Bullish Probability",
        line=dict(color="green")
    ))

    fig.add_trace(go.Scatter(
        x=predictions["Date"],
        y=predictions["prob_down"],
        name="Bearish Probability",
        line=dict(color="red")
    ))

    fig.update_layout(
        title="Short-Term Probability Conviction (per interval)",
        height=400,
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

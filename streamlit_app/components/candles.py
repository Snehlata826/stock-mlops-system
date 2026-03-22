import plotly.graph_objects as go
import streamlit as st


def render_candles(df, ticker):
    fig = go.Figure(go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        increasing_line_color="#68d391",
        increasing_fillcolor="#68d391",
        decreasing_line_color="#fc8181",
        decreasing_fillcolor="#fc8181",
        name="Price"
    ))

    if "Volume" in df.columns:
        fig.add_trace(go.Bar(
            x=df["Date"],
            y=df["Volume"],
            name="Volume",
            marker_color="rgba(99,179,237,0.25)",
            yaxis="y2",
        ))

    fig.update_layout(
        title=dict(text=f"{ticker} — Price Action", font=dict(size=16, color="#e2e8f0")),
        height=480,
        xaxis_rangeslider_visible=False,
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font=dict(color="#a0aec0"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            showgrid=True,
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            showgrid=True,
            title="Price",
            domain=[0.25, 1.0],
        ),
        yaxis2=dict(
            title="Volume",
            domain=[0.0, 0.2],
            showgrid=False,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

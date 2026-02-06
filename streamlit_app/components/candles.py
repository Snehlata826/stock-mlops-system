import plotly.graph_objects as go
import streamlit as st

def render_candles(df, ticker):
    fig = go.Figure(go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        increasing_line_color="green",
        decreasing_line_color="red"
    ))

    fig.update_layout(
        title=f"{ticker} Price Action",
        height=450,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

import streamlit as st

def render_header(ticker, interval, horizon):
    st.title("📈 Stock Prediction System")
    st.markdown(
        f"**{ticker} · {interval} candles · {horizon} forecast**"
    )

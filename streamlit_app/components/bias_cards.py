import streamlit as st

def render_bias_cards(summary):
    bias = "Bullish" if summary["direction"] == "UP" else "Bearish"
    color = "🟢" if bias == "Bullish" else "🔴"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market Bias", f"{color} {bias}")
    c2.metric("Confidence", f"{summary['confidence']:.2%}")
    c3.metric("Forecast Horizon", summary["horizon"])
    c4.metric("Intervals Aggregrated", summary["window"])
    st.caption(
          "Confidence represents probabilistic edge, not certainty. "
        "Values above ~55% .indicate directional bias."
    )
    

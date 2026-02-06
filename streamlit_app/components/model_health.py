import streamlit as st

def render_model_health(drift):
    drifted = drift["n_drifted_columns"]
    total = drift["total_columns"]
    drift_pct = drift["drift_percentage"]

    

    c1, c2, c3 = st.columns(3)
    c1.metric("Drifted Features", drifted)
    c2.metric("Total Features", total)
    c3.metric("Drift %", f"{drift_pct:.1f}%")

    # -------- Health Logic --------
    if drift_pct < 20:
        st.success("🟢 Model Stable\n\nPredictions are reliable.")
    elif drift_pct < 30:
        st.warning(
            "🟡 Model Degrading\n\n"
            "Market behavior is changing. Monitor closely."
        )
    else:
        st.error(
            "🔴 Model Unstable\n\n"
            "Predictions may be unreliable.\n"
            "Retraining is strongly recommended."
        )

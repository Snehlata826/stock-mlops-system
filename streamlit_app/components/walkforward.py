import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def render_walkforward_results(results: dict):
    agg = results["aggregate"]
    folds = results["folds"]

    st.markdown(f"""
    <style>
    .wf-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 18px;
    }}
    .wf-card {{
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 14px 18px;
    }}
    .wf-label {{
        font-size: 0.66rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.35);
        margin-bottom: 6px;
    }}
    .wf-value {{
        font-size: 1.4rem;
        font-weight: 800;
        color: #63b3ed;
    }}
    .wf-std {{
        font-size: 0.75rem;
        color: rgba(255,255,255,0.35);
        margin-top: 2px;
    }}
    </style>
    <div class="wf-grid">
        <div class="wf-card">
            <div class="wf-label">Accuracy (WF)</div>
            <div class="wf-value">{agg['accuracy_mean']:.1%}</div>
            <div class="wf-std">± {agg['accuracy_std']:.1%} across {agg['n_folds']} folds</div>
        </div>
        <div class="wf-card">
            <div class="wf-label">ROC-AUC (WF)</div>
            <div class="wf-value">{agg['roc_auc_mean']:.3f}</div>
            <div class="wf-std">± {agg['roc_auc_std']:.3f}</div>
        </div>
        <div class="wf-card">
            <div class="wf-label">F1 Score (WF)</div>
            <div class="wf-value">{agg['f1_mean']:.3f}</div>
            <div class="wf-std">± {agg['f1_std']:.3f}</div>
        </div>
        <div class="wf-card">
            <div class="wf-label">MAE (prob)</div>
            <div class="wf-value">{agg['mae_mean']:.4f}</div>
            <div class="wf-std">Mean absolute error on probabilities</div>
        </div>
        <div class="wf-card">
            <div class="wf-label">RMSE (prob)</div>
            <div class="wf-value">{agg['rmse_mean']:.4f}</div>
            <div class="wf-std">Root mean squared error</div>
        </div>
        <div class="wf-card">
            <div class="wf-label">MAPE</div>
            <div class="wf-value">{agg['mape_mean']:.1f}%</div>
            <div class="wf-std">Mean absolute pct error</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Fold-level accuracy chart
    fold_df = pd.DataFrame(folds)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"Fold {f['fold']}" for f in folds],
        y=[f["accuracy"] for f in folds],
        marker_color="#63b3ed",
        name="Accuracy",
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.3)",
                  annotation_text="Random baseline")
    fig.add_hline(y=agg["accuracy_mean"], line_dash="dash", line_color="#68d391",
                  annotation_text=f"Mean: {agg['accuracy_mean']:.1%}")
    fig.update_layout(
        title=dict(text="Accuracy per walk-forward fold", font=dict(size=14, color="#e2e8f0")),
        height=260,
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font=dict(color="#a0aec0"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[0, 1]),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_baseline_comparison(baselines: dict, wf_accuracy: float):
    st.markdown("""
    <style>
    .baseline-note {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.45);
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    rows = []
    rows.append({"Model": "Random guess", "Accuracy": 0.50, "F1": "—", "Type": "Baseline"})

    naive = baselines.get("naive", {})
    if naive and "accuracy" in naive:
        rows.append({"Model": "Naive (momentum)", "Accuracy": naive["accuracy"],
                     "F1": f"{naive['f1']:.3f}", "Type": "Baseline"})

    arima = baselines.get("arima", {})
    if arima and "accuracy" in arima:
        rows.append({"Model": arima.get("model", "ARIMA"), "Accuracy": arima["accuracy"],
                     "F1": f"{arima.get('f1', 0):.3f}", "Type": "Statistical"})

    rows.append({"Model": "XGBoost (walk-forward)", "Accuracy": wf_accuracy,
                 "F1": "—", "Type": "Our model"})

    fig = go.Figure()
    colors = {"Baseline": "#888", "Statistical": "#f6e05e", "Our model": "#68d391"}
    for row in rows:
        fig.add_trace(go.Bar(
            x=[row["Model"]],
            y=[row["Accuracy"]],
            marker_color=colors.get(row["Type"], "#888"),
            name=row["Type"],
            showlegend=False,
        ))

    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.25)",
                  annotation_text="50% (random)")
    fig.update_layout(
        title=dict(text="Model vs baselines", font=dict(size=14, color="#e2e8f0")),
        height=280,
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font=dict(color="#a0aec0"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", range=[0.3, 0.8], title="Accuracy"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<p class="baseline-note">Green = our XGBoost model. Yellow = ARIMA statistical baseline. Gray = naive baselines.</p>', unsafe_allow_html=True)

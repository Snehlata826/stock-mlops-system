"""
Walk-forward validation results: metrics cards + fold chart + baseline comparison.
"""
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from components.theme import apply_chart_theme
from components.ui import metric_row, section_header


def render_walkforward_results(results: dict):
    agg   = results["aggregate"]
    folds = results["folds"]

    # Aggregate metric cards — 3 + 2 layout
    metric_row([
        {"label": "Accuracy (WF)",  "value": f"{agg['accuracy_mean']:.1%}",
         "sub": f"±{agg['accuracy_std']:.1%} · {agg['n_folds']} folds", "variant": "accent"},
        {"label": "ROC-AUC",        "value": f"{agg['roc_auc_mean']:.3f}",
         "sub": f"±{agg['roc_auc_std']:.3f}"},
        {"label": "F1 Score",       "value": f"{agg['f1_mean']:.3f}",
         "sub": f"±{agg['f1_std']:.3f}"},
    ])
    metric_row([
        {"label": "MAE (prob)",     "value": f"{agg['mae_mean']:.4f}",
         "sub": "Mean absolute error on probabilities"},
        {"label": "RMSE (prob)",    "value": f"{agg['rmse_mean']:.4f}",
         "sub": "Root mean squared error"},
        {"label": "Overall AUC",    "value": f"{agg['overall_roc_auc']:.3f}",
         "sub": "Across all folds combined"},
    ])

    # Per-fold accuracy chart
    fig = go.Figure()
    fold_nums  = [f"Fold {f['fold']}" for f in folds]
    accuracies = [f["accuracy"]       for f in folds]
    roc_aucs   = [f["roc_auc"]        for f in folds]

    fig.add_trace(go.Bar(
        x=fold_nums, y=accuracies,
        name="Accuracy",
        marker=dict(
            color=accuracies,
            colorscale=[[0, "#3d4d6b"], [0.5, "#00d4ff"], [1, "#00e5a0"]],
            cmin=0.4, cmax=0.75,
        ),
        hovertemplate="%{y:.1%}<extra>Accuracy</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=fold_nums, y=roc_aucs,
        name="ROC-AUC",
        line=dict(color="#ffb547", width=2, dash="dot"),
        mode="lines+markers",
        marker=dict(size=6),
        yaxis="y2",
        hovertemplate="%{y:.3f}<extra>ROC-AUC</extra>",
    ))

    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.15)",
                  annotation_text="Random", annotation_font_size=9,
                  annotation_font_color="rgba(255,255,255,0.3)")
    fig.add_hline(y=agg["accuracy_mean"], line_dash="dash", line_color="#00e5a0",
                  annotation_text=f"Mean {agg['accuracy_mean']:.1%}",
                  annotation_font_size=9, annotation_font_color="#00e5a0")

    apply_chart_theme(fig, height=260, title="Per-Fold Accuracy & ROC-AUC")
    fig.update_layout(
        yaxis=dict(range=[0.3, 0.85], tickformat=".0%"),
        yaxis2=dict(overlaying="y", side="right", range=[0.3, 0.9],
                    showgrid=False, tickformat=".2f",
                    title="ROC-AUC", titlefont=dict(size=9)),
        barmode="group",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Fold detail table in expander
    with st.expander("📋  Fold-level detail"):
        fold_df = pd.DataFrame(folds)[[
            "fold", "train_size", "test_size",
            "accuracy", "roc_auc", "f1", "mae", "rmse"
        ]].copy()
        fold_df.columns = ["Fold", "Train", "Test", "Accuracy", "ROC-AUC", "F1", "MAE", "RMSE"]
        for col in ["Accuracy", "F1"]:
            fold_df[col] = fold_df[col].map("{:.1%}".format)
        for col in ["ROC-AUC", "MAE", "RMSE"]:
            fold_df[col] = fold_df[col].map("{:.4f}".format)
        st.dataframe(fold_df, use_container_width=True, hide_index=True)


def render_baseline_comparison(baselines: dict, wf_accuracy: float):
    rows = [
        {"Model": "Random Guess",            "Accuracy": 0.50,        "Type": "baseline"},
        {"Model": "Naive (Momentum)",         "Accuracy": baselines.get("naive", {}).get("accuracy", 0), "Type": "baseline"},
    ]
    arima = baselines.get("arima", {})
    if arima and "accuracy" in arima:
        rows.append({"Model": arima.get("model", "ARIMA"), "Accuracy": arima["accuracy"], "Type": "statistical"})
    rows.append({"Model": "XGBoost (Walk-fwd)", "Accuracy": wf_accuracy, "Type": "ours"})

    color_map = {"baseline": "#3d4d6b", "statistical": "#ffb547", "ours": "#00e5a0"}
    fig = go.Figure()
    for row in rows:
        fig.add_trace(go.Bar(
            x=[row["Model"]], y=[row["Accuracy"]],
            marker_color=color_map[row["Type"]],
            showlegend=False,
            hovertemplate=f"{row['Model']}: %{{y:.1%}}<extra></extra>",
        ))

    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.15)",
                  annotation_text="Random 50%", annotation_font_size=9,
                  annotation_font_color="rgba(255,255,255,0.3)")

    apply_chart_theme(fig, height=260, title="XGBoost vs Baselines")
    fig.update_layout(
        yaxis=dict(range=[0.3, 0.85], tickformat=".0%", title="Accuracy"),
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("🟢 XGBoost (ours)  ·  🟡 ARIMA statistical  ·  ⬛ Naive baselines")
"""
Backtest results: metrics + equity curve.
"""
import streamlit as st
import plotly.graph_objects as go
from components.theme import apply_chart_theme
from components.ui import metric_row


def render_backtest(result: dict):
    if "error" in result:
        st.error(f"Backtest failed: {result['error']}")
        return

    m  = result["metrics"]
    tl = result["trade_log"]

    is_alpha_pos  = m["alpha"] >= 0
    is_strat_pos  = m["total_return_strategy"] >= 0
    alpha_variant = "bull" if is_alpha_pos else "bear"
    strat_variant = "bull" if is_strat_pos else "bear"

    # Top metric row
    metric_row([
        {
            "label":   "Strategy Return",
            "value":   f"{m['total_return_strategy']:+.1%}",
            "sub":     f"vs B&H: {m['total_return_bah']:+.1%}",
            "variant": strat_variant,
        },
        {
            "label":   "Alpha vs B&H",
            "value":   f"{m['alpha']:+.1%}",
            "sub":     "Outperformance",
            "variant": alpha_variant,
        },
        {
            "label": "Sharpe Ratio",
            "value": f"{m['sharpe_strategy']:.2f}",
            "sub":   f"B&H: {m['sharpe_bah']:.2f}",
        },
        {
            "label":   "Max Drawdown",
            "value":   f"{m['max_drawdown_strategy']:.1%}",
            "sub":     f"B&H: {m['max_drawdown_bah']:.1%}",
            "variant": "bear",
        },
    ])

    # Secondary row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Win Rate",    f"{m['win_rate']:.1%}")
    with col2:
        st.metric("Total Trades", str(m["n_trades"]))
    with col3:
        st.metric("Test Days",   str(m["n_days"]))

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Equity curve
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=tl["date"].astype(str),
        y=tl["cum_strategy"],
        name=m["strategy"],
        line=dict(color="#00e5a0", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0,229,160,0.04)",
        hovertemplate="%{y:.3f}×<extra>Strategy</extra>",
    ))

    fig.add_trace(go.Scatter(
        x=tl["date"].astype(str),
        y=tl["cum_bah"],
        name="Buy & Hold",
        line=dict(color="#00d4ff", width=1.5, dash="dot"),
        hovertemplate="%{y:.3f}×<extra>Buy & Hold</extra>",
    ))

    # Drawdown fill
    fig.add_trace(go.Scatter(
        x=tl["date"].astype(str),
        y=tl["cum_strategy"],
        fill="tozeroy",
        fillcolor="rgba(255,77,109,0.03)",
        line=dict(color="rgba(0,0,0,0)", width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    apply_chart_theme(fig, height=320, title="Equity Curve — Out-of-Sample Backtest")
    fig.update_layout(
        yaxis=dict(title="Cumulative Return (×)", tickformat=".2f"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.caption(
        "⚠  Backtest is run on the held-out 20% of training data — never seen during model fitting. "
        "Past performance does not guarantee future results."
    )
import streamlit as st
import plotly.graph_objects as go


def render_backtest(result: dict):
    if "error" in result:
        st.error(f"Backtest failed: {result['error']}")
        return

    m = result["metrics"]
    tl = result["trade_log"]

    alpha_color = "#68d391" if m["alpha"] >= 0 else "#fc8181"
    strat_color = "#68d391" if m["total_return_strategy"] >= 0 else "#fc8181"

    st.markdown(f"""
    <style>
    .bt-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 18px;
    }}
    .bt-card {{
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 14px 16px;
    }}
    .bt-label {{
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.35);
        margin-bottom: 6px;
    }}
    .bt-value {{
        font-size: 1.3rem;
        font-weight: 800;
        color: #e2e8f0;
    }}
    .bt-sub {{
        font-size: 0.72rem;
        color: rgba(255,255,255,0.35);
        margin-top: 3px;
    }}
    </style>
    <div class="bt-grid">
        <div class="bt-card">
            <div class="bt-label">Strategy Return</div>
            <div class="bt-value" style="color:{strat_color}">{m['total_return_strategy']:.1%}</div>
            <div class="bt-sub">vs B&H: {m['total_return_bah']:.1%}</div>
        </div>
        <div class="bt-card">
            <div class="bt-label">Alpha vs B&H</div>
            <div class="bt-value" style="color:{alpha_color}">{m['alpha']:+.1%}</div>
            <div class="bt-sub">Outperformance</div>
        </div>
        <div class="bt-card">
            <div class="bt-label">Sharpe Ratio</div>
            <div class="bt-value">{m['sharpe_strategy']:.2f}</div>
            <div class="bt-sub">B&H: {m['sharpe_bah']:.2f}</div>
        </div>
        <div class="bt-card">
            <div class="bt-label">Max Drawdown</div>
            <div class="bt-value" style="color:#fc8181">{m['max_drawdown_strategy']:.1%}</div>
            <div class="bt-sub">B&H: {m['max_drawdown_bah']:.1%}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Win Rate", f"{m['win_rate']:.1%}")
    with col2:
        st.metric("Trades", str(m["n_trades"]))

    # Equity curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tl["date"].astype(str),
        y=tl["cum_strategy"],
        name=m["strategy"],
        line=dict(color="#68d391", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=tl["date"].astype(str),
        y=tl["cum_bah"],
        name="Buy & Hold",
        line=dict(color="#63b3ed", width=1.5, dash="dot"),
    ))
    fig.update_layout(
        title=dict(text="Equity curve (backtest period)", font=dict(size=14, color="#e2e8f0")),
        height=320,
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        font=dict(color="#a0aec0"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Cumulative return (×)"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "⚠️ Backtest uses the held-out 20% of training data — never seen during model fitting. "
        "Past performance does not guarantee future results."
    )

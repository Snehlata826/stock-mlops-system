# 📈 Stock MLOps System v2.1

A **production-ready MLOps pipeline** for stock price direction prediction using XGBoost, MLflow, Evidently AI, and Streamlit — with a FastAPI backend, walk-forward validation, backtesting, and real-time drift monitoring.

> ⚠️ **Disclaimer:** For educational purposes only. Not financial advice. Past performance does not guarantee future results.

---

## 🎯 Live Demo

**[▶ Launch Live Dashboard](https://stock-mlops-system-jbr8pfensdgxappkr9v9n9a.streamlit.app/)**

*Deployed on Streamlit Cloud. Supports AAPL, TSLA, MSFT, NVDA, GLD, SPY, IBIT, QQQ with real-time predictions.*

---

## ✨ Features

- **XGBoost classifier** trained on 40+ engineered technical indicators
- **MLflow model registry** with Production → Staging → Latest fallback
- **Walk-forward validation** — no data leakage from random splits
- **Backtesting** with equity curve, Sharpe ratio, max drawdown, win rate
- **ARIMA & naive baselines** for honest model benchmarking
- **Evidently AI drift monitoring** with fallback KS-test detection
- **FastAPI REST backend** with full async pipeline support
- **Streamlit dashboard** with tabbed dark-theme UI
- **Docker Compose** setup for one-command deployment
- **ngrok tunnelling** support for Streamlit Cloud connectivity

---

## 🏗️ Architecture

```
Data Ingestion → Feature Engineering → Training → Inference → Monitoring
      │                  │                │             │            │
  yfinance           40+ features      XGBoost      MLflow       Evidently
  Alpha Vantage      SMA/EMA/RSI       Walk-fwd     Registry     Drift KS
                     MACD/BB/ATR       Backtest     FastAPI      HTML report
                     Stochastic        Baselines    Streamlit
```

---

## 📁 Project Structure

```
stock-mlops-system/
├── src/
│   ├── common/
│   │   ├── config.py                    # All configuration & per-asset XGBoost params
│   │   ├── companies.py                 # Supported tickers (AAPL, TSLA, MSFT, NVDA, ...)
│   │   └── utils.py                     # Logger, dataframe & asset validators
│   ├── ingestion/
│   │   ├── fetch_historical.py          # yfinance + Alpha Vantage fallback
│   │   └── fetch_realtime.py            # Intraday data with TTL caching
│   ├── features/
│   │   └── feature_engineering.py       # 40+ technical indicators + target creation
│   ├── training/
│   │   ├── train.py                     # XGBoost + MLflow (time-ordered split)
│   │   ├── evaluate.py                  # Confusion matrix, ROC, feature importance plots
│   │   ├── walk_forward_validation.py   # No-leakage walk-forward validation
│   │   ├── baseline_models.py           # ARIMA + naive momentum baselines
│   │   └── backtesting.py               # Strategy simulation (long-only, long/short, B&H)
│   ├── inference/
│   │   ├── model_loader.py              # Production → Staging → Latest fallback loader
│   │   └── predict.py                   # Batch prediction pipeline
│   ├── monitoring/
│   │   └── drift_monitor.py             # Evidently drift detection + KS-test fallback
│   └── api.py                           # FastAPI REST endpoint (v2.1)
├── streamlit_app/
│   ├── app.py                           # Main dashboard — tabbed UI, API-first
│   ├── api_client.py                    # HTTP client wrapping all API endpoints
│   └── components/
│       ├── theme.py                     # Design tokens + CSS injection
│       ├── ui.py                        # Reusable primitives (cards, headers, etc.)
│       ├── sidebar.py                   # Asset/interval/horizon controls
│       ├── header.py                    # Page banner
│       ├── candles.py                   # Candlestick + volume chart
│       ├── bias_cards.py                # Bullish/bearish summary cards
│       ├── probability_chart.py         # Conviction over time
│       ├── model_health.py              # Drift health cards + progress bar
│       ├── walkforward.py               # WF results, fold chart, baseline comparison
│       └── backtest.py                  # Equity curve + metrics
├── docker/
│   ├── Dockerfile                       # Single-container (Streamlit only)
│   ├── Dockerfile.backend               # Multi-stage backend image
│   └── Dockerfile.frontend              # Multi-stage frontend image
├── docker-compose.yml                   # MLflow + backend services
├── scripts/
│   ├── run_training.sh                  # Full 6-step training pipeline
│   ├── run_streamlit.sh                 # Launch dashboard
│   ├── run_api.sh                       # Launch FastAPI
│   └── retrain_cron.sh                  # Scheduled retraining helper
├── data/
│   ├── raw/                             # OHLCV CSVs (gitignored)
│   ├── processed/                       # Engineered feature CSVs (gitignored)
│   └── drift/                           # Reference data for monitoring (gitignored)
├── reports/                             # Evidently HTML drift reports
├── evaluation/                          # ROC, confusion matrix, feature importance plots
├── mlruns/                              # MLflow experiment tracking (gitignored)
├── requirements.txt
├── requirements_streamlit.txt           # Frontend-only dependencies
├── .env.example
└── start.ps1                            # Windows Docker + ngrok launcher
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or 3.11
- Docker Desktop (for Docker path) or bare Python (for local path)

---

### Option A — Local Python

**1. Set up environment**

```bash
python3.10 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

**2. (Optional) Add API key**

```bash
cp .env.example .env
# Paste your free key from https://www.alphavantage.co
```

Without a key the system falls back to historical data — fully functional.

**3. Start MLflow**

```bash
mlflow ui --port 5000
# UI → http://localhost:5000
```

**4. Train a model**

```bash
bash scripts/run_training.sh AAPL 2y
```

This runs all 6 pipeline steps automatically:

| Step | Command |
|------|---------|
| 1 | Fetch 2 years of OHLCV data |
| 2 | Engineer 40+ technical features |
| 3 | Train XGBoost (time-ordered split, no shuffle) |
| 4 | Evaluate & save ROC / confusion matrix / feature importance |
| 5 | Walk-forward validation (5 folds) |
| 6 | ARIMA + naive baselines |

**5. Promote model to Production**

```bash
# Python (quick)
python - <<'EOF'
import mlflow
mlflow.set_tracking_uri("file:mlruns")
client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name="stock_direction_predictor_AAPL",
    version="1",
    stage="Production"
)
EOF
```

Or use the MLflow UI at http://localhost:5000 → **Models** → select version → **Transition to Production**.

**6. Launch the dashboard**

```bash
bash scripts/run_streamlit.sh
# Dashboard → http://localhost:8501
```

**7. (Optional) Launch the API**

```bash
bash scripts/run_api.sh
# API docs → http://localhost:8000/docs
```

---

### Option B — Docker Compose

```bash
# Build and start MLflow + backend
docker compose up backend mlflow -d

# Then launch Streamlit (separate terminal)
bash scripts/run_streamlit.sh
```

For full containerised deployment:

```bash
docker build -t stock-mlops -f docker/Dockerfile .

docker run -p 8501:8501 \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/data:/app/data \
  stock-mlops
```

---

## 📊 Dashboard Tabs

| Tab | What it shows |
|-----|---------------|
| **📊 Price Action** | Candlestick chart + volume, period summary metrics |
| **🔮 Model Outlook** | Bullish/bearish bias card, confidence score, conviction chart |
| **📐 Validation** | Walk-forward metrics (MAE/RMSE/MAPE/AUC), baseline comparison |
| **💹 Backtest** | Equity curve, Sharpe ratio, max drawdown, win rate, alpha vs B&H |
| **🩺 Health** | Feature drift detection, Evidently HTML report link |

---

## 🌐 FastAPI Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check + supported assets |
| `GET` | `/health` | Simple ping |
| `GET` | `/readyz` | Readiness check (model loader) |
| `POST` | `/predict` | Get predictions for a ticker |
| `GET` | `/assets` | List supported tickers |
| `GET` | `/assets/{ticker}/status` | Check data readiness |
| `GET` | `/assets/{ticker}/price` | Raw OHLCV data |
| `POST` | `/assets/{ticker}/run_pipeline` | Fetch + engineer features |
| `POST` | `/assets/{ticker}/backtest` | Run strategy backtest |
| `POST` | `/assets/{ticker}/walkforward` | Walk-forward validation |
| `POST` | `/assets/{ticker}/baselines` | ARIMA + naive baselines |
| `GET` | `/assets/{ticker}/drift` | Drift monitoring results |
| `GET` | `/model/{ticker}/info` | Model version + feature count |

**Example:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "top_n": 5}'
```

---

## 📐 Walk-Forward Validation

Unlike random `train_test_split`, walk-forward validation trains only on past data and tests on future data — eliminating look-ahead bias.

```
Fold 1:  [== train ==] [test]
Fold 2:  [=== train ===] [test]
Fold 3:  [==== train ====] [test]
```

Run independently:

```bash
python -m src.training.walk_forward_validation --ticker AAPL --n_splits 5
```

---

## 💹 Backtesting

Three strategies available:

| Strategy | Description |
|----------|-------------|
| `long_only` | Buy when model predicts UP, hold cash otherwise |
| `long_short` | Buy when UP, short when DOWN |
| `buy_and_hold` | Always long — passive baseline |

Metrics: total return, alpha vs buy-and-hold, Sharpe ratio, max drawdown, win rate, trade count.

```bash
python -m src.training.backtesting --ticker AAPL --strategy long_only
```

---

## 🔧 Supported Assets

| Name | Ticker | Type |
|------|--------|------|
| Apple | `AAPL` | Equity |
| Tesla | `TSLA` | Equity |
| Microsoft | `MSFT` | Equity |
| NVIDIA | `NVDA` | Equity |
| Gold ETF | `GLD` | Commodity |
| S&P 500 ETF | `SPY` | Index ETF |
| Bitcoin ETF | `IBIT` | Crypto |
| Nasdaq ETF | `QQQ` | Index ETF |

Per-asset XGBoost hyperparameter overrides are configured in `src/common/config.py` — momentum assets (TSLA, NVDA) use deeper trees; mean-reversion assets (GLD, SPY) use shallower, slower models.

---

## 📈 Expected Performance

| Metric | Typical Range |
|--------|---------------|
| Accuracy (walk-forward) | 52%–58% |
| ROC-AUC | 0.54–0.64 |
| Sharpe Ratio (long only) | 0.2–0.8 |
| ARIMA baseline accuracy | 49%–53% |

Stock direction prediction is inherently noisy. The goal is a consistent, statistically significant edge over 50% and over statistical baselines — not high absolute accuracy.

---

## 🔄 Retraining

**Manual:**

```bash
bash scripts/run_training.sh TSLA 2y
```

**Scheduled (cron):**

```bash
crontab -e
# Add: 0 2 * * 0 /absolute/path/to/scripts/retrain_cron.sh
```

---

## 🐛 Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**

```bash
export PYTHONPATH="$(pwd)"    # macOS/Linux
set PYTHONPATH=%cd%           # Windows CMD
```

**`Model not found in registry`**
- Ensure training completed: `bash scripts/run_training.sh AAPL`
- Promote the model in MLflow UI (http://localhost:5000) → Models → Transition to Production

**`No inference features for AAPL`**
- Click **▶ Run Market Analysis** in the dashboard sidebar first
- Or: `python -m src.ingestion.fetch_realtime --ticker AAPL`

**`ARIMA baseline is slow`**
- ARIMA runs rolling one-step-ahead forecasts — this is by design to avoid leakage
- Run in the background or reduce test size

**Port already in use**

```bash
lsof -i :8501    # find the PID
kill -9 <PID>
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| ML model | XGBoost 1.7 |
| Experiment tracking | MLflow 2.8 |
| Data drift | Evidently AI 0.4 |
| Dashboard | Streamlit 1.28 |
| REST API | FastAPI 0.104 + Uvicorn |
| Data source | yfinance + Alpha Vantage |
| Statistical baselines | statsmodels (ARIMA) |
| Visualisation | Plotly 5.17 |
| Logging | Loguru |
| Containerisation | Docker + Docker Compose |
| Tunnel | ngrok (optional) |

---

## 📊 Highlights of v2.1

### New Features
- ✨ **Live Streamlit Cloud deployment** — try predictions instantly
- 🔬 **Per-asset XGBoost tuning** — momentum vs mean-reversion strategies
- 📈 **Enhanced feature engineering** — 40+ indicators covering momentum, volatility, trend, reversion
- 🛡️ **Production fallback chain** — Production → Staging → Latest automatically
- ⏱️ **Intraday data caching** — TTL-based refresh for rapid inference
- 📊 **Drift detection twin system** — Evidently AI + KS-test fallback

### Quality Improvements
- ✅ **Time-ordered splits only** — never shuffles, zero look-ahead bias
- ✅ **Walk-forward validation** — fold-by-fold training on expanding windows
- ✅ **Baseline comparison** — ARIMA, naive momentum, and buy-and-hold references
- ✅ **Equity curve simulation** — 3 backtesting strategies with full metrics
- ✅ **Dark-theme Streamlit UI** — 5 tabbed interface with modern design tokens

---

## 🤝 Contributing

We welcome contributions! Please open issues for bugs or feature requests. Submit pull requests with:
1. Clear description of changes
2. Tests for new functionality
3. Update to documentation

---

## 📄 License

MIT License — see LICENSE file for details.

---

## ⚠️ Disclaimer

**Educational purposes only.** This system is a demonstration of MLOps engineering patterns and is not intended for live trading. Past performance does not guarantee future results. Consult a licensed financial advisor before making investment decisions.
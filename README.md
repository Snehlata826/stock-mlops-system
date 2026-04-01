# Stock MLOps System v2.1

A production-ready MLOps pipeline for stock price direction prediction using XGBoost, MLflow, Evidently AI, and Streamlit — with a FastAPI backend, walk-forward validation, backtesting, and real-time drift monitoring.

> Disclaimer: For educational purposes only. Not financial advice. Past performance does not guarantee future results.

---

## Contents

- [Live Demo](#live-demo)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Dashboard](#dashboard)
- [API Reference](#api-reference)
- [Advanced Features](#advanced-features)
- [Supported Assets](#supported-assets)
- [Performance](#performance)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Tech Stack](#tech-stack)
- [Contributing](#contributing)

---

## Live Demo

**[Launch Live Dashboard](https://stock-mlops-system-jbr8pfensdgxappkr9v9n9a.streamlit.app/)**

Deployed on Streamlit Cloud. Supports real-time predictions for AAPL, TSLA, MSFT, NVDA, GLD, SPY, IBIT, QQQ.

---

## Features

- XGBoost classifier trained on 40+ engineered technical indicators
- MLflow model registry with Production → Staging → Latest fallback chain
- Walk-forward validation eliminating data leakage from random splits
- Backtesting framework with equity curves, Sharpe ratio, max drawdown, win rate
- ARIMA and naive baseline models for honest benchmarking
- Evidently AI drift monitoring with KS-test detection fallback
- FastAPI REST backend with async pipeline support
- Streamlit dashboard with tabbed dark-theme interface
- Docker Compose setup for containerized deployment
- ngrok tunneling support for Streamlit Cloud connectivity

---

## Architecture

```
Data Ingestion → Feature Engineering → Training → Inference → Monitoring
      │                  │                │             │            │
  yfinance           40+ features      XGBoost      MLflow       Evidently
  Alpha Vantage      SMA/EMA/RSI       Walk-fwd     Registry     Drift KS
                     MACD/BB/ATR       Backtest     FastAPI      HTML report
                     Stochastic        Baselines    Streamlit
```
---

## Project Structure

```
stock-mlops-system/
├── src/
│   ├── common/
│   │   ├── config.py                    # Configuration & per-asset XGBoost params
│   │   ├── companies.py                 # Supported tickers
│   │   └── utils.py                     # Logger, validators
│   ├── ingestion/
│   │   ├── fetch_historical.py          # yfinance + Alpha Vantage fallback
│   │   └── fetch_realtime.py            # Intraday data with TTL caching
│   ├── features/
│   │   └── feature_engineering.py       # 40+ technical indicators
│   ├── training/
│   │   ├── train.py                     # XGBoost + MLflow training
│   │   ├── evaluate.py                  # Confusion matrix, ROC, feature importance
│   │   ├── walk_forward_validation.py   # Walk-forward validation
│   │   ├── baseline_models.py           # ARIMA + naive baselines
│   │   └── backtesting.py               # Strategy simulation
│   ├── inference/
│   │   ├── model_loader.py              # Production/Staging/Latest fallback
│   │   └── predict.py                   # Batch prediction pipeline
│   ├── monitoring/
│   │   └── drift_monitor.py             # Evidently + KS-test drift detection
│   └── api.py                           # FastAPI REST endpoints
├── streamlit_app/
│   ├── app.py                           # Main dashboard
│   ├── api_client.py                    # API client wrapper
│   └── components/
│       ├── theme.py                     # Design tokens
│       ├── ui.py                        # Reusable UI primitives
│       ├── sidebar.py                   # Controls
│       ├── header.py                    # Page banner
│       ├── candles.py                   # Candlestick chart
│       ├── bias_cards.py                # Prediction cards
│       ├── probability_chart.py         # Conviction chart
│       ├── model_health.py              # Drift monitoring
│       ├── walkforward.py               # Validation results
│       └── backtest.py                  # Strategy metrics
├── docker/
│   ├── Dockerfile                       # Streamlit container
│   ├── Dockerfile.backend               # API container
│   └── Dockerfile.frontend              # Frontend container
├── scripts/
│   ├── run_training.sh                  # Training pipeline
│   ├── run_streamlit.sh                 # Dashboard launcher
│   ├── run_api.sh                       # API launcher
│   └── retrain_cron.sh                  # Retraining scheduler
├── data/
│   ├── raw/                             # OHLCV CSVs
│   ├── processed/                       # Engineered features
│   └── drift/                           # Reference data
├── reports/                             # Evidently reports
├── evaluation/                          # Plots and metrics
├── mlruns/                              # MLflow tracking
├── requirements.txt
├── requirements_streamlit.txt
├── .env.example
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10 or 3.11
- Docker Desktop (optional)

### Option A: Local Python Setup

**Step 1 — Create Environment**

```bash
python3.10 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

**Step 2 — Configure API Key (Optional)**

```bash
cp .env.example .env
# Edit .env and add your Alpha Vantage key from https://www.alphavantage.co
```

**Step 3 — Start MLflow**

```bash
mlflow ui --port 5000
```

**Step 4 — Train Model**

```bash
bash scripts/run_training.sh AAPL 2y
```

**Step 5 — Launch Dashboard**

```bash
bash scripts/run_streamlit.sh
```

---

### Option B: Docker Compose

```bash
docker compose up backend mlflow -d
bash scripts/run_streamlit.sh
```

---

## Dashboard

| Tab | Content |
|-----|---------|
| Price Action | Candlestick chart with volume, period summary |
| Model Outlook | Bullish/bearish prediction, confidence score |
| Validation | Walk-forward metrics, baseline comparison |
| Backtest | Equity curve, Sharpe ratio, max drawdown |
| Health | Feature drift detection, Evidently report |

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Health check and supported assets |
| GET | /health | Service health ping |
| POST | /predict | Get prediction for ticker |
| GET | /assets | List supported tickers |
| POST | /assets/{ticker}/backtest | Run backtest |
| GET | /assets/{ticker}/drift | Drift monitoring results |

---

## Advanced Features

### Walk-Forward Validation

```bash
python -m src.training.walk_forward_validation --ticker AAPL --n_splits 5
```

### Backtesting

```bash
python -m src.training.backtesting --ticker AAPL --strategy long_only
```

---

## Supported Assets

| Name | Ticker | Type |
|------|--------|------|
| Apple | AAPL | Equity |
| Tesla | TSLA | Equity |
| Microsoft | MSFT | Equity |
| NVIDIA | NVDA | Equity |
| Gold ETF | GLD | Commodity |
| S&P 500 ETF | SPY | Index |
| Bitcoin ETF | IBIT | Crypto |
| Nasdaq ETF | QQQ | Index |

---

## Performance

| Metric | Range |
|--------|-------|
| Accuracy | 52–58% |
| ROC-AUC | 0.54–0.64 |
| Sharpe Ratio | 0.2–0.8 |
| ARIMA baseline | 49–53% |

---

## Deployment

### Manual Retraining

```bash
bash scripts/run_training.sh TSLA 2y
```

### Scheduled Retraining

```bash
crontab -e
# Add: 0 2 * * 0 /path/to/scripts/retrain_cron.sh
```

---

## Troubleshooting

### ModuleNotFoundError

```bash
export PYTHONPATH="$(pwd)"
```

### Model not found

- Ensure training completed: `bash scripts/run_training.sh AAPL`
- Check MLflow UI at http://localhost:5000
- Promote to Production stage

### Port already in use

```bash
lsof -i :8501
kill -9 <PID>
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| ML Model | XGBoost 1.7 |
| Tracking | MLflow 2.8 |
| Data Drift | Evidently AI 0.4 |
| Dashboard | Streamlit 1.28 |
| API | FastAPI 0.104 |
| Data | yfinance + Alpha Vantage |
| Baselines | statsmodels (ARIMA) |
| Visualization | Plotly 5.17 |

---

## Contributing

We welcome contributions! Please:

1. Open issues for bugs or features
2. Submit PRs with clear descriptions
3. Include tests for new functionality
4. Update documentation

---

## License

MIT License — see LICENSE file for details.

---

## Disclaimer

Educational purposes only. Not for live trading. Consult a licensed financial advisor before investment decisions.

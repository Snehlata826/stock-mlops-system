# 📈 Stock MLOps System v2.0

A **production-ready MLOps system** for stock price direction prediction.  
Built with XGBoost · MLflow · Evidently AI · Streamlit · FastAPI.

---

## 🆕 What's New in v2.0

| Feature | Status |
|---|---|
| Walk-forward validation (no data leakage) | ✅ New |
| ARIMA & naive baseline comparison | ✅ New |
| Backtesting with equity curve, Sharpe, drawdown | ✅ New |
| MAE / RMSE / MAPE on probability outputs | ✅ New |
| FastAPI REST endpoint `/predict` | ✅ New |
| 40+ engineered features (was 25) | ✅ New |
| Full dark-theme UI redesign with tabs | ✅ New |
| Time-ordered train/test split (no shuffle) | ✅ Fixed |

---

## 🏗️ Architecture

```
Data Ingestion → Feature Engineering → Training → Inference → Monitoring
     │                  │                │             │            │
 yfinance           40+ features      XGBoost      MLflow       Evidently
 Alpha Vantage      SMA/EMA/RSI       Walk-fwd     Registry     Drift KS
                    MACD/BB/ATR       Backtест     FastAPI      HTML report
                    Stochastic        Baselines    Streamlit
```

---

## 📁 Project Structure

```
stock-mlops-system/
├── src/
│   ├── common/
│   │   ├── config.py              # All configuration
│   │   ├── companies.py           # Supported tickers
│   │   └── utils.py               # Logger, validators
│   ├── ingestion/
│   │   ├── fetch_historical.py    # yfinance + Alpha Vantage fallback
│   │   └── fetch_realtime.py      # Intraday with caching
│   ├── features/
│   │   └── feature_engineering.py # 40+ technical indicators
│   ├── training/
│   │   ├── train.py               # XGBoost + MLflow (time-ordered split)
│   │   ├── evaluate.py            # Confusion matrix, ROC, feature importance
│   │   ├── walk_forward_validation.py  # ★ NEW: No-leakage WF validation
│   │   ├── baseline_models.py     # ★ NEW: ARIMA + naive baselines
│   │   └── backtesting.py         # ★ NEW: Strategy simulation
│   ├── inference/
│   │   ├── model_loader.py        # Production→Staging→Latest fallback
│   │   └── predict.py             # Batch prediction pipeline
│   ├── monitoring/
│   │   └── drift_monitor.py       # Evidently drift detection
│   └── api.py                     # ★ NEW: FastAPI REST endpoint
├── streamlit_app/
│   ├── app.py                     # Main dashboard (tabbed UI)
│   └── components/
│       ├── header.py
│       ├── sidebar.py
│       ├── candles.py             # Candlestick + volume chart
│       ├── bias_cards.py          # Bullish/bearish metric cards
│       ├── probability_chart.py   # Conviction over time
│       ├── model_health.py        # Drift health cards
│       ├── walkforward.py         # ★ NEW: WF results + baseline chart
│       └── backtest.py            # ★ NEW: Equity curve + metrics
├── docker/
│   └── Dockerfile
├── scripts/
│   ├── run_training.sh            # Full training pipeline
│   ├── run_streamlit.sh           # Launch dashboard
│   ├── run_api.sh                 # Launch FastAPI
│   └── retrain_cron.sh            # Scheduled retraining
├── data/
│   ├── raw/                       # OHLCV CSV files
│   ├── processed/                 # Engineered features
│   └── drift/                     # Reference data for monitoring
├── reports/                       # Evidently HTML drift reports
├── evaluation/                    # Plots: ROC, confusion matrix, importance
├── mlruns/                        # MLflow experiment tracking
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- Python 3.10 or 3.11
- Git (optional)

### Step 1 — Setup environment

```bash
# Unzip and enter the project
unzip stock-mlops-system.zip
cd stock-mlops-system

# Create virtual environment
python3.10 -m venv venv

# Activate
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2 — (Optional) Add Alpha Vantage key

```bash
cp .env.example .env
# Edit .env and paste your free key from https://www.alphavantage.co
```

Without a key the system falls back to historical data for inference — fully functional.

### Step 3 — Start MLflow

Open **Terminal 1** and keep it running:

```bash
mlflow ui --port 5000
```

✅ MLflow UI → http://localhost:5000

### Step 4 — Train the model

Open **Terminal 2**:

```bash
# macOS / Linux
chmod +x scripts/*.sh
bash scripts/run_training.sh AAPL 2y

# Windows (Git Bash or WSL)
bash scripts/run_training.sh AAPL 2y

# Or run steps manually:
python -m src.ingestion.fetch_historical --ticker AAPL --period 2y
python -m src.features.feature_engineering --mode train --ticker AAPL
python -m src.training.train --ticker AAPL
python -m src.training.evaluate --ticker AAPL
python -m src.training.walk_forward_validation --ticker AAPL --n_splits 5
python -m src.training.baseline_models --ticker AAPL
```

The training script runs all 6 steps automatically:
1. Fetch 2 years of OHLCV data
2. Engineer 40+ technical features
3. Train XGBoost (time-ordered split)
4. Evaluate & save plots
5. Walk-forward validation (5 folds)
6. ARIMA + naive baselines

### Step 5 — Promote model to Production

**Option A — MLflow UI (recommended):**
1. Open http://localhost:5000
2. Click **Models** tab → `stock_direction_predictor_AAPL`
3. Select latest version → **Stage → Transition to Production**

**Option B — Python:**
```python
import mlflow
mlflow.set_tracking_uri("file:mlruns")
client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name="stock_direction_predictor_AAPL",
    version="1",
    stage="Production"
)
```

### Step 6 — Launch the dashboard

Open **Terminal 3**:

```bash
bash scripts/run_streamlit.sh
```

✅ Dashboard → http://localhost:8501

### Step 7 — (Optional) Launch the API

Open **Terminal 4**:

```bash
bash scripts/run_api.sh
```

✅ API docs → http://localhost:8000/docs

---

## 📊 Dashboard Tabs

| Tab | What it shows |
|---|---|
| **Price Action** | Candlestick chart + volume, period summary |
| **Model Outlook** | Bullish/bearish bias card, confidence, probability chart |
| **Validation** | Walk-forward results (MAE/RMSE/MAPE), baseline comparison |
| **Backtest** | Equity curve, Sharpe ratio, max drawdown, win rate |
| **Health** | Data drift detection, Evidently HTML report |

---

## 🌐 FastAPI Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check + supported assets |
| GET | `/health` | Simple ping |
| POST | `/predict` | Get predictions for a ticker |
| GET | `/assets` | List supported tickers |
| GET | `/model/{ticker}/info` | Model version + feature count |

**Example:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "top_n": 5}'
```

---

## 🐳 Docker

```bash
# Build
docker build -t stock-mlops -f docker/Dockerfile .

# Run (mounts mlruns and data for persistence)
docker run -p 8501:8501 \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/data:/app/data \
  stock-mlops
```

---

## 📐 Walk-Forward Validation

Unlike random `train_test_split`, walk-forward validation:
- Trains only on past data
- Tests on future data
- Repeats across multiple folds
- Gives honest out-of-sample performance

```
Fold 1:  [====train====] [test]
Fold 2:  [=====train=====] [test]
Fold 3:  [======train======] [test]
```

Run independently:
```bash
python -m src.training.walk_forward_validation --ticker AAPL --n_splits 5
```

---

## 💹 Backtesting

Three strategies are available:

| Strategy | Description |
|---|---|
| Long Only | Buy when model predicts UP, hold cash otherwise |
| Long / Short | Buy when UP, short when DOWN |
| Buy & Hold | Always long — used as the baseline |

Metrics reported: total return, alpha vs B&H, Sharpe ratio, max drawdown, win rate, number of trades.

```bash
python -m src.training.backtesting --ticker AAPL --strategy long_only
```

---

## 📈 Expected Performance

| Metric | Typical Range |
|---|---|
| Accuracy (walk-forward) | 52%–58% |
| ROC-AUC | 0.54–0.64 |
| Sharpe Ratio (long only) | 0.2–0.8 |
| ARIMA baseline accuracy | 49%–53% |

Stock prediction is inherently hard. The goal is a consistent edge over 50% and over statistical baselines.

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
export PYTHONPATH="$(pwd)"   # macOS/Linux
set PYTHONPATH=%cd%          # Windows CMD
```

**`Model not found in registry`**
- Check MLflow UI at http://localhost:5000
- Ensure training ran: `bash scripts/run_training.sh AAPL`
- Promote the model to Production in MLflow UI

**`No inference features for AAPL`**
- Click **Run Market Analysis** in the dashboard first
- Or run: `python -m src.ingestion.fetch_realtime --ticker AAPL`

**`ARIMA baseline is slow`**
- ARIMA runs rolling one-step forecasts — this is intentional for correctness
- Reduce the test window or run in the background

**Port already in use**
```bash
lsof -i :8501   # macOS/Linux — find PID then kill
kill -9 <PID>
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| ML model | XGBoost |
| Experiment tracking | MLflow |
| Data drift | Evidently AI |
| Dashboard | Streamlit |
| REST API | FastAPI + Uvicorn |
| Data source | yfinance + Alpha Vantage |
| Statistical baselines | statsmodels (ARIMA) |
| Visualization | Plotly |
| Logging | Loguru |
| Containerization | Docker |

---

## ⚠️ Disclaimer

**Educational purposes only.** Not financial advice. Past performance does not guarantee future results. Consult a licensed financial advisor before making investment decisions.

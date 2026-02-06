# 📈 Stock MLOps System - Local-First Stock Prediction Platform

A complete **production-ready MLOps system** for stock price direction prediction, designed to run entirely on your local machine. Built with industry-standard tools: **XGBoost**, **MLflow**, **Evidently AI**, and **Streamlit**.

---

## 🎯 Overview

This system predicts stock movement direction (UP/DOWN) using machine learning and follows MLOps best practices:

- ✅ **End-to-end ML pipeline** (data ingestion → training → inference)
- ✅ **Experiment tracking** with MLflow
- ✅ **Model registry** (Staging/Production)
- ✅ **Data drift monitoring** with Evidently AI
- ✅ **Real-time predictions** via Streamlit dashboard
- ✅ **Dockerized inference** service
- ✅ **Automated retraining** scripts

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Ingestion Layer                     │
│  • Historical data (yfinance) • Real-time data (5-15min)    │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                  Feature Engineering Layer                   │
│  • Technical Indicators • Rolling Stats • Volatility         │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                     Training Layer                           │
│  • XGBoost Classifier • MLflow Tracking • Model Registry     │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                    Inference Layer                           │
│  • Production Model Loader • Batch Predictions               │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                   Monitoring Layer                           │
│  • Data Drift Detection • Model Performance Tracking         │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  • Streamlit Dashboard • Docker Container                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
stock-mlops-system/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── data/                              # Data storage
│   ├── raw/                          # Raw stock data
│   │   ├── historical_prices.csv
│   │   └── realtime_prices.csv
│   ├── processed/                    # Engineered features
│   │   ├── features_train.csv
│   │   └── features_inference.csv
│   └── drift/                        # Drift monitoring data
│       └── reference_data.csv
│
├── src/                              # Source code
│   ├── common/                       # Shared utilities
│   │   ├── config.py                # Configuration
│   │   └── utils.py                 # Helper functions
│   │
│   ├── ingestion/                   # Data fetching
│   │   ├── fetch_historical.py     # Historical data
│   │   └── fetch_realtime.py       # Real-time data
│   │
│   ├── features/                    # Feature engineering
│   │   └── feature_engineering.py  # Technical indicators
│   │
│   ├── training/                    # Model training
│   │   ├── train.py                # Training pipeline
│   │   └── evaluate.py             # Model evaluation
│   │
│   ├── monitoring/                  # Monitoring
│   │   └── drift_monitor.py        # Drift detection
│   │
│   └── inference/                   # Prediction
│       ├── model_loader.py         # Model loading
│       └── predict.py              # Inference pipeline
│
├── streamlit_app/                   # Web dashboard
│   └── app.py                       # Streamlit UI
│
├── docker/                          # Containerization
│   └── Dockerfile                   # Docker config
│
└── scripts/                         # Automation scripts
    ├── run_training.sh             # Full training pipeline
    ├── run_streamlit.sh            # Launch dashboard
    └── retrain_cron.sh             # Scheduled retraining
```

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites

- **Python 3.10 or 3.11** ([Download](https://www.python.org/downloads/))
- **Git** (optional, for cloning)
- **Docker** (optional, for containerization)

### Step 1: Setup Environment

```bash
# Extract the zip file
unzip stock-mlops-system.zip
cd stock-mlops-system

# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data (optional, for sentiment features)
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"
```

### Step 2: Start MLflow Server

**Open Terminal 1** (keep this running):

```bash
mlflow ui --port 5000
```

✅ Access MLflow UI at: **http://localhost:5000**

### Step 3: Train the Model

**Open Terminal 2**:

```bash
# Make scripts executable (Mac/Linux only)
chmod +x scripts/*.sh

# Run complete training pipeline
bash scripts/run_training.sh AAPL 2y

# On Windows (use Git Bash or WSL), or run manually:
# python src/ingestion/fetch_historical.py --ticker AAPL --period 2y
# python src/features/feature_engineering.py --mode train
# python src/training/train.py
# python src/training/evaluate.py
```

This will:
1. ✅ Fetch 2 years of AAPL historical data
2. ✅ Engineer technical features
3. ✅ Train XGBoost model
4. ✅ Log experiment to MLflow
5. ✅ Evaluate model performance

### Step 4: Promote Model to Production

**Option A - Via MLflow UI** (Recommended):
1. Open http://localhost:5000
2. Click "Models" tab
3. Click "stock_direction_predictor"
4. Select latest version
5. Click "Stage" → "Transition to Production"

**Option B - Via Python**:
```python
import mlflow
mlflow.set_tracking_uri("file:///path/to/stock-mlops-system/mlruns")
client = mlflow.MlflowClient()
client.transition_model_version_stage(
    name="stock_direction_predictor",
    version="1",
    stage="Production"
)
```

### Step 5: Run Streamlit Dashboard

**Open Terminal 3**:

```bash
bash scripts/run_streamlit.sh

# Or directly:
streamlit run streamlit_app/app.py --server.port 8501
```

✅ Access Dashboard at: **http://localhost:8501**

---

## 📊 Using the Dashboard

The Streamlit dashboard provides an interactive interface:

### Features:

1. **📥 Fetch Real-time Data**
   - Configure ticker symbol (e.g., AAPL, MSFT, GOOGL)
   - Select time period (1d, 5d, 1mo)
   - Choose interval (5m, 15m, 1h)

2. **⚙️ Engineer Features**
   - Automatically creates 30+ technical indicators
   - Rolling statistics, volatility, momentum

3. **🎯 Make Predictions**
   - Binary direction (UP/DOWN)
   - Confidence scores
   - Probability distributions

4. **📊 Drift Monitoring**
   - Detect data drift vs training data
   - HTML reports with detailed analysis
   - Retraining alerts

---

## 🧪 Features & Technical Indicators

The system creates **30+ features** including:

### Price-based Features
- Simple Moving Averages (SMA 5, 10, 20)
- Exponential Moving Averages (EMA 5, 10, 20)
- Price momentum (5-day, 10-day)
- High-Low spread
- Log returns

### Technical Indicators
- **RSI** (Relative Strength Index)
- **MACD** (Moving Average Convergence Divergence)
- **Bollinger Bands** (upper, lower, width)

### Volatility Features
- Rolling standard deviation
- Historical volatility

### Volume Features
- Volume SMA
- Volume ratio

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t stock-mlops-inference -f docker/Dockerfile .
```

### Run Container

```bash
docker run -p 8501:8501 \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/data:/app/data \
  stock-mlops-inference
```

✅ Access at: **http://localhost:8501**

### Docker Compose (Optional)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  streamlit:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8501:8501"
    volumes:
      - ./mlruns:/app/mlruns
      - ./data:/app/data
    environment:
      - PYTHONUNBUFFERED=1
```

Run with:
```bash
docker-compose up
```

---

## 🔄 Automated Retraining

### Manual Retraining

```bash
bash scripts/run_training.sh AAPL 2y
```

### Scheduled Retraining (Cron)

```bash
# Edit crontab
crontab -e

# Add line (runs every Sunday at 2 AM):
0 2 * * 0 /absolute/path/to/stock-mlops-system/scripts/retrain_cron.sh

# Or weekly on Monday at 3 AM:
0 3 * * 1 /absolute/path/to/stock-mlops-system/scripts/retrain_cron.sh
```

### Logs

Retraining logs are saved in `logs/retrain_YYYYMMDD_HHMMSS.log`

---

## 📈 Model Performance

Expected performance metrics:

| Metric      | Typical Range |
|-------------|---------------|
| Accuracy    | 52% - 58%     |
| Precision   | 50% - 60%     |
| Recall      | 50% - 60%     |
| F1 Score    | 50% - 58%     |
| ROC-AUC     | 55% - 65%     |

**Note**: Stock prediction is inherently difficult. The goal is to beat random guessing (50%) consistently.

---

## 🔧 Advanced Usage

### Custom Stock Ticker

```bash
# Fetch different stock
python src/ingestion/fetch_historical.py --ticker TSLA --period 1y

# Retrain with new ticker
bash scripts/run_training.sh TSLA 1y
```

### Manual Pipeline Execution

```bash
# 1. Fetch historical data
python src/ingestion/fetch_historical.py --ticker AAPL --period 2y

# 2. Engineer features for training
python src/features/feature_engineering.py --mode train

# 3. Train model
python src/training/train.py

# 4. Evaluate model
python src/training/evaluate.py

# 5. Fetch real-time data
python src/ingestion/fetch_realtime.py --ticker AAPL --period 5d --interval 5m

# 6. Engineer features for inference
python src/features/feature_engineering.py --mode inference

# 7. Make predictions
python src/inference/predict.py

# 8. Check drift
python src/monitoring/drift_monitor.py
```

### MLflow Model Registry API

```python
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("file:///path/to/mlruns")
client = MlflowClient()

# List all models
models = client.search_registered_models()
for model in models:
    print(f"Model: {model.name}")

# Get model versions
versions = client.search_model_versions("name='stock_direction_predictor'")
for v in versions:
    print(f"Version {v.version}: Stage={v.current_stage}")

# Promote to Production
client.transition_model_version_stage(
    name="stock_direction_predictor",
    version="1",
    stage="Production"
)

# Archive old versions
client.transition_model_version_stage(
    name="stock_direction_predictor",
    version="1",
    stage="Archived"
)
```

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Ensure you're in the project root
cd stock-mlops-system

# Add to PYTHONPATH (Mac/Linux)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run scripts from project root
python -m src.training.train
```

### Issue: `Model not found in registry`

**Solution**:
1. Check MLflow UI (http://localhost:5000)
2. Ensure model is trained: `python src/training/train.py`
3. Promote model to Production in MLflow UI
4. Model loader has fallback: Staging → Latest

### Issue: `No data retrieved for ticker`

**Solution**:
- Check internet connection
- Verify ticker symbol is valid
- Try different period: `--period 1y` instead of `2y`
- Check yfinance status:
  ```python
  import yfinance as yf
  ticker = yf.Ticker('AAPL')
  print(ticker.info)
  ```

### Issue: Permission denied on scripts (Mac/Linux)

**Solution**:
```bash
chmod +x scripts/*.sh
```

### Issue: MLflow UI not starting

**Solution**:
```bash
# Check if port 5000 is in use
lsof -i :5000  # Mac/Linux
netstat -ano | findstr :5000  # Windows

# Use different port
mlflow ui --port 5001
```

### Issue: Streamlit crashes

**Solution**:
```bash
# Clear cache
streamlit cache clear

# Check port availability
lsof -i :8501  # Mac/Linux

# Use different port
streamlit run streamlit_app/app.py --server.port 8502
```

### Issue: Docker build fails

**Solution**:
```bash
# Ensure Docker is running
docker --version

# Clean build
docker system prune -a
docker build --no-cache -t stock-mlops-inference -f docker/Dockerfile .
```

---

## 📚 Technology Stack

| Component           | Technology      | Purpose                    |
|---------------------|-----------------|----------------------------|
| ML Framework        | XGBoost         | Binary classification      |
| Experiment Tracking | MLflow          | Model versioning & registry|
| Data Fetching       | yfinance        | Stock market data          |
| Drift Detection     | Evidently AI    | Data quality monitoring    |
| Web Dashboard       | Streamlit       | Interactive UI             |
| Visualization       | Plotly          | Interactive charts         |
| Logging             | Loguru          | Structured logging         |
| Data Processing     | Pandas          | Data manipulation          |
| Containerization    | Docker          | Deployment                 |

---

## 🎓 Learning Resources

### MLOps Concepts
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Evidently AI Docs](https://docs.evidentlyai.com/)
- [XGBoost Guide](https://xgboost.readthedocs.io/)

### Stock Market Features
- [Technical Analysis Indicators](https://www.investopedia.com/terms/t/technicalindicator.asp)
- [Bollinger Bands](https://www.investopedia.com/terms/b/bollingerbands.asp)
- [RSI (Relative Strength Index)](https://www.investopedia.com/terms/r/rsi.asp)

### Best Practices
- [Google MLOps Guide](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [ML Model Registry Patterns](https://neptune.ai/blog/ml-model-registry)

---

## 🔐 Security & Privacy

- ✅ **No API keys required** - uses free yfinance data
- ✅ **Local-first** - all data stays on your machine
- ✅ **No cloud dependencies** - fully offline capable
- ✅ **Open source libraries** - transparent and auditable

---

## 🗺️ Roadmap

### Planned Features
- [ ] Multi-stock portfolio predictions
- [ ] Sentiment analysis from news headlines
- [ ] LSTM/Transformer model options
- [ ] FastAPI REST API endpoint
- [ ] PostgreSQL database integration
- [ ] Kubernetes deployment manifests
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Model explainability (SHAP values)

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

1. **Feature Engineering**: Add more technical indicators
2. **Model Architectures**: Try LSTM, Transformers, Ensemble methods
3. **Data Sources**: Integrate news sentiment, economic indicators
4. **Monitoring**: Enhanced alerting, Slack/email notifications
5. **Testing**: Unit tests, integration tests
6. **Documentation**: Jupyter notebooks, tutorials

---

## 📄 License

MIT License - free for personal and commercial use.

---

## ⚠️ Disclaimer

**This system is for educational purposes only.**

- Not financial advice
- Past performance ≠ future results
- Stock market prediction is inherently uncertain
- Use at your own risk
- Consult a licensed financial advisor for investment decisions

---

## 📞 Support

For issues or questions:

1. Check **Troubleshooting** section above
2. Review MLflow UI logs
3. Check `logs/` directory for error details
4. Verify all dependencies are installed

---

## 🎉 Acknowledgments

Built with:
- [XGBoost](https://github.com/dmlc/xgboost)
- [MLflow](https://github.com/mlflow/mlflow)
- [Evidently AI](https://github.com/evidentlyai/evidently)
- [Streamlit](https://github.com/streamlit/streamlit)
- [yfinance](https://github.com/ranaroussi/yfinance)

---

**Happy Predicting! 📈**

*Built with ❤️ for the MLOps community*

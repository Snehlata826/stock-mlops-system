# 🚀 Quick Start Guide (5 Minutes)

## Step 1: Setup (1 minute)

```bash
# Extract zip and navigate
cd stock-mlops-system

# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Mac/Linux
# OR
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Start MLflow (Terminal 1)

```bash
mlflow ui --port 5000
```

✅ Open: http://localhost:5000

## Step 3: Train Model (Terminal 2)

```bash
# Make scripts executable (Mac/Linux only)
chmod +x scripts/*.sh

# Run training pipeline
bash scripts/run_training.sh AAPL 2y

# Windows users, run manually:
# python src/ingestion/fetch_historical.py --ticker AAPL --period 2y
# python src/features/feature_engineering.py --mode train
# python src/training/train.py
# python src/training/evaluate.py
```

## Step 4: Promote Model

1. Open MLflow UI: http://localhost:5000
2. Click "Models" → "stock_direction_predictor"
3. Select latest version
4. Click "Stage" → "Transition to Production"

## Step 5: Run Dashboard (Terminal 3)

```bash
bash scripts/run_streamlit.sh

# OR
streamlit run streamlit_app/app.py
```

✅ Open: http://localhost:8501

## 🎉 Done!

Now you can:
- 🔄 Fetch real-time stock data
- ⚙️ Engineer features
- 🎯 Make predictions
- 📊 Monitor data drift

---

## Common Issues

### Import Errors
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Port Already in Use
```bash
# Use different port
mlflow ui --port 5001
streamlit run streamlit_app/app.py --server.port 8502
```

### Permission Denied (Mac/Linux)
```bash
chmod +x scripts/*.sh
```

---

**Need Help?** Check the full README.md for detailed troubleshooting.

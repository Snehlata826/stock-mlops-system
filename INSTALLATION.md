# 📦 Installation Guide

## System Requirements

- **Operating System**: Windows 10/11, macOS 10.15+, or Linux
- **Python**: 3.10 or 3.11 (required)
- **RAM**: Minimum 4GB (8GB recommended)
- **Disk Space**: ~500MB for dependencies + data
- **Internet**: Required for downloading stock data

---

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### 1. Install Python

**Windows:**
1. Download Python 3.10 from [python.org](https://www.python.org/downloads/)
2. Run installer
3. ✅ Check "Add Python to PATH"
4. Click "Install Now"

**macOS:**
```bash
# Using Homebrew
brew install python@3.10

# Verify installation
python3.10 --version
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

#### 2. Extract Project

```bash
# Extract the zip file
unzip stock-mlops-system.zip
cd stock-mlops-system
```

#### 3. Create Virtual Environment

**Mac/Linux:**
```bash
python3.10 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` in your terminal prompt.

#### 4. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# This may take 3-5 minutes
```

#### 5. Verify Installation

```bash
# Check Python packages
pip list | grep -E "xgboost|mlflow|streamlit|evidently"

# Expected output:
# evidently          0.4.8
# mlflow             2.8.1
# streamlit          1.28.1
# xgboost            1.7.6
```

---

### Method 2: Docker Installation

#### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed

#### Build Image

```bash
cd stock-mlops-system
docker build -t stock-mlops-inference -f docker/Dockerfile .
```

#### Run Container

```bash
docker run -p 8501:8501 \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/data:/app/data \
  stock-mlops-inference
```

Access dashboard at: http://localhost:8501

---

## Post-Installation Setup

### 1. Download NLTK Data (Optional)

For sentiment analysis features (if you plan to use them):

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('vader_lexicon')"
```

### 2. Verify Directory Structure

```bash
# Check directory structure
tree -L 2

# Should see:
# stock-mlops-system/
# ├── data/
# │   ├── raw/
# │   ├── processed/
# │   └── drift/
# ├── src/
# ├── streamlit_app/
# ├── docker/
# └── scripts/
```

### 3. Test Installation

```bash
# Test imports
python -c "import xgboost; import mlflow; import streamlit; print('✓ All imports successful')"

# Should print: ✓ All imports successful
```

---

## Troubleshooting Installation

### Issue: `python3.10: command not found`

**Solution:**
```bash
# Check available Python versions
python --version
python3 --version

# Use whichever version is 3.10 or 3.11
python3.11 -m venv venv  # if Python 3.11 is installed
```

### Issue: `pip: command not found`

**Solution:**
```bash
# Use python -m pip instead
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Issue: Permission denied installing packages

**Solution (Mac/Linux):**
```bash
# Don't use sudo! Use virtual environment instead
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Solution (Windows):**
```cmd
# Run Command Prompt as Administrator
# OR use virtual environment (recommended)
```

### Issue: `error: Microsoft Visual C++ 14.0 is required` (Windows)

**Solution:**
Download and install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

### Issue: `No module named 'distutils'` (Ubuntu/Debian)

**Solution:**
```bash
sudo apt install python3.10-distutils python3.10-dev
```

### Issue: SSL Certificate Error downloading packages

**Solution:**
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Issue: Out of Memory during installation

**Solution:**
```bash
# Install packages one at a time
pip install numpy pandas scikit-learn
pip install yfinance xgboost mlflow
pip install evidently streamlit
pip install matplotlib plotly
pip install textblob nltk python-dotenv joblib requests pytz loguru
```

---

## Version Compatibility

### Tested Configurations

| OS            | Python | Status |
|---------------|--------|--------|
| macOS 13+     | 3.10   | ✅     |
| macOS 13+     | 3.11   | ✅     |
| Ubuntu 22.04  | 3.10   | ✅     |
| Windows 11    | 3.10   | ✅     |
| Windows 10    | 3.10   | ✅     |

### Known Incompatibilities

- ❌ Python 3.9 or earlier (some dependencies won't work)
- ❌ Python 3.12 (XGBoost compatibility issues)
- ⚠️ Apple Silicon (M1/M2): Use `pip install --no-binary :all: xgboost`

---

## Environment Variables (Optional)

Create `.env` file in project root:

```bash
# Stock configuration
DEFAULT_TICKER=AAPL
HISTORICAL_PERIOD=2y

# MLflow
MLFLOW_TRACKING_URI=file:///absolute/path/to/mlruns

# Logging
LOG_LEVEL=INFO
```

---

## Next Steps

After successful installation:

1. ✅ Read `QUICKSTART.md` for 5-minute setup
2. ✅ Read `README.md` for full documentation
3. ✅ Run training pipeline: `bash scripts/run_training.sh`
4. ✅ Launch dashboard: `bash scripts/run_streamlit.sh`

---

## Getting Help

- **Installation issues**: Check troubleshooting section above
- **Usage questions**: See README.md
- **Package conflicts**: Try creating fresh virtual environment

---

**Happy Installing! 🚀**

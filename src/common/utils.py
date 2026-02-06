from loguru import logger
import sys
from pathlib import Path

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)

def setup_logger(log_file: Path = None):
    """Setup logger with optional file output"""
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, rotation="10 MB", retention="7 days")
    return logger

def validate_dataframe(df, required_columns=None):
    """Validate DataFrame has required columns"""
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty")
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    return True

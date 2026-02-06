import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.common.config import PROCESSED_DATA_DIR, TARGET_COLUMN
from src.common.utils import logger, validate_dataframe
from src.inference.model_loader import ModelLoader

def predict(data_path: Path = PROCESSED_DATA_DIR / "features_inference.csv"):
    """
    Make predictions on new data
    
    Returns:
        DataFrame with predictions and probabilities
    """
    logger.info(f"Making predictions from {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    validate_dataframe(df)
    logger.info(f"  Loaded {len(df)} records")
    
    # Prepare features (exclude non-feature columns)
    exclude_cols = ['Date', TARGET_COLUMN, 'Dividends', 'Stock Splits']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]
    
    # Load model
    loader = ModelLoader()
    model = loader.get_model()
    
    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'Date': df['Date'],
        'prediction': predictions,
        'prob_down': probabilities[:, 0],
        'prob_up': probabilities[:, 1],
        'confidence': np.max(probabilities, axis=1)
    })
    
    # Add direction label
    results['direction'] = results['prediction'].map({0: 'DOWN', 1: 'UP'})
    
    logger.info(f"✓ Predictions complete")
    logger.info(f"  UP predictions: {(predictions == 1).sum()}")
    logger.info(f"  DOWN predictions: {(predictions == 0).sum()}")
    logger.info(f"  Average confidence: {results['confidence'].mean():.4f}")
    
    return results

if __name__ == "__main__":
    predictions = predict()
    print("\nLatest Predictions:")
    print(predictions.tail(10))

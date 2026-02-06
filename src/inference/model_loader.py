import mlflow
import mlflow.xgboost
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.common.config import MLFLOW_TRACKING_URI, MODEL_NAME
from src.common.utils import logger

class ModelLoader:
    """Load and cache production model from MLflow"""
    
    def __init__(self):
        self.model = None
        self.model_version = None
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    def load_production_model(self):
        """Load model from Production stage"""
        try:
            model_uri = f"models:/{MODEL_NAME}/Production"
            logger.info(f"Loading production model: {model_uri}")
            
            self.model = mlflow.xgboost.load_model(model_uri)
            
            # Get model version
            client = mlflow.MlflowClient()
            model_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
            
            if model_versions:
                self.model_version = model_versions[0].version
                logger.info(f"✓ Loaded model version {self.model_version}")
            
            return self.model
            
        except Exception as e:
            logger.warning(f"Production model not found: {e}")
            logger.info("Trying Staging model...")
            return self.load_staging_model()
    
    def load_staging_model(self):
        """Fallback: Load model from Staging stage"""
        try:
            model_uri = f"models:/{MODEL_NAME}/Staging"
            logger.info(f"Loading staging model: {model_uri}")
            
            self.model = mlflow.xgboost.load_model(model_uri)
            
            client = mlflow.MlflowClient()
            model_versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
            
            if model_versions:
                self.model_version = model_versions[0].version
                logger.info(f"✓ Loaded model version {self.model_version} (Staging)")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Staging model not found: {e}")
            logger.info("Trying latest version...")
            return self.load_latest_model()
    
    def load_latest_model(self):
        """Fallback: Load latest model version"""
        try:
            model_uri = f"models:/{MODEL_NAME}/latest"
            logger.info(f"Loading latest model: {model_uri}")
            
            self.model = mlflow.xgboost.load_model(model_uri)
            logger.info(f"✓ Loaded latest model version")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load any model: {e}")
            raise
    
    def get_model(self):
        """Get loaded model (loads if not already loaded)"""
        if self.model is None:
            self.load_production_model()
        return self.model

if __name__ == "__main__":
    loader = ModelLoader()
    model = loader.get_model()
    print(f"Model loaded successfully: {type(model)}")

import mlflow
import mlflow.xgboost
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.common.config import MLFLOW_TRACKING_URI
from src.common.utils import logger


class ModelLoader:
    """
    Load and cache MLflow models per asset
    """

    def __init__(self):
        self.models = {}          # cache: model_name -> model
        self.model_versions = {} # cache: model_name -> version
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    def _load_model(self, model_name: str):
        """
        Internal: load model with Production → Staging → Latest fallback
        """
        client = mlflow.MlflowClient()

        # 1️⃣ Try Production
        try:
            model_uri = f"models:/{model_name}/Production"
            logger.info(f"Loading Production model: {model_uri}")
            model = mlflow.xgboost.load_model(model_uri)

            versions = client.get_latest_versions(model_name, stages=["Production"])
            version = versions[0].version if versions else None

            logger.info(f"✓ Loaded Production model {model_name} (v{version})")
            return model, version
        except Exception as e:
            logger.warning(f"Production model not found for {model_name}: {e}")

        # 2️⃣ Try Staging
        try:
            model_uri = f"models:/{model_name}/Staging"
            logger.info(f"Loading Staging model: {model_uri}")
            model = mlflow.xgboost.load_model(model_uri)

            versions = client.get_latest_versions(model_name, stages=["Staging"])
            version = versions[0].version if versions else None

            logger.info(f"✓ Loaded Staging model {model_name} (v{version})")
            return model, version
        except Exception as e:
            logger.warning(f"Staging model not found for {model_name}: {e}")

        # 3️⃣ Fallback: Latest
        try:
            model_uri = f"models:/{model_name}/latest"
            logger.info(f"Loading Latest model: {model_uri}")
            model = mlflow.xgboost.load_model(model_uri)

            logger.info(f"✓ Loaded Latest model {model_name}")
            return model, "latest"
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def get_model(self, model_name: str):
        """
        Get model by name (cached)
        """
        if model_name not in self.models:
            model, version = self._load_model(model_name)
            self.models[model_name] = model
            self.model_versions[model_name] = version

        return self.models[model_name]

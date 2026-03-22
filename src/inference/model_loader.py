import mlflow
import mlflow.xgboost
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.common.config import MLFLOW_TRACKING_URI
from src.common.utils import logger


class ModelLoader:
    def __init__(self):
        self.models = {}
        self.model_versions = {}
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    def _load_model(self, model_name: str):
        client = mlflow.MlflowClient()

        for stage in ["Production", "Staging"]:
            try:
                model_uri = f"models:/{model_name}/{stage}"
                model = mlflow.xgboost.load_model(model_uri)
                versions = client.get_latest_versions(model_name, stages=[stage])
                version = versions[0].version if versions else None
                logger.info(f"Loaded {stage} model {model_name} (v{version})")
                return model, version
            except Exception as e:
                logger.warning(f"{stage} model not found for {model_name}: {e}")

        try:
            model_uri = f"models:/{model_name}/latest"
            model = mlflow.xgboost.load_model(model_uri)
            logger.info(f"Loaded Latest model {model_name}")
            return model, "latest"
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def get_model(self, model_name: str):
        if model_name not in self.models:
            model, version = self._load_model(model_name)
            self.models[model_name] = model
            self.model_versions[model_name] = version
        return self.models[model_name]

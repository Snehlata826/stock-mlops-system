"""
API client — connects Streamlit frontend to FastAPI backend.
Reads API_BASE_URL from Streamlit secrets or environment variable.
"""
from __future__ import annotations
import os
from typing import Optional
import requests

def get_api_base_url() -> str:
    # Try Streamlit secrets first (production on Streamlit Cloud)
    try:
        import streamlit as st
        return st.secrets["API_BASE_URL"].rstrip("/")
    except Exception:
        pass
    # Fall back to environment variable (local Docker)
    return os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

API_BASE_URL = get_api_base_url()
_TIMEOUT = 30

class APIError(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API {status_code}: {detail}")

class APIClient:
    def __init__(self):
        self.base_url = get_api_base_url()
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "ngrok-skip-browser-warning": "true"
        })

    def _get(self, path: str) -> dict:
        try:
            r = self.session.get(f"{self.base_url}{path}", timeout=_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.ConnectionError:
            raise APIError(503, f"Cannot reach backend at {self.base_url}")
        except requests.Timeout:
            raise APIError(504, f"Request timed out")
        except requests.HTTPError as e:
            try:
                detail = e.response.json().get("detail", str(e))
            except Exception:
                detail = str(e)
            raise APIError(e.response.status_code, detail)

    def _post(self, path: str, payload: dict) -> dict:
        try:
            r = self.session.post(f"{self.base_url}{path}", json=payload, timeout=_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.ConnectionError:
            raise APIError(503, f"Cannot reach backend at {self.base_url}")
        except requests.Timeout:
            raise APIError(504, "Request timed out")
        except requests.HTTPError as e:
            try:
                detail = e.response.json().get("detail", str(e))
            except Exception:
                detail = str(e)
            raise APIError(e.response.status_code, detail)

    def is_alive(self) -> bool:
        try:
            return self._get("/health").get("status") == "ok"
        except APIError:
            return False

    def predict(self, ticker: str, top_n: int = 500) -> dict:
        return self._post("/predict", {"ticker": ticker, "top_n": top_n})

    def get_price_data(self, ticker: str, interval: str = "15min") -> dict:
        """Fetch OHLCV price data for candlestick chart."""
        return self._get(f"/assets/{ticker.upper()}/price?interval={interval}")

    def run_pipeline(self, ticker: str) -> dict:
        """Trigger data fetch + feature engineering on backend."""
        return self._post(f"/assets/{ticker.upper()}/run_pipeline", {})
    
    def asset_status(self, ticker: str) -> dict:
        return self._get(f"/assets/{ticker.upper()}/status")

_client: Optional[APIClient] = None

def get_client() -> APIClient:
    global _client
    if _client is None:
        _client = APIClient()
    return _client
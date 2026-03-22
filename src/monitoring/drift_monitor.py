import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.common.config import DRIFT_DATA_DIR, PROCESSED_DATA_DIR, TARGET_COLUMN, BASE_DIR, REPORTS_DIR
from src.common.utils import logger, validate_dataframe, validate_asset


def monitor_drift(ticker: str) -> dict:
    validate_asset(ticker)
    logger.info(f"Running drift monitoring for {ticker}...")

    reference_path = DRIFT_DATA_DIR / f"reference_features_{ticker}.csv"
    current_path = PROCESSED_DATA_DIR / f"features_inference_{ticker}.csv"

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference data not found: {reference_path}. Train the model first.")
    if not current_path.exists():
        raise FileNotFoundError(f"Current data not found: {current_path}. Run inference first.")

    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    validate_dataframe(reference_df)
    validate_dataframe(current_df)

    exclude_cols = ["Date", TARGET_COLUMN, "Dividends", "Stock Splits"]
    common_cols = [
        col for col in reference_df.columns
        if col in current_df.columns and col not in exclude_cols
    ]

    if not common_cols:
        raise ValueError("No common feature columns found for drift monitoring")

    reference_data = reference_df[common_cols]
    current_data = current_df[common_cols]

    logger.info(f"Monitoring {len(common_cols)} features across {len(current_data)} current rows")

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.test_suite import TestSuite
        from evidently.tests import TestNumberOfDriftedColumns

        drift_report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
        drift_report.run(reference_data=reference_data, current_data=current_data)

        REPORTS_DIR.mkdir(exist_ok=True)
        report_path = REPORTS_DIR / f"drift_report_{ticker}.html"
        drift_report.save_html(str(report_path))
        logger.info(f"Drift report saved → {report_path}")

        drift_tests = TestSuite(tests=[TestNumberOfDriftedColumns()])
        drift_tests.run(reference_data=reference_data, current_data=current_data)
        test_results = drift_tests.as_dict()

        n_drifted = 0
        for test in test_results.get("tests", []):
            params = test.get("parameters", {})
            if "number_of_drifted_columns" in params:
                n_drifted = params["number_of_drifted_columns"]
                break

        report_path_str = str(report_path)

    except ImportError:
        logger.warning("Evidently not installed — using statistical drift detection fallback")
        n_drifted, report_path_str = _statistical_drift_fallback(reference_data, current_data, ticker)

    drift_pct = (n_drifted / len(common_cols)) * 100

    if drift_pct > 30:
        logger.warning("SIGNIFICANT DRIFT DETECTED — Retraining recommended")
    else:
        logger.info("No significant drift detected")

    return {
        "ticker": ticker,
        "n_drifted_columns": n_drifted,
        "total_columns": len(common_cols),
        "drift_percentage": drift_pct,
        "report_path": report_path_str,
    }


def _statistical_drift_fallback(reference_data: pd.DataFrame, current_data: pd.DataFrame, ticker: str):
    """Simple KS-test based drift detection when Evidently is unavailable."""
    from scipy import stats

    n_drifted = 0
    for col in reference_data.columns:
        try:
            stat, p_value = stats.ks_2samp(
                reference_data[col].dropna(),
                current_data[col].dropna()
            )
            if p_value < 0.05:
                n_drifted += 1
        except Exception:
            continue

    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / f"drift_report_{ticker}.html"
    report_path.write_text(
        f"<html><body><h2>Drift Report ({ticker})</h2>"
        f"<p>Drifted columns (KS test p&lt;0.05): {n_drifted}/{len(reference_data.columns)}</p>"
        f"<p>Install evidently for a detailed report.</p></body></html>"
    )
    return n_drifted, str(report_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()
    results = monitor_drift(args.ticker)
    print(results)

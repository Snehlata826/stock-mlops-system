import pandas as pd
from pathlib import Path
import sys
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.common.config import DRIFT_DATA_DIR, PROCESSED_DATA_DIR, TARGET_COLUMN, BASE_DIR
from src.common.utils import logger, validate_dataframe, validate_asset


def monitor_drift(ticker: str):
    """
    Monitor data drift for a specific asset (Top 3 + Gold)

    Args:
        ticker: Asset ticker (AAPL, MSFT, TSLA, GLD)

    Returns:
        Drift metrics dictionary
    """
    logger.info(f"Running drift monitoring for {ticker}...")

    # ✅ Validate asset
    validate_asset(ticker)

    reference_path = DRIFT_DATA_DIR / f"reference_features_{ticker}.csv"
    current_path = PROCESSED_DATA_DIR / f"features_inference_{ticker}.csv"

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference data not found: {reference_path}")

    if not current_path.exists():
        raise FileNotFoundError(f"Current data not found: {current_path}")

    # Load data
    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    logger.info(f"  Reference data: {len(reference_df)} records")
    logger.info(f"  Current data: {len(current_df)} records")

    validate_dataframe(reference_df)
    validate_dataframe(current_df)

    # Exclude non-feature columns
    exclude_cols = ["Date", TARGET_COLUMN, "Dividends", "Stock Splits"]
    common_cols = [
        col for col in reference_df.columns
        if col in current_df.columns and col not in exclude_cols
    ]

    if not common_cols:
        raise ValueError("No common feature columns found for drift monitoring")

    reference_data = reference_df[common_cols]
    current_data = current_df[common_cols]

    logger.info(f"  Monitoring {len(common_cols)} features")

    # Create Evidently report
    drift_report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset()
    ])

    drift_report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    # Save report
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / f"drift_report_{ticker}.html"

    drift_report.save_html(str(report_path))
    logger.info(f"✓ Drift report saved to {report_path}")

    # Run drift tests
    drift_tests = TestSuite(tests=[
        TestNumberOfDriftedColumns()
    ])

    drift_tests.run(
        reference_data=reference_data,
        current_data=current_data
    )

    test_results = drift_tests.as_dict()

    # Extract drift count
    n_drifted = 0
    for test in test_results.get("tests", []):
        params = test.get("parameters", {})
        if "number_of_drifted_columns" in params:
            n_drifted = params["number_of_drifted_columns"]
            break

    drift_pct = (n_drifted / len(common_cols)) * 100

    logger.info(f"  Drifted columns: {n_drifted}/{len(common_cols)}")

    if drift_pct > 30:
        logger.warning("⚠ SIGNIFICANT DRIFT DETECTED — Retraining recommended")
    else:
        logger.info("✓ No significant drift detected")

    return {
        "ticker": ticker,
        "n_drifted_columns": n_drifted,
        "total_columns": len(common_cols),
        "drift_percentage": drift_pct,
        "report_path": str(report_path)
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()

    results = monitor_drift(args.ticker)
    print("\nDrift Results:")
    print(f"  Asset: {results['ticker']}")
    print(f"  Drifted columns: {results['n_drifted_columns']}/{results['total_columns']}")
    print(f"  Drift percentage: {results['drift_percentage']:.2f}%")

import pandas as pd
from pathlib import Path
import sys
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfDriftedColumns

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.common.config import (
    DRIFT_DATA_DIR, PROCESSED_DATA_DIR, TARGET_COLUMN, BASE_DIR
)
from src.common.utils import logger, validate_dataframe

def monitor_drift(
    reference_path: Path = DRIFT_DATA_DIR / "reference_data.csv",
    current_path: Path = PROCESSED_DATA_DIR / "features_inference.csv"
):
    """
    Monitor data drift between reference and current data
    
    Args:
        reference_path: Path to reference (training) data
        current_path: Path to current (inference) data
    
    Returns:
        Drift report results
    """
    logger.info("Running drift monitoring...")
    
    # Load data
    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)
    
    logger.info(f"  Reference data: {len(reference_df)} records")
    logger.info(f"  Current data: {len(current_df)} records")
    
    # Get common columns (exclude Date, target)
    exclude_cols = ['Date', TARGET_COLUMN, 'Dividends', 'Stock Splits']
    common_cols = [
        col for col in reference_df.columns 
        if col in current_df.columns and col not in exclude_cols
    ]
    
    reference_data = reference_df[common_cols]
    current_data = current_df[common_cols]
    
    logger.info(f"  Monitoring {len(common_cols)} features")
    
    # Create drift report
    drift_report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset()
    ])
    
    drift_report.run(
        reference_data=reference_data,
        current_data=current_data
    )
    
    # Save HTML report
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "drift_report.html"
    
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
    
    # Get test results
    test_results = drift_tests.as_dict()
    
    # Check if drift detected
    n_drifted = 0
    if 'tests' in test_results:
        for test in test_results['tests']:
            if 'parameters' in test and 'number_of_drifted_columns' in test['parameters']:
                n_drifted = test['parameters']['number_of_drifted_columns']
                break
    
    logger.info(f"  Drifted columns: {n_drifted}/{len(common_cols)}")
    
    if n_drifted > len(common_cols) * 0.3:  # >30% drift
        logger.warning("⚠ SIGNIFICANT DRIFT DETECTED - Consider retraining!")
    else:
        logger.info("✓ No significant drift detected")
    
    return {
        'n_drifted_columns': n_drifted,
        'total_columns': len(common_cols),
        'drift_percentage': (n_drifted / len(common_cols)) * 100 if len(common_cols) > 0 else 0
    }

if __name__ == "__main__":
    results = monitor_drift()
    print(f"\nDrift Results:")
    print(f"  Drifted columns: {results['n_drifted_columns']}/{results['total_columns']}")
    print(f"  Drift percentage: {results['drift_percentage']:.2f}%")

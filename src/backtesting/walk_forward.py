import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def walk_forward_validation(model, df, feature_cols, target_col,
                             train_window=200,
                             test_window=20):

    results = []
    start = 0

    while start + train_window + test_window <= len(df):

        train_df = df.iloc[start:start + train_window]
        test_df = df.iloc[start + train_window:
                          start + train_window + test_window]

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]

        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds)

        # Safe AUC
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, probs)
        else:
            auc = np.nan

        results.append({
            "start_index": start,
            "accuracy": acc,
            "roc_auc": auc
        })

        start += test_window

    return pd.DataFrame(results)
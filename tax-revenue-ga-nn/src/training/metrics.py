from typing import Dict
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Compute standard regression metrics.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(mse ** 0.5)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": float(mae),
        "r2": float(r2),
    }

"""Metrics computation utilities."""

import numpy as np
from typing import Tuple


def compute_metrics(
    predictions: np.ndarray, targets: np.ndarray
) -> dict:
    """Compute evaluation metrics.
    
    Args:
        predictions: Predicted heading angles
        targets: Ground truth heading angles
        
    Returns:
        Dictionary with metrics (MSE, MAE, RMSE)
    """
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
    }


"""
metrics.py – Evaluation metrics tự cài đặt (không dùng sklearn).

Cung cấp:
    mae(y_true, y_pred)
    rmse(y_true, y_pred)
    r2_score(y_true, y_pred)
"""

import numpy as np


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true, y_pred):
    """Coefficient of determination R²."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot

# Add alias for compatibility with tests
r2 = r2_score

def evaluate_metrics(y_true, y_pred):
    """
    Return dictionary with MAE, RMSE, and R2.
    For compatibility with ABC tests.
    """
    return {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }

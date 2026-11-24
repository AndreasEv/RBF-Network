"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Regression error measures on targets t and predictions y:

  MSE  = mean((t - y)^2)
  RMSE = sqrt(MSE)
  R^2  = 1 - sum((t - y)^2) / sum((t - mean(t))^2)

R^2 is 1 for a perfect fit, 0 when the network does no better than always
predicting the mean of t, and negative when it does worse than that.
"""

import math

import numpy as np


def _pair(targets, predictions):
    t = np.asarray(targets, dtype=float).reshape(-1)
    y = np.asarray(predictions, dtype=float).reshape(-1)
    if len(t) != len(y):
        raise ValueError(f"{len(t)} targets but {len(y)} predictions")
    if len(t) == 0:
        raise ValueError("no values")
    return t, y


def mse(targets, predictions):
    t, y = _pair(targets, predictions)
    return float(np.mean((t - y) ** 2))


def rmse(targets, predictions):
    return math.sqrt(mse(targets, predictions))


def r2(targets, predictions):
    """Coefficient of determination. If all targets are equal (no variance)
    it is 1.0 for a perfect fit and nan otherwise."""
    t, y = _pair(targets, predictions)
    ss_res = float(np.sum((t - y) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else float("nan")
    return 1.0 - ss_res / ss_tot

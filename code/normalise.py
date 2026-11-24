"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Min-max normalisation of the input features to [0, 1].

The statistics (minimum and range of every feature) come from the TRAINING
set only and are then applied unchanged to the test set, so the test set
never influences the preprocessing:

    x_norm = (x - min_train) / (max_train - min_train)

Every training value therefore lands in [0, 1]. A test value can fall a
little outside [0, 1] when it is outside the training range; we keep it
as it is (no clipping), so the network still sees how far out it is.

Constant features: if a feature has the same value in every training row,
its range is 0 and the formula would divide by zero. Such a feature carries
no information for training, so it is mapped to 0 for every row (training
and test). The guard uses a scale of 0 instead of 1/range for that column.

The activity Y is not normalised.
"""

import numpy as np


class MinMaxScaler:
    """Feature-wise min-max scaling with statistics fitted on one data set.

    Usage:
        scaler = MinMaxScaler.fit(X_train)
        X_train_n = scaler.transform(X_train)   # values in [0, 1]
        X_test_n = scaler.transform(X_test)     # same min / range as training
    """

    def __init__(self, minimum, value_range):
        self.minimum = np.asarray(minimum, dtype=float)
        self.range = np.asarray(value_range, dtype=float)
        # True for features that are constant in the training set
        self.constant = self.range == 0
        # 1 / range, or 0 for constant features (the divide-by-zero guard)
        self.scale = np.zeros_like(self.range)
        self.scale[~self.constant] = 1.0 / self.range[~self.constant]

    @classmethod
    def fit(cls, X):
        """Compute min and range of every column of X (shape (N, d))."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError(f"expected a non-empty 2-D array, got shape {X.shape}")
        minimum = X.min(axis=0)
        return cls(minimum, X.max(axis=0) - minimum)

    def transform(self, X):
        """Scale X (shape (N, d) or (d,)) with the fitted statistics."""
        X = np.asarray(X, dtype=float)
        if X.shape[-1] != self.minimum.shape[0]:
            raise ValueError(
                f"expected {self.minimum.shape[0]} features, got {X.shape[-1]}")
        return (X - self.minimum) * self.scale


def normalise_train_test(X_train, X_test):
    """Fit on X_train, apply to both. Returns (X_train_n, X_test_n, scaler)."""
    scaler = MinMaxScaler.fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), scaler

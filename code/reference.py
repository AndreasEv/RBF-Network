"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Two simple reference models, so the RBF scores have something to be
compared with. Both are fitted on the (normalised) training set only.

  mean    always predicts the mean training activity. Its test R^2 is
          close to 0 (slightly below 0, because the training mean is not
          the test mean): an R^2 above it means the model learned something.
  linear  y = w . x + b by least squares (numpy.linalg.lstsq). With 21
          molecules and 53 features the system is under-determined, so
          lstsq returns the minimum-norm solution, which fits the training
          set exactly. `features` restricts it to some columns (e.g. the
          ones chosen by features.select_features).
"""

import numpy as np


class MeanModel:
    """Predicts the mean training target for every input."""

    def __init__(self, Y_train):
        self.value = float(np.mean(np.asarray(Y_train, dtype=float)))

    def predict(self, X):
        return np.full(len(np.atleast_2d(X)), self.value)


class LinearModel:
    """Least-squares linear fit y = w . x[features] + b."""

    def __init__(self, X_train, Y_train, features=None):
        X = np.atleast_2d(np.asarray(X_train, dtype=float))
        Y = np.asarray(Y_train, dtype=float).reshape(-1)
        if len(X) != len(Y):
            raise ValueError(f"{len(X)} patterns but {len(Y)} targets")
        self.features = np.arange(X.shape[1]) if features is None else np.asarray(features)
        A = np.column_stack([X[:, self.features], np.ones(len(X))])
        solution = np.linalg.lstsq(A, Y, rcond=None)[0]
        self.weights, self.bias = solution[:-1], float(solution[-1])

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return X[:, self.features] @ self.weights + self.bias

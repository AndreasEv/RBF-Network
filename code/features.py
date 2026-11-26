"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Feature selection by correlation with the activity, fitted on training
data only.

With 21 training molecules and 53 descriptors most directions of the input
space carry no information about Y, but every one of them adds to the
Euclidean distance inside the Gaussians. select_features keeps the m
descriptors whose absolute Pearson correlation with Y is largest *on the
data it is given*. It must only ever be given training data (in
cross-validation: the training folds), otherwise the test molecules would
leak into the choice.
"""

import numpy as np


def correlations(X, Y):
    """|Pearson correlation| of every column of X (N, d) with Y (N,).

    A column that is constant in X has correlation 0 (it cannot help).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.asarray(Y, dtype=float).reshape(-1)
    if len(X) != len(Y):
        raise ValueError(f"{len(X)} patterns but {len(Y)} targets")
    xc = X - X.mean(axis=0)
    yc = Y - Y.mean()
    denom = np.sqrt((xc ** 2).sum(axis=0) * (yc ** 2).sum())
    corr = np.zeros(X.shape[1])
    ok = denom > 0
    corr[ok] = np.abs(xc[:, ok].T @ yc) / denom[ok]
    return corr


def select_features(X, Y, m):
    """Indices (sorted, ascending) of the m columns of X most correlated with Y.

    m = 0 or m >= number of columns keeps every column. Ties are broken by
    column order, so the result is deterministic.
    """
    d = np.atleast_2d(X).shape[1]
    if m < 0:
        raise ValueError(f"number of features must be >= 0, got {m}")
    if m == 0 or m >= d:
        return np.arange(d)
    order = np.argsort(-correlations(X, Y), kind="stable")
    return np.sort(order[:m])

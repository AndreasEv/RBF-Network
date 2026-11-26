"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Tests for cv.py (run from code/: python3 -m pytest tests -q).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv  # noqa: E402
from data import read_dataset  # noqa: E402
from parameters import read_parameters  # noqa: E402

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def base():
    params = read_parameters(os.path.join(CODE_DIR, "parameters.txt"))
    return params


@pytest.fixture(scope="module")
def train_set(base):
    _, Y, X = read_dataset(os.path.join(CODE_DIR, base["trainFile"]), 53)
    return X, Y


def test_folds_partition_the_patterns():
    parts = cv.fold_indices(21, 5, np.random.default_rng(0))
    assert len(parts) == 5
    assert sorted(np.concatenate(parts).tolist()) == list(range(21))
    assert {len(p) for p in parts} == {4, 5}
    with pytest.raises(ValueError):
        cv.fold_indices(21, 1, np.random.default_rng(0))


def test_fit_predict_is_fitted_on_the_training_part_only(base, train_set):
    X, Y = train_set
    config = {"model": "rbf", "features": 5, "K": 3, "sigma": 0.5, "rates": 1.0, "epochs": 3}
    a = cv.fit_predict(config, X[:16], Y[:16], X[16:], base, seed=0)
    # the held-out patterns only get predicted: predicting fewer of them, or
    # wildly different ones next to them, gives the same values
    b = cv.fit_predict(config, X[:16], Y[:16], X[16:18], base, seed=0)
    c = cv.fit_predict(config, X[:16], Y[:16], np.vstack([X[16:18], 100 * X[18:]]),
                       base, seed=0)
    assert a.shape == (5,) and np.allclose(a[:2], b) and np.allclose(a[:2], c[:2])


def test_mean_model_cv_equals_the_hand_computed_value(base, train_set):
    X, Y = train_set
    mean, std = cv.cv_mse({"model": "mean", "features": 0}, X, Y, base,
                          folds=3, repeats=1, seed=5)
    rng = np.random.default_rng(5)
    err = 0.0
    for part in cv.fold_indices(len(Y), 3, rng):
        rest = np.setdiff1d(np.arange(len(Y)), part)
        err += np.sum((Y[part] - Y[rest].mean()) ** 2)
    assert mean == pytest.approx(err / len(Y)) and std == 0.0


def test_cv_is_reproducible(base, train_set):
    X, Y = train_set
    config = {"model": "linear", "features": 3}
    assert cv.cv_mse(config, X, Y, base, 5, 2, 1) == cv.cv_mse(config, X, Y, base, 5, 2, 1)


def test_grid_has_rbf_linear_and_mean():
    grid = cv.make_grid(features=(5, 0), ks=(3,), sigmas=(1.0,), linear_features=(0,))
    assert [g["model"] for g in grid] == ["mean", "linear", "rbf", "rbf"]


def test_run_grid_sorts_and_writes(base, train_set, tmp_path):
    X, Y = train_set
    grid = cv.make_grid(features=(5,), ks=(3,), sigmas=(0.5, 1.0), linear_features=(3,),
                        epochs=3)
    rows = cv.run_grid(grid, X, Y, base, folds=3, repeats=1, seed=0, log=None)
    assert [r["cv_mse"] for r in rows] == sorted(r["cv_mse"] for r in rows)
    lines = cv.write_rows(rows, str(tmp_path), 3, 1, len(Y))
    assert (tmp_path / cv.OUT_CSV).is_file()
    assert len(lines) == 4 + len(grid)


def test_fine_grid_varies_rates_and_epochs(base, train_set):
    grid = cv.make_fine_grid(ks=(3,), sigmas=(1.0,), factors=(0.5, 2.0), epochs=(2, 4))
    assert len(grid) == 4 and all(g["features"] == 0 for g in grid)
    X, Y = train_set
    slow = cv.fit_predict(grid[0], X[:16], Y[:16], X[16:], base, seed=0)
    fast = cv.fit_predict(grid[3], X[:16], Y[:16], X[16:], base, seed=0)
    assert not np.allclose(slow, fast)

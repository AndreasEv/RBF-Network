"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Tests for reference.py (run from code/: python3 -m pytest tests -q).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metrics  # noqa: E402
from reference import LinearModel, MeanModel  # noqa: E402


def test_mean_model_predicts_the_training_mean():
    model = MeanModel([1.0, 2.0, 6.0])
    assert model.predict(np.zeros((4, 7))).tolist() == [3.0] * 4


def test_mean_model_has_r2_zero_on_its_own_training_set():
    Y = np.array([0.5, -1.0, 1.2, 0.3])
    assert metrics.r2(Y, MeanModel(Y).predict(np.zeros((4, 2)))) == pytest.approx(0.0)


def test_linear_model_recovers_an_exact_linear_relation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    Y = X @ np.array([1.5, -2.0, 0.5]) + 0.7
    model = LinearModel(X, Y)
    assert np.allclose(model.weights, [1.5, -2.0, 0.5])
    assert model.bias == pytest.approx(0.7)
    assert np.allclose(model.predict(X), Y)


def test_linear_model_fits_exactly_when_features_outnumber_patterns():
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(21, 53)), rng.normal(size=21)
    assert np.allclose(LinearModel(X, Y).predict(X), Y)   # minimum-norm interpolation


def test_linear_model_on_some_features():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(25, 4))
    Y = 3 * X[:, 2] - 1
    model = LinearModel(X, Y, features=[2])
    assert model.weights.tolist() == pytest.approx([3.0])
    assert np.allclose(model.predict(X), Y)


def test_linear_model_rejects_mismatched_sizes():
    with pytest.raises(ValueError):
        LinearModel(np.zeros((3, 2)), np.zeros(4))

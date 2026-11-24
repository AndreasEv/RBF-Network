"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Unit tests for metrics.py: MSE, RMSE and R^2
(run from code/: python3 -m pytest tests -q).
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import mse, r2, rmse  # noqa: E402


def test_known_values():
    t = [1.0, 2.0, 3.0, 4.0]
    y = [1.0, 2.0, 4.0, 2.0]
    # errors 0, 0, -1, 2 -> squared 0, 0, 1, 4
    assert mse(t, y) == pytest.approx(5 / 4)
    assert rmse(t, y) == pytest.approx(math.sqrt(5 / 4))
    # mean(t) = 2.5, SS_tot = 2.25 + 0.25 + 0.25 + 2.25 = 5
    assert r2(t, y) == pytest.approx(1 - 5 / 5)


def test_perfect_fit():
    t = [-1.0, 0.3, 2.5]
    assert mse(t, t) == 0.0
    assert rmse(t, t) == 0.0
    assert r2(t, t) == 1.0


def test_predicting_the_mean_gives_zero_r2_and_worse_is_negative():
    t = [1.0, 2.0, 3.0]
    assert r2(t, [2.0, 2.0, 2.0]) == pytest.approx(0.0)
    assert r2(t, [3.0, 2.0, 1.0]) < 0


def test_constant_targets():
    assert r2([2.0, 2.0], [2.0, 2.0]) == 1.0
    assert math.isnan(r2([2.0, 2.0], [1.0, 3.0]))


def test_bad_inputs():
    with pytest.raises(ValueError):
        mse([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        r2([], [])

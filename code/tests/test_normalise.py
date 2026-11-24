"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Unit tests for normalise.py and data.load_normalised
(run from code/: python3 -m pytest tests -q).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_normalised, read_dataset, write_dataset  # noqa: E402
from normalise import MinMaxScaler, normalise_train_test  # noqa: E402
from parameters import read_parameters  # noqa: E402

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN = np.array([[0.0, 10.0, 5.0],
                  [2.0, 20.0, 5.0],
                  [4.0, 30.0, 5.0]])   # last column is constant
TEST = np.array([[1.0, 40.0, 7.0],
                 [-2.0, 15.0, 5.0]])


def test_hand_computed_values():
    Xn = MinMaxScaler.fit(TRAIN).transform(TRAIN)
    assert Xn.tolist() == [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 1.0, 0.0]]


def test_training_values_land_in_0_1_with_min_0_and_max_1():
    rng = np.random.default_rng(0)
    X = rng.normal(50, 20, size=(21, 53))
    Xn = MinMaxScaler.fit(X).transform(X)
    assert Xn.min() >= 0.0 and Xn.max() <= 1.0
    assert np.allclose(Xn.min(axis=0), 0.0) and np.allclose(Xn.max(axis=0), 1.0)


def test_constant_feature_gives_zeros_not_nan():
    scaler = MinMaxScaler.fit(TRAIN)
    assert scaler.constant.tolist() == [False, False, True]
    tr, te = scaler.transform(TRAIN), scaler.transform(TEST)
    assert np.all(np.isfinite(tr)) and np.all(np.isfinite(te))
    assert np.all(tr[:, 2] == 0.0) and np.all(te[:, 2] == 0.0)


def test_all_constant_features():
    X = np.ones((4, 3))
    assert np.array_equal(MinMaxScaler.fit(X).transform(X), np.zeros((4, 3)))


def test_test_set_uses_training_statistics():
    tr, te, scaler = normalise_train_test(TRAIN, TEST)
    # (1 - 0) / 4 and (40 - 10) / 20 with the TRAINING min and range
    assert te[0, :2].tolist() == [0.25, 1.5]
    assert te[1, :2].tolist() == [-0.5, 0.25]
    assert scaler.minimum.tolist() == [0.0, 10.0, 5.0]
    # fitting on the test set would give different numbers
    assert not np.allclose(te, MinMaxScaler.fit(TEST).transform(TEST))


def test_test_values_outside_training_range_are_not_clipped():
    _, te, _ = normalise_train_test(TRAIN, TEST)
    assert te.max() > 1.0 and te.min() < 0.0


def test_single_row_and_wrong_width():
    scaler = MinMaxScaler.fit(TRAIN)
    assert scaler.transform(TRAIN[1]).tolist() == [0.5, 0.5, 0.0]
    with pytest.raises(ValueError, match="expected 3 features"):
        scaler.transform(np.zeros((2, 4)))
    with pytest.raises(ValueError):
        MinMaxScaler.fit(np.zeros((0, 3)))


def test_training_data_is_not_changed():
    X = TRAIN.copy()
    MinMaxScaler.fit(X).transform(X)
    assert np.array_equal(X, TRAIN)


def test_load_normalised_on_committed_files():
    params = read_parameters(os.path.join(CODE_DIR, "parameters.txt"))
    (tr_n, tr_y, tr_x), (te_n, te_y, te_x), scaler = load_normalised(params, CODE_DIR)
    assert tr_x.shape == (21, 53) and te_x.shape == (10, 53)
    assert tr_x.min() >= 0.0 and tr_x.max() <= 1.0
    assert np.all(np.isfinite(te_x))
    # Y stays unscaled
    raw = read_dataset(os.path.join(CODE_DIR, params["trainFile"]), 53)
    assert np.array_equal(tr_y, raw[1])
    assert np.allclose(scaler.transform(raw[2]), tr_x)


def test_load_normalised_stats_ignore_test_set(tmp_path):
    params = {"numInputNeurons": 2, "trainFile": "tr.txt", "testFile": "te.txt"}
    write_dataset(tmp_path / "tr.txt", ["a", "b"], np.array([1.0, 2.0]),
                  np.array([[0.0, 3.0], [10.0, 3.0]]))
    write_dataset(tmp_path / "te.txt", ["c"], np.array([0.0]), np.array([[100.0, 9.0]]))
    _, (_, _, te_x), _ = load_normalised(params, tmp_path)
    assert te_x.tolist() == [[10.0, 0.0]]

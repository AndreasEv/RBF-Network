"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Tests for features.py and the numFeatures parameter
(run from code/: python3 -m pytest tests -q).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_normalised, normalise_and_select, read_feature_names  # noqa: E402
from features import correlations, select_features  # noqa: E402
from parameters import ParameterError, parse_parameters  # noqa: E402
from prepare_data import prepare  # noqa: E402
from rbf import RBF  # noqa: E402

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_correlations_match_numpy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(21, 6))
    Y = rng.normal(size=21)
    expected = [abs(np.corrcoef(X[:, j], Y)[0, 1]) for j in range(6)]
    assert np.allclose(correlations(X, Y), expected)


def test_constant_feature_has_zero_correlation():
    X = np.column_stack([np.ones(5), np.arange(5.0)])
    assert correlations(X, np.arange(5.0)).tolist() == pytest.approx([0.0, 1.0])


def test_select_picks_the_most_correlated_columns():
    rng = np.random.default_rng(1)
    Y = rng.normal(size=30)
    noise = rng.normal(size=(30, 4))
    X = np.column_stack([noise[:, 0], Y, noise[:, 1], -2 * Y + 0.01 * noise[:, 2], noise[:, 3]])
    assert select_features(X, Y, 2).tolist() == [1, 3]   # sorted, sign ignored


def test_zero_or_too_many_keeps_everything():
    X = np.arange(12.0).reshape(4, 3)
    Y = np.arange(4.0)
    assert select_features(X, Y, 0).tolist() == [0, 1, 2]
    assert select_features(X, Y, 5).tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        select_features(X, Y, -1)


def test_choice_never_looks_at_the_test_set():
    rng = np.random.default_rng(2)
    X_train, Y_train = rng.normal(size=(21, 10)), rng.normal(size=21)
    params = {"numFeatures": 3}
    a = normalise_and_select(params, X_train, Y_train, rng.normal(size=(10, 10)))
    b = normalise_and_select(params, X_train, Y_train, 100 * rng.normal(size=(10, 10)))
    assert a[3].tolist() == b[3].tolist()
    assert a[0].shape == (21, 3) and a[1].shape == (10, 3)


def _params_text(**changes):
    values = {"numHiddenLayerNeurons": "3", "numInputNeurons": "53", "numOutputNeurons": "1",
              "learningRates": "0.01 0.001 0.0005", "sigmas": "0.5", "maxIterations": "3",
              "randomSeed": "42", "centresFile": "centerVectors.txt",
              "trainFile": "training.txt", "testFile": "test.txt"}
    values.update(changes)
    return [f"{k} {v}" for k, v in values.items()]


def test_num_features_parameter():
    assert parse_parameters(_params_text())["numNetworkInputs"] == 53
    params = parse_parameters(_params_text(numFeatures="5"))
    assert params["numFeatures"] == 5 and params["numNetworkInputs"] == 5
    with pytest.raises(ParameterError):
        parse_parameters(_params_text(numFeatures="54"))
    with pytest.raises(ParameterError):
        parse_parameters(_params_text(numFeatures="-1"))


def test_pipeline_with_selected_features(tmp_path):
    params = parse_parameters(_params_text(numFeatures="5"))
    selwood = os.path.join(CODE_DIR, "selwood.txt")
    _, _, centres = prepare(params, str(tmp_path), selwood_path=selwood)
    assert centres.shape == (3, 5)
    train, test, scaler = load_normalised(params, str(tmp_path))
    assert train[2].shape == (21, 5) and test[2].shape == (10, 5)
    assert scaler.columns.tolist() == prepare.columns.tolist()
    net = RBF.from_parameters(params, str(tmp_path))
    assert net.num_inputs == 5


def test_feature_names_from_the_header():
    names = read_feature_names(os.path.join(CODE_DIR, "selwood.txt"))
    assert len(names) == 53 and names[0] == "ATCH1"

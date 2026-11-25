"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Checks that the committed baseline files in code/ agree with each other:
parameters.txt, training.txt, test.txt, centerVectors.txt and the three
output files of main.py (results.txt, weights.txt, predictions.txt).
(run from code/: python3 -m pytest tests -q)
"""

import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metrics  # noqa: E402
from data import load_normalised, read_centres, read_dataset  # noqa: E402
from parameters import read_parameters  # noqa: E402
from prepare_data import prepare  # noqa: E402
from results import read_weights  # noqa: E402

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path(name):
    return os.path.join(CODE_DIR, name)


@pytest.fixture(scope="module")
def params():
    return read_parameters(path("parameters.txt"))


def read_predictions():
    rows = [line.split() for line in open(path("predictions.txt"), encoding="utf-8") if line.strip()]
    return [r[0] for r in rows], np.array([[float(r[1]), float(r[2])] for r in rows])


def read_results():
    return np.loadtxt(path("results.txt"), ndmin=2)


# ---------- parameters and input files ----------

def test_parameters_load(params):
    assert params["numInputNeurons"] == 53
    assert params["numOutputNeurons"] == 1
    assert len(params["sigmas"]) == params["numHiddenLayerNeurons"]
    assert len(params["learningRates"]) == 3


def test_centres_match_parameters(params):
    centres = read_centres(path(params["centresFile"]))
    assert centres.shape == (params["numHiddenLayerNeurons"], params["numInputNeurons"])
    assert np.all(np.isfinite(centres))


def test_split_is_21_train_10_test(params):
    train_names, _, train_x = read_dataset(path(params["trainFile"]), 53)
    test_names, _, test_x = read_dataset(path(params["testFile"]), 53)
    assert len(train_names) == 21 and len(test_names) == 10
    assert train_x.shape == (21, 53) and test_x.shape == (10, 53)
    assert not set(train_names) & set(test_names)


def test_committed_files_are_reproducible(params, tmp_path):
    """prepare_data.py with the committed parameters gives back exactly the
    committed training, test and centres files (no stale files)."""
    for name in ("parameters.txt", "selwood.txt"):
        shutil.copy(path(name), tmp_path / name)
    prepare(params, str(tmp_path))
    for name in (params["trainFile"], params["testFile"]):
        assert (tmp_path / name).read_text() == open(path(name)).read()
    np.testing.assert_allclose(read_centres(str(tmp_path / params["centresFile"])),
                               read_centres(path(params["centresFile"])))


# ---------- output files of the baseline run ----------

def test_results_has_one_line_per_epoch(params):
    results = read_results()
    assert results.shape == (params["maxIterations"], 3)
    np.testing.assert_array_equal(results[:, 0], np.arange(1, params["maxIterations"] + 1))
    assert np.all(results[:, 1:] >= 0)


def test_weights_match_parameters(params):
    net = read_weights(path("weights.txt"))
    k = params["numHiddenLayerNeurons"]
    assert net.weights.shape == (k,)
    assert net.centres.shape == (k, params["numInputNeurons"])
    assert net.sigmas.shape == (k,)
    assert np.all(net.sigmas > 0)


def test_predictions_are_the_test_set(params):
    names, values = read_predictions()
    test_names, test_y, _ = read_dataset(path(params["testFile"]), 53)
    assert names == list(test_names)
    np.testing.assert_allclose(values[:, 0], test_y, atol=1e-4)


def test_outputs_agree_with_each_other(params):
    """The saved weights reproduce predictions.txt and the last line of
    results.txt (final train and test MSE)."""
    net = read_weights(path("weights.txt"))
    train, test, _ = load_normalised(params, CODE_DIR)
    _, values = read_predictions()
    np.testing.assert_allclose(net.predict(test[2]), values[:, 1], atol=1e-4)
    last = read_results()[-1]
    assert metrics.mse(train[1], net.predict(train[2])) == pytest.approx(last[1], abs=1e-6)
    assert metrics.mse(test[1], net.predict(test[2])) == pytest.approx(last[2], abs=1e-6)


def test_baseline_does_not_overfit_badly():
    """Tuned baseline: test MSE clearly better than predicting the mean
    (R^2 > 0.4) and not more than 3x the training MSE."""
    _, values = read_predictions()
    assert metrics.r2(values[:, 0], values[:, 1]) > 0.4
    last = read_results()[-1]
    assert last[2] < 3 * last[1]

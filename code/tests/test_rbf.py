"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Unit tests for rbf.py, the forward pass of the RBF network
(run from code/: python3 -m pytest tests -q).
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import DataError, load_normalised, write_centres  # noqa: E402
from parameters import read_parameters  # noqa: E402
from rbf import INIT_RANGE, RBF  # noqa: E402

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def small():
    """2 centres in 2-D with known weights and bias."""
    return RBF(centres=[[0.0, 0.0], [1.0, 2.0]], sigmas=[1.0, 2.0],
               weights=[2.0, -1.0], bias=0.5)


def test_phi_is_one_at_the_centre(small):
    assert np.allclose(small.hidden([0.0, 0.0])[0], 1.0)
    assert np.allclose(small.hidden([1.0, 2.0])[1], 1.0)


def test_phi_drops_with_distance(small):
    phis = [small.hidden([d, 0.0])[0] for d in (0.0, 0.5, 1.0, 2.0, 4.0)]
    assert all(a > b for a, b in zip(phis, phis[1:]))
    assert all(0.0 < p <= 1.0 for p in phis)


def test_hand_computed_example(small):
    x = [1.0, 0.0]
    # ||x - c1||^2 = 1, sigma 1 -> exp(-1/2); ||x - c2||^2 = 4, sigma 2 -> exp(-4/8)
    phi1, phi2 = math.exp(-0.5), math.exp(-0.5)
    assert np.allclose(small.hidden(x), [phi1, phi2])
    assert math.isclose(small.predict(x), 2.0 * phi1 - 1.0 * phi2 + 0.5)


def test_gaussian_form_uses_two_sigma_squared():
    net = RBF([[0.0]], [3.0], weights=[1.0], bias=0.0)
    # at distance sigma the Gaussian is exp(-1/2)
    assert math.isclose(net.predict([3.0]), math.exp(-0.5))


def test_shapes_single_and_batch(small):
    assert small.hidden([0.0, 0.0]).shape == (2,)
    assert isinstance(small.predict([0.0, 0.0]), float)
    X = np.zeros((5, 2))
    assert small.hidden(X).shape == (5, 2)
    assert small.predict(X).shape == (5,)


def test_batch_matches_single(small):
    X = np.random.default_rng(0).normal(size=(7, 2))
    batch = small.predict(X)
    one_by_one = [small.predict(x) for x in X]
    assert np.allclose(batch, one_by_one)
    assert np.allclose(small.hidden(X), [small.hidden(x) for x in X])


def test_shared_sigma_equals_repeated_sigmas():
    C = [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]
    shared = RBF(C, 0.7, weights=[1.0, 2.0, 3.0], bias=0.1)
    per_neuron = RBF(C, [0.7, 0.7, 0.7], weights=[1.0, 2.0, 3.0], bias=0.1)
    assert shared.sigmas.tolist() == [0.7, 0.7, 0.7]
    X = np.random.default_rng(1).normal(size=(4, 2))
    assert np.allclose(shared.predict(X), per_neuron.predict(X))


def test_per_neuron_sigmas_are_used():
    # same centre, different widths: the wider one is larger away from it
    net = RBF([[0.0], [0.0]], [0.5, 2.0], weights=[1.0, 1.0], bias=0.0)
    phi = net.hidden([1.0])
    assert phi[1] > phi[0]
    assert np.allclose(phi, [math.exp(-1 / 0.5), math.exp(-1 / 8.0)])


def test_output_is_bias_far_from_all_centres(small):
    assert math.isclose(small.predict([1e3, 1e3]), 0.5)


def test_random_init_is_seeded_and_in_range():
    C = np.zeros((6, 3))
    a = RBF(C, 1.0, seed=42)
    b = RBF(C, 1.0, seed=42)
    c = RBF(C, 1.0, seed=7)
    assert np.array_equal(a.weights, b.weights) and a.bias == b.bias
    assert not np.array_equal(a.weights, c.weights)
    assert a.weights.shape == (6,)
    assert np.all(np.abs(a.weights) <= INIT_RANGE) and abs(a.bias) <= INIT_RANGE


def test_bad_arguments_are_rejected():
    with pytest.raises(ValueError):
        RBF([[0.0], [1.0]], [1.0, 1.0, 1.0])      # 3 sigmas for 2 centres
    with pytest.raises(ValueError):
        RBF([[0.0], [1.0]], [1.0, 0.0])           # sigma 0
    with pytest.raises(ValueError):
        RBF([[0.0], [1.0]], 1.0, weights=[1.0])   # 1 weight for 2 centres
    with pytest.raises(ValueError):
        RBF([[0.0, 0.0]], 1.0).predict([1.0, 2.0, 3.0])  # wrong feature count


def test_from_parameters_reads_centres_and_sigmas(tmp_path):
    centres = np.arange(6, dtype=float).reshape(2, 3) / 10
    write_centres(tmp_path / "c.txt", centres)
    params = {"centresFile": "c.txt", "numHiddenLayerNeurons": 2,
              "numInputNeurons": 3, "sigmas": [0.5, 1.5], "randomSeed": 3}
    net = RBF.from_parameters(params, str(tmp_path))
    assert np.allclose(net.centres, centres)
    assert net.sigmas.tolist() == [0.5, 1.5]
    again = RBF.from_parameters(params, str(tmp_path))
    assert np.array_equal(net.weights, again.weights) and net.bias == again.bias


def test_from_parameters_catches_wrong_centre_count(tmp_path):
    write_centres(tmp_path / "c.txt", np.zeros((3, 2)))
    params = {"centresFile": "c.txt", "numHiddenLayerNeurons": 4,
              "numInputNeurons": 2, "sigmas": [1.0] * 4, "randomSeed": 0}
    with pytest.raises(DataError):
        RBF.from_parameters(params, str(tmp_path))


def test_real_files_forward_pass():
    """The committed parameters.txt + centerVectors.txt build a working net."""
    params = read_parameters(os.path.join(CODE_DIR, "parameters.txt"))
    net = RBF.from_parameters(params, CODE_DIR)
    assert net.centres.shape == (params["numHiddenLayerNeurons"], params["numInputNeurons"])
    train, test, _ = load_normalised(params, CODE_DIR)
    for _, Y, X in (train, test):
        y = net.predict(X)
        phi = net.hidden(X)
        assert y.shape == Y.shape and np.all(np.isfinite(y))
        assert np.all((phi > 0) & (phi <= 1))

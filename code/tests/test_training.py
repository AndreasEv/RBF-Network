"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Unit tests for the variable-centres training: RBF.update() (eqs 6-9 of the
course notes) and the epoch loop in training.py
(run from code/: python3 -m pytest tests -q).
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parameters import read_parameters  # noqa: E402
from rbf import MIN_SIGMA, RBF  # noqa: E402
from training import mse, train, train_from_parameters  # noqa: E402

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_net(seed=0, k=3, d=2):
    rng = np.random.default_rng(seed)
    return RBF(rng.uniform(0, 1, (k, d)), rng.uniform(0.5, 1.5, k),
               weights=rng.uniform(-1, 1, k), bias=0.3)


def copy(net):
    return RBF(net.centres.copy(), net.sigmas.copy(), net.weights.copy(), net.bias)


def half_sq_error(net, x, t):
    return 0.5 * (t - net.predict(x)) ** 2


# ---------- one update step ----------

def test_hand_computed_single_step():
    # 1 centre at 0 in 1-D, sigma 1, w 2, b 0.5; pattern x = 1, t = 3
    net = RBF([[0.0]], [1.0], weights=[2.0], bias=0.5)
    phi = math.exp(-0.5)
    y = 2.0 * phi + 0.5
    eps = 3.0 - y
    ew, ec, es = 0.1, 0.2, 0.3
    got = net.update([1.0], 3.0, ew, ec, es)
    assert math.isclose(got, eps)
    assert math.isclose(net.weights[0], 2.0 + ew * eps * phi)                 # eq 6
    assert math.isclose(net.bias, 0.5 + ew * eps)                             # eq 7
    # eqs 8 and 9 use the OLD weight 2.0, not the updated one
    assert math.isclose(net.centres[0, 0], 0.0 + ec * eps * 2.0 * phi * 1.0 / 1.0)
    assert math.isclose(net.sigmas[0], 1.0 + es * eps * 2.0 * phi * 1.0 / 1.0)


def test_updates_use_old_values_in_higher_dimensions():
    net = make_net(seed=3, k=4, d=5)
    old = copy(net)
    x, t = np.full(5, 0.4), 1.7
    ew, ec, es = 0.05, 0.07, 0.09
    net.update(x, t, ew, ec, es)

    diff = x - old.centres
    dist2 = np.sum(diff ** 2, axis=1)
    phi = np.exp(-dist2 / (2 * old.sigmas ** 2))
    eps = t - (phi @ old.weights + old.bias)
    assert np.allclose(net.weights, old.weights + ew * eps * phi)
    assert math.isclose(net.bias, old.bias + ew * eps)
    assert np.allclose(net.centres, old.centres + ec * eps * (old.weights * phi
                       / old.sigmas ** 2)[:, None] * diff)
    assert np.allclose(net.sigmas, old.sigmas + es * eps * old.weights * phi
                       * dist2 / old.sigmas ** 3)


@pytest.mark.parametrize("group", ["weights", "bias", "centres", "sigmas"])
def test_update_is_gradient_descent_finite_differences(group):
    """Each change equals -eta * dE/dp with E = 1/2 eps^2 (numerical gradient)."""
    net = make_net(seed=7, k=3, d=4)
    x, t, eta, h = np.array([0.2, 0.9, 0.5, 0.1]), 0.8, 1e-3, 1e-6

    values = getattr(net, group)
    flat = np.atleast_1d(np.array(values, dtype=float)).reshape(-1)
    grad = np.zeros_like(flat)
    for i in range(flat.size):
        for sign in (+1, -1):
            probe = copy(net)
            v = np.atleast_1d(np.array(getattr(probe, group), dtype=float)).reshape(-1)
            v[i] += sign * h
            setattr(probe, group, v.reshape(np.shape(values)) if group != "bias" else v[0])
            grad[i] += sign * half_sq_error(probe, x, t) / (2 * h)

    rates = {"weights": (eta, 0, 0), "bias": (eta, 0, 0),
             "centres": (1.0, eta, 0), "sigmas": (1.0, 0, eta)}[group]
    stepped = copy(net)
    stepped.update(x, t, *rates)
    change = (np.atleast_1d(np.array(getattr(stepped, group), dtype=float)).reshape(-1)
              - flat)
    assert np.allclose(change, -eta * grad, rtol=1e-5, atol=1e-10)


def test_zero_rates_keep_centres_and_sigmas_fixed():
    net = make_net()
    c, s = net.centres.copy(), net.sigmas.copy()
    net.update([0.3, 0.6], 2.0, 0.1, 0.0, 0.0)
    assert np.array_equal(net.centres, c)
    assert np.array_equal(net.sigmas, s)


def test_sigma_never_goes_below_min():
    net = RBF([[0.0]], [0.5], weights=[1.0], bias=0.0)
    # target far below y -> eps < 0, so a huge sigma rate drives sigma down
    net.update([0.4], -100.0, 0.0001, 0.0, 1000.0)
    assert net.sigmas[0] == MIN_SIGMA


def test_update_rejects_wrong_length():
    with pytest.raises(ValueError):
        make_net(d=2).update([1.0, 2.0, 3.0], 0.0, 0.1, 0.1, 0.1)


# ---------- the epoch loop ----------

def sin_data(n=40):
    x = np.linspace(0, 2 * np.pi, n)
    return x[:, None], np.sin(x)


def sin_net(seed=1):
    centres = np.linspace(0, 2 * np.pi, 6)[:, None]
    return RBF(centres, 1.0, seed=seed)


def test_learns_sin_x():
    """Sanity check for the report: the error on sin(x), x in [0, 2pi], drops."""
    X, Y = sin_data()
    net = sin_net()
    before = mse(net, X, Y)
    history = train(net, X, Y, (0.05, 0.01, 0.01), 300, seed=1)
    after = history[-1][1]
    assert after < before / 20
    assert after < 0.01
    # generalises between the training points
    Xt = np.linspace(0.1, 2 * np.pi - 0.1, 17)[:, None]
    assert mse(net, Xt, np.sin(Xt[:, 0])) < 0.02


def test_history_format_and_test_error():
    X, Y = sin_data(20)
    Xt, Yt = sin_data(7)
    history = train(sin_net(), X, Y, (0.05, 0.01, 0.01), 5, seed=0, X_test=Xt, Y_test=Yt)
    assert [h[0] for h in history] == [1, 2, 3, 4, 5]
    assert all(isinstance(h[1], float) and isinstance(h[2], float) for h in history)
    no_test = train(sin_net(), X, Y, (0.05, 0, 0), 2, seed=0)
    assert all(h[2] is None for h in no_test)


def test_training_mse_matches_mse_function():
    X, Y = sin_data(15)
    net = sin_net()
    history = train(net, X, Y, (0.05, 0.01, 0.01), 3, seed=2)
    assert math.isclose(history[-1][1], mse(net, X, Y))


def test_same_seed_same_run_and_different_seed_differs():
    X, Y = sin_data(20)
    a, b, c = sin_net(), sin_net(), sin_net()
    ha = train(a, X, Y, (0.05, 0.01, 0.01), 10, seed=5)
    hb = train(b, X, Y, (0.05, 0.01, 0.01), 10, seed=5)
    hc = train(c, X, Y, (0.05, 0.01, 0.01), 10, seed=6)
    assert ha == hb
    assert np.array_equal(a.centres, b.centres) and np.array_equal(a.sigmas, b.sigmas)
    assert ha != hc  # only the pattern order differs


def test_each_epoch_visits_every_pattern_once(monkeypatch):
    X, Y = sin_data(9)
    net = sin_net()
    seen = []
    real = RBF.update

    def spy(self, x, t, *rates):
        seen.append(float(x[0]))
        return real(self, x, t, *rates)

    monkeypatch.setattr(RBF, "update", spy)
    train(net, X, Y, (0.05, 0, 0), 3, seed=0)
    epochs = [seen[i * 9:(i + 1) * 9] for i in range(3)]
    for e in epochs:
        assert sorted(e) == sorted(X[:, 0].tolist())
    assert epochs[0] != epochs[1] or epochs[1] != epochs[2]  # order is shuffled


def test_fixed_centres_run_keeps_centres_and_sigmas():
    X, Y = sin_data(20)
    net = sin_net()
    c, s = net.centres.copy(), net.sigmas.copy()
    train(net, X, Y, (0.05, 0.0, 0.0), 20, seed=0)
    assert np.array_equal(net.centres, c) and np.array_equal(net.sigmas, s)


def test_bad_arguments():
    X, Y = sin_data(5)
    with pytest.raises(ValueError):
        train(sin_net(), X, Y[:3], (0.1, 0, 0), 1)
    with pytest.raises(ValueError):
        train(sin_net(), X, Y, (0.1, 0, 0), 0)


def test_selwood_run_from_parameters(tmp_path):
    """Short run on the real files: finite errors and training MSE goes down."""
    params = read_parameters(os.path.join(CODE_DIR, "parameters.txt"))
    params["maxIterations"] = 30
    net, history, train_set, test_set = train_from_parameters(params, CODE_DIR)
    assert len(history) == 30
    assert all(np.isfinite(h[1]) and np.isfinite(h[2]) for h in history)
    start = RBF.from_parameters(params, CODE_DIR)
    assert history[-1][1] < mse(start, train_set[2], train_set[1])

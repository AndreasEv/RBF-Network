"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Training loop for the RBF network ("Learning Algorithm for variable
centres", lecture L7, slide 58):

  for each epoch (maxIterations of them):
      visit every training pattern once, in a new random order
      (the order comes from a generator seeded with randomSeed):
          compute y, then update w, b, centres and sigmas (RBF.update)
      at the end of the epoch record the training and test MSE

The MSE of a set is mean((t - y)^2) over its patterns, computed with the
network as it is at the end of the epoch.
"""

import numpy as np

from data import load_normalised
from rbf import RBF


def mse(net, X, Y):
    """Mean squared error of the network on patterns X with targets Y."""
    Y = np.asarray(Y, dtype=float).reshape(-1)
    return float(np.mean((Y - net.predict(np.atleast_2d(X))) ** 2))


def train(net, X_train, Y_train, learning_rates, max_iterations, seed=None,
          X_test=None, Y_test=None):
    """Train `net` in place for `max_iterations` epochs.

    learning_rates = (eta_w, eta_c, eta_sigma), the learningRates parameter.
    Returns a list of (epoch, train MSE, test MSE) with epoch = 1..max_iterations;
    the test MSE is None when no test set is given.
    """
    eta_w, eta_c, eta_sigma = learning_rates
    X_train = np.atleast_2d(np.asarray(X_train, dtype=float))
    Y_train = np.asarray(Y_train, dtype=float).reshape(-1)
    if len(X_train) != len(Y_train):
        raise ValueError(f"{len(X_train)} training patterns but {len(Y_train)} targets")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    has_test = X_test is not None and Y_test is not None

    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, max_iterations + 1):
        for p in rng.permutation(len(X_train)):
            net.update(X_train[p], Y_train[p], eta_w, eta_c, eta_sigma)
        test_err = mse(net, X_test, Y_test) if has_test else None
        history.append((epoch, mse(net, X_train, Y_train), test_err))
    return history


def train_from_parameters(params, folder="."):
    """Build the network from parameters.txt, train it and return
    (net, history, train, test), where train/test are (names, Y, X_normalised)."""
    train_set, test_set, _ = load_normalised(params, folder)
    net = RBF.from_parameters(params, folder)
    history = train(net, train_set[2], train_set[1], params["learningRates"],
                    params["maxIterations"], seed=params["randomSeed"],
                    X_test=test_set[2], Y_test=test_set[1])
    return net, history, train_set, test_set

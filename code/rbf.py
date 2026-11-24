"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

The RBF network: 53 inputs, one hidden layer of K Gaussian neurons and one
linear output neuron with a bias.

Hidden neuron j has a centre c_j (a point in the normalised input space) and
a width sigma_j. Its output for an input x is the Gaussian

    phi_j(x) = exp( -||x - c_j||^2 / (2 * sigma_j^2) )

which is 1 when x is exactly on the centre and drops towards 0 as x moves
away from it. sigma_j sets how fast it drops.

The output neuron is linear:

    y(x) = sum_j w_j * phi_j(x) + b

Both formulas are equations 1 and 2 of the course notes ("RBF Networks -
Learning Algorithm for variable centres", lecture L7), where the bias is the
weight w_bo from a bias unit.

Initial values:
  centres  read from centresFile (centerVectors.txt, made by prepare_data.py)
  sigmas   from the sigmas parameter (one shared value or one per neuron)
  w, b     small random numbers, uniform in [-INIT_RANGE, INIT_RANGE] = [-1, 1]
           as in step 3 of the course notes' pseudocode, drawn with
           randomSeed, so the same seed gives the same network

This module only does the forward pass. Training (the variable-centres
updates of w, b, c and sigma) is added on top of it.
"""

import os

import numpy as np

from data import read_centres
from kmeans import squared_distances

# initial output weights and bias are uniform in [-INIT_RANGE, INIT_RANGE];
# the course notes initialise the weights/coefficients in [-1, 1]
INIT_RANGE = 1.0


class RBF:
    """An RBF network with Gaussian hidden neurons and one linear output.

    Attributes (numpy arrays, changed in place by training):
      centres  (K, d)  one centre per hidden neuron
      sigmas   (K,)    Gaussian width of each hidden neuron, all > 0
      weights  (K,)    hidden -> output weights
      bias     float   output bias
    """

    def __init__(self, centres, sigmas, weights=None, bias=None, seed=None):
        """Build a network from its centres and sigmas.

        `sigmas` may be one number (shared by every neuron) or K numbers.
        If `weights` / `bias` are not given they are drawn uniformly from
        [-INIT_RANGE, INIT_RANGE] with a generator seeded by `seed`.
        """
        self.centres = np.array(centres, dtype=float, ndmin=2)
        k = self.centres.shape[0]

        sigmas = np.array(sigmas, dtype=float).reshape(-1)
        if sigmas.size == 1:
            sigmas = np.full(k, sigmas[0])
        if sigmas.size != k:
            raise ValueError(f"need 1 or {k} sigmas (one per centre), got {sigmas.size}")
        if np.any(sigmas <= 0):
            raise ValueError("every sigma must be > 0")
        self.sigmas = sigmas

        rng = np.random.default_rng(seed)
        if weights is None:
            weights = rng.uniform(-INIT_RANGE, INIT_RANGE, size=k)
        weights = np.array(weights, dtype=float).reshape(-1)
        if weights.size != k:
            raise ValueError(f"need {k} weights (one per centre), got {weights.size}")
        self.weights = weights
        self.bias = float(rng.uniform(-INIT_RANGE, INIT_RANGE) if bias is None else bias)

    @classmethod
    def from_parameters(cls, params, folder="."):
        """Build the initial network described by parameters.txt.

        Centres come from centresFile (it must have numHiddenLayerNeurons rows
        of numInputNeurons values), sigmas from `sigmas`, and the random
        weights and bias from randomSeed.
        """
        path = os.path.join(folder, params["centresFile"])
        centres = read_centres(path, params["numHiddenLayerNeurons"],
                               params["numInputNeurons"])
        return cls(centres, params["sigmas"], seed=params["randomSeed"])

    @property
    def num_hidden(self):
        return self.centres.shape[0]

    @property
    def num_inputs(self):
        return self.centres.shape[1]

    def _as_batch(self, X):
        """Return (X as a (N, d) array, whether the input was one pattern)."""
        X = np.asarray(X, dtype=float)
        single = X.ndim == 1
        X = np.atleast_2d(X)
        if X.ndim != 2 or X.shape[1] != self.num_inputs:
            raise ValueError(f"expected patterns with {self.num_inputs} features, "
                             f"got shape {np.shape(X)}")
        return X, single

    def hidden(self, X):
        """Gaussian outputs phi_j(x) of the hidden neurons.

        X is one pattern (shape (d,)) -> returns shape (K,), or
        N patterns (shape (N, d))     -> returns shape (N, K).
        """
        X, single = self._as_batch(X)
        dist2 = squared_distances(X, self.centres)             # (N, K)
        phi = np.exp(-dist2 / (2.0 * self.sigmas ** 2))
        return phi[0] if single else phi

    def predict(self, X):
        """Network output y(x) = sum_j w_j phi_j(x) + b.

        One pattern -> a float; N patterns -> an array of shape (N,).
        """
        phi = self.hidden(X)
        y = phi @ self.weights + self.bias
        return float(y) if np.ndim(y) == 0 else y

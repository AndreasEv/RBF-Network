"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Initial RBF centres. Two methods, chosen with the centresInit parameter:

  kmeans   our own k-means (no libraries other than numpy). The centres are
           the means of K groups of similar training patterns, so each
           Gaussian starts in a dense part of the input space and together
           they cover the training data.
  samples  K distinct training patterns picked at random (for comparison).

Both are seeded, so the same seed always gives the same centres.

k-means (Lloyd's algorithm) with k-means++ initialisation:
  1. Initialisation (k-means++): the first centre is a random training
     pattern; every next centre is a training pattern picked with
     probability proportional to its squared distance from the nearest
     centre chosen so far. This spreads the starting centres out.
  2. Assignment: every pattern joins the cluster of its nearest centre
     (Euclidean distance).
  3. Update: every centre moves to the mean of the patterns in its cluster.
     If a cluster is empty, its centre jumps to the pattern that is
     farthest from its own centre, so we always return K distinct centres.
  4. Repeat 2-3 until no assignment changes or maxIterations is reached.
"""

import numpy as np

CENTRE_METHODS = ("kmeans", "samples")


def squared_distances(X, C):
    """Matrix D with D[i, j] = ||X[i] - C[j]||^2. X: (N, d), C: (K, d)."""
    X = np.asarray(X, dtype=float)
    C = np.asarray(C, dtype=float)
    diff = X[:, None, :] - C[None, :, :]
    return np.einsum("nkd,nkd->nk", diff, diff)


def _check(X, k):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"expected a non-empty 2-D array, got shape {X.shape}")
    if not 1 <= k <= X.shape[0]:
        raise ValueError(
            f"number of centres must be between 1 and the number of training "
            f"patterns ({X.shape[0]}), got {k}")
    return X


def sample_centres(X, k, seed):
    """K distinct training patterns, chosen at random with `seed`."""
    X = _check(X, k)
    rng = np.random.default_rng(seed)
    return X[rng.choice(X.shape[0], size=k, replace=False)].copy()


def kmeans_plus_plus(X, k, rng):
    """k-means++ starting centres (step 1 above). Returns a (k, d) array."""
    n = X.shape[0]
    chosen = [int(rng.integers(n))]
    nearest = squared_distances(X, X[chosen])[:, 0]
    for _ in range(1, k):
        total = nearest.sum()
        if total == 0:  # every pattern is already a centre: pick an unused one
            unused = np.setdiff1d(np.arange(n), chosen)
            nxt = int(rng.choice(unused))
        else:
            nxt = int(rng.choice(n, p=nearest / total))
        chosen.append(nxt)
        nearest = np.minimum(nearest, squared_distances(X, X[nxt:nxt + 1])[:, 0])
    return X[chosen].copy()


def kmeans(X, k, seed, max_iterations=100):
    """Our own k-means on the rows of X.

    Returns (centres, labels, iterations):
      centres     (k, d) array, the final cluster means
      labels      (N,) array, the cluster of every pattern
      iterations  number of assignment/update rounds that were run
    """
    X = _check(X, k)
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
    rng = np.random.default_rng(seed)
    centres = kmeans_plus_plus(X, k, rng)
    labels = None
    for iteration in range(1, max_iterations + 1):
        # assignment step: nearest centre for every pattern
        distances = squared_distances(X, centres)
        new_labels = distances.argmin(axis=1)
        if labels is not None and np.array_equal(new_labels, labels):
            return centres, labels, iteration - 1
        labels = new_labels
        # update step: each centre becomes the mean of its cluster
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                centres[j] = members.mean(axis=0)
            else:
                # empty cluster: move it to the pattern worst served by its centre
                worst = distances[np.arange(len(X)), labels].argmax()
                centres[j] = X[worst]
                labels[worst] = j
                distances[worst] = 0.0
    return centres, labels, max_iterations


def initial_centres(X, k, method, seed, max_iterations=100):
    """Initial centres with the chosen method ("kmeans" or "samples")."""
    if method == "kmeans":
        return kmeans(X, k, seed, max_iterations)[0]
    if method == "samples":
        return sample_centres(X, k, seed)
    raise ValueError(f"unknown centre method {method!r} (use one of {CENTRE_METHODS})")


def quantisation_error(X, centres):
    """Mean squared distance from every pattern to its nearest centre."""
    return float(squared_distances(X, centres).min(axis=1).mean())

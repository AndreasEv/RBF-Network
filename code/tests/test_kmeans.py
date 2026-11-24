"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Unit tests for kmeans.py, the centres file and the centres made by
prepare_data.py (run from code/: python3 -m pytest tests -q).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import DataError, load_normalised, read_centres, write_centres  # noqa: E402
from kmeans import (initial_centres, kmeans, quantisation_error,  # noqa: E402
                    sample_centres, squared_distances)
from parameters import read_parameters  # noqa: E402
from prepare_data import prepare  # noqa: E402

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELWOOD = os.path.join(CODE_DIR, "selwood.txt")

# three obvious clusters of 10 points around these centres
TRUE_CENTRES = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])


@pytest.fixture
def blobs():
    rng = np.random.default_rng(1)
    X = np.vstack([c + rng.normal(0, 0.3, size=(10, 2)) for c in TRUE_CENTRES])
    return X, np.repeat([0, 1, 2], 10)


@pytest.fixture(scope="module")
def selwood_train():
    params = read_parameters(os.path.join(CODE_DIR, "parameters.txt"))
    return load_normalised(params, CODE_DIR)[0][2]


def test_squared_distances_by_hand():
    D = squared_distances([[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [3.0, 4.0]])
    assert D.tolist() == [[0.0, 25.0], [2.0, 13.0]]


def test_kmeans_finds_obvious_clusters(blobs):
    X, truth = blobs
    centres, labels, _ = kmeans(X, 3, seed=0)
    # each found centre is close to one true centre, and each true one is used
    match = squared_distances(centres, TRUE_CENTRES).argmin(axis=1)
    assert sorted(match) == [0, 1, 2]
    assert np.allclose(centres, TRUE_CENTRES[match], atol=0.3)
    # the clusters are exactly the three blobs (up to renaming)
    for j in range(3):
        assert len(set(truth[labels == j])) == 1


def test_centres_are_the_means_of_their_clusters(blobs):
    X, _ = blobs
    centres, labels, _ = kmeans(X, 4, seed=3)
    for j in range(4):
        assert np.allclose(centres[j], X[labels == j].mean(axis=0))
    assert np.array_equal(labels, squared_distances(X, centres).argmin(axis=1))


@pytest.mark.parametrize("k", [1, 3, 8, 21])
def test_shape_and_finite(selwood_train, k):
    centres = kmeans(selwood_train, k, seed=42)[0]
    assert centres.shape == (k, 53)
    assert np.all(np.isfinite(centres))
    # centres are means of normalised patterns, so they stay in [0, 1]
    assert centres.min() >= 0.0 and centres.max() <= 1.0


def test_k_equals_n_puts_a_centre_on_every_pattern(selwood_train):
    centres = kmeans(selwood_train, 21, seed=0)[0]
    assert quantisation_error(selwood_train, centres) == pytest.approx(0.0)


def test_one_centre_is_the_mean(selwood_train):
    centres = kmeans(selwood_train, 1, seed=0)[0]
    assert np.allclose(centres[0], selwood_train.mean(axis=0))


def test_same_seed_same_centres_other_seed_can_differ(selwood_train):
    a = kmeans(selwood_train, 8, seed=42)[0]
    b = kmeans(selwood_train, 8, seed=42)[0]
    assert np.array_equal(a, b)
    others = [kmeans(selwood_train, 8, seed=s)[0] for s in range(1, 6)]
    assert any(not np.allclose(a, o) for o in others)


def test_no_empty_clusters_with_duplicate_points():
    X = np.array([[0.0, 0.0]] * 5 + [[1.0, 1.0]] * 5 + [[5.0, 5.0]])
    centres, labels, _ = kmeans(X, 3, seed=0)
    assert sorted(set(labels.tolist())) == [0, 1, 2]
    assert len({tuple(c) for c in centres}) == 3


def test_stops_at_max_iterations(selwood_train):
    _, _, iterations = kmeans(selwood_train, 8, seed=42, max_iterations=1)
    assert iterations == 1


def test_kmeans_beats_random_samples(selwood_train):
    km = quantisation_error(selwood_train, kmeans(selwood_train, 8, seed=42)[0])
    for seed in range(5):
        assert km < quantisation_error(selwood_train, sample_centres(selwood_train, 8, seed))


def test_sample_centres_are_distinct_training_rows(selwood_train):
    centres = sample_centres(selwood_train, 8, seed=42)
    rows = {tuple(r) for r in selwood_train}
    assert all(tuple(c) in rows for c in centres)
    assert len({tuple(c) for c in centres}) == 8
    assert np.array_equal(centres, sample_centres(selwood_train, 8, seed=42))


@pytest.mark.parametrize("k", [0, 22])
def test_bad_number_of_centres(selwood_train, k):
    with pytest.raises(ValueError, match="number of centres"):
        kmeans(selwood_train, k, seed=0)
    with pytest.raises(ValueError, match="number of centres"):
        sample_centres(selwood_train, k, seed=0)


def test_initial_centres_methods(selwood_train):
    assert np.array_equal(initial_centres(selwood_train, 5, "kmeans", 7),
                          kmeans(selwood_train, 5, 7)[0])
    assert np.array_equal(initial_centres(selwood_train, 5, "samples", 7),
                          sample_centres(selwood_train, 5, 7))
    with pytest.raises(ValueError, match="unknown centre method"):
        initial_centres(selwood_train, 5, "random", 7)


# ---- centerVectors.txt ----

def test_centres_file_round_trip(tmp_path):
    centres = np.random.default_rng(0).random((4, 53))
    write_centres(tmp_path / "c.txt", centres)
    assert np.array_equal(read_centres(tmp_path / "c.txt", 4, 53), centres)


def test_centres_file_checks(tmp_path):
    path = tmp_path / "c.txt"
    path.write_text("1,2,3\n4,5\n")
    with pytest.raises(DataError, match="line 2"):
        read_centres(path)
    path.write_text("1,2,3\n4,5,6\n")
    with pytest.raises(DataError, match="numHiddenLayerNeurons"):
        read_centres(path, num_centres=8)
    with pytest.raises(DataError, match="expected 53"):
        read_centres(path, num_features=53)
    with pytest.raises(DataError, match="not found"):
        read_centres(tmp_path / "nope.txt")


def test_committed_centres_file_matches_parameters_and_data():
    params = read_parameters(os.path.join(CODE_DIR, "parameters.txt"))
    centres = read_centres(os.path.join(CODE_DIR, params["centresFile"]),
                           params["numHiddenLayerNeurons"], params["numInputNeurons"])
    X = load_normalised(params, CODE_DIR)[0][2]
    expected = initial_centres(X, params["numHiddenLayerNeurons"], params["centresInit"],
                               params["randomSeed"], params["kmeansIterations"])
    assert np.allclose(centres, expected)


def _params_in(tmp_path, **changes):
    params = read_parameters(os.path.join(CODE_DIR, "parameters.txt"))
    params.update(trainFile="tr.txt", testFile="te.txt", centresFile="c.txt", **changes)
    (tmp_path / "selwood.txt").write_text(open(SELWOOD, encoding="utf-8").read())
    return params


@pytest.mark.parametrize("method", ["kmeans", "samples"])
def test_prepare_writes_centres_from_training_set(tmp_path, method):
    params = _params_in(tmp_path, numHiddenLayerNeurons=5, centresInit=method)
    _, _, centres = prepare(params, folder=tmp_path)
    from_file = read_centres(tmp_path / "c.txt", 5, 53)
    assert np.array_equal(from_file, centres)
    X = load_normalised(params, tmp_path)[0][2]
    assert np.allclose(centres, initial_centres(X, 5, method, params["randomSeed"]))


def test_prepare_rejects_more_centres_than_training_rows(tmp_path):
    params = _params_in(tmp_path, numHiddenLayerNeurons=22)
    with pytest.raises(DataError, match="only 21 training molecules"):
        prepare(params, folder=tmp_path)

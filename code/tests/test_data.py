"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Unit tests for data.py and prepare_data.py (run from code/: python3 -m pytest tests -q).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import DataError, parse_selwood, read_dataset, read_selwood, write_dataset  # noqa: E402
from parameters import read_parameters  # noqa: E402
from prepare_data import NUM_TRAIN, prepare, split_dataset, split_indices  # noqa: E402

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELWOOD = os.path.join(CODE_DIR, "selwood.txt")

SMALL = """\
Small Data Set

Compound, Activity, F1, F2
A1,0.5,1.0,2.0
A2,<-1*,3.0,-4.0E-01
A3,1.25,5,6

*a footnote, with a comma
"""


@pytest.fixture(scope="module")
def selwood():
    return read_selwood(SELWOOD)


# ---- parsing selwood.txt ----

def test_selwood_has_31_rows_and_53_features(selwood):
    names, Y, X = selwood
    assert len(names) == 31
    assert Y.shape == (31,)
    assert X.shape == (31, 53)
    assert np.all(np.isfinite(X)) and np.all(np.isfinite(Y))


def test_names_are_unique(selwood):
    names = selwood[0]
    assert len(set(names)) == 31


def test_below_minus_one_activities_become_minus_one(selwood):
    names, Y, _ = selwood
    assert Y[names.index("K17")] == -1.0
    assert Y[names.index("D30")] == -1.0
    assert np.sum(Y == -1.0) >= 2


def test_known_values_of_first_numeric_row(selwood):
    names, Y, X = selwood
    i = names.index("B7")
    assert Y[i] == 1.84
    assert X[i, 0] == 0.2373
    assert X[i, 2] == pytest.approx(6.19e-02)


def test_small_file_skips_title_blank_and_footnote():
    names, Y, X = parse_selwood(SMALL.splitlines())
    assert names == ["A1", "A2", "A3"]
    assert Y.tolist() == [0.5, -1.0, 1.25]
    assert X.tolist() == [[1.0, 2.0], [3.0, -0.4], [5.0, 6.0]]


def test_wrong_number_of_values_is_an_error():
    bad = SMALL.replace("A3,1.25,5,6", "A3,1.25,5")
    with pytest.raises(DataError, match="line 6"):
        parse_selwood(bad.splitlines())


def test_bad_number_is_an_error():
    with pytest.raises(DataError, match="bad feature"):
        parse_selwood(SMALL.replace("5,6", "5,x").splitlines())
    with pytest.raises(DataError, match="bad activity"):
        parse_selwood(SMALL.replace("1.25", "high").splitlines())


def test_missing_header_is_an_error():
    with pytest.raises(DataError, match="header"):
        parse_selwood(["A1,0.5,1.0,2.0"])


# ---- split ----

def test_split_is_21_plus_10_without_overlap():
    train, test = split_indices(31, NUM_TRAIN, seed=42)
    assert len(train) == 21 and len(test) == 10
    assert set(train).isdisjoint(test)
    assert sorted(set(train) | set(test)) == list(range(31))


def test_same_seed_same_split_and_other_seed_other_split():
    a = split_indices(31, 21, seed=7)
    b = split_indices(31, 21, seed=7)
    c = split_indices(31, 21, seed=8)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_split_rejects_bad_sizes():
    for n_train in (0, 31, 40):
        with pytest.raises(ValueError):
            split_indices(31, n_train, seed=1)


def test_split_dataset_keeps_rows_together(selwood):
    names, Y, X = selwood
    (tr_n, tr_y, tr_x), (te_n, te_y, te_x) = split_dataset(names, Y, X, 21, seed=3)
    for part_n, part_y, part_x in ((tr_n, tr_y, tr_x), (te_n, te_y, te_x)):
        for name, y, x in zip(part_n, part_y, part_x):
            i = names.index(name)
            assert y == Y[i]
            assert np.array_equal(x, X[i])


# ---- writing and reading back ----

def test_written_file_loads_back_identically(tmp_path, selwood):
    names, Y, X = selwood
    path = tmp_path / "all.txt"
    write_dataset(path, names, Y, X)
    names2, Y2, X2 = read_dataset(path, num_features=53)
    assert names2 == names
    assert np.array_equal(Y2, Y)
    assert np.array_equal(X2, X)


def test_read_dataset_checks_feature_count(tmp_path):
    path = tmp_path / "d.txt"
    path.write_text("A,1.0,2.0,3.0\nB,1.0,2.0\n")
    with pytest.raises(DataError, match="line 2"):
        read_dataset(path)
    with pytest.raises(DataError, match="line 1"):
        read_dataset(path, num_features=53)


def test_missing_files_are_errors(tmp_path):
    with pytest.raises(DataError, match="not found"):
        read_selwood(tmp_path / "nope.txt")
    with pytest.raises(DataError, match="not found"):
        read_dataset(tmp_path / "nope.txt")


# ---- prepare_data.prepare (end to end, in a temp folder) ----

def test_prepare_writes_files_named_in_parameters(tmp_path, selwood):
    params = read_parameters(os.path.join(CODE_DIR, "parameters.txt"))
    params["trainFile"], params["testFile"] = "tr.txt", "te.txt"
    (tmp_path / "selwood.txt").write_text(open(SELWOOD, encoding="utf-8").read())

    train, test = prepare(params, folder=tmp_path)
    tr = read_dataset(tmp_path / "tr.txt", num_features=53)
    te = read_dataset(tmp_path / "te.txt", num_features=53)
    assert len(tr[0]) == 21 and len(te[0]) == 10
    assert set(tr[0]).isdisjoint(te[0])
    assert set(tr[0]) | set(te[0]) == set(selwood[0])
    assert tr[0] == train[0] and np.array_equal(tr[2], train[2])
    assert te[0] == test[0] and np.array_equal(te[1], test[1])

    # same seed again gives the same files; another seed a different test set
    again = prepare(params, folder=tmp_path)
    assert again[1][0] == test[0]
    other = prepare(params, folder=tmp_path, seed=params["randomSeed"] + 1)
    assert other[1][0] != test[0]


def test_prepare_checks_num_input_neurons(tmp_path):
    params = read_parameters(os.path.join(CODE_DIR, "parameters.txt"))
    params["numInputNeurons"] = 10
    (tmp_path / "selwood.txt").write_text(open(SELWOOD, encoding="utf-8").read())
    with pytest.raises(DataError, match="numInputNeurons"):
        prepare(params, folder=tmp_path)

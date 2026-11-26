"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Prepares the data files for training. Run from code/:

    python prepare_data.py

It reads parameters.txt and then:
  1. parses selwood.txt,
  2. makes a random split of the 31 molecules into 21 training and 10 test
     molecules using randomSeed, and writes them (raw, not normalised) to
     trainFile and testFile (training.txt / test.txt),
  3. min-max normalises the features with the training-set statistics
     (normalise.py),
  3b. with numFeatures > 0, keeps the numFeatures features most correlated
     with Y in the training set (features.py),
  4. chooses numHiddenLayerNeurons initial centres from the normalised
     training patterns (our own k-means, or random samples; centresInit,
     see kmeans.py), seeded with randomSeed, and writes them to centresFile
     (centerVectors.txt), one centre per line.
"""

import os
import sys

import numpy as np

from data import (DataError, normalise_and_select, read_feature_names, read_selwood,
                  write_centres, write_dataset)
from kmeans import initial_centres
from parameters import ParameterError, read_parameters

SELWOOD_FILE = "selwood.txt"
NUM_TRAIN = 21  # the assignment asks for 21 training / 10 test molecules


def split_indices(num_rows, num_train, seed):
    """Return (train_idx, test_idx): a random split of range(num_rows).

    The same seed always gives the same split. Both index arrays are sorted,
    so the files keep the original order of the molecules.
    """
    if not 1 <= num_train < num_rows:
        raise ValueError(f"num_train must be between 1 and {num_rows - 1}, got {num_train}")
    order = np.random.default_rng(seed).permutation(num_rows)
    return np.sort(order[:num_train]), np.sort(order[num_train:])


def split_dataset(names, Y, X, num_train, seed):
    """Split (names, Y, X) into a training and a test data set."""
    train_idx, test_idx = split_indices(len(names), num_train, seed)
    pick = lambda idx: ([names[i] for i in idx], Y[idx], X[idx])  # noqa: E731
    return pick(train_idx), pick(test_idx)


def make_centres(params, X_train_normalised, seed):
    """Initial centres (K x d) from the normalised training patterns."""
    k = params["numHiddenLayerNeurons"]
    if k > len(X_train_normalised):
        raise DataError(
            f"numHiddenLayerNeurons is {k} but there are only "
            f"{len(X_train_normalised)} training molecules to pick centres from")
    return initial_centres(X_train_normalised, k, params["centresInit"], seed,
                           params["kmeansIterations"])


def prepare(params, folder=".", seed=None, selwood_path=None):
    """Parse selwood.txt and write the training, test and centres files into `folder`.

    `seed` defaults to params["randomSeed"]; pass another one to make a
    different split (used by the experiments). The same seed is used for
    the centres. selwood.txt is read from `folder` unless `selwood_path`
    names another file (the experiments keep one copy in code/).
    Returns (train, test, centres) with train and test raw. With
    numFeatures > 0, prepare.columns holds the kept feature columns.
    """
    seed = params["randomSeed"] if seed is None else seed
    if selwood_path is None:
        selwood_path = os.path.join(folder, SELWOOD_FILE)
    names, Y, X = read_selwood(selwood_path)
    if X.shape[1] != params["numInputNeurons"]:
        raise DataError(
            f"{SELWOOD_FILE} has {X.shape[1]} features but numInputNeurons "
            f"is {params['numInputNeurons']}")
    train, test = split_dataset(names, Y, X, NUM_TRAIN, seed)
    write_dataset(os.path.join(folder, params["trainFile"]), *train)
    write_dataset(os.path.join(folder, params["testFile"]), *test)

    # centres live in the normalised (and, with numFeatures, reduced) space,
    # fitted on the training set only
    X_train_n, _, _, columns = normalise_and_select(params, train[2], train[1], test[2])
    prepare.columns = columns
    centres = make_centres(params, X_train_n, seed)
    write_centres(os.path.join(folder, params["centresFile"]), centres)
    return train, test, centres


def main():
    try:
        params = read_parameters("parameters.txt")
        train, test, centres = prepare(params)
    except (ParameterError, DataError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Wrote {params['trainFile']} ({len(train[0])} molecules) and "
          f"{params['testFile']} ({len(test[0])} molecules), seed {params['randomSeed']}")
    print("Test molecules:", " ".join(test[0]))
    print(f"Wrote {params['centresFile']} ({len(centres)} centres, "
          f"method {params['centresInit']})")
    if params["numFeatures"] > 0:
        names = read_feature_names(SELWOOD_FILE)
        print(f"Features used ({len(prepare.columns)}, chosen on the training set):",
              " ".join(names[c] for c in prepare.columns))
    return 0


if __name__ == "__main__":
    sys.exit(main())

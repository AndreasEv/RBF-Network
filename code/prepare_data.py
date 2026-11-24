"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Prepares the data files for training. Run from code/:

    python prepare_data.py

It reads parameters.txt, parses selwood.txt, makes a random split of the 31
molecules into 21 training and 10 test molecules using randomSeed, and writes
them to trainFile and testFile (training.txt / test.txt).
"""

import os
import sys

import numpy as np

from data import DataError, read_selwood, write_dataset
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


def prepare(params, folder=".", seed=None):
    """Parse selwood.txt in `folder` and write the training and test files.

    `seed` defaults to params["randomSeed"]; pass another one to make a
    different split (used by the experiments). Returns (train, test).
    """
    seed = params["randomSeed"] if seed is None else seed
    names, Y, X = read_selwood(os.path.join(folder, SELWOOD_FILE))
    if X.shape[1] != params["numInputNeurons"]:
        raise DataError(
            f"{SELWOOD_FILE} has {X.shape[1]} features but numInputNeurons "
            f"is {params['numInputNeurons']}")
    train, test = split_dataset(names, Y, X, NUM_TRAIN, seed)
    write_dataset(os.path.join(folder, params["trainFile"]), *train)
    write_dataset(os.path.join(folder, params["testFile"]), *test)
    return train, test


def main():
    try:
        params = read_parameters("parameters.txt")
        train, test = prepare(params)
    except (ParameterError, DataError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Wrote {params['trainFile']} ({len(train[0])} molecules) and "
          f"{params['testFile']} ({len(test[0])} molecules), seed {params['randomSeed']}")
    print("Test molecules:", " ".join(test[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

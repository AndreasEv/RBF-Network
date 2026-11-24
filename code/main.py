"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Runs the whole pipeline from parameters.txt in the current folder:

    python3 main.py

1. read parameters.txt, trainFile, testFile and centresFile
   (prepare_data.py must have been run first to make them),
2. normalise the features with training-set statistics,
3. train the RBF network for maxIterations epochs (variable centres),
4. write results.txt, weights.txt and predictions.txt,
5. print the final training/test MSE and the test RMSE and R^2.
"""

import os
import sys

import metrics
from parameters import read_parameters
from results import write_predictions, write_results, write_weights
from training import train_from_parameters

RESULTS_FILE = "results.txt"
WEIGHTS_FILE = "weights.txt"
PREDICTIONS_FILE = "predictions.txt"


def run(folder="."):
    """Train from folder/parameters.txt, write the 3 output files into
    `folder` and return a dict of final scores."""
    params = read_parameters(os.path.join(folder, "parameters.txt"))
    net, history, train_set, test_set = train_from_parameters(params, folder)

    test_names, test_y, test_x = test_set
    test_pred = net.predict(test_x)
    train_pred = net.predict(train_set[2])

    write_results(os.path.join(folder, RESULTS_FILE), history)
    write_weights(os.path.join(folder, WEIGHTS_FILE), net)
    write_predictions(os.path.join(folder, PREDICTIONS_FILE), test_names, test_y, test_pred)

    return {
        "epochs": len(history),
        "train_mse": metrics.mse(train_set[1], train_pred),
        "test_mse": metrics.mse(test_y, test_pred),
        "test_rmse": metrics.rmse(test_y, test_pred),
        "test_r2": metrics.r2(test_y, test_pred),
    }


def main(folder="."):
    try:
        scores = run(folder)
    except (OSError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Trained for {scores['epochs']} epochs.")
    print(f"Training MSE: {scores['train_mse']:.6f}")
    print(f"Test MSE:     {scores['test_mse']:.6f}")
    print(f"Test RMSE:    {scores['test_rmse']:.6f}")
    print(f"Test R^2:     {scores['test_r2']:.6f}")
    print(f"Wrote {RESULTS_FILE}, {WEIGHTS_FILE} and {PREDICTIONS_FILE}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

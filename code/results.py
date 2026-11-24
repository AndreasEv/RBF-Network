"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Writes the three output files of a training run (and reads weights.txt back).

results.txt      one line per epoch, 3 columns separated by spaces:
                     epoch  trainMSE  testMSE
predictions.txt  one line per test molecule, 3 columns separated by spaces:
                     molecule  trueY  predictedY
weights.txt      the final network, one labelled line per item:
                     weights w_1 ... w_K
                     bias b
                     centre 1 c_11 ... c_1d      (K lines, normalised space)
                     ...
                     sigmas s_1 ... s_K
Numbers in weights.txt are written with repr() so reading the file back
gives exactly the same network.
"""

import numpy as np

from rbf import RBF


def write_results(path, history):
    """history = [(epoch, train MSE, test MSE), ...] from training.train()."""
    with open(path, "w", encoding="utf-8") as f:
        for epoch, train_err, test_err in history:
            test_text = "nan" if test_err is None else f"{test_err:.8f}"
            f.write(f"{epoch} {train_err:.8f} {test_text}\n")


def write_predictions(path, names, targets, predictions):
    with open(path, "w", encoding="utf-8") as f:
        for name, t, y in zip(names, targets, predictions):
            f.write(f"{name} {float(t):.4f} {float(y):.4f}\n")


def write_weights(path, net):
    def nums(values):
        return " ".join(repr(float(v)) for v in values)

    with open(path, "w", encoding="utf-8") as f:
        f.write("weights " + nums(net.weights) + "\n")
        f.write("bias " + repr(float(net.bias)) + "\n")
        for j, centre in enumerate(net.centres, start=1):
            f.write(f"centre {j} " + nums(centre) + "\n")
        f.write("sigmas " + nums(net.sigmas) + "\n")


def read_weights(path):
    """Read weights.txt back into an RBF network."""
    weights = bias = sigmas = None
    centres = []
    with open(path, encoding="utf-8") as f:
        for number, raw in enumerate(f, start=1):
            fields = raw.split()
            if not fields:
                continue
            key, values = fields[0], fields[1:]
            if key == "weights":
                weights = [float(v) for v in values]
            elif key == "bias":
                bias = float(values[0])
            elif key == "centre":
                centres.append([float(v) for v in values[1:]])
            elif key == "sigmas":
                sigmas = [float(v) for v in values]
            else:
                raise ValueError(f"{path}, line {number}: unknown item {key!r}")
    if weights is None or bias is None or sigmas is None or not centres:
        raise ValueError(f"{path}: missing weights, bias, centres or sigmas")
    return RBF(np.array(centres), np.array(sigmas), np.array(weights), bias)

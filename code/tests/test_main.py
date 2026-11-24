"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

End-to-end tests for main.py and the output files written by results.py
(run from code/: python3 -m pytest tests -q).
"""

import os
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402
from data import read_dataset  # noqa: E402
from parameters import read_parameters  # noqa: E402
from rbf import RBF  # noqa: E402
from results import read_weights, write_results, write_weights  # noqa: E402

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPOCHS = 5


@pytest.fixture
def run_folder(tmp_path):
    """A copy of the real inputs with a short run (EPOCHS epochs)."""
    params_text = []
    with open(os.path.join(CODE_DIR, "parameters.txt"), encoding="utf-8") as f:
        for line in f:
            if line.split() and line.split()[0] == "maxIterations":
                line = f"maxIterations {EPOCHS}\n"
            params_text.append(line)
    (tmp_path / "parameters.txt").write_text("".join(params_text), encoding="utf-8")
    params = read_parameters(str(tmp_path / "parameters.txt"))
    for key in ("trainFile", "testFile", "centresFile"):
        shutil.copy(os.path.join(CODE_DIR, params[key]), tmp_path / params[key])
    return tmp_path, params


def test_main_writes_three_files_in_the_right_format(run_folder, capsys):
    folder, params = run_folder
    assert main.main(str(folder)) == 0
    k, d = params["numHiddenLayerNeurons"], params["numInputNeurons"]

    # results.txt: EPOCHS lines of "epoch trainMSE testMSE"
    rows = (folder / "results.txt").read_text().splitlines()
    assert len(rows) == EPOCHS
    for i, row in enumerate(rows, start=1):
        fields = row.split()
        assert len(fields) == 3
        assert int(fields[0]) == i
        assert np.isfinite(float(fields[1])) and float(fields[1]) >= 0
        assert np.isfinite(float(fields[2])) and float(fields[2]) >= 0

    # predictions.txt: one line per test molecule, in test-file order
    names, ys, _ = read_dataset(str(folder / params["testFile"]))
    rows = (folder / "predictions.txt").read_text().splitlines()
    assert len(rows) == len(names)
    for row, name, y in zip(rows, names, ys):
        fields = row.split()
        assert len(fields) == 3
        assert fields[0] == name
        assert float(fields[1]) == pytest.approx(y, abs=1e-4)
        assert np.isfinite(float(fields[2]))

    # weights.txt: K weights, a bias, K centres of d values, K sigmas
    net = read_weights(str(folder / "weights.txt"))
    assert net.weights.shape == (k,)
    assert net.centres.shape == (k, d)
    assert net.sigmas.shape == (k,)

    out = capsys.readouterr().out
    for label in ("Test MSE", "Test RMSE", "Test R^2"):
        assert label in out


def test_scores_match_output_files(run_folder):
    folder, _ = run_folder
    scores = main.run(str(folder))
    last = (folder / "results.txt").read_text().splitlines()[-1].split()
    assert float(last[1]) == pytest.approx(scores["train_mse"], abs=1e-7)
    assert float(last[2]) == pytest.approx(scores["test_mse"], abs=1e-7)
    assert scores["test_rmse"] == pytest.approx(np.sqrt(scores["test_mse"]))

    # predictions.txt must agree with the saved network
    params = read_parameters(str(folder / "parameters.txt"))
    from data import load_normalised
    _, test_set, _ = load_normalised(params, str(folder))
    net = read_weights(str(folder / "weights.txt"))
    pred = [float(r.split()[2]) for r in (folder / "predictions.txt").read_text().splitlines()]
    assert np.allclose(pred, net.predict(test_set[2]), atol=1e-4)


def test_same_parameters_give_same_files(run_folder):
    folder, _ = run_folder
    main.run(str(folder))
    first = [(folder / n).read_text() for n in ("results.txt", "weights.txt", "predictions.txt")]
    main.run(str(folder))
    second = [(folder / n).read_text() for n in ("results.txt", "weights.txt", "predictions.txt")]
    assert first == second


def test_weights_round_trip_exactly(tmp_path):
    rng = np.random.default_rng(1)
    net = RBF(rng.normal(size=(3, 4)), rng.uniform(0.5, 2, 3), rng.normal(size=3), -0.37)
    write_weights(str(tmp_path / "w.txt"), net)
    back = read_weights(str(tmp_path / "w.txt"))
    assert np.array_equal(back.weights, net.weights)
    assert back.bias == net.bias
    assert np.array_equal(back.centres, net.centres)
    assert np.array_equal(back.sigmas, net.sigmas)


def test_results_without_test_set(tmp_path):
    write_results(str(tmp_path / "r.txt"), [(1, 0.5, None), (2, 0.25, None)])
    assert (tmp_path / "r.txt").read_text().splitlines() == [
        "1 0.50000000 nan", "2 0.25000000 nan"]


def test_missing_parameters_file_returns_error(tmp_path, capsys):
    assert main.main(str(tmp_path)) == 1
    assert "Error" in capsys.readouterr().err

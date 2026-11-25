"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Tests for plots.py: the file readers and that every plot function writes a
non-empty PNG (Agg backend, no screen needed). Run from code/:
    python3 -m pytest tests -q
"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("matplotlib")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plots  # noqa: E402
from results import write_predictions, write_results  # noqa: E402

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

HISTORY = [(1, 0.9, 1.2), (2, 0.6, 0.9), (3, 0.4, 0.7), (4, 0.3, 0.65)]
NAMES = ["A1", "B2", "C3", "D4"]
TRUE_Y = [-1.0, 0.0, 0.5, 1.5]
PRED_Y = [-0.7, 0.2, 0.45, 1.1]


def is_png(path):
    assert os.path.isfile(path)
    assert os.path.getsize(path) > 1000
    with open(path, "rb") as f:
        return f.read(8) == PNG_SIGNATURE


@pytest.fixture
def run_folder(tmp_path):
    """A folder with small fake results.txt and predictions.txt."""
    write_results(tmp_path / "results.txt", HISTORY)
    write_predictions(tmp_path / "predictions.txt", NAMES, TRUE_Y, PRED_Y)
    return tmp_path


# ---------- readers ----------

def test_read_results(run_folder):
    epochs, train, test = plots.read_results(run_folder / "results.txt")
    assert list(epochs) == [1, 2, 3, 4]
    assert np.allclose(train, [h[1] for h in HISTORY])
    assert np.allclose(test, [h[2] for h in HISTORY])


def test_read_results_without_test_set(tmp_path):
    write_results(tmp_path / "results.txt", [(1, 0.5, None), (2, 0.4, None)])
    _, train, test = plots.read_results(tmp_path / "results.txt")
    assert np.allclose(train, [0.5, 0.4]) and np.all(np.isnan(test))


def test_read_predictions(run_folder):
    names, t, y = plots.read_predictions(run_folder / "predictions.txt")
    assert names == NAMES
    assert np.allclose(t, TRUE_Y) and np.allclose(y, PRED_Y)


@pytest.mark.parametrize("reader", [plots.read_results, plots.read_predictions])
def test_missing_and_empty_files(tmp_path, reader):
    with pytest.raises(plots.PlotError, match="not found"):
        reader(tmp_path / "nope.txt")
    (tmp_path / "empty.txt").write_text("\n")
    with pytest.raises(plots.PlotError, match="empty"):
        reader(tmp_path / "empty.txt")


def test_bad_results_columns(tmp_path):
    (tmp_path / "results.txt").write_text("1 0.5\n2 0.4\n")
    with pytest.raises(plots.PlotError, match="3 columns"):
        plots.read_results(tmp_path / "results.txt")


def test_bad_predictions_lines(tmp_path):
    (tmp_path / "p.txt").write_text("A1 0.5\n")
    with pytest.raises(plots.PlotError, match="line 1.*3 columns"):
        plots.read_predictions(tmp_path / "p.txt")
    (tmp_path / "p.txt").write_text("A1 0.5 0.4\nB2 x 0.1\n")
    with pytest.raises(plots.PlotError, match="line 2.*numbers"):
        plots.read_predictions(tmp_path / "p.txt")


# ---------- figures ----------

def test_error_curve_png(run_folder, tmp_path):
    path = plots.plot_error_curve(*plots.read_results(run_folder / "results.txt"),
                                  str(tmp_path / "out" / "error.png"))
    assert is_png(path)


def test_error_curve_single_epoch_and_no_test(tmp_path):
    assert is_png(plots.plot_error_curve([1], [0.5], [0.7], str(tmp_path / "a.png")))
    assert is_png(plots.plot_error_curve([1, 2], [0.5, 0.4], [np.nan, np.nan],
                                         str(tmp_path / "b.png")))


def test_error_curve_length_mismatch(tmp_path):
    with pytest.raises(ValueError):
        plots.plot_error_curve([1, 2], [0.5], [0.7, 0.6], str(tmp_path / "x.png"))
    with pytest.raises(ValueError):
        plots.plot_error_curve([], [], [], str(tmp_path / "x.png"))


def test_predictions_png(run_folder, tmp_path):
    path = plots.plot_predictions(*plots.read_predictions(run_folder / "predictions.txt"),
                                  str(tmp_path / "pred.png"))
    assert is_png(path)


def test_predictions_close_points_and_single_point(tmp_path):
    names = ["A", "B", "C"]
    assert is_png(plots.plot_predictions(names, [0.42, 0.43, 0.3], [0.434, 0.456, 0.32],
                                         str(tmp_path / "close.png")))
    assert is_png(plots.plot_predictions(["A"], [0.5], [0.5], str(tmp_path / "one.png")))


def test_predictions_length_mismatch(tmp_path):
    with pytest.raises(ValueError):
        plots.plot_predictions(["A", "B"], [0.1], [0.2, 0.3], str(tmp_path / "x.png"))


def test_sin_fit_learns():
    """Same check as in test_training.py: the fit of sin(x) gets very good."""
    x, y, net, initial, before, after = plots.sin_fit(epochs=300)
    assert len(x) == plots.SIN_POINTS and np.allclose(y, np.sin(x))
    assert len(initial) == net.num_hidden == plots.SIN_CENTRES
    assert after < before / 20 and after < 0.01


def test_sin_fit_png(tmp_path):
    assert is_png(plots.plot_sin_fit(str(tmp_path / "sin.png"), epochs=20))


# ---------- the whole script ----------

def test_main_writes_all_plots(run_folder, capsys):
    assert plots.main([str(run_folder)]) == 0
    folder = run_folder / plots.PLOTS_FOLDER
    for name in (plots.ERROR_PLOT, plots.PREDICTIONS_PLOT, plots.SIN_PLOT):
        assert is_png(folder / name)
    assert "Saved" in capsys.readouterr().out


def test_main_reports_missing_file(tmp_path, capsys):
    assert plots.main([str(tmp_path)]) == 1
    assert "not found" in capsys.readouterr().err

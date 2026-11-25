"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Tests for experiments.py (run from code/: python3 -m pytest tests -q).
"""

import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments  # noqa: E402
from data import read_centres, read_dataset  # noqa: E402
from parameters import read_parameters  # noqa: E402
from results import read_weights  # noqa: E402

EPOCHS = 3


@pytest.fixture(scope="module")
def baseline():
    return experiments.read_parameter_texts(experiments.BASELINE_FILE)


@pytest.fixture(scope="module")
def configs(baseline):
    return experiments.make_configs(baseline)


def _params(baseline, overrides, tmp_path, name="p.txt"):
    texts = dict(baseline)
    texts.update(overrides)
    path = tmp_path / name
    experiments.write_parameter_texts(str(path), texts)
    return read_parameters(str(path))


# ---- the experiment list covers every required comparison -----------------

def test_names_are_unique(configs):
    names = [c[0] for c in configs]
    assert len(names) == len(set(names))


def test_every_config_is_a_valid_parameters_file(baseline, configs, tmp_path):
    for name, _, overrides in configs:
        params = _params(baseline, overrides, tmp_path, f"{name}.txt")
        assert params["numInputNeurons"] == 53


def test_at_least_three_centre_counts(baseline, configs, tmp_path):
    ks = {_params(baseline, o, tmp_path)["numHiddenLayerNeurons"]
          for n, g, o in configs if g in ("baseline", "centres")}
    assert len(ks) >= 3


def test_at_least_two_sigmas(baseline, configs, tmp_path):
    sigmas = {_params(baseline, o, tmp_path)["sigmas"][0]
              for n, g, o in configs if g in ("baseline", "sigma")}
    assert len(sigmas) >= 2


def test_at_least_two_learning_rate_settings(baseline, configs, tmp_path):
    rates = {tuple(_params(baseline, o, tmp_path)["learningRates"])
             for n, g, o in configs if g in ("baseline", "rates")}
    assert len(rates) >= 2


def test_fixed_and_variable_centres(baseline, configs, tmp_path):
    base = _params(baseline, {}, tmp_path)
    fixed = [_params(baseline, o, tmp_path) for n, g, o in configs if g == "fixed"]
    assert base["etaCentres"] > 0
    assert fixed and all(p["etaCentres"] == 0 for p in fixed)
    # only the centre rate changes
    assert all(p["etaWeights"] == base["etaWeights"] and
               p["etaSigmas"] == base["etaSigmas"] for p in fixed)


def test_at_least_five_splits_variable_and_fixed(baseline, configs, tmp_path):
    for group, fixed in (("split", False), ("split_fixed", True)):
        ps = [_params(baseline, o, tmp_path) for n, g, o in configs if g == group]
        assert len({p["randomSeed"] for p in ps}) >= 5
        assert all((p["etaCentres"] == 0) == fixed for p in ps)


def test_one_change_per_experiment(configs):
    for name, group, overrides in configs:
        if group in ("centres", "sigma", "rates", "fixed", "split"):
            assert len(overrides) == 1, name


# ---- a tiny run writes the right files -----------------------------------

@pytest.fixture(scope="module")
def tiny_run(baseline, configs, tmp_path_factory):
    out = tmp_path_factory.mktemp("exp")
    picked = [c for c in configs
              if c[0] in ("baseline", "K_5", "fixed_centres", "split_1", "split_2",
                          "split_1_fixed", "split_2_fixed")]
    rows = experiments.run_experiments(picked, baseline, str(out), epochs=EPOCHS, log=None)
    return out, picked, rows


def test_each_experiment_has_its_own_folder(tiny_run):
    out, picked, _ = tiny_run
    for name, _, _ in picked:
        for f in ("parameters.txt", "training.txt", "test.txt", "centerVectors.txt",
                  "results.txt", "weights.txt", "predictions.txt", "run.txt"):
            assert (out / name / f).is_file(), f"{name}/{f}"
        assert len((out / name / "results.txt").read_text().splitlines()) == EPOCHS


def test_summary_csv_has_the_right_columns_and_rows(tiny_run):
    out, picked, _ = tiny_run
    with open(out / "summary.csv", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == experiments.COLUMNS
        rows = list(reader)
    assert [r["name"] for r in rows] == [c[0] for c in picked]
    for r in rows:
        assert int(r["epochs"]) == EPOCHS
        for c in experiments.SCORE_COLUMNS:
            float(r[c])
        assert float(r["test_rmse"]) ** 2 == pytest.approx(float(r["test_mse"]), rel=1e-4)


def test_summary_txt_has_split_mean_and_std(tiny_run):
    out, _, rows = tiny_run
    text = (out / "summary.txt").read_text()
    for name in ("baseline", "K_5", "split_1_fixed"):
        assert name in text
    assert "variable centres (2 splits): mean +- std" in text
    assert "fixed centres (2 splits): mean +- std" in text
    mean, std = experiments.split_stats(rows, "split")["test_rmse"]
    assert f"test_rmse  {mean:.4f} +- {std:.4f}" in text


def test_run_settings_match_the_config(tiny_run):
    out, _, rows = tiny_run
    by_name = {r["name"]: r for r in rows}
    assert by_name["K_5"]["K"] == 5
    assert read_centres(str(out / "K_5" / "centerVectors.txt")).shape == (5, 53)
    assert by_name["fixed_centres"]["eta_c"] == 0
    assert by_name["split_1"]["seed"] == 1


def test_fixed_centres_do_not_move(tiny_run):
    out, _, _ = tiny_run
    start = read_centres(str(out / "fixed_centres" / "centerVectors.txt"))
    final = read_weights(str(out / "fixed_centres" / "weights.txt")).centres
    assert np.allclose(start, final)


# ---- the splits differ and are reproducible ------------------------------

def _test_names(folder):
    return read_dataset(str(folder / "test.txt"))[0]


def test_splits_really_differ(tiny_run):
    out, _, _ = tiny_run
    assert _test_names(out / "split_1") != _test_names(out / "split_2")
    assert _test_names(out / "split_1") != _test_names(out / "baseline")
    for name in ("split_1", "split_2"):
        assert len(_test_names(out / name)) == 10
        assert len(read_dataset(str(out / name / "training.txt"))[0]) == 21


def test_fixed_and_variable_use_the_same_split(tiny_run):
    out, _, _ = tiny_run
    for seed in (1, 2):
        assert _test_names(out / f"split_{seed}") == _test_names(out / f"split_{seed}_fixed")
        assert (out / f"split_{seed}" / "centerVectors.txt").read_text() == \
            (out / f"split_{seed}_fixed" / "centerVectors.txt").read_text()


def test_each_split_gets_its_own_centres(tiny_run):
    out, _, _ = tiny_run
    assert (out / "split_1" / "centerVectors.txt").read_text() != \
        (out / "split_2" / "centerVectors.txt").read_text()


def test_runs_are_reproducible(baseline, configs, tiny_run, tmp_path):
    out, _, rows = tiny_run
    again = experiments.run_experiments(
        [c for c in configs if c[0] == "split_1"], baseline, str(tmp_path),
        epochs=EPOCHS, log=None)
    first = next(r for r in rows if r["name"] == "split_1")
    assert again[0] == first
    for f in ("test.txt", "centerVectors.txt", "predictions.txt"):
        assert (tmp_path / "split_1" / f).read_text() == (out / "split_1" / f).read_text()


# ---- command line --------------------------------------------------------

def test_cli_rejects_an_unknown_name(capsys):
    assert experiments.main_cli(["--only", "no_such_run"]) == 1
    assert "unknown experiment" in capsys.readouterr().err


def test_cli_only_and_epochs(tmp_path, capsys):
    assert experiments.main_cli(["--only", "baseline", "--epochs", "2",
                                 "--out", str(tmp_path)]) == 0
    assert (tmp_path / "summary.csv").is_file()
    assert len((tmp_path / "baseline" / "results.txt").read_text().splitlines()) == 2

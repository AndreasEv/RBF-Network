"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Runs the experiments of the report. Run from code/:

    python3 experiments.py                    # all experiments
    python3 experiments.py --only K_3 sigma_0.5
    python3 experiments.py --epochs 50        # quick check with fewer epochs
    python3 experiments.py --list             # print the experiment names

Every experiment starts from the baseline parameters.txt in code/ and
changes one setting (see make_configs). Each setting is run on the seed 42
split of the report and on 5 more random splits (seeds 1-5), so settings
are compared by their mean over splits, not by one 10-molecule test set.
Two reference models (reference.py: predict the training mean, and a
least-squares linear fit) are run on the same splits. Each one runs in its own folder
experiments/<name>/ with its own parameters.txt:

  1. prepare_data.prepare: split selwood.txt 21/10 with the experiment's
     randomSeed, normalise with that training set and make new k-means
     centres from it (training.txt, test.txt, centerVectors.txt),
  2. main.run: train and write results.txt, weights.txt, predictions.txt
     (a reference model writes predictions.txt only),
  3. write run.txt with the final scores.

Then it writes experiments/summary.csv (one row per experiment) and
experiments/summary.txt (the same table plus mean +- std over the random
splits). A run's randomSeed sets the split, the centres, the initial
weights and the pattern order, so the same seed always gives the same run.
"""

import argparse
import csv
import os
import sys

import numpy as np

import main
import metrics
from data import load_normalised
from parameters import ParameterError, read_parameters
from prepare_data import SELWOOD_FILE, prepare
from reference import LinearModel, MeanModel
from results import write_predictions

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_FILE = os.path.join(CODE_DIR, "parameters.txt")
OUT_DIR = os.path.join(CODE_DIR, "experiments")

SPLIT_SEEDS = (1, 2, 3, 4, 5)
FIXED_CENTRES_RATE = "0"  # eta_c = 0 keeps the centres where k-means put them

# One-setting changes of the baseline that are run on the seed 42 split AND
# on the 5 random splits, so they can be compared by their mean over splits.
VARIANTS = [
    ("K_3", "centres", {"numHiddenLayerNeurons": "3"}),
    ("K_5", "centres", {"numHiddenLayerNeurons": "5"}),
    ("K_15", "centres", {"numHiddenLayerNeurons": "15"}),
    ("sigma_0.5", "sigma", {"sigmas": "0.5"}),
    ("sigma_2.0", "sigma", {"sigmas": "2.0"}),
    ("rates_x4", "rates", {"learningRates": "0.01 0.001 0.0005"}),
    ("rates_x8", "rates", {"learningRates": "0.02 0.002 0.001"}),
]

# Feature selection (extra experiment): the setting with the lowest repeated
# cross-validation error on the seed 42 TRAINING set among the runs that use
# fewer features (cv.py, cv_results.txt). The features are chosen again on
# the training set of every split.
FEATURE_SELECTION = {"numFeatures": "20", "numHiddenLayerNeurons": "5", "sigmas": "3.0"}

# Reference models (reference.py), not RBF networks. The "model" key is not
# a parameter: it tells run_one to fit this model instead of the network.
MODEL_KEY = "model"
REFERENCES = (("mean", {MODEL_KEY: "mean"}), ("linear", {MODEL_KEY: "linear"}))

COLUMNS = ["name", "group", "features", "K", "sigma", "eta_w", "eta_c", "eta_sigma",
           "epochs", "seed", "train_mse", "test_mse", "test_rmse", "test_r2"]
SCORE_COLUMNS = ["train_mse", "test_mse", "test_rmse", "test_r2"]


def _with_centre_rate(rates, eta_c):
    """learningRates text with the centre rate (2nd value) replaced."""
    w, _, s = rates.split()
    return f"{w} {eta_c} {s}"


def make_configs(baseline):
    """The experiment list. `baseline` is the baseline parameters.txt as a
    dict name -> value text. Each config is (name, group, overrides), where
    overrides maps parameter names to new value texts."""
    fixed = {"learningRates": _with_centre_rate(baseline["learningRates"],
                                                FIXED_CENTRES_RATE)}
    configs = [("baseline", "baseline", {})]
    # number of centres (baseline K = 10), initial width (baseline sigma = 1.0)
    # and learning rates (baseline 0.0025 0.00025 0.000125), on the seed 42 split
    configs += [(name, group, dict(o)) for name, group, o in VARIANTS]
    # fixed vs variable centres on the baseline split
    configs.append(("fixed_centres", "fixed", dict(fixed)))
    # feature selection and the reference models on the baseline split
    configs.append(("features", "features", dict(FEATURE_SELECTION)))
    configs += [(name, "reference", dict(o)) for name, o in REFERENCES]
    # random 21/10 splits, each with variable and with fixed centres
    for seed in SPLIT_SEEDS:
        configs.append((f"split_{seed}", "split", {"randomSeed": str(seed)}))
    for seed in SPLIT_SEEDS:
        configs.append((f"split_{seed}_fixed", "split_fixed",
                        {"randomSeed": str(seed), **fixed}))
    # every other setting on the same 5 splits ("split:<name>" groups)
    others = [(n, o) for n, _, o in VARIANTS] + [("features", FEATURE_SELECTION)]
    others += list(REFERENCES)
    for name, overrides in others:
        for seed in SPLIT_SEEDS:
            configs.append((f"{name}_split_{seed}", f"split:{name}",
                            {"randomSeed": str(seed), **overrides}))
    return configs


def read_parameter_texts(path):
    """parameters.txt as an ordered dict name -> value text (comments dropped)."""
    texts = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if line:
                name, value = line.split(None, 1)
                texts[name] = value
    return texts


def write_parameter_texts(path, texts):
    with open(path, "w", encoding="utf-8") as f:
        for name, value in texts.items():
            f.write(f"{name} {value}\n")


def run_one(name, group, overrides, baseline, out_dir, epochs=None,
            selwood_path=None):
    """Run one experiment in out_dir/name and return its summary row (dict)."""
    folder = os.path.join(out_dir, name)
    os.makedirs(folder, exist_ok=True)
    overrides = dict(overrides)
    model = overrides.pop(MODEL_KEY, None)
    texts = dict(baseline)
    texts.update(overrides)
    if epochs is not None:
        texts["maxIterations"] = str(epochs)
    write_parameter_texts(os.path.join(folder, "parameters.txt"), texts)

    params = read_parameters(os.path.join(folder, "parameters.txt"))
    if selwood_path is None:
        selwood_path = os.path.join(CODE_DIR, SELWOOD_FILE)
    prepare(params, folder, selwood_path=selwood_path)
    scores = main.run(folder) if model is None else run_reference(model, params, folder)

    row = {
        "name": name,
        "group": group,
        "features": params["numNetworkInputs"],
        "K": params["numHiddenLayerNeurons"],
        "sigma": texts["sigmas"],
        "eta_w": params["etaWeights"],
        "eta_c": params["etaCentres"],
        "eta_sigma": params["etaSigmas"],
        "epochs": scores["epochs"],
        "seed": params["randomSeed"],
    }
    if model is not None:   # not an RBF network: no K, sigma or rates
        row.update(K="-", sigma="-", eta_w="-", eta_c="-", eta_sigma="-")
    row.update({c: scores[c] for c in SCORE_COLUMNS})
    with open(os.path.join(folder, "run.txt"), "w", encoding="utf-8") as f:
        for c in ("epochs",) + tuple(SCORE_COLUMNS):
            f.write(f"{c} {row[c]:.6f}\n" if c != "epochs" else f"{c} {row[c]}\n")
    return row


def run_reference(model, params, folder):
    """Fit a reference model (reference.py) on the folder's training set,
    write predictions.txt and return the same scores as main.run."""
    (_, train_y, train_x), (test_names, test_y, test_x), _ = load_normalised(params, folder)
    net = MeanModel(train_y) if model == "mean" else LinearModel(train_x, train_y)
    train_pred, test_pred = net.predict(train_x), net.predict(test_x)
    write_predictions(os.path.join(folder, main.PREDICTIONS_FILE), test_names, test_y,
                      test_pred)
    return {"epochs": 0,
            "train_mse": metrics.mse(train_y, train_pred),
            "test_mse": metrics.mse(test_y, test_pred),
            "test_rmse": metrics.rmse(test_y, test_pred),
            "test_r2": metrics.r2(test_y, test_pred)}


def split_stats(rows, group):
    """{column: (mean, std)} over the rows of `group` (population std)."""
    picked = [r for r in rows if r["group"] == group]
    if not picked:
        return None
    return {c: (float(np.mean([r[c] for r in picked])),
                float(np.std([r[c] for r in picked]))) for c in SCORE_COLUMNS}


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) >= 1e-3 or value == 0 else f"{value:.5g}"
    return str(value)


def write_summary(rows, out_dir):
    """Write summary.csv and summary.txt into out_dir."""
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: (f"{r[c]:.6f}" if c in SCORE_COLUMNS else r[c])
                             for c in COLUMNS})

    table = [COLUMNS] + [[_fmt(r[c]) for c in COLUMNS] for r in rows]
    widths = [max(len(line[i]) for line in table) for i in range(len(COLUMNS))]
    lines = ["RBF experiments (each row changes one setting of the baseline)", ""]
    for i, line in enumerate(table):
        lines.append("  ".join(v.ljust(w) for v, w in zip(line, widths)).rstrip())
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))

    for group, title in (("split", "variable centres"), ("split_fixed", "fixed centres")):
        stats = split_stats(rows, group)
        if stats is None:
            continue
        n = sum(r["group"] == group for r in rows)
        lines += ["", f"Random 21/10 splits, {title} ({n} splits): mean +- std"]
        for c in SCORE_COLUMNS:
            mean, std = stats[c]
            lines.append(f"  {c:<10} {mean:.4f} +- {std:.4f}")

    # every setting that was run on the random splits, side by side
    groups = [g for g in dict.fromkeys(r["group"] for r in rows)
              if g in ("split", "split_fixed") or g.startswith("split:")]
    if groups:
        lines += ["", "Mean +- std over the random splits (seeds "
                  + " ".join(str(s) for s in SPLIT_SEEDS) + "), every setting", ""]
        header = f"{'setting':<18}{'n':>3}" + "".join(f"{c:>20}" for c in SCORE_COLUMNS)
        lines += [header, "-" * len(header)]
        for g in groups:
            stats = split_stats(rows, g)
            label = {"split": "baseline", "split_fixed": "fixed_centres"}.get(g, g[6:])
            n = sum(r["group"] == g for r in rows)
            lines.append(f"{label:<18}{n:>3}" + "".join(
                f"{stats[c][0]:.4f} +- {stats[c][1]:.4f}".rjust(20) for c in SCORE_COLUMNS))
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return lines


def run_experiments(configs, baseline, out_dir=OUT_DIR, epochs=None,
                    selwood_path=None, log=print):
    """Run every config and write the summary; return the rows."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for i, (name, group, overrides) in enumerate(configs, start=1):
        row = run_one(name, group, overrides, baseline, out_dir, epochs, selwood_path)
        rows.append(row)
        if log:
            log(f"[{i}/{len(configs)}] {name:<22} test MSE {row['test_mse']:.4f}  "
                f"RMSE {row['test_rmse']:.4f}  R^2 {row['test_r2']:.4f}")
    write_summary(rows, out_dir)
    return rows


def main_cli(argv=None):
    parser = argparse.ArgumentParser(description="Run the RBF experiments.")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="run only these experiments")
    parser.add_argument("--epochs", type=int,
                        help="override maxIterations (for a quick check)")
    parser.add_argument("--out", default=OUT_DIR, help="output folder")
    parser.add_argument("--list", action="store_true", help="list the experiments")
    args = parser.parse_args(argv)

    try:
        read_parameters(BASELINE_FILE)  # validate the baseline first
        baseline = read_parameter_texts(BASELINE_FILE)
        configs = make_configs(baseline)
        if args.list:
            for name, group, overrides in configs:
                print(f"{name:<24} {group:<18} {overrides}")
            return 0
        if args.only:
            known = {c[0] for c in configs}
            unknown = [n for n in args.only if n not in known]
            if unknown:
                print(f"Error: unknown experiment(s): {', '.join(unknown)}", file=sys.stderr)
                return 1
            configs = [c for c in configs if c[0] in args.only]
        if args.epochs is not None and args.epochs < 1:
            print("Error: --epochs must be >= 1", file=sys.stderr)
            return 1
        run_experiments(configs, baseline, args.out, args.epochs)
    except (OSError, ValueError, ParameterError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Wrote {os.path.join(args.out, 'summary.txt')} and summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())

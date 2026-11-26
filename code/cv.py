"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Repeated k-fold cross-validation on the TRAINING set only (trainFile of
parameters.txt; the test file is never read). Run from code/:

    python3 cv.py                      # coarse grid -> cv_results.txt / .csv
    python3 cv.py --grid fine          # fine grid  -> cv_fine.txt / .csv
    python3 cv.py --folds 5 --repeats 4 --jobs 4
    python3 cv.py --epochs 100         # quick check

Two grids. "coarse": number of selected features x K x sigma for the RBF
network, plus the linear and mean reference models. "fine": all features,
around the best coarse settings, also varying the learning rates and the
number of epochs. For every setting and every fold:

  1. the min-max scaler is fitted on the training folds only,
  2. the features are chosen on the training folds only
     (features.select_features: largest |correlation| with Y),
  3. the k-means centres are made from the training folds only,
  4. the model is trained on the training folds and scored on the held-out
     fold.

The CV error of a setting is the mean squared error over all held-out
predictions, averaged over the repeats (each repeat shuffles the molecules
into new folds); the std over the repeats shows how noisy it is. With 21
molecules a single 5-fold split is very noisy, hence the repeats.
"""

import argparse
import csv
import itertools
import os
import sys

import numpy as np

from data import read_dataset
from features import select_features
from kmeans import initial_centres
from normalise import normalise_train_test
from parameters import ParameterError, read_parameters
from rbf import RBF
from reference import LinearModel, MeanModel
from training import train

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILES = {"coarse": ("cv_results.txt", "cv_results.csv"),
             "fine": ("cv_fine.txt", "cv_fine.csv")}
OUT_TXT, OUT_CSV = OUT_FILES["coarse"]

# Learning rates of the grids are multiples ("rates" factor) of these
# (weights, centres, sigmas); fixed here so the grids do not change when
# parameters.txt does.
GRID_RATES = (0.01, 0.001, 0.0005)
COARSE_EPOCHS = 2000

# The coarse grid. 0 features = all 53.
GRID_FEATURES = (5, 10, 20, 0)
GRID_K = (3, 5, 8)
GRID_SIGMA = (0.5, 1.0, 2.0, 3.0)
LINEAR_FEATURES = (3, 5, 10, 0)

# The fine grid (all features). Rates are multiples of the baseline rates.
FINE_K = (5, 6, 8, 10)
FINE_SIGMA = (0.75, 1.0, 1.5)
FINE_RATE_FACTORS = (0.25, 0.5, 1.0, 2.0)
FINE_EPOCHS = (250, 500, 1000, 2000, 4000)

COLUMNS = ["model", "features", "K", "sigma", "rates", "epochs", "cv_mse", "cv_std"]


def fold_indices(n, folds, rng):
    """A random partition of range(n) into `folds` nearly equal parts."""
    if not 2 <= folds <= n:
        raise ValueError(f"folds must be between 2 and {n}, got {folds}")
    return np.array_split(rng.permutation(n), folds)


def fit_predict(config, X_train, Y_train, X_eval, base, seed):
    """Fit one model on (X_train, Y_train) (raw features) and predict X_eval.

    `config` has "model" ("rbf", "linear" or "mean"), "features" (0 = all)
    and for the RBF "K", "sigma", "rates" (factor times GRID_RATES) and
    "epochs"; `base` is the parameters dict (centre method, k-means rounds). Everything that is fitted,
    scaler, feature choice, centres, weights, sees X_train only.
    """
    if config["model"] == "mean":
        return MeanModel(Y_train).predict(X_eval)
    Xn, Xe, _ = normalise_train_test(X_train, X_eval)
    cols = select_features(Xn, Y_train, config["features"])
    Xn, Xe = Xn[:, cols], Xe[:, cols]
    if config["model"] == "linear":
        return LinearModel(Xn, Y_train).predict(Xe)
    if config["model"] != "rbf":
        raise ValueError(f"unknown model {config['model']!r}")
    centres = initial_centres(Xn, config["K"], base["centresInit"], seed,
                              base["kmeansIterations"])
    net = RBF(centres, config["sigma"], seed=seed)
    rates = [config["rates"] * r for r in GRID_RATES]
    train(net, Xn, Y_train, rates, config["epochs"], seed=seed)
    return net.predict(Xe)


def cv_mse(config, X, Y, base, folds=5, repeats=4, seed=0):
    """(mean, std) over `repeats` of the k-fold CV mean squared error."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(repeats):
        sq_err = 0.0
        for part in fold_indices(len(Y), folds, rng):
            rest = np.setdiff1d(np.arange(len(Y)), part)
            pred = fit_predict(config, X[rest], Y[rest], X[part], base, seed)
            sq_err += float(np.sum((Y[part] - pred) ** 2))
        scores.append(sq_err / len(Y))
    return float(np.mean(scores)), float(np.std(scores))


def make_grid(features=GRID_FEATURES, ks=GRID_K, sigmas=GRID_SIGMA,
              linear_features=LINEAR_FEATURES, epochs=None):
    """The coarse grid (GRID_RATES, COARSE_EPOCHS)."""
    blank = {"K": "-", "sigma": "-", "rates": "-", "epochs": "-"}
    grid = [dict(blank, model="mean", features=0)]
    grid += [dict(blank, model="linear", features=m) for m in linear_features]
    grid += [{"model": "rbf", "features": m, "K": k, "sigma": s, "rates": 1.0,
              "epochs": epochs or COARSE_EPOCHS}
             for m, k, s in itertools.product(features, ks, sigmas)]
    return grid


def make_fine_grid(ks=FINE_K, sigmas=FINE_SIGMA, factors=FINE_RATE_FACTORS,
                   epochs=FINE_EPOCHS):
    """The fine grid: all features, K x sigma x rate factor x epochs."""
    return [{"model": "rbf", "features": 0, "K": k, "sigma": s, "rates": f, "epochs": e}
            for k, s, f, e in itertools.product(ks, sigmas, factors, epochs)]


def _score(job):
    config, X, Y, base, folds, repeats, seed = job
    mean, std = cv_mse(config, X, Y, base, folds, repeats, seed)
    return dict(config, cv_mse=mean, cv_std=std)


def run_grid(grid, X, Y, base, folds=5, repeats=4, seed=0, jobs=1, log=print):
    """Score every config; return the rows sorted by CV MSE (best first)."""
    work = [(c, X, Y, base, folds, repeats, seed) for c in grid]
    if jobs > 1:
        import multiprocessing
        with multiprocessing.Pool(jobs) as pool:
            rows = pool.map(_score, work)
    else:
        rows = []
        for i, job in enumerate(work, start=1):
            rows.append(_score(job))
            if log:
                r = rows[-1]
                log(f"[{i}/{len(work)}] {describe(r):<34} CV MSE {r['cv_mse']:.4f}")
    return sorted(rows, key=lambda r: r["cv_mse"])


def describe(row):
    m = "all" if row["features"] == 0 else row["features"]
    if row["model"] == "rbf":
        return (f"rbf  features {m:<3} K {row['K']:<2} sigma {row['sigma']} "
                f"rates x{row['rates']} epochs {row['epochs'] or 'base'}")
    return f"{row['model']:<6} features {m}"


def write_rows(rows, folder, folds, repeats, n, base=None, grid="coarse"):
    out_txt, out_csv = OUT_FILES[grid]
    with open(os.path.join(folder, out_csv), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: (f"{r[c]:.6f}" if c.startswith("cv_") else r[c])
                             for c in COLUMNS})
    lines = [f"{folds}-fold cross-validation x {repeats} repeats on the {n} training "
             f"molecules only, best first ({grid} grid)",
             "features: all = 53; rates: factor times "
             + " / ".join(f"{r:g}" for r in GRID_RATES) + " (weights / centres / sigmas)", "",
             f"{'model':<7}{'features':>9}{'K':>4}{'sigma':>7}{'rates':>7}{'epochs':>8}"
             f"{'CV MSE':>9}{'+- std':>9}"]
    for r in rows:
        m = "all" if r["features"] == 0 else str(r["features"])
        ep = r["epochs"]
        rates = r["rates"] if r["rates"] == "-" else f"x{r['rates']:g}"
        lines.append(f"{r['model']:<7}{m:>9}{str(r['K']):>4}{str(r['sigma']):>7}"
                     f"{rates:>7}{str(ep):>8}{r['cv_mse']:>9.4f}{r['cv_std']:>9.4f}")
    with open(os.path.join(folder, out_txt), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return lines


def main_cli(argv=None):
    parser = argparse.ArgumentParser(description="Cross-validation on the training set.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--epochs", type=int, help="override maxIterations (quick check)")
    parser.add_argument("--jobs", type=int, default=1, help="parallel processes")
    parser.add_argument("--grid", choices=("coarse", "fine"), default="coarse")
    parser.add_argument("--out", default=CODE_DIR, help="folder for the result files")
    args = parser.parse_args(argv)
    try:
        base = read_parameters(os.path.join(CODE_DIR, "parameters.txt"))
        if args.epochs is not None:   # quick check: same epochs everywhere
            global COARSE_EPOCHS, FINE_EPOCHS
            COARSE_EPOCHS, FINE_EPOCHS = args.epochs, (args.epochs,)
        _, Y, X = read_dataset(os.path.join(CODE_DIR, base["trainFile"]),
                               base["numInputNeurons"])
        grid = (make_grid() if args.grid == "coarse" else
                make_fine_grid(epochs=FINE_EPOCHS))
        rows = run_grid(grid, X, Y, base, args.folds, args.repeats,
                        base["randomSeed"], args.jobs)
    except (OSError, ValueError, ParameterError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    lines = write_rows(rows, args.out, args.folds, args.repeats, len(Y), base, args.grid)
    print("\n".join(lines[:9]))
    print(f"Wrote {os.path.join(args.out, OUT_FILES[args.grid][0])} and "
          f"{OUT_FILES[args.grid][1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main_cli())

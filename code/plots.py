"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Plots for the report, drawn from the output files of a run so they work for
any run (any number of epochs, centres or test molecules):

  error_curve.png  training and test MSE per epoch (from results.txt)
  predictions.png  actual vs predicted activity Y of every test molecule,
                   with the y = x line of a perfect prediction and the test
                   MSE, RMSE and R^2 (from predictions.txt)
  sin_fit.png      sanity check of the implementation: an RBF network with
                   6 centres trained on sin(x), x in [0, 2pi], with the same
                   variable-centres algorithm; shows the target, the fitted
                   curve and the centres before and after training

To draw all three into code/plots/ after python3 main.py:
    python3 plots.py [folder]

matplotlib is used only here, with the non-interactive "Agg" backend, so
plotting works without a screen. It is imported only when a plot is drawn.
"""

import os
import sys

import numpy as np

import metrics
from main import PREDICTIONS_FILE, RESULTS_FILE
from rbf import RBF
from training import mse, train

PLOTS_FOLDER = "plots"
ERROR_PLOT = "error_curve.png"
PREDICTIONS_PLOT = "predictions.png"
SIN_PLOT = "sin_fit.png"
DPI = 300

# The figures are drawn at (about) the size they are printed in the report,
# so a font size here is the printed size in points: >= 8 pt, readable on paper.
FONT_SIZES = {"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
              "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
              "legend.fontsize": 8.5, "legend.title_fontsize": 8.5}
ERROR_FIGSIZE = (3.75, 2.55)        # inches
PREDICTIONS_FIGSIZE = (3.0, 2.9)
SIN_FIGSIZE = (3.75, 2.45)
LABEL_FONTSIZE = 8

# The sin(x) check: same settings as the test in tests/test_training.py.
SIN_POINTS = 40
SIN_CENTRES = 6
SIN_SIGMA = 1.0
SIN_RATES = (0.05, 0.01, 0.01)   # weights, centres, sigmas
SIN_EPOCHS = 300
SIN_SEED = 1

# Colours: blue and orange are easy to tell apart, also for colour-blind readers.
TRAIN_COLOUR = "#2a78d6"
TEST_COLOUR = "#eb6834"
INK = "#0b0b0b"
MUTED_INK = "#52514e"
GRID_COLOUR = "#dddcd8"


class PlotError(Exception):
    """An output file is missing or not in the expected format."""


# ---------------------------------------------------------------- reading files

def _check_file(path):
    if not os.path.isfile(path):
        raise PlotError(f"{path} not found")
    with open(path, encoding="utf-8") as f:
        if not f.read().strip():
            raise PlotError(f"{path} is empty")


def read_results(path):
    """Read results.txt. Returns (epochs, train_mse, test_mse) as arrays
    (test_mse is nan where the run had no test set)."""
    _check_file(path)
    try:
        table = np.loadtxt(path, ndmin=2)
    except ValueError as error:
        raise PlotError(f"{path}: {error}") from None
    if table.shape[1] != 3:
        raise PlotError(f"{path}: expected 3 columns (epoch trainMSE testMSE), "
                        f"found {table.shape[1]}")
    return table[:, 0].astype(int), table[:, 1], table[:, 2]


def read_predictions(path):
    """Read predictions.txt. Returns (names, true_y, predicted_y)."""
    _check_file(path)
    names, true_y, pred_y = [], [], []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 3:
                raise PlotError(f"{path} line {number}: expected 3 columns "
                                f"(molecule trueY predictedY), found {len(parts)}")
            try:
                t, y = float(parts[1]), float(parts[2])
            except ValueError:
                raise PlotError(f"{path} line {number}: trueY and predictedY "
                                f"must be numbers") from None
            names.append(parts[0])
            true_y.append(t)
            pred_y.append(y)
    return names, np.array(true_y), np.array(pred_y)


# ---------------------------------------------------------------- drawing

def _pyplot():
    """Import pyplot with the file-only Agg backend (works without a screen)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(FONT_SIZES)
    return plt


def _style(ax):
    """Light grid behind the data, no top/right frame."""
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID_COLOUR, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED_INK)
    ax.tick_params(colors=MUTED_INK)


def _save(fig, path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    _pyplot().close(fig)
    return path


def plot_error_curve(epochs, train_mse, test_mse, path):
    """Figure 1: training and test MSE per epoch."""
    epochs = np.asarray(epochs)
    train_mse = np.asarray(train_mse, dtype=float)
    test_mse = np.asarray(test_mse, dtype=float)
    if not (len(epochs) == len(train_mse) == len(test_mse)) or len(epochs) == 0:
        raise ValueError("epochs, train_mse and test_mse must have the same, non-zero length")
    has_test = not np.all(np.isnan(test_mse))

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=ERROR_FIGSIZE)
    marker = "o" if len(epochs) == 1 else None   # a single point is not visible as a line
    ax.plot(epochs, train_mse, color=TRAIN_COLOUR, linewidth=1.4, marker=marker,
            label="Training set")
    final = f"final: train {train_mse[-1]:.4f}"
    if has_test:
        ax.plot(epochs, test_mse, color=TEST_COLOUR, linewidth=1.4, linestyle="--",
                marker=marker, label="Test set")
        final += f", test {test_mse[-1]:.4f}"
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Training and test MSE per epoch", color=INK)
    ax.set_xlim(epochs[0], epochs[-1] if len(epochs) > 1 else epochs[0] + 1)
    ax.set_ylim(bottom=0)
    _style(ax)
    ax.legend(frameon=False, title=final)
    return _save(fig, path)


# label offsets in points (dx, dy, horizontal alignment), tried in this order
_LABEL_SPOTS = [(5, 4, "left"), (5, -12, "left"), (-5, 4, "right"), (-5, -12, "right"),
                (0, 9, "center"), (0, -17, "center")]


def _label_points(fig, ax, names, xs, ys, fontsize=LABEL_FONTSIZE):
    """Write each name next to its point, at the first spot that does not
    cover another point or an earlier label (in screen units)."""
    ax.apply_aspect()                               # final axes box before measuring
    scale = 72.0 / fig.dpi                          # pixels -> points
    pts = ax.transData.transform(np.column_stack([xs, ys])) * scale
    width_per_char, height, dot = 0.62 * fontsize, fontsize, 4.0
    taken = []                                      # boxes of placed labels

    def box(px, py, dx, dy, ha, text):
        w = width_per_char * len(text)
        x0 = px + dx - (w if ha == "right" else w / 2 if ha == "center" else 0)
        return (x0, py + dy, x0 + w, py + dy + height)

    def overlaps(b, others_pts):
        for bx0, by0, bx1, by1 in taken:
            if b[0] < bx1 and bx0 < b[2] and b[1] < by1 and by0 < b[3]:
                return True
        return any(b[0] - dot < qx < b[2] + dot and b[1] - dot < qy < b[3] + dot
                   for qx, qy in others_pts)

    for i, name in enumerate(names):
        px, py = pts[i]
        others = [q for j, q in enumerate(pts) if j != i]
        choice = _LABEL_SPOTS[0]
        for spot in _LABEL_SPOTS:
            if not overlaps(box(px, py, *spot, name), others):
                choice = spot
                break
        dx, dy, ha = choice
        taken.append(box(px, py, dx, dy, ha, name))
        ax.annotate(name, (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="bottom", fontsize=fontsize, color=INK)


def plot_predictions(names, true_y, pred_y, path):
    """Figure 2: actual vs predicted Y of the test molecules, y = x line,
    each point labelled with its molecule, MSE/RMSE/R^2 in the legend."""
    true_y = np.asarray(true_y, dtype=float)
    pred_y = np.asarray(pred_y, dtype=float)
    if not (len(names) == len(true_y) == len(pred_y)) or len(names) == 0:
        raise ValueError("names, true_y and pred_y must have the same, non-zero length")
    test_mse = metrics.mse(true_y, pred_y)
    test_rmse = metrics.rmse(true_y, pred_y)
    test_r2 = metrics.r2(true_y, pred_y)

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=PREDICTIONS_FIGSIZE)
    low = min(true_y.min(), pred_y.min())
    high = max(true_y.max(), pred_y.max())
    pad = 0.1 * (high - low) if high > low else 0.5
    low, high = low - pad, high + pad
    ax.plot([low, high], [low, high], color=MUTED_INK, linewidth=1, linestyle="--",
            label="y = x (perfect)")
    ax.scatter(true_y, pred_y, s=18, color=TRAIN_COLOUR, zorder=3,
               label=f"Test molecules ({len(names)})")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual Y")
    ax.set_ylabel("Predicted Y")
    _label_points(fig, ax, names, true_y, pred_y)
    ax.set_title("Test set: actual vs predicted", color=INK)
    _style(ax)
    ax.legend(loc="upper left", frameon=False, handlelength=1.2, borderaxespad=0.2,
              title=f"RMSE {test_rmse:.3f}, R² {test_r2:.2f}")
    return _save(fig, path)


def sin_fit(points=SIN_POINTS, centres=SIN_CENTRES, sigma=SIN_SIGMA,
            rates=SIN_RATES, epochs=SIN_EPOCHS, seed=SIN_SEED):
    """Train an RBF network on sin(x), x in [0, 2pi] (evenly spaced points,
    centres evenly spaced over the same range). Returns
    (x, y, net, initial_centres, mse_before, mse_after)."""
    x = np.linspace(0, 2 * np.pi, points)
    y = np.sin(x)
    initial = np.linspace(0, 2 * np.pi, centres)
    net = RBF(initial[:, None], sigma, seed=seed)
    before = mse(net, x[:, None], y)
    history = train(net, x[:, None], y, rates, epochs, seed=seed)
    return x, y, net, initial, before, history[-1][1]


def plot_sin_fit(path, **settings):
    """Figure 3: the sin(x) sanity check (target, RBF output, centres)."""
    x, y, net, initial, before, after = sin_fit(**settings)
    epochs = settings.get("epochs", SIN_EPOCHS)
    dense = np.linspace(x[0], x[-1], 400)

    plt = _pyplot()
    fig, ax = plt.subplots(figsize=SIN_FIGSIZE)
    ax.plot(dense, np.sin(dense), color=MUTED_INK, linewidth=1.2, label="sin(x) (target)")
    ax.scatter(x, y, s=12, color=MUTED_INK, zorder=3, label=f"Points ({len(x)})")
    ax.plot(dense, net.predict(dense[:, None]), color=TEST_COLOUR, linewidth=1.8,
            linestyle="--", label="RBF output")
    base = -1.35
    ax.plot(initial, np.full(len(initial), base), "|", color=MUTED_INK, markersize=9,
            label="Centres before")
    ax.plot(net.centres[:, 0], np.full(net.num_hidden, base), "v", color=TRAIN_COLOUR,
            markersize=5, label="Centres after")
    ax.set_ylim(-1.55, 1.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"RBF fit of sin(x), {net.num_hidden} centres", color=INK)
    _style(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
              handlelength=1.3,
              title=f"MSE {before:.3f} \u2192 {after:.5f}\n({epochs} epochs)")
    return _save(fig, path)


# ---------------------------------------------------------------- all plots of a run

def save_plots(folder="."):
    """Draw the three figures for the run whose results.txt and
    predictions.txt are in `folder`. Images go to folder/plots/.
    Returns the list of written paths."""
    out = os.path.join(folder, PLOTS_FOLDER)
    return [
        plot_error_curve(*read_results(os.path.join(folder, RESULTS_FILE)),
                         os.path.join(out, ERROR_PLOT)),
        plot_predictions(*read_predictions(os.path.join(folder, PREDICTIONS_FILE)),
                         os.path.join(out, PREDICTIONS_PLOT)),
        plot_sin_fit(os.path.join(out, SIN_PLOT)),
    ]


def main(argv=None):
    """python3 plots.py [folder]: draw the figures into folder/plots/."""
    argv = sys.argv[1:] if argv is None else argv
    folder = argv[0] if argv else "."
    try:
        written = save_plots(folder)
    except PlotError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except ImportError:
        print("Error: matplotlib is not installed (pip install matplotlib)", file=sys.stderr)
        return 1
    for path in written:
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

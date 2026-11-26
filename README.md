# RBF Network (EPL442 Assignment 4)

Radial Basis Function network written from scratch in Python + NumPy
(matplotlib only for plots). It does regression on the Selwood data set
(31 molecules, 53 features, activity Y), split 21 training / 10 test.

## Layout

```
code/
  parameters.txt   all run settings (read from the current folder)
  parameters.py    reads and validates parameters.txt
  selwood.txt      the original Selwood data set
  data.py          reads selwood.txt, training/test and centres files ("<-1*" -> -1)
  normalise.py     min-max scaling to [0, 1] with training-set statistics only
  kmeans.py        our own k-means (k-means++ start) and random-sample centres
  prepare_data.py  seeded 21/10 split -> training.txt, test.txt; centres -> centerVectors.txt
  rbf.py           the RBF network: Gaussian hidden neurons, linear output + bias;
                   forward pass and the online variable-centres update
  training.py      epoch loop: shuffled online updates, train/test MSE per epoch
  metrics.py       MSE, RMSE and R^2
  results.py       writes results.txt, weights.txt, predictions.txt (reads weights.txt back)
  main.py          runs the whole pipeline: python3 main.py
  plots.py         draws the report figures into plots/: python3 plots.py
  experiments.py   runs every experiment of the report into experiments/: python3 experiments.py
  cv.py            repeated k-fold cross-validation on the training set only: python3 cv.py
  features.py      feature selection: the m features most correlated with Y (training set only)
  reference.py     reference models: predict the training mean; least-squares linear fit
  cv_results.txt, cv_fine.txt  cross-validation tables (coarse and fine grid), best first
  training.txt     21 training molecules, "name,Y,53 features" per line (raw values)
  test.txt         10 test molecules, same format
  centerVectors.txt  initial centres, one per line, 53 normalised values
  results.txt, weights.txt, predictions.txt  output of the baseline run
  plots/           figures of the baseline run (error_curve, predictions, sin_fit)
  experiments/     one folder per experiment + summary.txt / summary.csv
  tests/           unit tests (pytest)
```

## The network (rbf.py)

Hidden neuron j: phi_j(x) = exp(-||x - c_j||^2 / (2 sigma_j^2)).
Output: y(x) = sum_j w_j phi_j(x) + b.
Both follow the course notes (lecture L7, "Learning Algorithm for variable
centres"). Centres are read from centresFile, sigmas from `sigmas`; the
initial output weights and bias are uniform in [-1, 1] (as in the notes),
drawn with randomSeed.

## Training (rbf.py, training.py)

Variable-centres learning from the course notes (L7, slides 58-59). After
every pattern, with eps = t - y and all right-hand sides taken from the old
values (every change is computed first, then all are applied):

    w_j'     = w_j     + eta_w     * eps * phi_j
    b'       = b       + eta_w     * eps
    c_ji'    = c_ji    + eta_c     * eps * w_j * phi_j * (x_i - c_ji) / sigma_j^2
    sigma_j' = sigma_j + eta_sigma * eps * w_j * phi_j * ||x - c_j||^2 / sigma_j^3

These are gradient descent on E = 1/2 eps^2 (checked against finite
differences in the tests). Each epoch visits the training patterns in a new
random order (seeded by randomSeed) and records train and test MSE. A
learning rate of 0 keeps the centres (or sigmas) fixed. Our addition: a sigma
is never allowed below 0.001 (MIN_SIGMA in rbf.py).

## Parameters

`parameters.txt` has one `name value(s)` line per setting. Blank lines and
anything after `#` are ignored.

| name | meaning |
|---|---|
| numHiddenLayerNeurons | number of RBF centres K (>= 1) |
| numInputNeurons | number of input features (53 for Selwood) |
| numOutputNeurons | number of outputs (must be 1, regression on Y) |
| learningRates | 3 rates: weights, centres, sigmas (weights > 0, the others >= 0; 0 keeps them fixed) |
| sigmas | initial width: 1 value for all neurons, or K values (one per neuron), all > 0 |
| maxIterations | number of training epochs (>= 1) |
| randomSeed | seed for all random choices (>= 0) |
| centresFile | initial centres file (centerVectors.txt) |
| trainFile / testFile | training and test set files |
| centresInit | our addition (optional, default kmeans): `kmeans` or `samples` (K random training molecules) |
| kmeansIterations | our addition (optional, default 100): max k-means rounds |
| numFeatures | our addition (optional, default 0 = all): use only the numFeatures features most correlated with Y in the training set |

## Preparing the data

```
cd code
python3 prepare_data.py
```

Uses `randomSeed`, `trainFile`, `testFile`, `numHiddenLayerNeurons`,
`centresInit`, `kmeansIterations` and `centresFile` from `parameters.txt`.

Normalisation: every feature is min-max scaled to [0, 1] with the minimum and
range of the training set only; the test set uses the same numbers, so test
values may fall slightly outside [0, 1] (they are not clipped). A feature that
is constant in the training set is mapped to 0 (no division by zero). Y is not
scaled. training.txt / test.txt keep the raw values; the program normalises
them when it loads them, and the centres are in the normalised space.

## Running

```
cd code
python3 prepare_data.py   # only when the split, K or centresInit change
python3 main.py
```

`main.py` reads `parameters.txt` from the current folder, trains for
`maxIterations` epochs, prints the final training MSE and the test MSE, RMSE
and R^2, and writes three files next to it (all columns separated by spaces):

| file | contents |
|---|---|
| results.txt | one line per epoch: `epoch trainMSE testMSE` |
| weights.txt | `weights w_1 ... w_K`, `bias b`, then `centre j c_j1 ... c_jd` for j = 1..K (normalised space), then `sigmas s_1 ... s_K` |
| predictions.txt | one line per test molecule: `molecule trueY predictedY` |

R^2 = 1 - sum((t - y)^2) / sum((t - mean(t))^2): 1 is a perfect fit, 0 is no
better than predicting the mean test activity, negative is worse.

## Baseline (tuned by cross-validation)

The committed `parameters.txt` is the baseline: K = 10 centres (k-means),
initial sigma 1.0, learning rates 0.0025 / 0.00025 / 0.000125 (weights /
centres / sigmas), 1000 epochs, all 53 features, randomSeed 42 (test
molecules D30 J19 J1 A10 F15 G9 I26 B8 E24 B28). `results.txt`,
`weights.txt` and `predictions.txt` in code/ are the output of
`python3 main.py` with it.

| | train MSE | test MSE | test RMSE | test R^2 |
|---|---|---|---|---|
| first try (K 8, sigma 1.0, 2000 epochs) | 0.010 | 0.464 | 0.681 | 0.41 |
| baseline, seed 42 split | 0.118 | 0.526 | 0.725 | 0.33 |
| baseline, 5 random splits (mean +- std) | 0.194 | 0.376 +- 0.170 | 0.598 +- 0.136 | 0.37 +- 0.18 |
| predict the training mean, 5 splits | 0.680 | 0.642 +- 0.201 | 0.792 +- 0.121 | -0.10 +- 0.14 |

How it was chosen (`cv.py`, results in `cv_results.txt` and `cv_fine.txt`):
5-fold cross-validation repeated 4 times with new folds, on the 21 training
molecules only (the test file is never read). In every fold the scaler, the
feature choice, the k-means centres and the network are fitted on the
training folds only. The coarse grid tries 5, 10, 20 or all 53 features,
K = 3, 5, 8 and sigma 0.5-3 (2000 epochs); the fine grid tries, with all
features, K = 5-10, sigma 0.75-1.5, the learning rates x0.25-x2 and 250-4000
epochs. Lowest CV MSE: 0.406 +- 0.055 (K 10, sigma 1.0, rates x0.25, 1000
epochs). More training raises the CV error, so stopping early is the
regulariser. Feature selection did not beat all features (best 0.455 with
20 features vs 0.452), so the baseline uses all 53.

```
cd code
python3 cv.py --jobs 4                 # coarse grid, about 5 minutes on 4 cores
python3 cv.py --grid fine --jobs 4     # fine grid, about 15 minutes on 4 cores
```

The seed 42 test split is a hard one (the mean predictor has RMSE 0.944 on
it against 0.792 on average), so the report compares settings by their mean
over 5 random splits.

## Plots

```
cd code
python3 plots.py          # after python3 main.py; optional: python3 plots.py <folder>
```

Writes three PNG files to `plots/` (matplotlib, Agg backend, no screen needed):

| file | shows |
|---|---|
| error_curve.png | training and test MSE per epoch (from results.txt) |
| predictions.png | actual vs predicted Y for every test molecule, labelled, with the y = x line and the test MSE, RMSE and R^2 (from predictions.txt) |
| sin_fit.png | sanity check: 6 centres trained on sin(x), x in [0, 2pi] (40 points, rates 0.05/0.01/0.01, 300 epochs, seed 1): target, network output and the centres before and after training; MSE 1.159 -> 0.00065 |

## Experiments

```
cd code
python3 experiments.py                 # all 72 runs, about 1 minute
python3 experiments.py --list          # the experiment names
python3 experiments.py --only K_3 K_5  # some of them (the summary then has only these)
python3 experiments.py --epochs 50     # quick check with fewer epochs
```

Every experiment is the baseline `parameters.txt` with one setting changed.
Each runs in its own folder `experiments/<name>/`: it writes its
`parameters.txt`, makes its own 21/10 split, normalisation, feature choice and
k-means centres from its own training set (`prepare_data.prepare` with its
randomSeed), trains (`main.run`) and writes the usual output files plus
`run.txt` (final scores). Every setting is run on the seed 42 split and on the
random splits 1-5 (`<name>_split_<seed>`), so settings are compared by their
mean over splits. `experiments/summary.csv` has one row per run;
`experiments/summary.txt` has the same table plus mean +- std (population
std) over the random splits for every setting.

| group | runs | changes |
|---|---|---|
| baseline | baseline, split_1..5 | nothing |
| centres | K_3, K_5, K_15 | numHiddenLayerNeurons |
| sigma | sigma_0.5, sigma_2.0 | initial sigma |
| rates | rates_x4 (0.01/0.001/0.0005), rates_x8 (0.02/0.002/0.001) | learningRates |
| fixed | fixed_centres, split_1..5_fixed | centre learning rate 0 (sigmas still learn) |
| features | features | numFeatures 20, K 5, sigma 3.0 (best reduced setting by CV) |
| reference | mean, linear | not an RBF: predict the training mean; least-squares linear fit |

Results (test RMSE; mean +- std over the random splits 1-5):

| run | seed 42 | 5 splits RMSE | 5 splits R^2 |
|---|---|---|---|
| baseline (K 10, sigma 1.0) | 0.725 | 0.598 +- 0.136 | 0.37 +- 0.18 |
| K 3 | 0.751 | 0.601 +- 0.148 | 0.36 +- 0.21 |
| K 5 | 0.732 | 0.593 +- 0.141 | 0.38 +- 0.21 |
| K 15 | 0.654 | 0.592 +- 0.136 | 0.38 +- 0.19 |
| sigma 0.5 | 0.834 | 0.682 +- 0.142 | 0.17 +- 0.24 |
| sigma 2.0 | 0.698 | 0.600 +- 0.187 | 0.36 +- 0.29 |
| rates x4 | 0.692 | 0.590 +- 0.149 | 0.38 +- 0.22 |
| rates x8 | 0.691 | 0.592 +- 0.132 | 0.38 +- 0.19 |
| fixed centres | 0.737 | 0.606 +- 0.130 | 0.36 +- 0.17 |
| 20 selected features | 0.738 | 0.605 +- 0.188 | 0.35 +- 0.29 |
| mean predictor | 0.944 | 0.792 +- 0.121 | -0.10 +- 0.14 |
| linear least squares | 1.174 | 1.260 +- 0.580 | -2.60 +- 3.30 |

Apart from sigma 0.5, every RBF setting is within 0.02 of the baseline on
average, far less than the spread between splits (test RMSE 0.42 to 0.81 for
the baseline). All of them clearly beat the mean predictor and the linear
fit.

## Tests

```
cd code
python3 -m pytest tests -q
```

225 tests; the 212 outside `tests/test_make_zip.py` are the ones that go into
the zip (make_zip.py itself is not part of the submission).

## Report and submission zip

The report source is `report/report.html` (A4, 11pt; the figures come from
`code/plots/`). To turn it into `report.pdf` in the repository root, open it
in Chrome/Chromium and print to PDF, or run (no page header/footer):

```
chromium --headless --no-pdf-header-footer --print-to-pdf=report.pdf report/report.html
```

`README.txt` is the plain-text readme that goes into the zip. From the
repository root, after `python3 main.py` in code/:

```
python3 make_zip.py 1134609 --name "Andreas Evagorou"
```

writes `EPL442_ex4_1134609.zip` with `code/` (source, tests, data,
parameters, the baseline output files, plots, the experiment summary tables
and the cross-validation tables), `README.txt` and `report.pdf`. It fills the name and student ID into
the header of every file inside the zip only (the repository files keep the
placeholders), leaves out caches and the per-run experiment folders, and
warns if the report has more than 5 pages. The zip is ignored by git.


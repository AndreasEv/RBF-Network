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
  training.txt     21 training molecules, "name,Y,53 features" per line (raw values)
  test.txt         10 test molecules, same format
  centerVectors.txt  initial centres, one per line, 53 normalised values
  results.txt, weights.txt, predictions.txt  output of the baseline run
  tests/           unit tests (pytest)
```

More modules (plots, experiments) are added as the work goes on.

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

## Baseline (tuned)

The committed `parameters.txt` is the baseline: K = 3 centres (k-means),
initial sigma 2.0 for every neuron, learning rates 0.01 / 0.001 / 0.0005
(weights / centres / sigmas), 2000 epochs, randomSeed 42 (test molecules
D30 J19 J1 A10 F15 G9 I26 B8 E24 B28). `results.txt`, `weights.txt` and
`predictions.txt` in code/ are the output of `python3 main.py` with it.

| | train MSE | test MSE | test RMSE | test R^2 |
|---|---|---|---|---|
| first try (K 8, sigma 1.0) | 0.010 | 0.464 | 0.681 | 0.41 |
| baseline (K 3, sigma 2.0) | 0.179 | 0.376 | 0.613 | 0.52 |

How it was chosen: the first try fitted the 21 training molecules almost
perfectly but generalised badly (overfitting). We tried K = 3, 4, 5, 6, 8,
10, sigma = 1.0, 1.5, 2.0, 2.5, 3.0, four learning-rate settings (including
fixed centres and sigmas) and 100 to 2000 epochs, and picked the setting with
the lowest 5-fold cross-validation MSE on the training set only (0.338), so
the test set was not used for the choice. Fewer, wider Gaussians generalise
better here: sigma 2.0 is a bit below the mean distance between the centres
(about 2.5 in the normalised space), in line with the width hint of the
course notes (L7). The CV error is flat between 2000 and 4000 epochs and
rises after that, so we keep 2000. The worst test prediction is D30
(true -1, the "<-1*" molecule, predicted 0.69).

## Tests

```
cd code
python3 -m pytest tests -q
```

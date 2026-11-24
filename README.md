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
  training.txt     21 training molecules, "name,Y,53 features" per line (raw values)
  test.txt         10 test molecules, same format
  centerVectors.txt  initial centres, one per line, 53 normalised values
  tests/           unit tests (pytest)
```

More modules (data preparation, the RBF network, training, outputs, plots,
experiments) are added as the work goes on.

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

## Tests

```
cd code
python3 -m pytest tests -q
```

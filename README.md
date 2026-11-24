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
  data.py          reads selwood.txt and training/test files ("<-1*" -> -1)
  prepare_data.py  seeded 21/10 split -> training.txt, test.txt
  training.txt     21 training molecules, "name,Y,53 features" per line
  test.txt         10 test molecules, same format
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

## Preparing the data

```
cd code
python3 prepare_data.py
```

Uses `randomSeed`, `trainFile` and `testFile` from `parameters.txt`.

## Tests

```
cd code
python3 -m pytest tests -q
```

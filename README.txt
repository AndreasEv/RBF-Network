EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

A Radial Basis Function (RBF) network written from scratch in Python with
NumPy. It predicts the activity Y of the 31 molecules of the Selwood data
set (53 features each) from a random split of 21 training and 10 test
molecules, using the "variable centres" learning algorithm of the course
notes (lecture L7): the output weights, the bias, the centres and the
widths (sigmas) of the Gaussian neurons are all learned.


1. Language and version
-----------------------
Python 3 (tested with Python 3.13; any Python 3.8 or newer works).
Python is interpreted, so there is no compile step.


2. Required libraries
---------------------
  numpy        all matrix operations (required)
  matplotlib   only for the plots (plots.py)
  pytest       only for the unit tests

Install them with:
  pip3 install numpy matplotlib pytest
(or: pip3 install -r requirements.txt from the folder above code/)

No RBF or machine learning library is used (no scikit-learn, PyTorch, ...).
The k-means that chooses the initial centres is also our own (kmeans.py).


3. Files
--------
code/
  main.py            entry point: trains one network with parameters.txt
  parameters.py      reads and validates parameters.txt
  prepare_data.py    split 21/10 -> training.txt, test.txt; centres ->
                     centerVectors.txt
  data.py            reads selwood.txt ("<-1*" is read as -1), the
                     training/test files and the centres file
  normalise.py       min-max scaling to [0, 1] with training-set statistics
  kmeans.py          our own k-means (k-means++ start) for the initial centres
  rbf.py             the RBF network: Gaussian neurons, linear output + bias,
                     forward pass and the variable-centres update
  training.py        epoch loop: shuffled online updates, MSE per epoch
  metrics.py         MSE, RMSE and R^2
  results.py         writes results.txt, weights.txt and predictions.txt
  plots.py           draws the figures of the report into plots/
  experiments.py     runs the parameter comparisons into experiments/
  cv.py              repeated k-fold cross-validation on the training set
                     only (how the baseline was chosen)
  features.py        feature selection: the m features most correlated
                     with Y in the training set (numFeatures)
  reference.py       reference models: predict the training mean, and a
                     least-squares linear fit
  cv_results.txt, cv_fine.txt   the cross-validation tables (best first)
  parameters.txt     the settings of the baseline run presented in the report
  selwood.txt        the original data set
  training.txt       the 21 training molecules (name, Y, 53 features; raw)
  test.txt           the 10 test molecules (same format)
  centerVectors.txt  the initial centres, one per line (normalised space)
  results.txt, weights.txt, predictions.txt   output of the baseline run
  plots/             the figures of the baseline run
  experiments/       summary.txt / summary.csv: tables of all experiments
  tests/             unit tests (pytest)
README.txt           this file
report.pdf           the report


4. How to run
-------------
Every program reads parameters.txt from the CURRENT folder and takes every
setting from it, so run them from inside code/:

  cd code
  python3 prepare_data.py
  python3 main.py

prepare_data.py makes training.txt, test.txt and centerVectors.txt. It is
needed only when randomSeed, numHiddenLayerNeurons or centresInit change
(the files of the baseline are included, so main.py can be run directly).
main.py trains for maxIterations epochs and prints the final training MSE
and the test MSE, RMSE and R^2. The baseline run takes about 1 second.

parameters.txt has one "name value(s)" line per setting (anything after #
is a comment):
  numHiddenLayerNeurons   number of RBF centres K
  numInputNeurons         number of input features (53)
  numOutputNeurons        number of outputs (1)
  learningRates           3 rates: output weights, centres, sigmas
                          (a rate of 0 keeps the centres or sigmas fixed)
  sigmas                  initial sigma: 1 value for all neurons, or K values
  maxIterations           number of epochs
  randomSeed              seed for the split, the k-means, the initial
                          weights and the order of the patterns
  centresFile             initial centres (centerVectors.txt)
  trainFile, testFile     training and test files
  centresInit             our addition (optional, default kmeans): kmeans,
                          or samples (K random training molecules)
  kmeansIterations        our addition (optional, default 100)
  numFeatures             our addition (optional, default 0 = all 53):
                          use only the numFeatures features most
                          correlated with Y in the training set

The baseline parameters.txt: K = 10, sigma 1.0, learning rates
0.0025 / 0.00025 / 0.000125, 1000 epochs, randomSeed 42, centresInit
kmeans, all 53 features. It was chosen by cross-validation on the training
set only:
  python3 cv.py --jobs 4               coarse grid -> cv_results.txt
  python3 cv.py --grid fine --jobs 4   fine grid   -> cv_fine.txt
(about 5 and 15 minutes on 4 processor cores).


5. Output files
---------------
All columns are separated by spaces.
  results.txt       one line per epoch: epoch, training MSE, test MSE
  weights.txt       "weights w_1 ... w_K", "bias b", then "centre j ..."
                    (53 values, normalised space) for every centre, then
                    "sigmas s_1 ... s_K": the trained network
  predictions.txt   one line per test molecule: name, true Y, predicted Y

The files in code/ are from the baseline run: final training MSE 0.118,
test MSE 0.526, test RMSE 0.725, test R^2 0.33 (over 5 other random
splits: test RMSE 0.598 +- 0.136, R^2 0.37 +- 0.18).


6. How the plots are made
-------------------------
  cd code
  python3 plots.py

(after main.py) reads results.txt and predictions.txt and writes:
  plots/error_curve.png   training and test MSE per epoch
  plots/predictions.png   true vs predicted Y of the test molecules
  plots/sin_fit.png       sanity check: an RBF network trained on sin(x)
python3 plots.py <folder> draws the plots of another run folder (for
example experiments/K_5).


7. Experiments
--------------
  cd code
  python3 experiments.py

runs the 72 experiments of the report (about 1 minute): the baseline,
K = 3, 5, 15 centres, sigma 0.5 and 2.0, two other learning-rate settings,
fixed centres (centre learning rate 0), feature selection (20 features) and
the two reference models (training mean, least-squares linear fit), each
on the seed 42 split and on 5 other random 21/10 splits (randomSeed 1..5). Each run gets its
own folder experiments/<name>/ with its own split, normalisation and
centres, and the tables go to experiments/summary.txt and summary.csv
(with mean +- std over the splits).
  python3 experiments.py --list           the names of the experiments
  python3 experiments.py --only K_3 K_5   only some of them


8. Design choices
-----------------
Normalisation: every feature is min-max scaled to [0, 1] with the minimum
and range of the TRAINING set only; the test set uses the same numbers (not
clipped). A feature that is constant in the training set becomes 0. Y is
not scaled.
Hidden neuron h:  phi_h(x) = exp( -||x - c_h||^2 / (2 sigma_h^2) )
Output:           y(x) = sum_h w_h phi_h(x) + b
Update after every pattern, with eps = t - y and the OLD values on the
right-hand side (all changes are computed first, then applied):
  w_h'     = w_h     + eta_w     eps phi_h
  b'       = b       + eta_w     eps
  c_hi'    = c_hi    + eta_c     eps w_h phi_h (x_i - c_hi) / sigma_h^2
  sigma_h' = sigma_h + eta_sigma eps w_h phi_h ||x - c_h||^2 / sigma_h^3
Initial weights and bias: uniform in [-1, 1]. Each epoch visits the 21
training molecules in a new random order. Our addition: a sigma is never
allowed below 0.001.


9. Tests
--------
  cd code
  python3 -m pytest tests -q

The tests cover the parameter reader, the data files, the normalisation,
k-means, the network formulas (the updates are checked against numerical
derivatives), training (including the sin(x) check), the output files, the
plots, feature selection, the reference models, cross-validation and the
experiment runner. (The tests of make_zip.py stay in the repository, with
the script, and are not part of this archive.)


10. Submission archive
----------------------
From the folder above code/, after main.py has been run and with
report.pdf next to this file:

  python3 make_zip.py 123456 --name "Name Surname"

(with your own student ID instead of 123456) writes EPL442_ex4_123456.zip
with code/, README.txt and report.pdf. The name and student ID are filled
into the header of every file inside the zip; the files in the repository
are not changed.

"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Reading and writing the data files.

selwood.txt (the original data set) looks like this:
  line 1   title ("Selwood Data Set (31 compounds, 53 properties)")
  line 2   blank
  line 3   header: Compound, Activity, ATCH1, ..., (53 feature names)
  then     one row per molecule: name, activity Y, 53 feature values
  last     a footnote starting with '*' (not data)

Two molecules (K17, D30) have the activity "<-1*" (below -1, marked with the
footnote). We use -1 for them.

training.txt / test.txt (made by prepare_data.py) have no header: one line
per molecule, "name,Y,f1,...,f53", comma separated.

A data set is returned as a tuple (names, Y, X):
  names  list of molecule names (str), length N
  Y      numpy array of activities, shape (N,)
  X      numpy array of features, shape (N, 53)
"""

import os

import numpy as np

# Activity values in selwood.txt that are not plain numbers, and what we use.
SPECIAL_ACTIVITY = {"<-1*": -1.0, "<-1": -1.0}


class DataError(ValueError):
    """Raised when a data file has a bad line."""


def _parse_activity(text, where):
    text = text.strip()
    if text in SPECIAL_ACTIVITY:
        return SPECIAL_ACTIVITY[text]
    try:
        return float(text)
    except ValueError:
        raise DataError(f"{where}: bad activity value {text!r}") from None


def _parse_row(fields, where, num_features):
    """Turn [name, Y, f1..fn] into (name, y, features). Checks the count."""
    fields = [f.strip() for f in fields]
    if len(fields) != num_features + 2:
        raise DataError(
            f"{where}: expected name, activity and {num_features} features "
            f"({num_features + 2} values), got {len(fields)}")
    name = fields[0]
    if not name:
        raise DataError(f"{where}: missing molecule name")
    y = _parse_activity(fields[1], where)
    try:
        features = [float(v) for v in fields[2:]]
    except ValueError as e:
        raise DataError(f"{where}: bad feature value ({e})") from None
    return name, y, features


def _to_arrays(rows):
    names = [r[0] for r in rows]
    Y = np.array([r[1] for r in rows], dtype=float)
    X = np.array([r[2] for r in rows], dtype=float)
    return names, Y, X


def parse_selwood(lines, source="selwood.txt"):
    """Parse the lines of the original Selwood file into (names, Y, X).

    The header line (starting with "Compound") gives the number of features.
    Lines before it (title, blank) are skipped; after it, blank lines and the
    footnote (a line starting with '*') are skipped.
    """
    lines = list(lines)
    header_at = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("compound"):
            header_at = i
            break
    if header_at is None:
        raise DataError(f"{source}: no header line starting with 'Compound'")
    header = [h.strip() for h in lines[header_at].split(",")]
    num_features = len(header) - 2

    rows = []
    for number in range(header_at + 1, len(lines)):
        line = lines[number].strip()
        if not line or line.startswith("*"):
            continue
        rows.append(_parse_row(line.split(","), f"{source}, line {number + 1}",
                               num_features))
    if not rows:
        raise DataError(f"{source}: no data rows")
    return _to_arrays(rows)


def read_selwood(path):
    """Read the original Selwood file at `path`; return (names, Y, X)."""
    if not os.path.isfile(path):
        raise DataError(f"{path}: file not found")
    with open(path, encoding="utf-8") as f:
        return parse_selwood(f, source=os.path.basename(path))


def write_dataset(path, names, Y, X):
    """Write a data set as "name,Y,f1,...,fn" lines (no header).

    Numbers are written with repr(), the shortest text that reads back as
    exactly the same float, so a write followed by a read loses nothing.
    """
    with open(path, "w", encoding="utf-8") as f:
        for name, y, x in zip(names, Y, X):
            values = [repr(float(y))] + [repr(float(v)) for v in x]
            f.write(name + "," + ",".join(values) + "\n")


def read_dataset(path, num_features=None):
    """Read a file written by write_dataset (training.txt / test.txt).

    If `num_features` is given (numInputNeurons), every line must have exactly
    that many features. Otherwise the first line sets the count.
    Returns (names, Y, X).
    """
    if not os.path.isfile(path):
        raise DataError(f"{path}: file not found")
    source = os.path.basename(path)
    rows = []
    with open(path, encoding="utf-8") as f:
        for number, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            fields = line.split(",")
            if num_features is None:
                num_features = len(fields) - 2
            rows.append(_parse_row(fields, f"{source}, line {number}", num_features))
    if not rows:
        raise DataError(f"{source}: no data rows")
    return _to_arrays(rows)

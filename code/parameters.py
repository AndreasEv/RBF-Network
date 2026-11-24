"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Reads and validates parameters.txt. Every run setting comes from this file,
so no value is hard-coded in the rest of the program.

File format: one "name value(s)" line per setting, separated by whitespace.
Blank lines are ignored, and so is anything after a '#'. Every name in
SCHEMA must appear exactly once, except the extra settings in OPTIONAL,
which may be left out and then take their default value. Most settings take
one value; the list settings (learningRates, sigmas) take several.

The returned dict has these extra conveniences:
  learningRates -> also split into etaWeights, etaCentres, etaSigmas
  sigmas        -> always a list of numHiddenLayerNeurons values (a single
                   value in the file is repeated for every neuron)

To add a new parameter, add one entry to SCHEMA (and to OPTIONAL with a
default if it may be left out).
"""

import os


class ParameterError(ValueError):
    """Raised when parameters.txt is missing a setting or has a bad value."""


def _parse_int(text):
    """Parse a whole number. '8' is accepted, '8.5' and 'abc' are not."""
    return int(text)


def _parse_float(text):
    """Parse a real number, e.g. '0.5' or '1e-2'. Rejects nan and inf."""
    value = float(text)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("not a finite number")
    return value


def _parse_str(text):
    """Keep the value as text (used for file names)."""
    return text


def _parse_centre_method(text):
    """'kmeans' or 'samples' (how the initial centres are chosen)."""
    if text not in ("kmeans", "samples"):
        raise ValueError(text)
    return text


# name -> (parser, how many values ("one", "list"), check on each parsed
#          value, description of a valid value)
SCHEMA = {
    "numHiddenLayerNeurons": (_parse_int, "one", lambda v: v >= 1, "an integer >= 1"),
    "numInputNeurons":       (_parse_int, "one", lambda v: v >= 1, "an integer >= 1"),
    "numOutputNeurons":      (_parse_int, "one", lambda v: v == 1,
                              "1 (the network predicts one value, Y)"),
    "learningRates":         (_parse_float, "list", lambda v: v >= 0, "numbers >= 0"),
    "sigmas":                (_parse_float, "list", lambda v: v > 0, "numbers > 0"),
    "maxIterations":         (_parse_int, "one", lambda v: v >= 1, "an integer >= 1"),
    "randomSeed":            (_parse_int, "one", lambda v: v >= 0, "an integer >= 0"),
    "centresFile":           (_parse_str, "one", lambda v: len(v) > 0, "a file name"),
    "trainFile":             (_parse_str, "one", lambda v: len(v) > 0, "a file name"),
    "testFile":              (_parse_str, "one", lambda v: len(v) > 0, "a file name"),
    # our own additions (explained in the report):
    "centresInit":           (_parse_centre_method, "one", lambda v: True,
                              "kmeans or samples"),
    "kmeansIterations":      (_parse_int, "one", lambda v: v >= 1, "an integer >= 1"),
}

# Settings that are not in the assignment's example file. They may be left
# out, and then take these values. The committed parameters.txt lists them.
OPTIONAL = {
    "centresInit": "kmeans",       # initial centres: our k-means or random samples
    "kmeansIterations": 100,       # max assignment/update rounds of k-means
}


def _parse_line_values(name, texts, where):
    """Parse the value(s) of one line according to SCHEMA[name]."""
    parser, count, is_valid, expected = SCHEMA[name]
    if count == "one" and len(texts) != 1:
        raise ParameterError(f"{where}: {name} takes exactly one value, got {len(texts)}")
    values = []
    for text in texts:
        try:
            value = parser(text)
        except ValueError:
            raise ParameterError(f"{where}: {name} must be {expected}, got {text!r}") from None
        if not is_valid(value):
            raise ParameterError(f"{where}: {name} must be {expected}, got {text!r}")
        values.append(value)
    return values[0] if count == "one" else values


def parse_parameters(lines, source="parameters.txt"):
    """Parse the lines of a parameters file into a dict of typed values.

    `source` is only used in error messages. Raises ParameterError on an
    unknown, duplicated, missing or invalid setting, naming the line.
    """
    params = {}
    for number, raw in enumerate(lines, start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        where = f"{source}, line {number}"
        parts = line.split()
        if len(parts) < 2:
            raise ParameterError(f"{where}: expected 'name value', got {line!r}")
        name, texts = parts[0], parts[1:]
        if name not in SCHEMA:
            known = ", ".join(SCHEMA)
            raise ParameterError(f"{where}: unknown parameter {name!r} (known: {known})")
        if name in params:
            raise ParameterError(f"{where}: {name} is given more than once")
        params[name] = _parse_line_values(name, texts, where)

    for name, default in OPTIONAL.items():
        params.setdefault(name, default)
    missing = [name for name in SCHEMA if name not in params]
    if missing:
        raise ParameterError(f"{source}: missing parameter(s): {', '.join(missing)}")

    # learningRates: exactly 3 values, in the order weights, centres, sigmas.
    # The centre and sigma rates may be 0 (keeps them fixed); the weight rate
    # must be > 0, otherwise the network never learns.
    rates = params["learningRates"]
    if len(rates) != 3:
        raise ParameterError(
            f"{source}: learningRates needs 3 values (weights centres sigmas), "
            f"got {len(rates)}")
    if rates[0] <= 0:
        raise ParameterError(
            f"{source}: the weights learning rate (first learningRates value) "
            f"must be > 0, got {rates[0]}")
    params["etaWeights"], params["etaCentres"], params["etaSigmas"] = rates

    # sigmas: one shared value, or one value per hidden neuron.
    k = params["numHiddenLayerNeurons"]
    sigmas = params["sigmas"]
    if len(sigmas) == 1:
        params["sigmas"] = sigmas * k
    elif len(sigmas) != k:
        raise ParameterError(
            f"{source}: sigmas needs 1 value or {k} values (one per hidden neuron, "
            f"numHiddenLayerNeurons = {k}), got {len(sigmas)}")
    return params


def read_parameters(path):
    """Read and validate the parameters file at `path`; return a dict.

    Example: read_parameters("parameters.txt")["numHiddenLayerNeurons"] -> 8
    """
    if not os.path.isfile(path):
        raise ParameterError(f"{path}: file not found")
    with open(path, encoding="utf-8") as f:
        return parse_parameters(f, source=os.path.basename(path))

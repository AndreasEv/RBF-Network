"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Unit tests for parameters.py (run from code/: python3 -m pytest tests -q).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parameters import ParameterError, parse_parameters, read_parameters  # noqa: E402

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXAMPLE = """\
numHiddenLayerNeurons 8
numInputNeurons 53
numOutputNeurons 1
learningRates 0.01 0.001 0.0005
sigmas 1.0
maxIterations 2000
randomSeed 42
centresFile centerVectors.txt
trainFile training.txt
testFile test.txt
"""


def parse(text):
    return parse_parameters(text.splitlines())


def replace(name, new_line):
    """EXAMPLE with the line for `name` replaced (or removed if new_line is None)."""
    lines = []
    for line in EXAMPLE.splitlines():
        if line.split()[0] == name:
            if new_line is not None:
                lines.append(new_line)
        else:
            lines.append(line)
    return "\n".join(lines)


# ---- valid files ---------------------------------------------------------

def test_example_file_values_and_types():
    p = parse(EXAMPLE)
    assert p["numHiddenLayerNeurons"] == 8
    assert p["numInputNeurons"] == 53
    assert p["numOutputNeurons"] == 1
    assert p["learningRates"] == [0.01, 0.001, 0.0005]
    assert (p["etaWeights"], p["etaCentres"], p["etaSigmas"]) == (0.01, 0.001, 0.0005)
    assert p["maxIterations"] == 2000 and isinstance(p["maxIterations"], int)
    assert p["randomSeed"] == 42
    assert p["centresFile"] == "centerVectors.txt"
    assert p["trainFile"] == "training.txt"
    assert p["testFile"] == "test.txt"


def test_committed_parameters_file_loads():
    p = read_parameters(os.path.join(CODE_DIR, "parameters.txt"))
    assert len(p["sigmas"]) == p["numHiddenLayerNeurons"]
    assert len(p["learningRates"]) == 3


def test_single_sigma_is_repeated_for_every_neuron():
    p = parse(EXAMPLE)
    assert p["sigmas"] == [1.0] * 8


def test_one_sigma_per_neuron():
    text = replace("numHiddenLayerNeurons", "numHiddenLayerNeurons 3")
    text = text.replace("sigmas 1.0", "sigmas 0.5 1 2.5")
    assert parse(text)["sigmas"] == [0.5, 1.0, 2.5]


def test_comments_blank_lines_and_order_do_not_matter():
    lines = EXAMPLE.splitlines()
    text = "# my settings\n\n" + "\n".join(reversed(lines)) + "\n  \n"
    text = text.replace("randomSeed 42", "randomSeed 42   # seed")
    assert parse(text) == parse(EXAMPLE)


def test_zero_centre_and_sigma_rates_are_allowed():
    p = parse(replace("learningRates", "learningRates 0.01 0 0"))
    assert (p["etaCentres"], p["etaSigmas"]) == (0.0, 0.0)


def test_scientific_notation():
    p = parse(replace("learningRates", "learningRates 1e-2 5E-4 0"))
    assert p["learningRates"] == [0.01, 0.0005, 0.0]


def test_read_parameters_from_file(tmp_path):
    path = tmp_path / "parameters.txt"
    path.write_text(EXAMPLE, encoding="utf-8")
    assert read_parameters(str(path)) == parse(EXAMPLE)


# ---- errors --------------------------------------------------------------

@pytest.mark.parametrize("line", [
    "learningRates 0.01 0.001",
    "learningRates 0.01",
    "learningRates 0.01 0.001 0.0005 0.1",
])
def test_wrong_learning_rates_count(line):
    with pytest.raises(ParameterError, match="learningRates needs 3 values"):
        parse(replace("learningRates", line))


def test_zero_weight_learning_rate_rejected():
    with pytest.raises(ParameterError, match="weights learning rate"):
        parse(replace("learningRates", "learningRates 0 0.001 0.001"))


def test_negative_learning_rate_rejected():
    with pytest.raises(ParameterError, match="learningRates must be numbers >= 0"):
        parse(replace("learningRates", "learningRates 0.01 -0.001 0.001"))


def test_wrong_sigmas_count():
    with pytest.raises(ParameterError, match="sigmas needs 1 value or 8 values"):
        parse(replace("sigmas", "sigmas 1 2 3"))


@pytest.mark.parametrize("value", ["0", "-1", "abc", "nan", "inf"])
def test_bad_sigma_value(value):
    with pytest.raises(ParameterError, match="sigmas must be numbers > 0"):
        parse(replace("sigmas", f"sigmas {value}"))


@pytest.mark.parametrize("name", [
    "numHiddenLayerNeurons", "numInputNeurons", "numOutputNeurons", "learningRates",
    "sigmas", "maxIterations", "randomSeed", "centresFile", "trainFile", "testFile",
])
def test_missing_key(name):
    with pytest.raises(ParameterError, match=f"missing parameter.*{name}"):
        parse(replace(name, None))


@pytest.mark.parametrize("line, message", [
    ("numHiddenLayerNeurons 0", "numHiddenLayerNeurons must be an integer >= 1"),
    ("numHiddenLayerNeurons 8.5", "numHiddenLayerNeurons must be an integer >= 1"),
    ("numHiddenLayerNeurons eight", "numHiddenLayerNeurons must be an integer >= 1"),
    ("numInputNeurons -53", "numInputNeurons must be an integer >= 1"),
    ("numOutputNeurons 2", "numOutputNeurons must be 1"),
    ("maxIterations 0", "maxIterations must be an integer >= 1"),
    ("maxIterations 1e3", "maxIterations must be an integer >= 1"),
    ("randomSeed -1", "randomSeed must be an integer >= 0"),
    ("learningRates 0.01 x 0.001", "learningRates must be numbers >= 0"),
])
def test_bad_numbers(line, message):
    name = line.split()[0]
    with pytest.raises(ParameterError, match=message):
        parse(replace(name, line))


def test_error_names_the_line():
    with pytest.raises(ParameterError, match="line 7"):
        parse(replace("randomSeed", "randomSeed abc"))


def test_single_value_key_with_two_values():
    with pytest.raises(ParameterError, match="maxIterations takes exactly one value"):
        parse(replace("maxIterations", "maxIterations 10 20"))


def test_key_without_value():
    with pytest.raises(ParameterError, match="expected 'name value'"):
        parse(replace("trainFile", "trainFile"))


def test_unknown_key():
    with pytest.raises(ParameterError, match="unknown parameter 'learningRate'"):
        parse(EXAMPLE + "learningRate 0.1\n")


def test_duplicate_key():
    with pytest.raises(ParameterError, match="randomSeed is given more than once"):
        parse(EXAMPLE + "randomSeed 1\n")


def test_missing_file(tmp_path):
    with pytest.raises(ParameterError, match="file not found"):
        read_parameters(str(tmp_path / "nope.txt"))

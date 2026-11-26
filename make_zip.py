"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Builds the submission archive EPL442_ex4_<AM>.zip with the layout the
assignment asks for:

    code/          source, tests, parameters.txt, selwood.txt, training.txt,
                   test.txt, centerVectors.txt, the output files of the
                   baseline run (results.txt, weights.txt, predictions.txt),
                   the plots and the experiment summary tables
    README.txt
    report.pdf

Run it from the repository root, after `cd code && python3 main.py` (and
optionally `python3 plots.py` and `python3 experiments.py`) have produced the
output files:

    python3 make_zip.py 1134609 --name "Name Surname"

The author placeholders at the top of every file are filled in with --name
and the AM in the copies inside the zip only; the files in the repository are
not changed. Caches and the per-run experiment folders are left out. The
report may have at most MAX_REPORT_PAGES pages; a longer one gives a warning.
"""

import argparse
import os
import re
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))

NAME_PLACEHOLDER = "<Name Surname>"
AM_PLACEHOLDER = "<AM>"
MAX_REPORT_PAGES = 5

# Files that must be in code/ (the program, its input and the baseline output).
REQUIRED_CODE_FILES = [
    "main.py", "parameters.py", "data.py", "normalise.py", "kmeans.py",
    "prepare_data.py", "rbf.py", "training.py", "metrics.py", "results.py",
    "plots.py", "experiments.py", "features.py", "reference.py", "cv.py",
    "parameters.txt", "selwood.txt", "training.txt", "test.txt", "centerVectors.txt",
    "results.txt", "weights.txt", "predictions.txt",
]
# Optional extras that are included when they exist.
OPTIONAL_CODE_FILES = [
    "plots/error_curve.png", "plots/predictions.png", "plots/sin_fit.png",
    "experiments/summary.txt", "experiments/summary.csv",
    "cv_results.txt", "cv_results.csv", "cv_fine.txt", "cv_fine.csv",
]
# Text files whose author placeholders are filled in.
TEXT_SUFFIXES = (".py", ".txt", ".csv")


class PackageError(Exception):
    """Raised when a file the submission needs is missing."""


def collect_files(root=ROOT):
    """Return [(path on disk, path inside the zip)] for the submission.

    Raises PackageError if a required file is missing. report.pdf is not
    checked here (see build_zip).
    """
    code = os.path.join(root, "code")
    missing = [f for f in REQUIRED_CODE_FILES if not os.path.isfile(os.path.join(code, f))]
    if not os.path.isfile(os.path.join(root, "README.txt")):
        missing.append("../README.txt")
    if missing:
        raise PackageError("missing in code/: " + ", ".join(missing) +
                           " (run `python3 prepare_data.py` and `python3 main.py` in code/ "
                           "first for the data and output files)")

    files = [(os.path.join(code, f), "code/" + f) for f in REQUIRED_CODE_FILES]
    files += [(os.path.join(code, f), "code/" + f) for f in OPTIONAL_CODE_FILES
              if os.path.isfile(os.path.join(code, f))]
    tests = os.path.join(code, "tests")
    if os.path.isdir(tests):
        for name in sorted(os.listdir(tests)):
            # test_make_zip.py tests this script, which is not in the zip.
            if name.endswith(".py") and name != "test_make_zip.py":
                files.append((os.path.join(tests, name), "code/tests/" + name))
    files.append((os.path.join(root, "README.txt"), "README.txt"))
    return files


def fill_placeholders(text, name, am):
    """Replace the author placeholders in `text` (a value of None leaves it)."""
    if name:
        text = text.replace(NAME_PLACEHOLDER, name)
    if am:
        text = text.replace(AM_PLACEHOLDER, am)
    return text


def count_pdf_pages(path):
    """Return the number of pages of a PDF file (no PDF library needed).

    Uses the largest /Count of a /Pages node (the root page tree holds the
    total); falls back to counting /Type /Page objects.
    """
    with open(path, "rb") as f:
        data = f.read()
    counts = [int(c) for c in re.findall(rb"/Type\s*/Pages\b[^>]*?/Count\s+(\d+)", data)]
    counts += [int(c) for c in re.findall(rb"/Count\s+(\d+)[^>]*?/Type\s*/Pages\b", data)]
    if counts:
        return max(counts)
    return len(re.findall(rb"/Type\s*/Page\b(?!s)", data))


def build_zip(am, name=None, report=None, root=ROOT, out_dir=None):
    """Write EPL442_ex4_<am>.zip and return (zip path, list of warnings)."""
    files = collect_files(root)
    warnings = []
    report = report or os.path.join(root, "report.pdf")
    if os.path.isfile(report):
        files.append((report, "report.pdf"))
        pages = count_pdf_pages(report)
        if pages > MAX_REPORT_PAGES:
            warnings.append(f"report.pdf has {pages} pages (at most {MAX_REPORT_PAGES} allowed)")
    else:
        warnings.append(f"report.pdf not found ({report}): the zip has no report")

    path = os.path.join(out_dir or root, f"EPL442_ex4_{am}.zip")
    left = set()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for disk, arc in files:
            if disk.endswith(TEXT_SUFFIXES):
                with open(disk, encoding="utf-8") as f:
                    text = fill_placeholders(f.read(), name, am)
                if NAME_PLACEHOLDER in text or f"Student ID: {AM_PLACEHOLDER}" in text:
                    left.add(arc)
                zf.writestr(arc, text)
            else:
                zf.write(disk, arc)
    if left:
        warnings.append("author placeholders not filled in (use --name): " +
                        ", ".join(sorted(left)))
    return path, warnings


def main(argv=None):
    """Command line entry point; return the exit code."""
    parser = argparse.ArgumentParser(description="Build EPL442_ex4_<AM>.zip for Moodle.")
    parser.add_argument("am", help="student ID (AM), used in the zip name and the file headers")
    parser.add_argument("--name", help='"Name Surname" for the file headers')
    parser.add_argument("--report", help="path of the report PDF (default: ./report.pdf)")
    args = parser.parse_args(argv)
    try:
        path, warnings = build_zip(args.am, args.name, args.report)
    except PackageError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    print(f"Wrote {path} ({len(names)} files)")
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

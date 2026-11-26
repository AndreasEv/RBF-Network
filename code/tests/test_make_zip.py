"""
EPL442 Assignment 4: RBF network for regression on the Selwood data
Author: <Name Surname>, Student ID: <AM>

Tests for make_zip.py (in the repository root): the archive has the name and
layout the assignment asks for (code/, README.txt, report.pdf), the author
placeholders are filled in only inside the zip, caches and per-run experiment
folders are left out, a missing file is reported, and the committed
report.pdf has at most 5 pages. Run from code/:
    python3 -m pytest tests -q
"""

import importlib.util
import os
import zipfile

import pytest

REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_spec = importlib.util.spec_from_file_location("make_zip", os.path.join(REPO_DIR, "make_zip.py"))
make_zip = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_zip)

HEADER = '"""\nAuthor: <Name Surname>, Student ID: <AM>\n"""\n'


def fake_pdf(pages):
    """A minimal PDF with the given number of pages (enough for the counter)."""
    kids = " ".join(f"{i + 3} 0 R" for i in range(pages))
    body = [b"%PDF-1.4\n", b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
            f"2 0 obj << /Type /Pages /Kids [{kids}] /Count {pages} >> endobj\n".encode()]
    body += [f"{i + 3} 0 obj << /Type /Page /Parent 2 0 R >> endobj\n".encode()
             for i in range(pages)]
    return b"".join(body) + b"%%EOF\n"


def make_fake_repo(root, with_report=True, pages=3):
    """Create every file make_zip needs, plus some it must leave out."""
    code = os.path.join(root, "code")
    for sub in ("tests", "plots", "experiments/K_5", "__pycache__", "tests/__pycache__"):
        os.makedirs(os.path.join(code, sub), exist_ok=True)
    for name in make_zip.REQUIRED_CODE_FILES:
        with open(os.path.join(code, name), "w", encoding="utf-8") as f:
            f.write(HEADER if name.endswith(".py") else "data\n")
    files = {
        "tests/__init__.py": "", "tests/test_rbf.py": HEADER, "tests/test_make_zip.py": HEADER,
        "plots/error_curve.png": "png", "experiments/summary.txt": "table\n",
        "experiments/K_5/results.txt": "1 0.3 0.3\n",
        "__pycache__/rbf.cpython-313.pyc": "x", "tests/__pycache__/t.pyc": "x",
        "baseline_run.log": "log\n",
    }
    for name, text in files.items():
        with open(os.path.join(code, name), "w", encoding="utf-8") as f:
            f.write(text)
    with open(os.path.join(root, "README.txt"), "w", encoding="utf-8") as f:
        f.write("readme\n" + HEADER)
    if with_report:
        with open(os.path.join(root, "report.pdf"), "wb") as f:
            f.write(fake_pdf(pages))


def build(root, **kwargs):
    path, warnings = make_zip.build_zip("123456", root=root, **kwargs)
    with zipfile.ZipFile(path) as zf:
        return path, warnings, zf.namelist(), {n: zf.read(n) for n in zf.namelist()}


def test_layout_and_name(tmp_path):
    make_fake_repo(str(tmp_path))
    path, warnings, names, _ = build(str(tmp_path), name="Name Surname")
    assert os.path.basename(path) == "EPL442_ex4_123456.zip"
    assert warnings == []
    assert "README.txt" in names and "report.pdf" in names
    for f in make_zip.REQUIRED_CODE_FILES:
        assert "code/" + f in names
    assert "code/tests/test_rbf.py" in names
    assert "code/tests/test_make_zip.py" not in names
    assert "code/plots/error_curve.png" in names
    assert "code/experiments/summary.txt" in names
    # Every entry is at the top level or under code/.
    assert all(n in ("README.txt", "report.pdf") or n.startswith("code/") for n in names)


def test_leaves_out_caches_logs_and_run_folders(tmp_path):
    make_fake_repo(str(tmp_path))
    _, _, names, _ = build(str(tmp_path), name="N S")
    assert not [n for n in names if "__pycache__" in n or n.endswith(".pyc")]
    assert "code/experiments/K_5/results.txt" not in names
    assert "code/baseline_run.log" not in names


def test_placeholders_filled_in_zip_only(tmp_path):
    make_fake_repo(str(tmp_path))
    _, _, _, content = build(str(tmp_path), name="Andreas Test")
    assert b"Author: Andreas Test, Student ID: 123456" in content["code/main.py"]
    assert b"Author: Andreas Test, Student ID: 123456" in content["README.txt"]
    with open(os.path.join(str(tmp_path), "code", "main.py"), encoding="utf-8") as f:
        assert "<Name Surname>" in f.read()


def test_warns_without_name(tmp_path):
    make_fake_repo(str(tmp_path))
    _, warnings, _, content = build(str(tmp_path))
    assert b"Author: <Name Surname>, Student ID: 123456" in content["code/main.py"]
    assert any("placeholders" in w for w in warnings)


def test_warns_without_report(tmp_path):
    make_fake_repo(str(tmp_path), with_report=False)
    _, warnings, names, _ = build(str(tmp_path), name="N S")
    assert "report.pdf" not in names
    assert any("report.pdf" in w for w in warnings)


def test_warns_when_report_too_long(tmp_path):
    make_fake_repo(str(tmp_path), pages=6)
    _, warnings, _, _ = build(str(tmp_path), name="N S")
    assert any("6 pages" in w for w in warnings)


@pytest.mark.parametrize("pages", [1, 5, 7])
def test_count_pdf_pages(tmp_path, pages):
    pdf = tmp_path / "r.pdf"
    pdf.write_bytes(fake_pdf(pages))
    assert make_zip.count_pdf_pages(str(pdf)) == pages


def test_missing_output_is_an_error(tmp_path):
    make_fake_repo(str(tmp_path))
    os.remove(os.path.join(str(tmp_path), "code", "weights.txt"))
    with pytest.raises(make_zip.PackageError, match="weights.txt"):
        make_zip.build_zip("1", root=str(tmp_path))


def test_main_writes_zip(tmp_path, monkeypatch, capsys):
    make_fake_repo(str(tmp_path))
    monkeypatch.setattr(make_zip, "ROOT", str(tmp_path))
    monkeypatch.setattr(make_zip.build_zip, "__defaults__", (None, None, str(tmp_path), None))
    assert make_zip.main(["123456", "--name", "N S"]) == 0
    assert "EPL442_ex4_123456.zip" in capsys.readouterr().out
    assert os.path.isfile(os.path.join(str(tmp_path), "EPL442_ex4_123456.zip"))


def test_real_repository_is_complete():
    # The repository itself has every required file (the baseline output
    # files are committed) and README.txt.
    arcs = [arc for _, arc in make_zip.collect_files(REPO_DIR)]
    assert "code/results.txt" in arcs and "README.txt" in arcs


def test_real_report_has_at_most_5_pages():
    report = os.path.join(REPO_DIR, "report.pdf")
    if not os.path.isfile(report):
        pytest.skip("report.pdf not built yet")
    assert 1 <= make_zip.count_pdf_pages(report) <= make_zip.MAX_REPORT_PAGES

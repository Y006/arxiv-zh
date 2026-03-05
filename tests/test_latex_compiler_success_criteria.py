"""Tests for strict LaTeX compiler success criteria."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

from arxiv_translate.compiler.latex_compiler import LaTeXCompiler


def _write_pdf(pdf_path: Path, *, with_eof: bool) -> None:
    payload = b"%PDF-1.5\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n"
    if with_eof:
        payload += b"%%EOF\n"
    pdf_path.write_bytes(payload)


def test_single_pass_fails_when_returncode_nonzero_even_if_pdf_exists(
    tmp_path: Path, monkeypatch
) -> None:
    compiler = LaTeXCompiler()
    source_file = tmp_path / "main.tex"
    source_file.write_text("test", encoding="utf-8")

    def fake_run(*args, **kwargs):
        _write_pdf(Path(kwargs["cwd"]) / "main.pdf", with_eof=True)
        return SimpleNamespace(returncode=1, stdout="out", stderr="err")

    monkeypatch.setattr(subprocess, "run", fake_run)

    success, _log, error = compiler._run_single_pass("xelatex", source_file, tmp_path)

    assert success is False
    assert error is not None
    assert "exited with code 1" in error


def test_single_pass_fails_when_pdf_missing_eof(tmp_path: Path, monkeypatch) -> None:
    compiler = LaTeXCompiler()
    source_file = tmp_path / "main.tex"
    source_file.write_text("test", encoding="utf-8")

    def fake_run(*args, **kwargs):
        _write_pdf(Path(kwargs["cwd"]) / "main.pdf", with_eof=False)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    success, _log, error = compiler._run_single_pass("xelatex", source_file, tmp_path)

    assert success is False
    assert error == "Generated PDF failed integrity check (%PDF/%%EOF)."


def test_single_pass_succeeds_when_returncode_zero_and_pdf_healthy(
    tmp_path: Path, monkeypatch
) -> None:
    compiler = LaTeXCompiler()
    source_file = tmp_path / "main.tex"
    source_file.write_text("test", encoding="utf-8")

    def fake_run(*args, **kwargs):
        _write_pdf(Path(kwargs["cwd"]) / "main.pdf", with_eof=True)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    success, _log, error = compiler._run_single_pass("xelatex", source_file, tmp_path)

    assert success is True
    assert error is None

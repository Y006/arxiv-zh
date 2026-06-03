from pathlib import Path


def test_compile_error_summary_extracts_key_errors_and_tail():
    from arxiv_translate.compiler.latex_compiler import build_compile_error_summary

    log = "\n".join(
        [
            "line 1",
            "! LaTeX Error: File `missing.sty' not found.",
            "l.10 \\usepackage{missing}",
            "! Undefined control sequence.",
            "l.12 \\badcommand",
            "! Missing $ inserted.",
            "l.13 x_y",
        ]
        + [f"tail {i}" for i in range(100)]
    )

    summary = build_compile_error_summary(log)

    assert "First LaTeX Error" in summary
    assert "missing.sty" in summary
    assert "First Undefined control sequence" in summary
    assert "First Missing $ inserted" in summary
    assert "First Missing file" in summary
    assert "Last 80 log lines" in summary
    assert "tail 20" in summary
    assert "tail 99" in summary
    assert "tail 19" not in summary


def test_write_compile_error_summary_file(tmp_path: Path):
    from arxiv_translate.compiler.latex_compiler import write_compile_error_summary

    log_path = tmp_path / "compile.log"
    summary_path = tmp_path / "compile_error_summary.md"
    log_path.write_text("! Undefined control sequence.\nl.1 \\bad\n", encoding="utf-8")

    write_compile_error_summary(log_path, summary_path)

    assert summary_path.exists()
    assert "Undefined control sequence" in summary_path.read_text(encoding="utf-8")

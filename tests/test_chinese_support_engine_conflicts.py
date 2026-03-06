from arxiv_translate.compiler.chinese_support import (
    _strip_engine_conflict_primitives,
    inject_chinese_support,
)


def test_strip_engine_conflict_primitives_removes_pdfoutput_in_preamble():
    source = (
        "\\pdfoutput=1\n"
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "text\n"
        "\\end{document}\n"
    )
    patched = _strip_engine_conflict_primitives(source)
    assert "\\pdfoutput=1" not in patched
    assert "\\documentclass{article}" in patched


def test_strip_engine_conflict_primitives_keeps_commented_command():
    source = (
        "% \\pdfoutput=1\n"
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "text\n"
        "\\end{document}\n"
    )
    patched = _strip_engine_conflict_primitives(source)
    assert "% \\pdfoutput=1" in patched


def test_strip_engine_conflict_primitives_does_not_touch_document_body():
    source = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\pdfoutput=1\n"
        "\\end{document}\n"
    )
    patched = _strip_engine_conflict_primitives(source)
    assert "\\pdfoutput=1" in patched


def test_strip_engine_conflict_primitives_removes_multiple_primitives():
    source = (
        "\\pdfoutput=1\n"
        "\\pdfminorversion=7\n"
        "\\pdfcompresslevel=9\n"
        "\\pdfobjcompresslevel=3\n"
        "\\documentclass{article}\n"
    )
    patched = _strip_engine_conflict_primitives(source)
    assert "\\pdfoutput=1" not in patched
    assert "\\pdfminorversion=7" not in patched
    assert "\\pdfcompresslevel=9" not in patched
    assert "\\pdfobjcompresslevel=3" not in patched
    assert "\\documentclass{article}" in patched


def test_inject_chinese_support_strips_conflicts_before_xecjk_guard():
    source = (
        "\\pdfoutput=1\n"
        "\\documentclass{article}\n"
        "\\usepackage{xeCJK}\n"
        "\\begin{document}\n"
        "text\n"
        "\\end{document}\n"
    )
    patched = inject_chinese_support(source)
    assert "\\pdfoutput=1" not in patched
    assert patched.count("\\usepackage{xeCJK}") == 1

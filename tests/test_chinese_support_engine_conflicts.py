from arxiv_translate.compiler.chinese_support import (
    _guard_control_word_cjk_boundaries,
    _strip_engine_conflict_primitives,
    _strip_unicode_engine_driver_options,
    _unescape_reference_command_keys,
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


def test_strip_engine_conflict_primitives_removes_pdfinfo_block():
    source = (
        "\\documentclass{article}\n"
        "\\pdfinfo{\n"
        "  /Title (Old title)\n"
        "  /Subject (旧主题)\n"
        "}\n"
        "\\begin{document}\n"
        "text\n"
        "\\end{document}\n"
    )
    patched = _strip_engine_conflict_primitives(source)
    assert "\\pdfinfo" not in patched
    assert "/Title" not in patched
    assert "\\begin{document}" in patched


def test_strip_engine_conflict_primitives_removes_pdfinfo_after_begin_document():
    source = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\title{Title}\n"
        "\\pdfinfo{\n"
        "  /Title (Old title)\n"
        "}\n"
        "text\n"
        "\\end{document}\n"
    )
    patched = _strip_engine_conflict_primitives(source)
    assert "\\pdfinfo" not in patched
    assert "/Title" not in patched
    assert "\\title{Title}" in patched
    assert "text" in patched


def test_strip_unicode_engine_driver_options_removes_pdftex_graphicx_option():
    source = (
        "\\documentclass{article}\n"
        "\\usepackage[pdftex]{graphicx}\n"
        "\\usepackage[pdftex,bookmarks=true]{hyperref}\n"
    )
    patched = _strip_unicode_engine_driver_options(source)
    assert "\\usepackage{graphicx}" in patched
    assert "\\usepackage[bookmarks=true]{hyperref}" in patched
    assert "pdftex" not in patched


def test_guard_control_word_cjk_boundaries_adds_empty_group():
    source = "\\ModelSymbol\\架构 与 \\Robots\\不同机器人，另见 \\ModelSymbol模型。"
    patched = _guard_control_word_cjk_boundaries(source)
    assert "\\ModelSymbol{}架构" in patched
    assert "\\Robots{}不同机器人" in patched
    assert "\\ModelSymbol{}模型" in patched


def test_guard_control_word_cjk_boundaries_handles_chinese_punctuation():
    source = "评估\\ModelSymbol\\。下一句"
    patched = _guard_control_word_cjk_boundaries(source)
    assert "\\ModelSymbol{}。下一句" in patched


def test_unescape_reference_command_keys_restores_escaped_underscores():
    source = (
        "Figure~\\ref{fig:iter\\_throughput}, "
        "\\label{sec:rl\\_setup}, and \\citep{smith\\_2024,doe2025}."
    )
    patched = _unescape_reference_command_keys(source)
    assert "\\ref{fig:iter_throughput}" in patched
    assert "\\label{sec:rl_setup}" in patched
    assert "\\citep{smith_2024,doe2025}" in patched


def test_inject_chinese_support_strips_conflicts_before_xecjk_guard():
    source = (
        "\\pdfoutput=1\n"
        "\\documentclass{article}\n"
        "\\usepackage{xeCJK}\n"
        "\\usepackage[pdftex]{graphicx}\n"
        "\\begin{document}\n"
        "\\ModelSymbol\\架构\n"
        "\\end{document}\n"
    )
    patched = inject_chinese_support(source)
    assert "\\pdfoutput=1" not in patched
    assert "\\usepackage{graphicx}" in patched
    assert "\\ModelSymbol{}架构" in patched
    assert patched.count("\\usepackage{xeCJK}") == 1

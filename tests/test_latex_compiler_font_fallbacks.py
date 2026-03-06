from arxiv_translate.compiler.latex_compiler import LaTeXCompiler


def test_extract_missing_font_name_handles_wrapped_fontspec_word():
    compiler = LaTeXCompiler()
    log = (
        'Package fontsp\n'
        'ec Error: The font "Source Han Sans SC" cannot be found.\n'
    )
    assert compiler._extract_missing_font_name(log) == "Source Han Sans SC"


def test_apply_missing_font_fallback_replaces_cjk_aux_fonts_with_main():
    compiler = LaTeXCompiler()
    source = r"""
\documentclass{article}
\usepackage{xeCJK}
\setCJKmainfont{Songti SC}
\setCJKsansfont{Source Han Sans SC}
\setCJKmonofont{Source Han Mono SC}
\begin{document}
text
\end{document}
"""
    patched, reason = compiler._apply_missing_font_fallback(
        source,
        "Source Han Sans SC",
    )
    assert reason == "fallback_cjk_aux_fonts_to_main"
    assert r"\setCJKsansfont{Songti SC}" in patched
    assert r"\setCJKmonofont{Songti SC}" in patched


def test_apply_missing_font_fallback_removes_unused_fontawesome_package():
    compiler = LaTeXCompiler()
    source = r"""
\documentclass{article}
\usepackage{fontawesome}
\begin{document}
text
\end{document}
"""
    patched, reason = compiler._apply_missing_font_fallback(source, "FontAwesome")
    assert reason == "remove_unused_fontawesome"
    assert r"\usepackage{fontawesome}" not in patched


def test_apply_missing_font_fallback_keeps_fontawesome_if_commands_present():
    compiler = LaTeXCompiler()
    source = r"""
\documentclass{article}
\usepackage{fontawesome}
\begin{document}
\faGithub
\end{document}
"""
    patched, reason = compiler._apply_missing_font_fallback(source, "FontAwesome")
    assert reason == "no_fallback"
    assert patched == source

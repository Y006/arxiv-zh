from pathlib import Path
from types import SimpleNamespace

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


def test_apply_missing_font_fallback_removes_explicit_cjk_fonts_when_main_missing():
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
    patched, reason = compiler._apply_missing_font_fallback(source, "Songti SC")
    assert reason == "fallback_remove_explicit_cjk_fonts"
    assert r"\setCJKmainfont" not in patched
    assert r"\setCJKsansfont" not in patched
    assert r"\setCJKmonofont" not in patched


def test_compile_prefers_informative_error_instead_of_unknown(monkeypatch, tmp_path: Path):
    compiler = LaTeXCompiler()
    compiler.engines = ["xelatex", "lualatex"]
    call_count = {"n": 0}

    def fake_run_engine(engine, source_file, cwd, latex_source):
        call_count["n"] += 1
        if engine == "xelatex":
            return (
                False,
                "log",
                'Compilation command exited with code 1. Package fontspec Error: The font "Songti SC" cannot be found.',
            )
        return (
            False,
            "log",
            "Compilation command exited with code 1. Unknown error (check full logs)",
        )

    monkeypatch.setattr(
        "arxiv_translate.compiler.latex_compiler.shutil.which", lambda _engine: "/usr/bin/true"
    )
    monkeypatch.setattr(compiler, "_run_engine", fake_run_engine)

    result = compiler.compile(
        latex_source=r"\documentclass{article}\begin{document}x\end{document}",
        output_path=tmp_path / "out.pdf",
    )

    assert result.success is False
    assert call_count["n"] == 2
    assert result.error_message is not None
    assert "fontspec" in result.error_message.lower()
    assert "[xelatex]" in result.error_message


def test_run_single_pass_sets_osfontdir_when_fonts_dir_exists(monkeypatch, tmp_path: Path):
    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    source_file = tmp_path / "main.tex"
    source_file.write_text("test", encoding="utf-8")
    compiler = LaTeXCompiler(fonts_dir=fonts_dir)
    captured = {}

    def fake_run(*args, **kwargs):
        captured["env"] = kwargs.get("env", {})
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setenv("OSFONTDIR", "/existing/fonts")
    monkeypatch.setattr("subprocess.run", fake_run)

    compiler._run_single_pass("xelatex", source_file, tmp_path)

    assert "OSFONTDIR" in captured["env"]
    assert str(fonts_dir) in captured["env"]["OSFONTDIR"]
    assert "/existing/fonts" in captured["env"]["OSFONTDIR"]


def test_apply_missing_file_fallback_sets_bxcoloremoji_names_false(tmp_path: Path):
    compiler = LaTeXCompiler()
    source = r"""
\documentclass{article}
\usepackage{bxcoloremoji}
\begin{document}
text
\end{document}
"""
    workspace_file = tmp_path / "mystyle.cls"
    workspace_file.write_text(r"\RequirePackage{bxcoloremoji}", encoding="utf-8")

    patched, reason, changed = compiler._apply_missing_file_fallback(
        source,
        "bxcoloremoji-names.def",
        workspace_dir=tmp_path,
    )

    assert changed is True
    assert reason == "fallback_bxcoloremoji_names_false"
    assert r"\usepackage[names=false]{bxcoloremoji}" in patched
    assert (
        r"\RequirePackage[names=false]{bxcoloremoji}"
        in workspace_file.read_text(encoding="utf-8")
    )


def test_apply_microtype_tracking_fallback_patches_source_and_workspace(tmp_path: Path):
    compiler = LaTeXCompiler()
    source = r"""
\documentclass{article}
\usepackage[tracking=smallcaps]{microtype}
\begin{document}
text
\end{document}
"""
    workspace_file = tmp_path / "mystyle.cls"
    workspace_file.write_text(
        r"\AtEndOfClass{\RequirePackage[tracking=smallcaps]{microtype}}",
        encoding="utf-8",
    )

    patched, reason, changed = compiler._apply_microtype_tracking_fallback(
        source,
        workspace_dir=tmp_path,
    )

    assert changed is True
    assert reason == "fallback_disable_microtype_tracking"
    assert r"\usepackage{microtype}" in patched
    assert "tracking=smallcaps" not in patched

    workspace_text = workspace_file.read_text(encoding="utf-8")
    assert r"\RequirePackage{microtype}" in workspace_text
    assert "tracking=smallcaps" not in workspace_text


def test_has_microtype_tracking_error_handles_wrapped_pdftex_word():
    compiler = LaTeXCompiler()
    log = (
        "./main.tex:377: Package microtype Error: The tracking feature only works "
        "with p\n"
        "(microtype)                dftex 1.40\n"
    )
    assert compiler._has_microtype_tracking_error(log) is True


def test_extract_error_detects_package_error_without_bang():
    compiler = LaTeXCompiler()
    log = (
        "Random line\n"
        "./main.tex:377: Package microtype Error: The tracking feature only works "
        "with pdftex 1.40\n"
        "(microtype)                or newer. Switching it off.\n"
        "See the microtype package documentation for explanation.\n"
    )
    extracted = compiler._extract_error(log)
    assert "Package microtype Error" in extracted
    assert "Switching it off." in extracted


def test_compile_reports_latest_round_error_after_fallback(monkeypatch, tmp_path: Path):
    compiler = LaTeXCompiler()
    compiler.engines = ["xelatex"]
    call_count = {"n": 0}
    microtype_log = (
        "./main.tex:377: Package microtype Error: The tracking feature only works "
        "with p\n"
        "(microtype)                dftex 1.40\n"
        "(microtype)                or newer. Switching it off.\n"
    )

    def fake_run_engine(engine, source_file, cwd, latex_source):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return (
                False,
                "! LaTeX Error: File `bxcoloremoji-names.def' not found.\n",
                "Compilation command exited with code 1. ! LaTeX Error: File `bxcoloremoji-names.def' not found.",
            )
        return (
            False,
            microtype_log,
            "Compilation command exited with code 1. "
            "Package microtype Error: The tracking feature only works with p dftex 1.40",
        )

    monkeypatch.setattr(
        "arxiv_translate.compiler.latex_compiler.shutil.which", lambda _engine: "/usr/bin/true"
    )
    monkeypatch.setattr(compiler, "_run_engine", fake_run_engine)

    result = compiler.compile(
        latex_source=(
            r"\documentclass{article}"
            r"\usepackage{bxcoloremoji}"
            r"\begin{document}x\end{document}"
        ),
        output_path=tmp_path / "out.pdf",
    )

    assert result.success is False
    assert call_count["n"] == 2
    assert result.error_message is not None
    assert "microtype error" in result.error_message.lower()
    assert "bxcoloremoji-names.def" not in result.error_message

import json
import os
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
        "arxiv_translate.compiler.latex_compiler.shutil.which",
        lambda _engine, path=None: "/usr/bin/true",
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


def test_compile_file_sets_osfontdir_when_fonts_dir_exists(monkeypatch, tmp_path: Path):
    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir(parents=True, exist_ok=True)
    tex_file = tmp_path / "main_zh.tex"
    tex_file.write_text(
        r"\documentclass{article}\begin{document}x\end{document}",
        encoding="utf-8",
    )
    compiler = LaTeXCompiler(fonts_dir=fonts_dir)
    captured = {}

    def fake_run(*args, **kwargs):
        captured["env"] = kwargs.get("env", {})
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setenv("OSFONTDIR", "/existing/fonts")
    monkeypatch.setattr(
        "arxiv_translate.compiler.latex_compiler.shutil.which",
        lambda name, path=None: "/usr/bin/latexmk" if name == "latexmk" else None,
    )
    monkeypatch.setattr("subprocess.run", fake_run)

    compiler.compile_file(
        tex_file=tex_file,
        output_path=tmp_path / "pdf" / "main_zh.pdf",
        logs_dir=tmp_path / "logs",
    )

    assert "OSFONTDIR" in captured["env"]
    assert str(fonts_dir) in captured["env"]["OSFONTDIR"]
    assert "/existing/fonts" in captured["env"]["OSFONTDIR"]


def test_tinytex_paths_are_prepended_to_path(monkeypatch, tmp_path: Path):
    tinytex_bin = tmp_path / "TinyTeX" / "bin" / "universal-darwin"
    tinytex_bin.mkdir(parents=True)
    monkeypatch.setenv("PATH", "/usr/bin")

    compiler = LaTeXCompiler(tinytex_paths=[str(tinytex_bin)])
    env = compiler._build_env()

    assert env["PATH"].split(os.pathsep)[0] == str(tinytex_bin.resolve())


def test_compile_file_latexmk_does_not_run_python_tlmgr_auto_install(
    monkeypatch,
    tmp_path: Path,
):
    build_dir = tmp_path / "build"
    tex_file = tmp_path / "main_zh.tex"
    tex_file.write_text(
        "\\documentclass{article}\n"
        "\\usepackage{enumitem}\n"
        "\\begin{document}x\\end{document}\n",
        encoding="utf-8",
    )
    commands = []

    def fake_which(name, path=None):
        if name == "latexmk":
            return "/tinytex/bin/latexmk"
        if name == "tlmgr":
            return "/tinytex/bin/tlmgr"
        return None

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        if cmd[0] == "/tinytex/bin/tlmgr" and cmd[1] == "search":
            return SimpleNamespace(
                returncode=0,
                stdout="enumitem:\n\ttexmf-dist/tex/latex/enumitem/enumitem.sty\n",
                stderr="",
            )
        if cmd[0] == "/tinytex/bin/tlmgr" and cmd[1] == "install":
            return SimpleNamespace(returncode=0, stdout="installed", stderr="")
        if cmd[0] == "latexmk":
            return SimpleNamespace(
                returncode=1,
                stdout="! LaTeX Error: File `enumitem.sty' not found.\n",
                stderr="",
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(
        "arxiv_translate.compiler.latex_compiler.shutil.which",
        fake_which,
    )
    monkeypatch.setattr("subprocess.run", fake_run)

    result = LaTeXCompiler().compile_file(
        tex_file=tex_file,
        output_path=tmp_path / "pdf" / "main_zh.pdf",
        logs_dir=tmp_path / "logs",
        build_dir=build_dir,
        max_repair_rounds=0,
    )

    assert result.success is False
    assert not [cmd for cmd in commands if cmd[0] == "/tinytex/bin/tlmgr"]
    assert not any(
        repair.startswith("tlmgr_install:")
        for attempt in result.attempts
        for repair in attempt.repairs
    )


def test_compile_file_auto_uses_r_tinytex_when_available(monkeypatch, tmp_path: Path):
    build_dir = tmp_path / "build"
    tex_file = tmp_path / "main_zh.tex"
    tex_file.write_text(
        "\\documentclass{article}\\begin{document}x\\end{document}",
        encoding="utf-8",
    )
    commands = []

    def fake_which(name, path=None):
        _ = path
        if name == "Rscript":
            return "/usr/bin/Rscript"
        return None

    def fake_run(cmd, **kwargs):
        commands.append((cmd, kwargs))
        if cmd[0] == "/usr/bin/Rscript" and "requireNamespace" in cmd[3]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[0] == "/usr/bin/Rscript":
            assert kwargs["timeout"] == 4321
            candidate = Path(cmd[4])
            output_dir = Path(cmd[6])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{candidate.stem}.pdf").write_bytes(
                b"%PDF-1.5\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
            )
            return SimpleNamespace(returncode=0, stdout="compiled", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(
        "arxiv_translate.compiler.latex_compiler.shutil.which",
        fake_which,
    )
    monkeypatch.setattr("subprocess.run", fake_run)

    result = LaTeXCompiler(total_timeout=4321).compile_file(
        tex_file=tex_file,
        output_path=tmp_path / "pdf" / "main_zh.pdf",
        logs_dir=tmp_path / "logs",
        build_dir=build_dir,
    )

    assert result.success is True
    assert result.engine_used == "xelatex"
    assert result.pdf_path == tmp_path / "pdf" / "main_zh.pdf"
    assert any("tinytex::latexmk" in cmd[3] for cmd, _kwargs in commands)
    assert not [cmd for cmd, _kwargs in commands if cmd[0] == "latexmk"]


def test_compile_file_r_tinytex_uses_engine_log_for_error_and_attempt_metadata(
    monkeypatch,
    tmp_path: Path,
):
    build_dir = tmp_path / "build"
    tex_file = tmp_path / "main_zh.tex"
    tex_file.write_text(
        "\\documentclass{article}\\begin{document}x\\end{document}",
        encoding="utf-8",
    )

    def fake_which(name, path=None):
        _ = path
        if name == "Rscript":
            return "/usr/bin/Rscript"
        return None

    def fake_run(cmd, **kwargs):
        if cmd[0] == "/usr/bin/Rscript" and "requireNamespace" in cmd[3]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[0] == "/usr/bin/Rscript":
            candidate = Path(cmd[4])
            engine_log = Path(cmd[6]) / f"{candidate.stem}.log"
            engine_log.parent.mkdir(parents=True, exist_ok=True)
            engine_log.write_text(
                "! LaTeX Error: File `bbding.sty' not found.\n"
                "l.17 \\usepackage{bbding}\n",
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(
        "arxiv_translate.compiler.latex_compiler.shutil.which",
        fake_which,
    )
    monkeypatch.setattr("subprocess.run", fake_run)

    result = LaTeXCompiler().compile_file(
        tex_file=tex_file,
        output_path=tmp_path / "pdf" / "main_zh.pdf",
        logs_dir=tmp_path / "logs",
        build_dir=build_dir,
        max_repair_rounds=0,
    )

    attempts = json.loads(
        (tmp_path / "logs" / "compile_attempts.json").read_text(encoding="utf-8")
    )["attempts"]
    compile_log = (tmp_path / "logs" / "compile.log").read_text(encoding="utf-8")
    assert result.success is False
    assert result.error_message is not None
    assert "bbding.sty" in result.error_message
    assert "Unknown error" not in result.error_message
    assert "bbding.sty" in compile_log
    assert attempts[0]["missing_file"] == "bbding.sty"
    assert attempts[0]["engine_log_path"].endswith(".log")
    assert "bbding.sty" in attempts[0]["first_error"]


def test_compile_file_r_tinytex_applies_log_fallback_and_retries(
    monkeypatch,
    tmp_path: Path,
):
    build_dir = tmp_path / "build"
    tex_file = tmp_path / "main_zh.tex"
    tex_file.write_text(
        "\\documentclass{article}\n"
        "\\usepackage{axessibility}\n"
        "\\begin{document}x\\end{document}\n",
        encoding="utf-8",
    )
    compile_candidates = []

    def fake_which(name, path=None):
        _ = path
        if name == "Rscript":
            return "/usr/bin/Rscript"
        return None

    def fake_run(cmd, **kwargs):
        if cmd[0] == "/usr/bin/Rscript" and "requireNamespace" in cmd[3]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[0] == "/usr/bin/Rscript":
            candidate = Path(cmd[4])
            output_dir = Path(cmd[6])
            compile_candidates.append(candidate)
            output_dir.mkdir(parents=True, exist_ok=True)
            engine_log = output_dir / f"{candidate.stem}.log"
            if len(compile_candidates) == 1:
                assert "\\usepackage{axessibility}" in candidate.read_text(
                    encoding="utf-8"
                )
                engine_log.write_text(
                    "/tinytex/texmf-dist/tex/latex/axessibility/axessibility.sty:349: "
                    "Undefined control sequence.\n"
                    "l.349 \\pdfcompresslevel\n",
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=1, stdout="", stderr="")
            assert "\\usepackage{axessibility}" not in candidate.read_text(
                encoding="utf-8"
            )
            (output_dir / f"{candidate.stem}.pdf").write_bytes(
                b"%PDF-1.5\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
            )
            engine_log.write_text("ok", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="compiled", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(
        "arxiv_translate.compiler.latex_compiler.shutil.which",
        fake_which,
    )
    monkeypatch.setattr("subprocess.run", fake_run)

    result = LaTeXCompiler().compile_file(
        tex_file=tex_file,
        output_path=tmp_path / "pdf" / "main_zh.pdf",
        logs_dir=tmp_path / "logs",
        build_dir=build_dir,
        max_repair_rounds=1,
    )

    assert result.success is True
    assert len(compile_candidates) == 2
    assert any(
        "fallback_remove_axessibility" in attempt.repairs
        for attempt in result.attempts
    )


def test_compile_file_r_tinytex_recovers_pdf_from_nonzero_wrapper_exit(
    monkeypatch,
    tmp_path: Path,
):
    build_dir = tmp_path / "build"
    tex_file = tmp_path / "main_zh.tex"
    tex_file.write_text(
        "\\documentclass{article}\\begin{document}x\\end{document}",
        encoding="utf-8",
    )

    def fake_which(name, path=None):
        _ = path
        if name == "Rscript":
            return "/usr/bin/Rscript"
        return None

    def fake_run(cmd, **kwargs):
        if cmd[0] == "/usr/bin/Rscript" and "requireNamespace" in cmd[3]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[0] == "/usr/bin/Rscript":
            candidate = Path(cmd[4])
            output_dir = Path(cmd[6])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{candidate.stem}.pdf").write_bytes(
                b"%PDF-1.5\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
            )
            (output_dir / f"{candidate.stem}.log").write_text(
                "Output written on main.pdf.\n", encoding="utf-8"
            )
            return SimpleNamespace(
                returncode=1,
                stdout="Latexmk: Errors, so I did not complete making targets\n",
                stderr="",
            )
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(
        "arxiv_translate.compiler.latex_compiler.shutil.which",
        fake_which,
    )
    monkeypatch.setattr("subprocess.run", fake_run)

    result = LaTeXCompiler().compile_file(
        tex_file=tex_file,
        output_path=tmp_path / "pdf" / "main_zh.pdf",
        logs_dir=tmp_path / "logs",
        build_dir=build_dir,
    )

    attempts = json.loads(
        (tmp_path / "logs" / "compile_attempts.json").read_text(encoding="utf-8")
    )["attempts"]
    assert result.success is True
    assert result.pdf_path == tmp_path / "pdf" / "main_zh.pdf"
    assert result.warning_message is not None
    assert attempts[0]["category"] == "success_with_wrapper_warning"


def test_compile_file_r_tinytex_sets_texinputs(monkeypatch, tmp_path: Path):
    source_dir = tmp_path / "source"
    translated_dir = tmp_path / "translated"
    build_dir = tmp_path / "build"
    source_dir.mkdir()
    translated_dir.mkdir()
    tex_file = translated_dir / "main_zh.tex"
    tex_file.write_text(
        "\\documentclass{article}\\begin{document}x\\end{document}",
        encoding="utf-8",
    )
    captured_env = {}

    def fake_which(name, path=None):
        _ = path
        if name == "Rscript":
            return "/usr/bin/Rscript"
        return None

    def fake_run(cmd, **kwargs):
        if cmd[0] == "/usr/bin/Rscript" and "requireNamespace" in cmd[3]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[0] == "/usr/bin/Rscript":
            captured_env.update(kwargs["env"])
            candidate = Path(cmd[4])
            output_dir = Path(cmd[6])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{candidate.stem}.pdf").write_bytes(
                b"%PDF-1.5\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
            )
            return SimpleNamespace(returncode=0, stdout="compiled", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(
        "arxiv_translate.compiler.latex_compiler.shutil.which",
        fake_which,
    )
    monkeypatch.setattr("subprocess.run", fake_run)

    result = LaTeXCompiler().compile_file(
        tex_file=tex_file,
        output_path=tmp_path / "pdf" / "main_zh.pdf",
        logs_dir=tmp_path / "logs",
        build_dir=build_dir,
    )

    assert result.success is True
    texinputs = captured_env["TEXINPUTS"]
    assert str(build_dir) in texinputs
    assert str(translated_dir) in texinputs
    assert str(source_dir) in texinputs


def test_compile_file_r_tinytex_honors_disabled_package_install(
    monkeypatch,
    tmp_path: Path,
):
    build_dir = tmp_path / "build"
    tex_file = tmp_path / "main_zh.tex"
    tex_file.write_text(
        "\\documentclass{article}\\begin{document}x\\end{document}",
        encoding="utf-8",
    )
    commands = []

    def fake_which(name, path=None):
        _ = path
        if name == "Rscript":
            return "/usr/bin/Rscript"
        return None

    def fake_run(cmd, **kwargs):
        commands.append((cmd, kwargs))
        if cmd[0] == "/usr/bin/Rscript" and "requireNamespace" in cmd[3]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if cmd[0] == "/usr/bin/Rscript":
            candidate = Path(cmd[4])
            output_dir = Path(cmd[6])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{candidate.stem}.pdf").write_bytes(
                b"%PDF-1.5\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
            )
            return SimpleNamespace(returncode=0, stdout="compiled", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(
        "arxiv_translate.compiler.latex_compiler.shutil.which",
        fake_which,
    )
    monkeypatch.setattr("subprocess.run", fake_run)

    result = LaTeXCompiler(install_missing_packages=False).compile_file(
        tex_file=tex_file,
        output_path=tmp_path / "pdf" / "main_zh.pdf",
        logs_dir=tmp_path / "logs",
        build_dir=build_dir,
    )

    compile_commands = [
        cmd for cmd, _kwargs in commands if cmd[0] == "/usr/bin/Rscript" and cmd[4:]
    ]
    assert result.success is True
    assert compile_commands
    assert "install_packages = FALSE" in compile_commands[0][3]


def test_compile_file_auto_falls_back_to_latexmk_when_r_tinytex_missing(
    monkeypatch,
    tmp_path: Path,
):
    build_dir = tmp_path / "build"
    tex_file = tmp_path / "main_zh.tex"
    tex_file.write_text(
        "\\documentclass{article}\\begin{document}x\\end{document}",
        encoding="utf-8",
    )
    commands = []

    def fake_which(name, path=None):
        _ = path
        if name == "latexmk":
            return "/tinytex/bin/latexmk"
        return None

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        if cmd[0] == "latexmk":
            tex_arg = Path(cmd[-1])
            build_dir.mkdir(parents=True, exist_ok=True)
            (build_dir / f"{tex_arg.stem}.pdf").write_bytes(
                b"%PDF-1.5\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
            )
            return SimpleNamespace(returncode=0, stdout="ok", stderr="")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(
        "arxiv_translate.compiler.latex_compiler.shutil.which",
        fake_which,
    )
    monkeypatch.setattr("subprocess.run", fake_run)

    result = LaTeXCompiler().compile_file(
        tex_file=tex_file,
        output_path=tmp_path / "pdf" / "main_zh.pdf",
        logs_dir=tmp_path / "logs",
        build_dir=build_dir,
    )

    assert result.success is True
    assert commands and commands[0][0] == "latexmk"


def test_compile_file_forced_r_tinytex_reports_missing_wrapper(
    monkeypatch,
    tmp_path: Path,
):
    tex_file = tmp_path / "main_zh.tex"
    tex_file.write_text(
        "\\documentclass{article}\\begin{document}x\\end{document}",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "arxiv_translate.compiler.latex_compiler.shutil.which",
        lambda _name, path=None: None,
    )

    result = LaTeXCompiler(tinytex_driver="r_tinytex").compile_file(
        tex_file=tex_file,
        output_path=tmp_path / "pdf" / "main_zh.pdf",
        logs_dir=tmp_path / "logs",
        build_dir=tmp_path / "build",
    )

    assert result.success is False
    assert "R tinytex driver requested but unavailable" in result.error_message


def test_prepare_compile_inputs_sanitizes_unicode_engine_conflicts(tmp_path: Path):
    tex_file = tmp_path / "main_zh.tex"
    tex_file.write_text(
        "\\documentclass{article}\n"
        "\\usepackage[pdftex]{graphicx}\n"
        "\\pdfinfo{\n"
        "  /Title (Old)\n"
        "}\n"
        "\\begin{document}\n"
        "\\ModelSymbol\\架构 与 \\ModelSymbol模型\n"
        "\\end{document}\n",
        encoding="utf-8",
    )

    compiler = LaTeXCompiler()
    compiler._prepare_compile_inputs(tex_file, tmp_path / "build")

    patched = tex_file.read_text(encoding="utf-8")
    assert "\\pdfinfo" not in patched
    assert "\\usepackage{graphicx}" in patched
    assert "\\ModelSymbol{}架构" in patched
    assert "\\ModelSymbol{}模型" in patched


def test_prepare_compile_inputs_aliases_precompiled_bbl_to_tex_stem(tmp_path: Path):
    build_dir = tmp_path / "build"
    tex_file = tmp_path / "main_zh.tex"
    tex_file.write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\bibliography{references}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    (tmp_path / "main.bbl").write_text(
        "\\begin{thebibliography}{1}\\bibitem{x} X\\end{thebibliography}",
        encoding="utf-8",
    )

    compiler = LaTeXCompiler()
    compiler._prepare_compile_inputs(tex_file, build_dir)

    patched = tex_file.read_text(encoding="utf-8")
    assert "\\bibliography{references}" not in patched
    assert "\\input{main_zh.bbl}" in patched
    assert (tmp_path / "main_zh.bbl").read_text(encoding="utf-8").startswith(
        "\\begin{thebibliography}"
    )
    assert (build_dir / "main_zh.bbl").exists()


def test_compile_file_disables_bibtex_when_using_precompiled_bbl(
    monkeypatch,
    tmp_path: Path,
):
    build_dir = tmp_path / "build"
    tex_file = tmp_path / "main_zh.tex"
    tex_file.write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\citep{key}\n"
        "\\bibliography{references}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    (tmp_path / "main.bbl").write_text(
        "\\begin{thebibliography}{1}\\bibitem{key} Key\\end{thebibliography}",
        encoding="utf-8",
    )
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        build_dir.mkdir(parents=True, exist_ok=True)
        (build_dir / "main_zh.pdf").write_bytes(
            b"%PDF-1.5\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(
        "arxiv_translate.compiler.latex_compiler.shutil.which",
        lambda name, path=None: "/usr/bin/latexmk" if name == "latexmk" else None,
    )
    monkeypatch.setattr("subprocess.run", fake_run)

    result = LaTeXCompiler().compile_file(
        tex_file=tex_file,
        output_path=tmp_path / "pdf" / "main_zh.pdf",
        logs_dir=tmp_path / "logs",
        build_dir=build_dir,
    )

    assert result.success is True
    assert "-bibtex-" in captured["cmd"]


def test_compile_file_falls_back_from_xelatex_to_lualatex(monkeypatch, tmp_path: Path):
    build_dir = tmp_path / "build"
    tex_file = tmp_path / "main_zh.tex"
    original_source = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "中文\n"
        "\\end{document}\n"
    )
    tex_file.write_text(original_source, encoding="utf-8")
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        tex_arg = Path(cmd[-1])
        if "-xelatex" in cmd:
            return SimpleNamespace(
                returncode=1,
                stdout='Package fontspec Error: The font "Missing" cannot be found.',
                stderr="",
            )
        build_dir.mkdir(parents=True, exist_ok=True)
        (build_dir / f"{tex_arg.stem}.pdf").write_bytes(
            b"%PDF-1.5\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
        )
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "arxiv_translate.compiler.latex_compiler.shutil.which",
        lambda name, path=None: f"/usr/bin/{name}" if name == "latexmk" else None,
    )
    monkeypatch.setattr("subprocess.run", fake_run)

    result = LaTeXCompiler().compile_file(
        tex_file=tex_file,
        output_path=tmp_path / "pdf" / "main_zh.pdf",
        logs_dir=tmp_path / "logs",
        build_dir=build_dir,
        max_repair_rounds=0,
    )

    assert result.success is True
    assert result.engine_used == "lualatex"
    assert len(result.attempts) == 2
    assert result.attempts[0].engine == "xelatex"
    assert result.attempts[1].engine == "lualatex"
    assert any("-lualatex" in call for call in calls)
    assert (tmp_path / "main_zh.before_compile.tex").read_text(
        encoding="utf-8"
    ) == original_source
    assert "\\usepackage{luatexja-fontspec}" in tex_file.read_text(encoding="utf-8")
    assert result.diagnostic_path == tmp_path / "logs" / "compile_attempts.json"
    data = json.loads(result.diagnostic_path.read_text(encoding="utf-8"))
    assert [attempt["engine"] for attempt in data["attempts"]] == [
        "xelatex",
        "lualatex",
    ]


def test_compile_file_writes_attempt_diagnostics_when_all_engines_fail(
    monkeypatch,
    tmp_path: Path,
):
    build_dir = tmp_path / "build"
    tex_file = tmp_path / "main_zh.tex"
    original_source = (
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "中文\n"
        "\\end{document}\n"
    )
    tex_file.write_text(original_source, encoding="utf-8")

    def fake_run(cmd, **kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout=(
                "Package minted Error: You must invoke LaTeX with the "
                "-shell-escape flag.\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(
        "arxiv_translate.compiler.latex_compiler.shutil.which",
        lambda name, path=None: f"/usr/bin/{name}" if name == "latexmk" else None,
    )
    monkeypatch.setattr("subprocess.run", fake_run)

    result = LaTeXCompiler().compile_file(
        tex_file=tex_file,
        output_path=tmp_path / "pdf" / "main_zh.pdf",
        logs_dir=tmp_path / "logs",
        build_dir=build_dir,
        max_repair_rounds=0,
    )

    assert result.success is False
    assert tex_file.read_text(encoding="utf-8") == original_source
    assert result.diagnostic_path == tmp_path / "logs" / "compile_attempts.json"
    data = json.loads(result.diagnostic_path.read_text(encoding="utf-8"))
    assert len(data["attempts"]) == 2
    assert all(attempt["returncode"] == 1 for attempt in data["attempts"])
    assert data["attempts"][0]["category"] == "shell_escape"
    summary = (tmp_path / "logs" / "compile_error_summary.md").read_text(
        encoding="utf-8"
    )
    assert "Shell Escape Required" in summary


def test_restricted_write18_success_log_is_not_shell_escape_required():
    compiler = LaTeXCompiler()

    category = compiler._classify_compile_log("restricted \\write18 enabled.", None)

    assert category == "unknown"


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


def test_extract_error_detects_missing_endcsname_inserted():
    compiler = LaTeXCompiler()
    log = (
        "Some line\n"
        "! Missing \\endcsname inserted.\n"
        "<to be read again>\n"
        "                   \\protect \n"
        "l.104 ...{（我们将在\\autoref{related\\_works}中详细讨论相关工作）}\n"
    )
    extracted = compiler._extract_error(log)
    assert "Missing \\endcsname inserted." in extracted
    assert "\\autoref{related\\_works}" in extracted


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
        "arxiv_translate.compiler.latex_compiler.shutil.which",
        lambda _engine, path=None: "/usr/bin/true",
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


def test_extract_error_detects_undefined_control_sequence():
    compiler = LaTeXCompiler()
    log = (
        "./main.tex:1: Undefined control sequence.\n"
        "l.1 \\pdfoutput\n"
        "              =1\n"
    )
    extracted = compiler._extract_error(log)
    assert "Undefined control sequence" in extracted
    assert "\\pdfoutput" in extracted


def test_extract_error_detects_misplaced_noalign():
    compiler = LaTeXCompiler()
    log = (
        "./main.tex:1304: Misplaced \\noalign.\n"
        "\\midrule ->\\noalign\n"
        "                    {\\ifnum 0=`}\\fi\n"
    )
    extracted = compiler._extract_error(log)
    assert "Misplaced \\noalign." in extracted
    assert "\\midrule" in extracted


def test_extract_error_detects_critical_package_error():
    compiler = LaTeXCompiler()
    log = (
        "/usr/local/texlive/2023/texmf-dist/tex/xelatex/xecjk/xeCJK.sty:43: "
        "Critical Package xeCJK Error: The xeCJK package requires XeTeX to function.\n"
        "(xeCJK) You must change your typesetting engine to \"xelatex\".\n"
    )
    extracted = compiler._extract_error(log)
    assert "Critical Package xeCJK Error" in extracted
    assert "requires XeTeX" in extracted


def test_compile_strips_pdftex_primitive_before_first_engine_run(monkeypatch, tmp_path: Path):
    compiler = LaTeXCompiler()
    compiler.engines = ["xelatex"]
    call_count = {"n": 0}

    def fake_run_engine(engine, source_file, cwd, latex_source):
        call_count["n"] += 1
        assert r"\pdfoutput=1" not in latex_source
        (cwd / "main.pdf").write_bytes(b"%PDF-1.5\n1 0 obj\n<<>>\nendobj\n%%EOF\n")
        return (True, "ok", None)

    monkeypatch.setattr(
        "arxiv_translate.compiler.latex_compiler.shutil.which",
        lambda _engine, path=None: "/usr/bin/true",
    )
    monkeypatch.setattr(compiler, "_run_engine", fake_run_engine)

    result = compiler.compile(
        latex_source=(
            "\\pdfoutput=1\n"
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            "x\n"
            "\\end{document}\n"
        ),
        output_path=tmp_path / "out.pdf",
    )

    assert result.success is True
    assert call_count["n"] == 1

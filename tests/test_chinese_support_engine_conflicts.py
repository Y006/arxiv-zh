from arxiv_translate.compiler.chinese_support import (
    detect_cjk_fonts,
    get_fonts_from_dir,
    _guard_control_word_cjk_boundaries,
    _strip_engine_conflict_primitives,
    _strip_unicode_engine_driver_options,
    _unescape_reference_command_keys,
    inject_chinese_support,
    inject_chinese_support_for_engine,
)


def test_get_fonts_from_dir_falls_back_to_local_files_without_fontconfig(
    monkeypatch,
    tmp_path,
):
    import arxiv_translate.compiler.chinese_support as chinese_support

    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    (fonts_dir / "STSONG.TTF").write_bytes(b"not a real font")
    (fonts_dir / "STXIHEI.TTF").write_bytes(b"not a real font")

    monkeypatch.setattr(chinese_support.shutil, "which", lambda _name: None)
    monkeypatch.setattr(chinese_support, "TTFont", None)

    fonts = get_fonts_from_dir(fonts_dir)

    assert str((fonts_dir / "STSONG.TTF").resolve()) in fonts
    assert str((fonts_dir / "STXIHEI.TTF").resolve()) in fonts
    assert "STSONG" in fonts
    assert "STXIHEI" in fonts


def test_detect_cjk_fonts_does_not_guess_unavailable_noto():
    assert detect_cjk_fonts([]) == {}


def test_detect_cjk_fonts_prefers_project_font_files(tmp_path):
    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    main = fonts_dir / "STSONG.TTF"
    sans = fonts_dir / "STXIHEI.TTF"
    mono = fonts_dir / "STKAITI.TTF"
    for font_file in (main, sans, mono):
        font_file.write_bytes(b"not a real font")

    detected = detect_cjk_fonts(
        [str(main.resolve()), str(sans.resolve()), str(mono.resolve())]
    )

    assert detected == {
        "main": str(main.resolve()),
        "sans": str(sans.resolve()),
        "mono": str(mono.resolve()),
    }


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
        "\\pdfmapline{+NVIDIASans_Rg < NVIDIA-Sans-Font-TTF/NVIDIASans_Rg.ttf}\n"
        "\\documentclass{article}\n"
    )
    patched = _strip_engine_conflict_primitives(source)
    assert "\\pdfoutput=1" not in patched
    assert "\\pdfminorversion=7" not in patched
    assert "\\pdfcompresslevel=9" not in patched
    assert "\\pdfobjcompresslevel=3" not in patched
    assert "\\pdfmapline" not in patched
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


def test_inject_chinese_support_for_engine_uses_luatexja_for_lualatex():
    source = (
        "\\documentclass{article}\n"
        "\\usepackage{xeCJK}\n"
        "\\setCJKmainfont{Songti SC}\n"
        "\\begin{document}\n"
        "中文\n"
        "\\end{document}\n"
    )

    patched = inject_chinese_support_for_engine(
        source,
        engine="lualatex",
        font_config={"main": "FandolSong", "auto_detect": False},
    )

    assert "\\usepackage{luatexja-fontspec}" in patched
    assert "\\setmainjfont{FandolSong}" in patched
    assert "\\usepackage{xeCJK}" not in patched
    assert "\\setCJKmainfont" not in patched


def test_inject_chinese_support_adds_luatexja_fonts_when_package_exists(tmp_path):
    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    for filename in ("STSONG.TTF", "STXIHEI.TTF", "STKAITI.TTF"):
        (fonts_dir / filename).write_bytes(b"not a real font")

    source = (
        "\\documentclass{article}\n"
        "\\usepackage{luatexja-fontspec}\n"
        "\\begin{document}\n"
        "中文\n"
        "\\end{document}\n"
    )

    patched = inject_chinese_support_for_engine(
        source,
        engine="lualatex",
        font_config={
            "dir": str(fonts_dir),
            "main": "STSONG.TTF",
            "sans": "STXIHEI.TTF",
            "mono": "STKAITI.TTF",
            "auto_detect": False,
        },
    )

    font_path_option = f"Path={{{fonts_dir.as_posix()}/}}"
    assert patched.count("\\usepackage{luatexja-fontspec}") == 1
    assert f"\\setmainjfont[{font_path_option}]{{STSONG.TTF}}" in patched
    assert f"\\setsansjfont[{font_path_option}]{{STXIHEI.TTF}}" in patched
    assert f"\\setmonojfont[{font_path_option}]{{STKAITI.TTF}}" in patched


def test_inject_chinese_support_uses_font_file_paths(tmp_path):
    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    main = fonts_dir / "STSONG.TTF"
    sans = fonts_dir / "STXIHEI.TTF"
    mono = fonts_dir / "STKAITI.TTF"
    for font_file in (main, sans, mono):
        font_file.write_bytes(b"not a real font")

    source = "\\documentclass{article}\n\\begin{document}\n中文\n\\end{document}\n"

    patched = inject_chinese_support_for_engine(
        source,
        engine="xelatex",
        font_config={
            "dir": str(fonts_dir),
            "main": "STSONG.TTF",
            "sans": "STXIHEI.TTF",
            "mono": "STKAITI.TTF",
            "auto_detect": False,
        },
    )

    font_path_option = f"Path={{{fonts_dir.as_posix()}/}}"
    assert f"\\setCJKmainfont[{font_path_option}]{{STSONG.TTF}}" in patched
    assert f"\\setCJKsansfont[{font_path_option}]{{STXIHEI.TTF}}" in patched
    assert f"\\setCJKmonofont[{font_path_option}]{{STKAITI.TTF}}" in patched


def test_inject_chinese_support_auto_detects_local_font_files_without_fontconfig(
    monkeypatch,
    tmp_path,
):
    import arxiv_translate.compiler.chinese_support as chinese_support

    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    for filename in ("STSONG.TTF", "STXIHEI.TTF", "STKAITI.TTF"):
        (fonts_dir / filename).write_bytes(b"not a real font")

    monkeypatch.setattr(chinese_support.shutil, "which", lambda _name: None)
    monkeypatch.setattr(chinese_support, "TTFont", None)

    source = "\\documentclass{article}\n\\begin{document}\n中文\n\\end{document}\n"
    patched = inject_chinese_support_for_engine(
        source,
        engine="xelatex",
        font_config={"dir": str(fonts_dir), "auto_detect": True},
    )

    font_path_option = f"Path={{{fonts_dir.as_posix()}/}}"
    assert f"\\setCJKmainfont[{font_path_option}]{{STSONG.TTF}}" in patched
    assert f"\\setCJKsansfont[{font_path_option}]{{STXIHEI.TTF}}" in patched
    assert f"\\setCJKmonofont[{font_path_option}]{{STKAITI.TTF}}" in patched


def test_inject_chinese_support_for_engine_skips_pdflatex_cjk_by_default():
    source = "\\documentclass{article}\n\\begin{document}\n中文\n\\end{document}\n"

    patched = inject_chinese_support_for_engine(source, engine="pdflatex")

    assert "\\usepackage{CJKutf8}" not in patched
    assert "\\begin{CJK}" not in patched


def test_inject_chinese_support_for_engine_can_wrap_pdflatex_cjk():
    source = "\\documentclass{article}\n\\begin{document}\n中文\n\\end{document}\n"

    patched = inject_chinese_support_for_engine(
        source,
        engine="pdflatex",
        allow_pdflatex_cjk=True,
    )

    assert "\\usepackage{CJKutf8}" in patched
    assert "\\begin{CJK}{UTF8}{gbsn}" in patched
    assert "\\end{CJK}" in patched

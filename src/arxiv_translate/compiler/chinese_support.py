import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

try:
    from fontTools.ttLib import TTFont
except ImportError:  # pragma: no cover - exercised through fallback tests
    TTFont = None  # type: ignore[assignment]

ENGINE_CONFLICT_PRIMITIVES = (
    "pdfoutput",
    "pdfminorversion",
    "pdfcompresslevel",
    "pdfobjcompresslevel",
)
ENGINE_CONFLICT_BRACED_COMMANDS = ("pdfinfo",)
_ENGINE_CONFLICT_PATTERN = re.compile(
    r"^\s*\\(?:"
    + "|".join(ENGINE_CONFLICT_PRIMITIVES)
    + r")\b",
    re.IGNORECASE,
)
_CJK_CHAR_CLASS = "\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef"
_UNICODE_ENGINE_DRIVER_OPTIONS = {"pdftex", "dvips", "dvipdfmx", "xetex", "luatex"}
_DRIVER_OPTION_PACKAGES = {"graphicx", "graphics", "color", "xcolor", "hyperref"}
_UNICODE_ENGINES = {"xelatex", "lualatex"}
_CHINESE_PACKAGE_CHOICES = {"auto", "xeCJK", "luatexja", "ctex", "CJKutf8"}


def _find_matching_brace(text: str, open_index: int) -> Optional[int]:
    depth = 0
    index = open_index
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return None


def _strip_braced_conflict_commands(preamble: str) -> str:
    if not preamble:
        return preamble

    command_pattern = re.compile(
        r"\\(?:" + "|".join(ENGINE_CONFLICT_BRACED_COMMANDS) + r")(?![A-Za-z@])",
        re.IGNORECASE,
    )
    pieces: List[str] = []
    cursor = 0
    for match in command_pattern.finditer(preamble):
        line_start = preamble.rfind("\n", 0, match.start()) + 1
        line_prefix = preamble[line_start : match.start()]
        if line_prefix.lstrip().startswith("%"):
            continue

        arg_start = match.end()
        while arg_start < len(preamble) and preamble[arg_start].isspace():
            arg_start += 1
        if arg_start >= len(preamble) or preamble[arg_start] != "{":
            continue

        close_index = _find_matching_brace(preamble, arg_start)
        if close_index is None:
            continue

        remove_start = line_start if not line_prefix.strip() else match.start()
        remove_end = close_index + 1
        if remove_end < len(preamble) and preamble[remove_end] == "\n":
            remove_end += 1
        pieces.append(preamble[cursor:remove_start])
        cursor = remove_end

    if cursor == 0:
        return preamble
    pieces.append(preamble[cursor:])
    return "".join(pieces)


def _strip_engine_conflict_primitives(latex_source: str) -> str:
    """Remove pdfTeX primitive commands that conflict with XeLaTeX/LuaLaTeX."""
    if not latex_source:
        return latex_source

    begin_doc_match = re.search(r"\\begin\{document\}", latex_source)
    if begin_doc_match:
        preamble = latex_source[: begin_doc_match.start()]
        body = latex_source[begin_doc_match.start() :]
    else:
        preamble = latex_source
        body = ""

    kept_lines: List[str] = []
    for line in preamble.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("%"):
            kept_lines.append(line)
            continue
        if _ENGINE_CONFLICT_PATTERN.match(line):
            continue
        kept_lines.append(line)

    return _strip_braced_conflict_commands("".join(kept_lines) + body)


def _strip_unicode_engine_driver_options(latex_source: str) -> str:
    """Remove explicit graphics/PDF driver options before XeLaTeX/LuaLaTeX runs."""
    if not latex_source:
        return latex_source

    pattern = re.compile(
        r"\\(?P<cmd>usepackage|RequirePackage)\[(?P<opts>[^\]]*)\]\{(?P<pkgs>[^}]*)\}"
    )

    def replace(match: re.Match[str]) -> str:
        packages = {pkg.strip().lower() for pkg in match.group("pkgs").split(",")}
        if not packages & _DRIVER_OPTION_PACKAGES:
            return match.group(0)

        options = [part.strip() for part in match.group("opts").split(",") if part.strip()]
        kept = [
            option
            for option in options
            if option.lower() not in _UNICODE_ENGINE_DRIVER_OPTIONS
        ]
        if len(kept) == len(options):
            return match.group(0)
        if kept:
            return f"\\{match.group('cmd')}[{','.join(kept)}]" + "{" + match.group("pkgs") + "}"
        return f"\\{match.group('cmd')}" + "{" + match.group("pkgs") + "}"

    return pattern.sub(replace, latex_source)


def _guard_control_word_cjk_boundaries(latex_source: str) -> str:
    """Separate ASCII control words from following CJK text after translation."""
    if not latex_source:
        return latex_source

    cjk = _CJK_CHAR_CLASS
    latex_source = re.sub(
        rf"\\([A-Za-z@]+)(?:\{{\}})?\\(?=[{cjk}])",
        r"\\\1{}",
        latex_source,
    )
    latex_source = re.sub(
        rf"\\([A-Za-z@]+)(?=[{cjk}])",
        r"\\\1{}",
        latex_source,
    )
    return latex_source


def _unescape_reference_command_keys(latex_source: str) -> str:
    """Undo prose escaping inside command arguments that are LaTeX keys."""
    if not latex_source:
        return latex_source

    key_command_pattern = re.compile(
        r"\\(?P<cmd>cite[A-Za-z]*|ref|eqref|autoref|pageref|nameref|cref|Cref|label)"
        r"(?P<star>\*)?"
        r"(?P<opts>(?:\[[^\]]*\])*)"
        r"\{(?P<body>[^{}]*)\}"
    )

    def replace(match: re.Match[str]) -> str:
        body = match.group("body")
        patched_body = body.replace(r"\_", "_")
        if patched_body == body:
            return match.group(0)
        star = match.group("star") or ""
        return (
            f"\\{match.group('cmd')}{star}{match.group('opts')}"
            + "{"
            + patched_body
            + "}"
        )

    return key_command_pattern.sub(replace, latex_source)


def normalize_unicode_engine_source(latex_source: str) -> str:
    """Apply non-destructive source cleanups needed by XeLaTeX/LuaLaTeX."""
    latex_source = _strip_engine_conflict_primitives(latex_source)
    latex_source = _strip_unicode_engine_driver_options(latex_source)
    latex_source = _guard_control_word_cjk_boundaries(latex_source)
    latex_source = _unescape_reference_command_keys(latex_source)
    return latex_source


def contains_cjk_text(latex_source: str) -> bool:
    """Return whether the source contains CJK or full-width characters."""
    return bool(re.search(rf"[{_CJK_CHAR_CLASS}]", latex_source or ""))


FONT_FILE_SUFFIXES = {".ttf", ".ttc", ".otf"}


def _split_font_families(output: str) -> List[str]:
    fonts = set()
    for line in output.splitlines():
        for family in line.split(","):
            family = family.strip()
            if family:
                fonts.add(family)
    return sorted(fonts)


def _dedupe_font_values(values: List[str]) -> List[str]:
    seen = set()
    deduped: List[str] = []
    for value in values:
        value = value.strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def find_font_files(font_dir: Optional[Union[str, Path]]) -> List[Path]:
    """Return local font files from a configured font directory."""
    if not font_dir:
        return []

    root = Path(font_dir).expanduser()
    if not root.exists() or not root.is_dir():
        return []

    return sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in FONT_FILE_SUFFIXES
    )


def _fonttools_family_names(font_file: Path) -> List[str]:
    if TTFont is None:
        return []

    try:
        font = TTFont(str(font_file), fontNumber=0, lazy=True)
    except Exception:
        try:
            font = TTFont(str(font_file), lazy=True)
        except Exception:
            return []

    try:
        names = []
        for name in font["name"].names:
            if name.nameID not in {1, 4, 16}:
                continue
            try:
                value = name.toUnicode().strip()
            except Exception:
                continue
            if value:
                names.append(value)
        return _dedupe_font_values(names)
    except Exception:
        return []
    finally:
        try:
            font.close()
        except Exception:
            pass


def _filename_font_fallback_names(font_file: Path) -> List[str]:
    return _dedupe_font_values([font_file.stem, font_file.name])


def get_fonts_from_dir(font_dir: Optional[Union[str, Path]]) -> List[str]:
    """Detect usable font values from local font files."""
    font_files = find_font_files(font_dir)
    if not font_files:
        return []

    fonts: List[str] = [str(path) for path in font_files]

    if shutil.which("fc-scan"):
        try:
            result = subprocess.run(
                [
                    "fc-scan",
                    "--format",
                    "%{family}\n",
                    *[str(path) for path in font_files],
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                fonts.extend(_split_font_families(result.stdout))
        except Exception:
            pass

    for font_file in font_files:
        family_names = _fonttools_family_names(font_file)
        fonts.extend(family_names or _filename_font_fallback_names(font_file))

    return _dedupe_font_values(fonts)


def get_system_fonts() -> List[str]:
    """Detect available CJK fonts using fc-list."""
    if not shutil.which("fc-list"):
        return []

    try:
        result = subprocess.run(
            ["fc-list", ":lang=zh", "family"], capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return []
        return _split_font_families(result.stdout)
    except Exception:
        return []


def get_available_fonts(
    font_dir: Optional[Union[str, Path]] = None,
    include_system: bool = True,
) -> List[str]:
    """Detect fonts from a local font directory first, then system fonts."""
    fonts: List[str] = []
    seen = set()

    def add_many(values: List[str]) -> None:
        for value in values:
            key = value.lower()
            if key in seen:
                continue
            seen.add(key)
            fonts.append(value)

    add_many(get_fonts_from_dir(font_dir))
    if include_system:
        add_many(get_system_fonts())
    return fonts


def detect_cjk_fonts(available_fonts: List[str]) -> Dict[str, str]:
    """Select best available CJK fonts from actually detected fonts."""
    # Priority groups: (Serif, Sans, Mono)
    font_candidates = [
        # Project-local sample bundle
        {
            "main": ["STSong", "SimSun"],
            "sans": ["STXihei", "SimHei"],
            "mono": ["STKaiti", "KaiTi"],
        },
        # Noto CJK (Google/Adobe) - Preferred
        {
            "main": ["Noto Serif CJK SC", "Noto Serif CJK"],
            "sans": ["Noto Sans CJK SC", "Noto Sans CJK"],
            "mono": ["Noto Sans Mono CJK SC", "Noto Sans Mono CJK"],
        },
        # Source Han (Adobe)
        {
            "main": ["Source Han Serif SC", "Source Han Serif"],
            "sans": ["Source Han Sans SC", "Source Han Sans"],
            "mono": ["Source Han Sans HW SC", "Source Han Sans HW"],
        },
        # macOS Chinese Fonts
        {
            "main": ["Songti SC", "STSong"],
            "sans": ["PingFang SC", "Heiti SC", "Hiragino Sans GB", "STHeiti"],
            "mono": ["STFangsong", "FangSong_GB2312", "FZFangSong-Z02"],
        },
        # Windows Chinese Fonts
        {
            "main": ["SimSun", "SongTi"],
            "sans": ["SimHei", "Microsoft YaHei"],
            "mono": ["FangSong", "KaiTi"],
        },
        # Fandol (TeX Live default)
        {"main": ["FandolSong"], "sans": ["FandolHei"], "mono": ["FandolKai"]},
    ]

    def font_value_matches(candidate: str, value: str) -> bool:
        candidate_key = candidate.lower()
        value_key = value.lower()
        if candidate_key == value_key or candidate_key in value_key:
            return True
        if Path(value).suffix.lower() in FONT_FILE_SUFFIXES:
            return candidate_key in Path(value).stem.lower()
        return False

    def find_match(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            for avail in available_fonts:
                if font_value_matches(cand, avail):
                    return avail
        return None

    for group in font_candidates:
        main = find_match(group["main"])
        if main:
            sans = find_match(group["sans"]) or main
            mono = find_match(group["mono"]) or sans
            return {"main": main, "sans": sans, "mono": mono}

    return {}


def _config_value(config: Optional[Any], key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _font_available(font_name: Optional[str], available_fonts: List[str]) -> bool:
    if not font_name:
        return False
    font_key = font_name.lower()
    return any(
        font_key == available.lower()
        or font_key in available.lower()
        or (
            Path(available).suffix.lower() in FONT_FILE_SUFFIXES
            and font_key in Path(available).stem.lower()
        )
        for available in available_fonts
    )


def _is_font_file_value(font_value: Optional[str]) -> bool:
    return bool(font_value and Path(font_value).suffix.lower() in FONT_FILE_SUFFIXES)


def _resolve_font_file_value(
    font_value: str,
    font_dir: Optional[Union[str, Path]],
) -> Path:
    raw_path = Path(font_value).expanduser()
    candidates: List[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append((Path.cwd() / raw_path).resolve())
        if font_dir:
            base_dir = Path(font_dir).expanduser()
            candidates.append((base_dir / raw_path).resolve())
            candidates.append((base_dir / raw_path.name).resolve())
        candidates.append(raw_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def _fontspec_path_option(directory: Path) -> str:
    path = directory.as_posix()
    if path and not path.endswith("/"):
        path += "/"
    return f"Path={{{path}}}"


def _font_command(
    command: str,
    font_value: str,
    font_dir: Optional[Union[str, Path]],
) -> str:
    if not _is_font_file_value(font_value):
        return f"\\{command}{{{font_value}}}"

    font_path = _resolve_font_file_value(font_value, font_dir)
    return (
        f"\\{command}[{_fontspec_path_option(font_path.parent)}]"
        f"{{{font_path.name}}}"
    )


def _resolved_font_commands(
    font_config: Optional[Any],
    *,
    engine: str,
    font_dir: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Build engine-specific font commands only for configured or detected fonts."""
    cfg_main = _config_value(font_config, "main")
    cfg_sans = _config_value(font_config, "sans")
    cfg_mono = _config_value(font_config, "mono")
    cfg_dir = font_dir or _config_value(font_config, "dir")
    use_auto = bool(_config_value(font_config, "auto_detect", True))

    available = get_available_fonts(cfg_dir) if use_auto else []
    detected = detect_cjk_fonts(available) if available else {}

    def choose(configured: Optional[str], detected_key: str) -> Optional[str]:
        if configured:
            return configured
        candidate = detected.get(detected_key)
        if candidate and _font_available(candidate, available):
            return candidate
        return None

    main_font = choose(cfg_main, "main")
    sans_font = choose(cfg_sans, "sans")
    mono_font = choose(cfg_mono, "mono")

    if engine == "lualatex":
        commands = []
        if main_font:
            commands.append(_font_command("setmainjfont", main_font, cfg_dir))
        if sans_font:
            commands.append(_font_command("setsansjfont", sans_font, cfg_dir))
        if mono_font:
            commands.append(_font_command("setmonojfont", mono_font, cfg_dir))
        return commands

    commands = []
    if main_font:
        commands.append(_font_command("setCJKmainfont", main_font, cfg_dir))
    if sans_font:
        commands.append(_font_command("setCJKsansfont", sans_font, cfg_dir))
    if mono_font:
        commands.append(_font_command("setCJKmonofont", mono_font, cfg_dir))
    return commands


def _remove_unicode_input_packages(latex_source: str) -> str:
    latex_source = re.sub(
        r"^[ \t]*\\usepackage\s*\[T1\]\s*\{fontenc\}\s*\n?",
        "",
        latex_source,
        flags=re.MULTILINE,
    )
    latex_source = re.sub(
        r"^[ \t]*\\usepackage\s*\[utf8\]\s*\{inputenc\}\s*\n?",
        "",
        latex_source,
        flags=re.MULTILINE,
    )
    return latex_source


def _package_present(latex_source: str, package_name: str) -> bool:
    return bool(
        re.search(
            rf"\\(?:usepackage|RequirePackage)(?:\[[^\]]*\])?\{{{re.escape(package_name)}\}}",
            latex_source,
        )
    )


def _strip_engine_specific_chinese_support(latex_source: str, *, keep: str) -> str:
    """Remove Chinese-support commands that are incompatible with the target engine."""
    package_lines = {
        "xeCJK": r"^[ \t]*\\(?:usepackage|RequirePackage)(?:\[[^\]]*\])?\{xeCJK\}[ \t]*\n?",
        "luatexja": r"^[ \t]*\\(?:usepackage|RequirePackage)(?:\[[^\]]*\])?\{luatexja-fontspec\}[ \t]*\n?",
        "CJKutf8": r"^[ \t]*\\(?:usepackage|RequirePackage)(?:\[[^\]]*\])?\{CJKutf8\}[ \t]*\n?",
    }
    font_lines = {
        "xeCJK": r"^[ \t]*\\setCJK(?:main|sans|mono)font(?:\[[^\]]*\])?\{[^}]*\}[ \t]*\n?",
        "luatexja": r"^[ \t]*\\set(?:main|sans|mono)jfont(?:\[[^\]]*\])?\{[^}]*\}[ \t]*\n?",
    }

    patched = latex_source
    for package_key, pattern in package_lines.items():
        if package_key != keep:
            patched = re.sub(pattern, "", patched, flags=re.MULTILINE)
    for package_key, pattern in font_lines.items():
        if package_key != keep:
            patched = re.sub(pattern, "", patched, flags=re.MULTILINE)
    if keep != "CJKutf8":
        patched = re.sub(r"^[ \t]*\\begin\{CJK\}\{UTF8\}\{[^}]*\}[ \t]*\n?", "", patched, flags=re.MULTILINE)
        patched = re.sub(r"^[ \t]*\\end\{CJK\}[ \t]*\n?", "", patched, flags=re.MULTILINE)
    return patched


def _insert_after_documentclass_or_before_document(
    latex_source: str,
    insertion: str,
) -> str:
    docclass_match = re.search(
        r"(\\documentclass\s*(?:\[[^\]]*\])?\s*\{[^}]+\}\s*\n?)", latex_source
    )
    if docclass_match:
        insert_pos = docclass_match.end()
        return latex_source[:insert_pos] + insertion + latex_source[insert_pos:]

    begin_doc_match = re.search(r"\\begin\{document\}", latex_source)
    if begin_doc_match:
        insert_pos = begin_doc_match.start()
        return latex_source[:insert_pos] + insertion + "\n" + latex_source[insert_pos:]

    return latex_source + "\n" + insertion


def _wrap_pdflatex_cjk_document(latex_source: str) -> str:
    if "\\begin{CJK}" in latex_source:
        return latex_source
    begin_doc_match = re.search(r"\\begin\{document\}", latex_source)
    end_doc_match = list(re.finditer(r"\\end\{document\}", latex_source))
    if not begin_doc_match or not end_doc_match:
        return latex_source

    end_match = end_doc_match[-1]
    patched = (
        latex_source[: begin_doc_match.end()]
        + "\n\\begin{CJK}{UTF8}{gbsn}\n"
        + latex_source[begin_doc_match.end() : end_match.start()]
        + "\n\\end{CJK}\n"
        + latex_source[end_match.start() :]
    )
    return patched


def inject_chinese_support_for_engine(
    latex_source: str,
    *,
    engine: str = "xelatex",
    font_config: Optional[Any] = None,
    allow_pdflatex_cjk: bool = False,
    chinese_package: str = "auto",
    font_dir: Optional[Union[str, Path]] = None,
) -> str:
    """Inject Chinese support that matches the selected LaTeX engine."""
    engine = (engine or "xelatex").lower()
    if chinese_package not in _CHINESE_PACKAGE_CHOICES:
        chinese_package = "auto"

    latex_source = normalize_unicode_engine_source(latex_source)

    if engine in _UNICODE_ENGINES:
        latex_source = _remove_unicode_input_packages(latex_source)

    if chinese_package == "ctex":
        latex_source = _strip_engine_specific_chinese_support(latex_source, keep="ctex")
        if _package_present(latex_source, "ctex"):
            return latex_source
        return _insert_after_documentclass_or_before_document(
            latex_source,
            "\n% Auto-injected Chinese Support\n\\usepackage[UTF8]{ctex}\n",
        )

    if engine == "lualatex":
        latex_source = _strip_engine_specific_chinese_support(
            latex_source,
            keep="luatexja",
        )
        if _package_present(latex_source, "luatexja-fontspec"):
            return latex_source
        font_commands = _resolved_font_commands(
            font_config,
            engine="lualatex",
            font_dir=font_dir,
        )
        injection_lines = [
            "",
            "% Auto-injected Chinese Support",
            r"\usepackage{luatexja-fontspec}",
            *font_commands,
        ]
        return _insert_after_documentclass_or_before_document(
            latex_source,
            "\n".join(injection_lines) + "\n",
        )

    if engine == "pdflatex":
        latex_source = _strip_engine_specific_chinese_support(
            latex_source,
            keep="CJKutf8" if allow_pdflatex_cjk else "pdflatex",
        )
        if not allow_pdflatex_cjk:
            return latex_source
        if not _package_present(latex_source, "CJKutf8"):
            latex_source = _insert_after_documentclass_or_before_document(
                latex_source,
                "\n% Auto-injected Chinese Support\n\\usepackage{CJKutf8}\n",
            )
        return _wrap_pdflatex_cjk_document(latex_source)

    latex_source = _strip_engine_specific_chinese_support(latex_source, keep="xeCJK")
    if _package_present(latex_source, "xeCJK"):
        return latex_source
    font_commands = _resolved_font_commands(
        font_config,
        engine="xelatex",
        font_dir=font_dir,
    )
    injection_lines = [
        "",
        "% Auto-injected Chinese Support",
        r"\usepackage{fontspec}",
        r"\usepackage{xeCJK}",
        *font_commands,
    ]
    return _insert_after_documentclass_or_before_document(
        latex_source,
        "\n".join(injection_lines) + "\n",
    )


def inject_chinese_support(latex_source: str, font_config: Optional[Any] = None) -> str:
    r"""
    Injects xeCJK package and Noto CJK font settings into the LaTeX source.

    This function inserts the necessary LaTeX commands to support Chinese characters
    using the xeCJK package. It attempts to insert these commands
    immediately after the \documentclass declaration.

    Args:
        latex_source (str): The original LaTeX source code.
        font_config (Optional[Any]): Configuration object or dict with font settings.

    Returns:
        str: The modified LaTeX source code with Chinese support injected.
    """
    return inject_chinese_support_for_engine(
        latex_source,
        engine="xelatex",
        font_config=font_config,
    )

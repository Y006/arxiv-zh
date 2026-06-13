import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

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


FONT_FILE_SUFFIXES = {".ttf", ".ttc", ".otf"}


def _split_font_families(output: str) -> List[str]:
    fonts = set()
    for line in output.splitlines():
        for family in line.split(","):
            family = family.strip()
            if family:
                fonts.add(family)
    return sorted(fonts)


def get_fonts_from_dir(font_dir: Optional[Union[str, Path]]) -> List[str]:
    """Detect font family names from font files in a local directory."""
    if not font_dir or not shutil.which("fc-scan"):
        return []

    root = Path(font_dir).expanduser()
    if not root.exists() or not root.is_dir():
        return []

    font_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in FONT_FILE_SUFFIXES
    )
    if not font_files:
        return []

    try:
        result = subprocess.run(
            ["fc-scan", "--format", "%{family}\n", *[str(path) for path in font_files]],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return []
        return _split_font_families(result.stdout)
    except Exception:
        return []


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
    """Select best available CJK fonts based on priority."""
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

    def find_match(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            for avail in available_fonts:
                if cand.lower() == avail.lower():
                    return avail
                if cand.lower() in avail.lower():
                    return avail
        return None

    for group in font_candidates:
        main = find_match(group["main"])
        if main:
            sans = find_match(group["sans"]) or main
            mono = find_match(group["mono"]) or sans
            return {"main": main, "sans": sans, "mono": mono}

    # Fallback
    return {
        "main": "Noto Serif CJK SC",
        "sans": "Noto Sans CJK SC",
        "mono": "Noto Sans Mono CJK SC",
    }


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
    latex_source = normalize_unicode_engine_source(latex_source)

    # Check if xeCJK is already present to avoid duplication
    if "xeCJK" in latex_source:
        return latex_source

    # Default: auto-detect fonts
    avail = get_available_fonts()
    detected = detect_cjk_fonts(avail)
    main_font = detected["main"]
    sans_font = detected["sans"]
    mono_font = detected["mono"]

    # Override with config if provided
    if font_config:
        # Support both Pydantic model and dict
        if isinstance(font_config, dict):
            cfg_main = font_config.get("main")
            cfg_sans = font_config.get("sans")
            cfg_mono = font_config.get("mono")
            use_auto = font_config.get("auto_detect", True)
        else:
            # Assume Pydantic model
            cfg_main = getattr(font_config, "main", None)
            cfg_sans = getattr(font_config, "sans", None)
            cfg_mono = getattr(font_config, "mono", None)
            use_auto = getattr(font_config, "auto_detect", True)

        # If auto_detect is disabled, use configured fonts
        if not use_auto:
            if cfg_main:
                main_font = cfg_main
            if cfg_sans:
                sans_font = cfg_sans
            if cfg_mono:
                mono_font = cfg_mono
        else:
            # Auto-detect is enabled, but override with any explicitly set fonts
            if cfg_main:
                main_font = cfg_main
            if cfg_sans:
                sans_font = cfg_sans
            if cfg_mono:
                mono_font = cfg_mono

    latex_source = re.sub(
        r"\\usepackage\s*\[T1\]\s*\{fontenc\}\s*\n?", "", latex_source
    )
    latex_source = re.sub(
        r"\\usepackage\s*\[utf8\]\s*\{inputenc\}\s*\n?", "", latex_source
    )
    latex_source = re.sub(
        r"\\usepackage\s*\[utf8\]\s*\{inputenc\}\s*\n?", "", latex_source
    )

    injection = (
        "\n% Auto-injected Chinese Support\n"
        r"\usepackage{xeCJK}" + "\n"
        f"\\setCJKmainfont{{{main_font}}}\n"
        f"\\setCJKsansfont{{{sans_font}}}\n"
        f"\\setCJKmonofont{{{mono_font}}}\n"
    )

    # Inject immediately after \documentclass (more canonical position)
    docclass_match = re.search(
        r"(\\documentclass\s*(?:\[[^\]]*\])?\s*\{[^}]+\}\s*\n?)", latex_source
    )
    if docclass_match:
        insert_pos = docclass_match.end()
        return latex_source[:insert_pos] + injection + latex_source[insert_pos:]

    # Fallback: inject before \begin{document}
    begin_doc_match = re.search(r"\\begin\{document\}", latex_source)
    if begin_doc_match:
        insert_pos = begin_doc_match.start()
        return latex_source[:insert_pos] + injection + "\n" + latex_source[insert_pos:]

    # Last resort: append at end if no \begin{document} found
    return latex_source + "\n" + injection

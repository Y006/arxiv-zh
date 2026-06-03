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
_ENGINE_CONFLICT_PATTERN = re.compile(
    r"^\s*\\(?:"
    + "|".join(ENGINE_CONFLICT_PRIMITIVES)
    + r")\b",
    re.IGNORECASE,
)


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

    return "".join(kept_lines) + body


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
    latex_source = _strip_engine_conflict_primitives(latex_source)

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

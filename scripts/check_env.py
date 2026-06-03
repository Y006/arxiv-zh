#!/usr/bin/env python3
import os
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

TINY_TEX_PATHS = [
    Path("~/Library/TinyTeX/bin/universal-darwin/").expanduser(),
    Path("/Library/TeX/texbin/"),
]
LOCAL_FONT_DIR = PROJECT_ROOT / "fonts"
RECOMMENDED_FONT_FAMILIES = [
    "STSong",
    "STXihei",
    "STKaiti",
    "SimSun",
    "SimHei",
    "KaiTi",
]


def status(value: bool) -> str:
    return "yes" if value else "no"


def which(name: str) -> str:
    return shutil.which(name) or "not found"


def scan_local_fonts(font_dir: Path) -> list[str]:
    try:
        from arxiv_translate.compiler.chinese_support import get_available_fonts

        return get_available_fonts(font_dir=font_dir, include_system=False)
    except Exception:
        return []


def detect_recommended_fonts(font_dir: Path) -> dict[str, str]:
    try:
        from arxiv_translate.compiler.chinese_support import (
            detect_cjk_fonts,
            get_available_fonts,
        )

        return detect_cjk_fonts(get_available_fonts(font_dir=font_dir))
    except Exception:
        return {"main": "unknown", "sans": "unknown", "mono": "unknown"}


def main() -> int:
    config_path = Path("~/.config/arxiv-translate/config.yaml").expanduser()
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    local_fonts = scan_local_fonts(LOCAL_FONT_DIR)
    recommended_fonts = detect_recommended_fonts(LOCAL_FONT_DIR)
    sample_fonts = [
        name for name in RECOMMENDED_FONT_FAMILIES if name in set(local_fonts)
    ]

    print("arxiv-zh environment check")
    print("==========================")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Virtual environment: {status(in_venv)}")
    print(f"sys.prefix: {sys.prefix}")
    print(f"DEEPSEEK_API_KEY present: {status(bool(os.getenv('DEEPSEEK_API_KEY')))}")
    print(f"xelatex: {which('xelatex')}")
    print(f"latexmk: {which('latexmk')}")
    print(f"tlmgr: {which('tlmgr')}")
    print(f"Config exists: {status(config_path.exists())} ({config_path})")
    print(f"Project fonts dir: {status(LOCAL_FONT_DIR.exists())} ({LOCAL_FONT_DIR})")
    print(
        "Local CJK families: "
        + (", ".join(sample_fonts) if sample_fonts else "not found")
    )
    print(
        "Recommended CJK fonts: "
        f"main={recommended_fonts['main']}, "
        f"sans={recommended_fonts['sans']}, "
        f"mono={recommended_fonts['mono']}"
    )
    print("")
    print("TinyTeX common paths:")
    for path in TINY_TEX_PATHS:
        print(f"- {path}: {status(path.exists())}")
    print("")
    print("Recommended next commands:")
    print("1. export DEEPSEEK_API_KEY=你的_key")
    print(
        "2. arxiv-zh 2605.28486 --provider deepseek --compile "
        "--max-chunks 2 --output ./output/mag-vla-font-test "
        "--font-dir ./fonts --cjk-main-font STSong "
        "--cjk-sans-font STXihei --cjk-mono-font STKaiti"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

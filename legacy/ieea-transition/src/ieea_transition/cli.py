from __future__ import annotations

import warnings

from arxiv_translate.cli import main as modern_main


DEPRECATION_MESSAGE = (
    "`ieeA` is deprecated and will stop receiving feature updates. "
    "Please migrate to `arxiv-translate` (commands: `arx` / `arxiv-translate`)."
)


def main() -> None:
    warnings.warn(DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
    modern_main()


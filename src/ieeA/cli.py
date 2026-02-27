from __future__ import annotations

import warnings

from arxiv_translate import cli as _modern_cli
from arxiv_translate.cli import *  # noqa: F401,F403

DEPRECATION_MESSAGE = (
    "`ieeA` CLI is deprecated and will be removed in the next release. "
    "Use `arx` or `arxiv-translate` instead."
)


def main() -> None:
    warnings.warn(DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
    _modern_cli.main()


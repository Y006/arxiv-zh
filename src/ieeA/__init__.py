from __future__ import annotations

import warnings

from arxiv_translate import *  # noqa: F401,F403

warnings.warn(
    "`ieeA` namespace is deprecated and will be removed in the next release. "
    "Use `arxiv_translate` instead.",
    DeprecationWarning,
    stacklevel=2,
)

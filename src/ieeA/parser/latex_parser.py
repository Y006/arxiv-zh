from __future__ import annotations

import importlib as _importlib
import sys as _sys

_module = _importlib.import_module("arxiv_translate.parser.latex_parser")
_sys.modules[__name__] = _module

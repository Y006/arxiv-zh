from __future__ import annotations

import importlib as _importlib
import sys as _sys

_module = _importlib.import_module("arxiv_translate.rules.validation_rules")
_sys.modules[__name__] = _module

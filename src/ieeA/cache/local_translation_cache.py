from __future__ import annotations

import importlib as _importlib
import sys as _sys

_module = _importlib.import_module("arxiv_translate.cache.local_translation_cache")
_sys.modules[__name__] = _module

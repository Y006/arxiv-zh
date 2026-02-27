import importlib
import sys
import warnings


def test_import_ieea_emits_deprecation_warning():
    sys.modules.pop("ieeA", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import ieeA  # noqa: F401

    messages = [str(item.message) for item in caught]
    assert any(
        "deprecated" in message.lower() and "arxiv_translate" in message
        for message in messages
    )


def test_legacy_module_aliases_new_namespace_module():
    legacy = importlib.import_module("ieeA.parser.latex_parser")
    modern = importlib.import_module("arxiv_translate.parser.latex_parser")
    assert legacy is modern

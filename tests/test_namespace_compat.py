import importlib
import sys
import warnings

from typer.testing import CliRunner


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


def test_legacy_cli_warns_and_delegates(monkeypatch):
    legacy_cli = importlib.import_module("ieeA.cli")
    modern_cli = importlib.import_module("arxiv_translate.cli")

    called = {"value": False}

    def fake_main():
        called["value"] = True

    monkeypatch.setattr(modern_cli, "main", fake_main)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy_cli.main()

    assert called["value"] is True
    messages = [str(item.message) for item in caught]
    assert any("deprecated" in message.lower() and "arxiv-translate" in message for message in messages)


def test_legacy_cli_app_supports_version_option(monkeypatch):
    legacy_cli = importlib.import_module("ieeA.cli")
    monkeypatch.setattr("arxiv_translate.cli._resolve_cli_version", lambda: "8.8.8")
    runner = CliRunner()

    result = runner.invoke(legacy_cli.app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "8.8.8"

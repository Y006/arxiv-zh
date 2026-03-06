from typer.testing import CliRunner
from importlib.metadata import PackageNotFoundError

from arxiv_translate.cli import _resolve_cli_version, app


def test_root_version_option_prints_version_and_exits_zero(monkeypatch):
    monkeypatch.setattr("arxiv_translate.cli._resolve_cli_version", lambda: "9.9.9")
    runner = CliRunner()

    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "9.9.9"


def test_help_still_available_after_version_option_added():
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Usage: arx" in result.stdout
    assert "--version" in result.stdout


def test_no_args_behavior_unchanged():
    runner = CliRunner()

    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert "Usage: arx" in result.stdout


def test_version_resolution_fallback_when_metadata_missing(monkeypatch):
    def _raise_not_found(_name: str):
        raise PackageNotFoundError

    monkeypatch.setattr("arxiv_translate.cli.metadata.version", _raise_not_found)

    assert _resolve_cli_version() == "unknown"

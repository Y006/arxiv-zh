from pathlib import Path
from textwrap import dedent

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner


def _write_config(path: Path, content: str) -> Path:
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return path


def test_prepare_arxiv_zh_output_dirs_creates_expected_layout(tmp_path: Path):
    from arxiv_translate.cli import _prepare_arxiv_zh_output_dirs

    layout = _prepare_arxiv_zh_output_dirs(tmp_path / "paper")

    assert layout.root == tmp_path / "paper"
    assert layout.source_dir == tmp_path / "paper" / "source"
    assert layout.translated_dir == tmp_path / "paper" / "translated"
    assert layout.pdf_dir == tmp_path / "paper" / "pdf"
    assert layout.cache_dir == tmp_path / "paper" / "cache"
    assert layout.logs_dir == tmp_path / "paper" / "logs"
    assert layout.translate_log == tmp_path / "paper" / "logs" / "translate.log"
    for path in (
        layout.source_dir,
        layout.translated_dir,
        layout.pdf_dir,
        layout.cache_dir,
        layout.logs_dir,
    ):
        assert path.is_dir()


def test_prepare_arxiv_zh_output_dirs_resolves_relative_output(tmp_path: Path, monkeypatch):
    from arxiv_translate.cli import _prepare_arxiv_zh_output_dirs

    monkeypatch.chdir(tmp_path)

    layout = _prepare_arxiv_zh_output_dirs(Path("paper"))

    assert layout.root == tmp_path / "paper"
    assert layout.cache_dir == tmp_path / "paper" / "cache"


def test_arxiv_zh_options_require_configured_key_env(monkeypatch, tmp_path: Path):
    import arxiv_translate.cli as cli_module

    config_path = _write_config(
        tmp_path / "config.yaml",
        """
        llm:
          key_env: CUSTOM_DEEPSEEK_KEY
        fonts:
          auto_detect: false
        """,
    )
    monkeypatch.delenv("CUSTOM_DEEPSEEK_KEY", raising=False)
    monkeypatch.setattr(cli_module, "_arxiv_zh_dotenv_paths", lambda: [])

    with pytest.raises(ValueError, match="CUSTOM_DEEPSEEK_KEY"):
        cli_module._resolve_arxiv_zh_options(
            arxiv_id="2501.12345",
            config=config_path,
        )


def test_arxiv_zh_options_load_key_from_dotenv(monkeypatch, tmp_path: Path):
    import arxiv_translate.cli as cli_module

    config_path = _write_config(
        tmp_path / "config.yaml",
        """
        fonts:
          auto_detect: false
        """,
    )
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("DEEPSEEK_API_KEY=sk-dotenv-test\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(cli_module, "_arxiv_zh_dotenv_paths", lambda: [dotenv_path])

    options, _config = cli_module._resolve_arxiv_zh_options(
        arxiv_id="2501.12345",
        config=config_path,
    )

    assert options.api_key == "sk-dotenv-test"


def test_arxiv_zh_options_prefers_shell_env_over_dotenv(monkeypatch, tmp_path: Path):
    import arxiv_translate.cli as cli_module

    config_path = _write_config(
        tmp_path / "config.yaml",
        """
        fonts:
          auto_detect: false
        """,
    )
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("DEEPSEEK_API_KEY=sk-dotenv-test\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-shell-test")
    monkeypatch.setattr(cli_module, "_arxiv_zh_dotenv_paths", lambda: [dotenv_path])

    options, _config = cli_module._resolve_arxiv_zh_options(
        arxiv_id="2501.12345",
        config=config_path,
    )

    assert options.api_key == "sk-shell-test"


def test_arxiv_zh_options_come_from_single_config(monkeypatch, tmp_path: Path):
    import arxiv_translate.cli as cli_module

    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()
    config_path = _write_config(
        tmp_path / "config.yaml",
        """
        llm:
          models: deepseek-reasoner
          endpoint: https://api.deepseek.com
        translation:
          concurrency: 5
          max_chunks: 2
        paths:
          output_dir: ./translated-output
        fonts:
          dir: ./fonts
          auto_detect: false
          main: STSong
          sans: STXihei
          mono: STKaiti
        compilation:
          enabled: true
        """,
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    options, config = cli_module._resolve_arxiv_zh_options(
        arxiv_id="2501.12345",
        config=config_path,
    )

    assert options.output == tmp_path / "translated-output" / "2501.12345"
    assert options.concurrency == 5
    assert options.max_chunks == 2
    assert options.compile_pdf is True
    assert options.model == "deepseek-reasoner"
    assert config.fonts.dir == str(fonts_dir)
    assert config.fonts.main == "STSong"
    assert config.fonts.sans == "STXihei"
    assert config.fonts.mono == "STKaiti"


def test_arxiv_zh_rejects_non_deepseek_config(monkeypatch, tmp_path: Path):
    import arxiv_translate.cli as cli_module

    config_path = _write_config(
        tmp_path / "config.yaml",
        """
        llm:
          sdk: openai
        fonts:
          auto_detect: false
        """,
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    with pytest.raises(ValueError, match="llm.sdk: deepseek"):
        cli_module._resolve_arxiv_zh_options(
            arxiv_id="2501.12345",
            config=config_path,
        )


def test_arxiv_zh_default_config_uses_detected_cjk_fonts(monkeypatch, tmp_path: Path):
    import arxiv_translate.cli as cli_module

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(
        cli_module,
        "get_available_fonts",
        lambda font_dir=None, include_system=True: [
            "Songti SC",
            "Heiti SC",
            "Hiragino Sans GB",
        ],
    )

    config = cli_module._load_config_for_arxiv_zh(None)

    assert config.fonts.auto_detect is True
    assert config.fonts.main == "Songti SC"
    assert config.fonts.sans == "Heiti SC"
    assert config.fonts.mono == "Heiti SC"


def test_arxiv_zh_default_config_prefers_project_fonts(monkeypatch, tmp_path: Path):
    import arxiv_translate.cli as cli_module

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    project_fonts = tmp_path / "fonts"
    project_fonts.mkdir()
    monkeypatch.setattr(cli_module, "_project_font_dir", lambda: project_fonts)
    monkeypatch.setattr(
        cli_module,
        "get_available_fonts",
        lambda font_dir=None, include_system=True: (
            ["STSong", "STXihei", "STKaiti", "Songti SC"]
            if font_dir == project_fonts
            else ["Songti SC", "Heiti SC"]
        ),
    )

    config = cli_module._load_config_for_arxiv_zh(None)

    assert config.fonts.dir == str(project_fonts)
    assert config.fonts.main == "STSong"
    assert config.fonts.sans == "STXihei"
    assert config.fonts.mono == "STKaiti"


def test_arxiv_zh_config_font_values_are_used(tmp_path: Path):
    from arxiv_translate.cli import _load_config_for_arxiv_zh

    font_dir = tmp_path / "fonts"
    font_dir.mkdir()
    config_path = _write_config(
        tmp_path / "config.yaml",
        """
        fonts:
          dir: ./fonts
          auto_detect: false
          main: STSong
          sans: STXihei
          mono: STKaiti
        """,
    )

    config = _load_config_for_arxiv_zh(config_path)

    assert config.fonts.dir == str(font_dir)
    assert config.fonts.auto_detect is False
    assert config.fonts.main == "STSong"
    assert config.fonts.sans == "STXihei"
    assert config.fonts.mono == "STKaiti"


def test_config_example_matches_runtime_schema():
    from arxiv_translate.rules.config import Config

    data = yaml.safe_load(Path("config.example.yaml").read_text(encoding="utf-8"))

    config = Config(**data)

    assert config.llm.sdk == "deepseek"
    assert config.llm.key is None
    assert config.llm.key_env == "DEEPSEEK_API_KEY"


def test_config_rejects_unknown_top_level_sections():
    from arxiv_translate.rules.config import Config

    with pytest.raises(ValidationError, match="provider"):
        Config(provider={"name": "deepseek"})


def test_config_set_writes_valid_schema_key(monkeypatch, tmp_path: Path):
    from arxiv_translate.cli import app

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    runner = CliRunner()

    result = runner.invoke(app, ["config", "set", "llm.models", "deepseek-reasoner"])

    assert result.exit_code == 0
    config_path = tmp_path / "xdg" / "arxiv-translate" / "config.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["llm"]["models"] == "deepseek-reasoner"


def test_config_set_rejects_unknown_schema_key(monkeypatch, tmp_path: Path):
    from arxiv_translate.cli import app

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    runner = CliRunner()

    result = runner.invoke(app, ["config", "set", "provider.name", "deepseek"])

    assert result.exit_code == 1
    assert "Invalid config value" in result.stdout
    assert not (tmp_path / "xdg" / "arxiv-translate" / "config.yaml").exists()

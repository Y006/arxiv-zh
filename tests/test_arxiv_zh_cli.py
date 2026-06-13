from pathlib import Path

import pytest
from typer.testing import CliRunner


def test_arxiv_zh_entry_rejects_non_deepseek_provider():
    from arxiv_translate.cli import zh_app

    runner = CliRunner()

    result = runner.invoke(
        zh_app,
        ["2501.12345", "--provider", "openai", "--output", "out"],
    )

    assert result.exit_code == 1
    assert "only supports --provider deepseek" in result.stdout


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


def test_arxiv_zh_options_require_deepseek_key(monkeypatch, tmp_path: Path):
    import arxiv_translate.cli as cli_module

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(cli_module, "_arxiv_zh_dotenv_paths", lambda: [])

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        cli_module._resolve_arxiv_zh_options(
            provider="deepseek",
            output=tmp_path / "paper",
            config=None,
            concurrency=3,
        )


def test_arxiv_zh_options_load_deepseek_key_from_dotenv(monkeypatch, tmp_path: Path):
    import arxiv_translate.cli as cli_module

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("DEEPSEEK_API_KEY=sk-dotenv-test\n", encoding="utf-8")
    monkeypatch.setattr(cli_module, "_project_root", lambda: tmp_path)

    options = cli_module._resolve_arxiv_zh_options(
        provider="deepseek",
        output=tmp_path / "paper",
        config=None,
        concurrency=3,
    )

    assert options.api_key == "sk-dotenv-test"


def test_arxiv_zh_options_prefers_shell_env_over_dotenv(monkeypatch, tmp_path: Path):
    import arxiv_translate.cli as cli_module

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-shell-test")
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("DEEPSEEK_API_KEY=sk-dotenv-test\n", encoding="utf-8")
    monkeypatch.setattr(cli_module, "_project_root", lambda: tmp_path)

    options = cli_module._resolve_arxiv_zh_options(
        provider="deepseek",
        output=tmp_path / "paper",
        config=None,
        concurrency=3,
    )

    assert options.api_key == "sk-shell-test"


def test_arxiv_zh_options_store_cli_font_overrides(monkeypatch, tmp_path: Path):
    from arxiv_translate.cli import _resolve_arxiv_zh_options

    font_dir = tmp_path / "fonts"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    options = _resolve_arxiv_zh_options(
        provider="deepseek",
        output=tmp_path / "paper",
        config=None,
        concurrency=3,
        font_dir=font_dir,
        cjk_main_font="STSong",
        cjk_sans_font="STXihei",
        cjk_mono_font="STKaiti",
        font_auto=False,
    )

    assert options.font_dir == font_dir
    assert options.cjk_main_font == "STSong"
    assert options.cjk_sans_font == "STXihei"
    assert options.cjk_mono_font == "STKaiti"
    assert options.font_auto is False


def test_arxiv_zh_options_store_cli_model_override(monkeypatch, tmp_path: Path):
    from arxiv_translate.cli import _resolve_arxiv_zh_options

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")

    options = _resolve_arxiv_zh_options(
        provider="deepseek",
        output=tmp_path / "paper",
        config=None,
        concurrency=3,
        model="deepseek-reasoner",
    )

    assert options.model == "deepseek-reasoner"


def test_arxiv_zh_default_config_uses_detected_cjk_fonts(monkeypatch):
    import arxiv_translate.cli as cli_module

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


def test_arxiv_zh_cli_font_overrides_apply_to_config(tmp_path: Path):
    from arxiv_translate.cli import _load_config_for_arxiv_zh

    font_dir = tmp_path / "fonts"
    font_dir.mkdir()

    config = _load_config_for_arxiv_zh(
        None,
        font_dir=font_dir,
        cjk_main_font="STSong",
        cjk_sans_font="STXihei",
        cjk_mono_font="STKaiti",
        font_auto=False,
    )

    assert config.fonts.dir == str(font_dir)
    assert config.fonts.auto_detect is False
    assert config.fonts.main == "STSong"
    assert config.fonts.sans == "STXihei"
    assert config.fonts.mono == "STKaiti"

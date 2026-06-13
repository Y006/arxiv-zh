from pathlib import Path

from arxiv_translate.rules.env import get_env_value, parse_dotenv_file


def test_get_env_value_honors_empty_dotenv_file_list(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("DEEPSEEK_API_KEY=sk-dotenv-test\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert get_env_value("DEEPSEEK_API_KEY", dotenv_files=[]) is None


def test_get_env_value_reads_explicit_dotenv_files(monkeypatch, tmp_path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("DEEPSEEK_API_KEY=sk-dotenv-test\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert get_env_value("DEEPSEEK_API_KEY", dotenv_files=[dotenv_path]) == (
        "sk-dotenv-test"
    )


def test_parse_dotenv_file_handles_export_and_quotes(tmp_path: Path):
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        'export DEEPSEEK_API_KEY="sk-dotenv-test"\n',
        encoding="utf-8",
    )

    assert parse_dotenv_file(dotenv_path)["DEEPSEEK_API_KEY"] == "sk-dotenv-test"

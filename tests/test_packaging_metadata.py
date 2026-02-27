from pathlib import Path


def test_pyproject_project_name_and_scripts_are_consistent():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'name = "arxiv-translate"' in pyproject
    assert 'arx = "arxiv_translate.cli:main"' in pyproject
    assert 'arxiv-translate = "arxiv_translate.cli:main"' in pyproject
    assert 'ieeA = "ieeA.cli:main"' not in pyproject

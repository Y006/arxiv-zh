from pathlib import Path

import yaml


def test_pyproject_project_name_and_scripts_are_consistent():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'name = "arxiv-zh"' in pyproject
    assert 'license = "GPL-3.0-or-later"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject
    assert 'Repository = "https://github.com/Y006/arxiv-zh"' in pyproject
    assert 'arx = "arxiv_translate.cli:main"' in pyproject
    assert 'arxiv-translate = "arxiv_translate.cli:main"' in pyproject
    assert 'arxiv-zh = "arxiv_translate.cli:zh_main"' in pyproject
    assert 'ieeA = "ieeA.cli:main"' not in pyproject
    assert 'include = ["arxiv_translate*"]' in pyproject
    assert "namespaces = false" in pyproject
    assert 'arxiv_translate = ["defaults/*.yaml"]' in pyproject


def test_sdist_manifest_prunes_non_runtime_repository_content():
    manifest = Path("MANIFEST.in").read_text(encoding="utf-8")

    assert "include environment.yml" in manifest
    assert "prune tests" in manifest
    assert "prune fonts" in manifest
    assert "exclude AGENTS.md" in manifest
    assert "recursive-include src/arxiv_translate/defaults *.yaml" in manifest


def test_conda_environment_file_exposes_user_runtime_dependencies():
    environment = yaml.safe_load(Path("environment.yml").read_text(encoding="utf-8"))
    dependencies = environment["dependencies"]

    assert environment["name"] == "arxiv-zh"
    assert "conda-forge" in environment["channels"]
    assert "nodefaults" in environment["channels"]
    for dependency in ("python=3.12", "uv", "r-base", "r-tinytex", "latexmk"):
        assert dependency in dependencies


def test_user_docs_are_conda_first_not_uv_run():
    readme = Path("README.md").read_text(encoding="utf-8")
    agents = Path("AGENTS.md").read_text(encoding="utf-8")

    assert "mamba env create -f environment.yml" in readme
    assert "uv pip install -e ." in readme
    assert "uv sync" not in readme
    assert "uv run arxiv-zh" not in readme
    assert "uv run arxiv-zh" not in agents
    assert "arxiv-zh --doctor --config config.yaml" in agents

from pathlib import Path


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

    assert "prune tests" in manifest
    assert "prune fonts" in manifest
    assert "exclude AGENTS.md" in manifest
    assert "exclude uv.lock" in manifest
    assert "recursive-include src/arxiv_translate/defaults *.yaml" in manifest

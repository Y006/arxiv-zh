import importlib


def test_migrate_legacy_files_to_xdg_config(monkeypatch, tmp_path):
    home = tmp_path / "home"
    xdg = tmp_path / "xdg"
    legacy_dir = home / ".ieeA"

    legacy_dir.mkdir(parents=True)
    xdg.mkdir(parents=True)

    (legacy_dir / "config.yaml").write_text("llm:\n  sdk: openai\n", encoding="utf-8")
    (legacy_dir / "glossary.yaml").write_text("AI: 人工智能\n", encoding="utf-8")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

    paths_mod = importlib.import_module("arxiv_translate.rules.user_paths")
    paths_mod = importlib.reload(paths_mod)

    migrated = paths_mod.migrate_legacy_files()
    new_dir = xdg / "arxiv-translate"

    assert (new_dir / "config.yaml").exists()
    assert (new_dir / "glossary.yaml").exists()
    assert all(path.parent == new_dir for path in migrated)


def test_resolve_user_file_prefers_new_then_legacy(monkeypatch, tmp_path):
    home = tmp_path / "home"
    xdg = tmp_path / "xdg"
    legacy_dir = home / ".ieeA"
    new_dir = xdg / "arxiv-translate"

    legacy_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)

    legacy = legacy_dir / "config.yaml"
    modern = new_dir / "config.yaml"
    legacy.write_text("old", encoding="utf-8")
    modern.write_text("new", encoding="utf-8")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

    paths_mod = importlib.import_module("arxiv_translate.rules.user_paths")
    paths_mod = importlib.reload(paths_mod)

    assert paths_mod.resolve_user_file("config.yaml") == modern

    modern.unlink()
    assert paths_mod.resolve_user_file("config.yaml") == legacy

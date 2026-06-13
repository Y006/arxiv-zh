import importlib


def test_get_config_dir_uses_xdg_config_home(monkeypatch, tmp_path):
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

    paths_mod = importlib.import_module("arxiv_translate.rules.user_paths")
    paths_mod = importlib.reload(paths_mod)

    assert paths_mod.get_config_dir() == xdg / "arxiv-translate"


def test_resolve_user_file_stays_under_current_config_dir(monkeypatch, tmp_path):
    xdg = tmp_path / "xdg"
    home = tmp_path / "home"
    legacy_dir = home / ".ieeA"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "config.yaml").write_text("old", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

    paths_mod = importlib.import_module("arxiv_translate.rules.user_paths")
    paths_mod = importlib.reload(paths_mod)

    assert paths_mod.resolve_user_file("config.yaml") == (
        xdg / "arxiv-translate" / "config.yaml"
    )


def test_ensure_config_dir_creates_current_config_dir(monkeypatch, tmp_path):
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))

    paths_mod = importlib.import_module("arxiv_translate.rules.user_paths")
    paths_mod = importlib.reload(paths_mod)

    config_dir = paths_mod.ensure_config_dir()

    assert config_dir == xdg / "arxiv-translate"
    assert config_dir.is_dir()

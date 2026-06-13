from __future__ import annotations

import os
from pathlib import Path

APP_DIR_NAME = "arxiv-translate"


def get_config_dir() -> Path:
    """Return the new user config directory."""
    xdg_config_home = Path(
        os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    )
    return xdg_config_home / APP_DIR_NAME


def ensure_config_dir() -> Path:
    """Ensure new config directory exists and return it."""
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def resolve_user_file(filename: str) -> Path:
    """Resolve a user configuration file under the current app config dir."""
    return get_config_dir() / filename

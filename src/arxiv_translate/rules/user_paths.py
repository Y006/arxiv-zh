from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path

APP_DIR_NAME = "arxiv-translate"
LEGACY_DIR_NAME = ".ieeA"
MIGRATION_FILES = ("config.yaml", "glossary.yaml", "examples.yaml")

_migration_checked = False


def get_config_dir() -> Path:
    """Return the new user config directory."""
    xdg_config_home = Path(
        os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    )
    return xdg_config_home / APP_DIR_NAME


def get_legacy_config_dir() -> Path:
    """Return the legacy user config directory."""
    return Path.home() / LEGACY_DIR_NAME


def ensure_config_dir() -> Path:
    """Ensure new config directory exists and return it."""
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def migrate_legacy_files() -> list[Path]:
    """Copy legacy user files to the new config directory once."""
    global _migration_checked
    if _migration_checked:
        return []
    _migration_checked = True

    legacy_dir = get_legacy_config_dir()
    if not legacy_dir.exists():
        return []

    config_dir = ensure_config_dir()
    migrated: list[Path] = []
    for filename in MIGRATION_FILES:
        legacy_file = legacy_dir / filename
        new_file = config_dir / filename
        if legacy_file.exists() and not new_file.exists():
            shutil.copy2(legacy_file, new_file)
            migrated.append(new_file)

    if migrated:
        warnings.warn(
            "Migrated legacy user files from ~/.ieeA to ~/.config/arxiv-translate.",
            UserWarning,
            stacklevel=2,
        )
    return migrated


def resolve_user_file(filename: str) -> Path:
    """Resolve user file path with new-dir priority and legacy fallback."""
    migrate_legacy_files()

    new_file = get_config_dir() / filename
    if new_file.exists():
        return new_file

    legacy_file = get_legacy_config_dir() / filename
    if legacy_file.exists():
        return legacy_file

    return new_file


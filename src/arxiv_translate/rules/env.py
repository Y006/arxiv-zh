import os
from pathlib import Path
from typing import Iterable, Optional


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def dotenv_paths(*, cwd: Optional[Path] = None) -> list[Path]:
    paths = [project_root() / ".env"]
    current_dir = Path(cwd or Path.cwd()).resolve()
    cwd_env = current_dir / ".env"
    if cwd_env not in paths:
        paths.append(cwd_env)
    return paths


def parse_dotenv_file(path: Path) -> dict[str, str]:
    if not path.exists() or not path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def get_env_value(
    key: str,
    *,
    dotenv_files: Optional[Iterable[Path]] = None,
) -> Optional[str]:
    env_value = os.getenv(key)
    if env_value:
        return env_value

    files = dotenv_paths() if dotenv_files is None else dotenv_files
    for dotenv_file in files:
        dotenv_value = parse_dotenv_file(Path(dotenv_file)).get(key)
        if dotenv_value:
            return dotenv_value
    return None

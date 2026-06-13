from pathlib import Path
from typing import Optional

from arxiv_translate.rules.env import get_env_value

from .openai_provider import OpenAIProvider


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _deepseek_dotenv_paths() -> list[Path]:
    paths = [_project_root() / ".env"]
    cwd_env = Path.cwd().resolve() / ".env"
    if cwd_env not in paths:
        paths.append(cwd_env)
    return paths


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek chat provider using the OpenAI-compatible API."""

    DEFAULT_BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-v4-flash"
    DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        **kwargs,
    ):
        resolved_key = api_key or get_env_value(
            api_key_env,
            dotenv_files=_deepseek_dotenv_paths(),
        )
        if not resolved_key:
            raise ValueError(
                f"{api_key_env} is required for DeepSeekProvider. "
                f"Export it or put it in .env before running arxiv-zh."
            )

        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.api_key_env = api_key_env
        super().__init__(
            model=model or self.DEFAULT_MODEL,
            api_key=resolved_key,
            base_url=self.base_url,
            **kwargs,
        )

    def __repr__(self) -> str:
        return (
            "DeepSeekProvider("
            f"model={self.model!r}, "
            f"base_url={self.base_url!r}, "
            f"api_key={self._masked_api_key()!r})"
        )

    def _masked_api_key(self) -> str:
        if not self.api_key:
            return "<missing>"
        if len(self.api_key) <= 8:
            return "****"
        return f"{self.api_key[:4]}...{self.api_key[-4:]}"

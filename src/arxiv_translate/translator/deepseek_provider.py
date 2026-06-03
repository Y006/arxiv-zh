import os
from typing import Optional

from .openai_provider import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek chat provider using the OpenAI-compatible API."""

    DEFAULT_BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-chat"
    DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        **kwargs,
    ):
        resolved_key = api_key or os.getenv(api_key_env)
        if not resolved_key:
            raise ValueError(
                f"{api_key_env} is required for DeepSeekProvider. "
                f"Set it in the environment before running arxiv-zh."
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

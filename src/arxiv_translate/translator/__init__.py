import os
from typing import Optional, Any
from urllib.parse import urlsplit, urlunsplit
from .llm_base import LLMProvider
from .openai_provider import OpenAIProvider
from .openai_coding_provider import OpenAICodingProvider
from .anthropic_provider import AnthropicProvider
from .anthropic_coding_provider import AnthropicCodingProvider
from .http_provider import DirectHTTPProvider
from .ark_provider import ArkProvider
from .deepseek_provider import DeepSeekProvider

DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"


def _normalize_openai_base_url(endpoint: Optional[str]) -> Optional[str]:
    """Normalize OpenAI-compatible base URL to avoid duplicate route suffix."""
    if not endpoint:
        return endpoint
    endpoint = endpoint.rstrip("/")
    suffix = "/chat/completions"
    if endpoint.endswith(suffix):
        return endpoint[: -len(suffix)]
    return endpoint


def _normalize_ark_base_url(endpoint: Optional[str]) -> Optional[str]:
    """Normalize Ark endpoint to API base URL.

    Supports:
    - https://ark.xx.volces.com/api/v3
    - https://ark.xx.volces.com/api/v3/chat/completions
    """
    if not endpoint:
        return endpoint

    parsed = urlsplit(endpoint)
    if not parsed.scheme or not parsed.netloc:
        return endpoint.rstrip("/")

    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")]
    normalized = urlunsplit((parsed.scheme, parsed.netloc, path or "/", "", ""))
    return normalized.rstrip("/")


def _normalize_anthropic_base_url(endpoint: Optional[str]) -> Optional[str]:
    if not endpoint:
        return endpoint
    normalized = endpoint.rstrip("/")
    for suffix in ("/v1/messages", "/v1", "/messages"):
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def is_ark_endpoint(endpoint: Optional[str]) -> bool:
    """Return True when endpoint host strictly matches ark.*.volces.com."""
    if not endpoint:
        return False
    try:
        parsed = urlsplit(endpoint)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    if not host.startswith("ark."):
        return False
    if not host.endswith(".volces.com"):
        return False
    middle = host[len("ark.") : -len(".volces.com")]
    return bool(middle)


def should_use_ark_autoroute(sdk: Optional[str], endpoint: Optional[str]) -> bool:
    return sdk in ("openai", "openai-coding", None) and is_ark_endpoint(endpoint)


def get_sdk_client(
    sdk: Optional[str],
    model: str,
    key: Optional[str] = None,
    endpoint: Optional[str] = None,
    **kwargs: Any,
) -> LLMProvider:
    """
    Factory function to get an LLM SDK client instance.

    Args:
        sdk: The SDK to use (openai, openai-coding, anthropic, anthropic-coding, bailian, or None for direct HTTP).
        model: The model name to use.
        key: Optional API key.
        endpoint: Optional API endpoint URL.
        **kwargs: Additional keyword arguments to pass to the provider constructor.

    Returns:
        An instance of LLMProvider.
    """
    if sdk == "ark":
        raise ValueError(
            "sdk=ark has been removed. Please use openai-style config "
            "with an Ark endpoint (ark.*.volces.com); Ark routing is now automatic."
        )

    if should_use_ark_autoroute(sdk, endpoint):
        normalized_endpoint = _normalize_ark_base_url(endpoint)
        return ArkProvider(model=model, api_key=key, base_url=normalized_endpoint, **kwargs)

    if sdk == "openai":
        normalized_endpoint = _normalize_openai_base_url(endpoint)
        return OpenAIProvider(
            model=model, api_key=key, base_url=normalized_endpoint, **kwargs
        )
    elif sdk == "deepseek":
        return DeepSeekProvider(
            model=model or DEEPSEEK_DEFAULT_MODEL,
            api_key=key or os.getenv("DEEPSEEK_API_KEY"),
            base_url=endpoint or DEEPSEEK_DEFAULT_BASE_URL,
            **kwargs,
        )
    elif sdk == "openai-coding":
        normalized_endpoint = _normalize_openai_base_url(endpoint)
        return OpenAICodingProvider(
            model=model, api_key=key, base_url=normalized_endpoint, **kwargs
        )
    elif sdk == "anthropic":
        normalized_endpoint = _normalize_anthropic_base_url(endpoint)
        return AnthropicProvider(
            model=model,
            api_key=key,
            base_url=normalized_endpoint,
            **kwargs,
        )
    elif sdk == "anthropic-coding":
        normalized_endpoint = _normalize_anthropic_base_url(endpoint)
        return AnthropicCodingProvider(
            model=model,
            api_key=key,
            base_url=normalized_endpoint,
            **kwargs,
        )
    elif sdk == "bailian":
        from .bailian_provider import BailianProvider

        return BailianProvider(
            model=model,
            api_key=key,
            base_url=endpoint or "https://dashscope.aliyuncs.com/compatible-mode/v1",
            **kwargs,
        )
    elif sdk is None:
        return DirectHTTPProvider(model=model, api_key=key, endpoint=endpoint, **kwargs)
    else:
        raise ValueError(
            "Unknown sdk: "
            f"{sdk}. Supported: openai, openai-coding, anthropic, anthropic-coding, bailian, deepseek, None"
        )


__all__ = [
    "LLMProvider",
    "OpenAIProvider",
    "OpenAICodingProvider",
    "AnthropicProvider",
    "AnthropicCodingProvider",
    "DirectHTTPProvider",
    "ArkProvider",
    "BailianProvider",
    "DeepSeekProvider",
    "is_ark_endpoint",
    "should_use_ark_autoroute",
    "get_sdk_client",
]

"""Tests for Ark endpoint auto-routing and normalization."""

import pytest
import typer


def test_is_ark_endpoint_strict_match():
    """Only ark.*.volces.com endpoints should be detected as Ark."""
    from arxiv_translate.translator import is_ark_endpoint

    assert is_ark_endpoint("https://ark.cn-beijing.volces.com/api/v3")
    assert is_ark_endpoint("https://ark.cn-beijing.volces.com/api/v3/chat/completions")
    assert not is_ark_endpoint("https://cn-beijing.volces.com/api/v3")
    assert not is_ark_endpoint("https://foo.ark.cn-beijing.volces.com/api/v3")
    assert not is_ark_endpoint("https://ark.cn-beijing.volcengine.com/api/v3")


def test_factory_rejects_legacy_ark_sdk():
    """sdk=ark should be rejected with migration guidance."""
    from arxiv_translate.translator import get_sdk_client

    with pytest.raises(ValueError, match="sdk=ark"):
        get_sdk_client(
            "ark",
            model="ep-test",
            key="test-key",
            endpoint="https://ark.cn-beijing.volces.com/api/v3",
        )


def test_factory_autoroutes_openai_to_ark_provider(monkeypatch):
    """openai sdk should auto-route to ArkProvider on Ark endpoint."""
    import arxiv_translate.translator as translator_mod

    class FakeArkProvider:
        def __init__(self, model, api_key=None, base_url=None, **kwargs):
            self.model = model
            self.api_key = api_key
            self.base_url = base_url
            self.kwargs = kwargs

    monkeypatch.setattr(translator_mod, "ArkProvider", FakeArkProvider)

    provider = translator_mod.get_sdk_client(
        "openai",
        model="ep-test",
        key="test-key",
        endpoint="https://ark.cn-beijing.volces.com/api/v3/chat/completions",
        temperature=0.2,
    )

    assert isinstance(provider, FakeArkProvider)
    assert provider.base_url == "https://ark.cn-beijing.volces.com/api/v3"
    assert provider.kwargs["temperature"] == 0.2


def test_factory_autoroutes_openai_coding_to_ark_provider(monkeypatch):
    """openai-coding sdk should auto-route to ArkProvider on Ark endpoint."""
    import arxiv_translate.translator as translator_mod

    class FakeArkProvider:
        def __init__(self, model, api_key=None, base_url=None, **kwargs):
            self.model = model
            self.api_key = api_key
            self.base_url = base_url
            self.kwargs = kwargs

    monkeypatch.setattr(translator_mod, "ArkProvider", FakeArkProvider)

    provider = translator_mod.get_sdk_client(
        "openai-coding",
        model="ep-test",
        key="test-key",
        endpoint="https://ark.cn-beijing.volces.com/api/v3",
    )

    assert isinstance(provider, FakeArkProvider)
    assert provider.base_url == "https://ark.cn-beijing.volces.com/api/v3"


def test_factory_autoroutes_null_sdk_to_ark_provider(monkeypatch):
    """null sdk should auto-route to ArkProvider on Ark endpoint."""
    import arxiv_translate.translator as translator_mod

    class FakeArkProvider:
        def __init__(self, model, api_key=None, base_url=None, **kwargs):
            self.model = model
            self.api_key = api_key
            self.base_url = base_url
            self.kwargs = kwargs

    monkeypatch.setattr(translator_mod, "ArkProvider", FakeArkProvider)

    provider = translator_mod.get_sdk_client(
        None,
        model="ep-test",
        key="test-key",
        endpoint="https://ark.cn-beijing.volces.com/api/v3",
    )

    assert isinstance(provider, FakeArkProvider)
    assert provider.base_url == "https://ark.cn-beijing.volces.com/api/v3"


def test_config_rejects_legacy_ark_sdk():
    """LLMConfig should reject sdk=ark after migration."""
    from arxiv_translate.rules.config import LLMConfig

    with pytest.raises(ValueError, match="sdk must be"):
        LLMConfig(sdk="ark")


def test_cli_validation_rejects_legacy_ark_sdk():
    """CLI argument validation should reject sdk=ark early."""
    from arxiv_translate.cli import _validate_provider_args

    with pytest.raises(typer.Exit):
        _validate_provider_args(
            sdk_name="ark",
            key_val="test-key",
            endpoint_val="https://ark.cn-beijing.volces.com/api/v3",
        )


def test_cli_validation_requires_key_for_ark_autoroute():
    """Ark auto-route should require key even when sdk is null."""
    from arxiv_translate.cli import _validate_provider_args

    with pytest.raises(typer.Exit):
        _validate_provider_args(
            sdk_name=None,
            key_val=None,
            endpoint_val="https://ark.cn-beijing.volces.com/api/v3",
        )


def test_cli_validation_allows_null_sdk_without_ark_endpoint():
    """null sdk + non-Ark endpoint should not require key."""
    from arxiv_translate.cli import _validate_provider_args

    _validate_provider_args(
        sdk_name=None,
        key_val=None,
        endpoint_val="https://openrouter.ai/api/v1/chat/completions",
    )

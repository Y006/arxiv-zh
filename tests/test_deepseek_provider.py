import pytest


def test_deepseek_provider_uses_env_key_and_defaults(monkeypatch):
    from arxiv_translate.translator.deepseek_provider import DeepSeekProvider

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")

    provider = DeepSeekProvider()

    assert provider.model == "deepseek-chat"
    assert provider.api_key == "sk-deepseek-test"
    assert provider.base_url == "https://api.deepseek.com"


def test_deepseek_provider_uses_dotenv_key(monkeypatch, tmp_path):
    import arxiv_translate.translator.deepseek_provider as provider_mod

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        'export DEEPSEEK_API_KEY="sk-dotenv-test"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(provider_mod, "_project_root", lambda: tmp_path)

    provider = provider_mod.DeepSeekProvider()

    assert provider.api_key == "sk-dotenv-test"


def test_deepseek_provider_requires_env_key(monkeypatch):
    import arxiv_translate.translator.deepseek_provider as provider_mod

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(provider_mod, "_deepseek_dotenv_paths", lambda: [])

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        provider_mod.DeepSeekProvider()


def test_deepseek_provider_masks_key_in_repr(monkeypatch):
    from arxiv_translate.translator.deepseek_provider import DeepSeekProvider

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-secret-value")

    provider = DeepSeekProvider()

    assert "sk-deepseek-secret-value" not in repr(provider)
    assert "sk-d..." in repr(provider)


def test_factory_builds_deepseek_provider(monkeypatch):
    import arxiv_translate.translator as translator_mod

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")

    class FakeDeepSeekProvider:
        def __init__(self, model, api_key=None, base_url=None, **kwargs):
            self.model = model
            self.api_key = api_key
            self.base_url = base_url
            self.kwargs = kwargs

    monkeypatch.setattr(translator_mod, "DeepSeekProvider", FakeDeepSeekProvider)

    provider = translator_mod.get_sdk_client(
        "deepseek",
        model="deepseek-chat",
        key=None,
        endpoint=None,
        temperature=0.1,
    )

    assert isinstance(provider, FakeDeepSeekProvider)
    assert provider.api_key == "sk-deepseek-test"
    assert provider.base_url == "https://api.deepseek.com"
    assert provider.kwargs["temperature"] == 0.1

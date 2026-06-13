import pytest

from arxiv_translate.rules.config import CompilationConfig


def test_compilation_config_exposes_compile_fallback_defaults():
    config = CompilationConfig()

    assert config.engine_policy == "auto"
    assert config.fallback_engines == ["xelatex", "lualatex"]
    assert config.allow_pdflatex_cjk is False
    assert config.allow_shell_escape is False
    assert config.max_repair_rounds == 3
    assert config.chinese_package == "auto"


def test_compilation_config_rejects_unknown_engine_policy():
    with pytest.raises(ValueError, match="engine_policy"):
        CompilationConfig(engine_policy="tectonic")

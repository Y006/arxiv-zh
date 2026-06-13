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
    assert config.timeout == 600
    assert config.prefer_latexmk is True
    assert config.use_tinytex is True
    assert config.install_missing_packages is True
    assert config.install_timeout == 1200
    assert config.max_package_install_rounds == 8
    assert any("TinyTeX" in path for path in config.tinytex_paths)


def test_compilation_config_rejects_unknown_engine_policy():
    with pytest.raises(ValueError, match="engine_policy"):
        CompilationConfig(engine_policy="tectonic")


def test_compilation_config_rejects_invalid_timeouts():
    with pytest.raises(ValueError, match="positive"):
        CompilationConfig(timeout=0)
    with pytest.raises(ValueError, match="positive"):
        CompilationConfig(install_timeout=0)

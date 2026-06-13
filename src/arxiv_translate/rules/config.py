import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator

from arxiv_translate.rules.user_paths import resolve_user_file


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMConfig(StrictConfigModel):
    sdk: Optional[str] = "deepseek"
    models: Union[str, List[str]] = "deepseek-chat"
    key: Optional[str] = None
    key_env: Optional[str] = "DEEPSEEK_API_KEY"
    endpoint: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4000

    @field_validator("sdk")
    @classmethod
    def validate_sdk(cls, v):
        if v is not None and v not in (
            "openai",
            "openai-coding",
            "anthropic",
            "anthropic-coding",
            "bailian",
            "deepseek",
        ):
            raise ValueError(
                "sdk must be 'openai', 'openai-coding', 'anthropic', "
                f"'anthropic-coding', 'bailian', 'deepseek', or None, got '{v}'"
            )
        return v

    @field_validator("models")
    @classmethod
    def validate_models(cls, v):
        if isinstance(v, str) and not v.strip():
            raise ValueError("models cannot be empty")
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("models list cannot be empty")
        return v

    @field_validator("key_env")
    @classmethod
    def validate_key_env(cls, v):
        if v is not None and not v.strip():
            raise ValueError("key_env cannot be empty")
        return v

    def get_model(self) -> str:
        if isinstance(self.models, list):
            return self.models[0]
        return self.models


class CompilationConfig(StrictConfigModel):
    enabled: bool = False
    engine: str = "xelatex"
    timeout: int = 120
    clean_aux: bool = True
    engine_policy: str = "auto"
    fallback_engines: List[str] = Field(default_factory=lambda: ["xelatex", "lualatex"])
    allow_pdflatex_cjk: bool = False
    allow_shell_escape: bool = False
    max_repair_rounds: int = 3
    chinese_package: str = "auto"

    @field_validator("engine_policy")
    @classmethod
    def validate_engine_policy(cls, v):
        allowed = {"auto", "xelatex", "lualatex", "pdflatex"}
        if v not in allowed:
            raise ValueError(
                f"engine_policy must be one of {sorted(allowed)}, got '{v}'"
            )
        return v

    @field_validator("fallback_engines")
    @classmethod
    def validate_fallback_engines(cls, v):
        allowed = {"xelatex", "lualatex", "pdflatex"}
        invalid = [engine for engine in v if engine not in allowed]
        if invalid:
            raise ValueError(
                f"fallback_engines contains unsupported engine(s): {invalid}"
            )
        return v

    @field_validator("chinese_package")
    @classmethod
    def validate_chinese_package(cls, v):
        allowed = {"auto", "xeCJK", "luatexja", "ctex", "CJKutf8"}
        if v not in allowed:
            raise ValueError(
                f"chinese_package must be one of {sorted(allowed)}, got '{v}'"
            )
        return v

    @field_validator("max_repair_rounds")
    @classmethod
    def validate_max_repair_rounds(cls, v):
        if v < 0:
            raise ValueError("max_repair_rounds must be non-negative")
        return v


class PathsConfig(StrictConfigModel):
    output_dir: str = "output"
    cache_dir: str = ".cache"


class FontConfig(StrictConfigModel):
    """CJK font configuration."""

    main: Optional[str] = None
    sans: Optional[str] = None
    mono: Optional[str] = None
    dir: Optional[str] = None
    auto_detect: bool = True


class TranslationConfig(StrictConfigModel):
    custom_system_prompt: Optional[str] = None
    custom_user_prompt: Optional[str] = None
    preserve_terms: List[str] = Field(default_factory=list)
    quality_mode: str = "standard"
    examples_path: Optional[str] = None
    batch_short_threshold: int = 300
    batch_max_chars: int = 2000
    concurrency: int = 3
    max_chunks: Optional[int] = None

    @field_validator("concurrency")
    @classmethod
    def validate_concurrency(cls, v):
        if v < 1:
            raise ValueError("concurrency must be at least 1")
        return v

    @field_validator("max_chunks")
    @classmethod
    def validate_max_chunks(cls, v):
        if v is not None and v < 1:
            raise ValueError("max_chunks must be at least 1 when provided")
        return v


class ParserConfig(StrictConfigModel):
    """LaTeX parser configuration."""

    extra_protected_environments: List[str] = Field(default_factory=list)
    extra_translatable_environments: List[str] = Field(default_factory=list)


class CacheConfig(StrictConfigModel):
    enabled: bool = True
    max_size_mb: int = 2048
    ttl_days: int = 30
    compression: str = "zstd"
    key_mode: str = "relaxed_chunk"


class Config(StrictConfigModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    compilation: CompilationConfig = Field(default_factory=CompilationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    fonts: FontConfig = Field(default_factory=FontConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)


def load_defaults() -> Dict[str, Any]:
    """Load default configuration from the package."""
    base_path = Path(__file__).parent.parent
    default_path = base_path / "defaults" / "config.yaml"

    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_user_config() -> Dict[str, Any]:
    """Load user configuration from ~/.config/arxiv-translate/config.yaml."""
    config_path = resolve_user_file("config.yaml")

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = base.copy()
    for k, v in update.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config() -> Config:
    """Load and merge configuration from defaults and user overrides."""
    config_data = load_defaults()
    user_data = load_user_config()
    merged_data = deep_merge(config_data, user_data)
    return Config(**merged_data)

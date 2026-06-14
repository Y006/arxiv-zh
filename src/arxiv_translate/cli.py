import asyncio
import importlib.metadata as metadata
import json
import os
import re
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from arxiv_translate.cache.local_translation_cache import LocalTranslationCache
from arxiv_translate.compiler import LaTeXCompiler
from arxiv_translate.compiler.chinese_support import (
    detect_cjk_fonts,
    find_font_files,
    get_available_fonts,
)
from arxiv_translate.downloader.arxiv import ArxivDownloader, DownloadResult
from arxiv_translate.parser.latex_parser import LaTeXParser
from arxiv_translate.rules.config import Config, deep_merge, load_config, load_defaults
from arxiv_translate.rules.env import get_env_value
from arxiv_translate.rules.glossary import load_glossary
from arxiv_translate.rules.examples import load_examples
from arxiv_translate.rules.user_paths import ensure_config_dir
from arxiv_translate.translator import get_sdk_client, should_use_ark_autoroute
from arxiv_translate.translator.pipeline import TranslationPipeline, TranslatedChunk
from arxiv_translate.translator.postprocess import sanitize_markdown_bold_safe
from arxiv_translate.parser.structure import validate_translated_placeholders
from arxiv_translate.validator.engine import ValidationEngine
from arxiv_translate.validator.rules import BuiltInRules

app = typer.Typer(
    name="arx",
    help="arxiv-translate - arXiv Paper Translator",
    add_completion=False,
    no_args_is_help=False,
)
zh_app = typer.Typer(
    name="arxiv-zh",
    help="arxiv-zh - local DeepSeek-powered arXiv paper translator",
    add_completion=False,
    no_args_is_help=True,
)
config_app = typer.Typer(help="Manage configuration")
glossary_app = typer.Typer(help="Manage glossary terms")
cache_app = typer.Typer(help="Manage local translation cache")
app.add_typer(config_app, name="config")
app.add_typer(glossary_app, name="glossary")
app.add_typer(cache_app, name="cache")

console = Console()


@dataclass
class ArxivZhOutputLayout:
    root: Path
    source_dir: Path
    translated_dir: Path
    pdf_dir: Path
    cache_dir: Path
    logs_dir: Path
    translate_log: Path
    compile_log: Path
    metadata: Path


@dataclass
class ArxivZhOptions:
    output: Path
    config: Optional[Path]
    concurrency: int
    api_key: str
    model: str
    endpoint: str
    compile_pdf: bool
    max_chunks: Optional[int] = None


@dataclass
class ArxivZhDoctorCheck:
    name: str
    status: str
    message: str


def _append_text_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(message.rstrip() + "\n")


def _prepare_arxiv_zh_output_dirs(output: Path) -> ArxivZhOutputLayout:
    root = Path(output).expanduser().resolve()
    layout = ArxivZhOutputLayout(
        root=root,
        source_dir=root / "source",
        translated_dir=root / "translated",
        pdf_dir=root / "pdf",
        cache_dir=root / "cache",
        logs_dir=root / "logs",
        translate_log=root / "logs" / "translate.log",
        compile_log=root / "logs" / "compile.log",
        metadata=root / "metadata.json",
    )
    for path in (
        layout.source_dir,
        layout.translated_dir,
        layout.pdf_dir,
        layout.cache_dir,
        layout.logs_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return layout


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _project_font_dir() -> Path:
    return _project_root() / "fonts"


def _arxiv_zh_dotenv_paths() -> list[Path]:
    paths = [_project_root() / ".env"]
    cwd_env = Path.cwd().resolve() / ".env"
    if cwd_env not in paths:
        paths.append(cwd_env)
    return paths


def _resolve_path_from_config(path_value: str | Path, config_path: Optional[Path]) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if config_path is not None:
        return (config_path.expanduser().resolve().parent / path).resolve()
    return (Path.cwd().resolve() / path).resolve()


def _resolve_arxiv_zh_api_key(config: Config) -> tuple[Optional[str], str]:
    api_key_env = config.llm.key_env or "DEEPSEEK_API_KEY"
    api_key = config.llm.key or get_env_value(
        api_key_env,
        dotenv_files=_arxiv_zh_dotenv_paths(),
    )
    return api_key, api_key_env


def _split_arxiv_id_version(arxiv_id: str) -> tuple[str, Optional[str]]:
    match = re.match(r"^(?P<base>\d{4}\.\d{4,5})(?P<version>v\d+)?$", arxiv_id)
    if match:
        return match.group("base"), match.group("version")
    return arxiv_id, None


def _arxiv_zh_output_dir_name(arxiv_id_or_url: str) -> str:
    parsed_id = ArxivDownloader.parse_id(arxiv_id_or_url)
    return f"arxiv-{parsed_id.replace('/', '_')}"


def _arxiv_zh_metadata_path(output_dir: Path) -> Path:
    return output_dir / "metadata.json"


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _metadata_arxiv_identity(
    metadata_data: dict[str, Any],
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    arxiv_data = metadata_data.get("arxiv")
    if not isinstance(arxiv_data, dict):
        return None, None, None
    arxiv_id = arxiv_data.get("id")
    base_id = arxiv_data.get("base_id")
    version = arxiv_data.get("version")
    return (
        str(arxiv_id) if arxiv_id else None,
        str(base_id) if base_id else None,
        str(version) if version else None,
    )


def _arxiv_identity_compatible(
    *,
    requested_id: str,
    existing_id: Optional[str],
    existing_base_id: Optional[str],
    existing_version: Optional[str],
) -> bool:
    if existing_id == requested_id:
        return True

    requested_base_id, requested_version = _split_arxiv_id_version(requested_id)
    if existing_base_id != requested_base_id:
        return False

    return requested_version is None or existing_version is None


def _find_existing_arxiv_zh_output_dir(
    output_root: Path,
    parsed_id: str,
) -> Optional[Path]:
    if not output_root.exists():
        return None

    exact_dir = output_root / _arxiv_zh_output_dir_name(parsed_id)
    if exact_dir.exists():
        return exact_dir

    for candidate in sorted(output_root.glob("arxiv-*")):
        if not candidate.is_dir():
            continue
        metadata_data = _load_json_file(_arxiv_zh_metadata_path(candidate))
        existing_id, existing_base_id, existing_version = _metadata_arxiv_identity(
            metadata_data
        )
        if _arxiv_identity_compatible(
            requested_id=parsed_id,
            existing_id=existing_id,
            existing_base_id=existing_base_id,
            existing_version=existing_version,
        ):
            return candidate

        candidate_id = candidate.name.removeprefix("arxiv-").replace("_", "/")
        candidate_base_id, candidate_version = _split_arxiv_id_version(candidate_id)
        if _arxiv_identity_compatible(
            requested_id=parsed_id,
            existing_id=candidate_id,
            existing_base_id=candidate_base_id,
            existing_version=candidate_version,
        ):
            return candidate

    return None


def _resolve_arxiv_zh_output_dir(
    *,
    arxiv_id_or_url: str,
    output_root: Path,
) -> Path:
    parsed_id = ArxivDownloader.parse_id(arxiv_id_or_url)
    existing = _find_existing_arxiv_zh_output_dir(output_root, parsed_id)
    if existing is not None:
        return existing
    return output_root / _arxiv_zh_output_dir_name(parsed_id)


def _load_config_for_arxiv_zh(config_path: Optional[Path]) -> Config:
    if config_path is None:
        config = load_config()
    else:
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as config_file:
            user_data = yaml.safe_load(config_file) or {}
        config = Config(**deep_merge(load_defaults(), user_data))

    resolved_font_dir = None
    configured_dir = getattr(config.fonts, "dir", None)
    if configured_dir:
        configured_path = Path(configured_dir).expanduser()
        if (
            config_path is None
            and not configured_path.is_absolute()
            and configured_path == Path("fonts")
            and _project_font_dir().exists()
        ):
            resolved_font_dir = _project_font_dir().resolve()
        else:
            resolved_font_dir = _resolve_path_from_config(configured_path, config_path)

    project_font_dir = _project_font_dir()
    if resolved_font_dir is None and project_font_dir.exists():
        resolved_font_dir = project_font_dir

    if resolved_font_dir is not None:
        config.fonts.dir = str(resolved_font_dir)

    if config_path is None:
        config.fonts.auto_detect = True

    if config.fonts.auto_detect:
        detected_fonts = detect_cjk_fonts(get_available_fonts(font_dir=resolved_font_dir))
        if detected_fonts:
            config.fonts.main = detected_fonts["main"]
            config.fonts.sans = detected_fonts["sans"]
            config.fonts.mono = detected_fonts["mono"]

    return config


def _resolve_arxiv_zh_options(
    *,
    arxiv_id: str,
    config: Optional[Path],
    require_api_key: bool = True,
) -> tuple[ArxivZhOptions, Config]:
    resolved_config = _load_config_for_arxiv_zh(config)

    if resolved_config.llm.sdk != "deepseek":
        raise ValueError("arxiv-zh only supports llm.sdk: deepseek in config.")

    api_key, api_key_env = _resolve_arxiv_zh_api_key(resolved_config)
    if require_api_key and not api_key:
        raise ValueError(
            f"{api_key_env} is required. Export it, put it in .env, or set "
            "llm.key in the config before running arxiv-zh."
        )

    output_root = _resolve_path_from_config(resolved_config.paths.output_dir, config)
    output = _resolve_arxiv_zh_output_dir(
        arxiv_id_or_url=arxiv_id,
        output_root=output_root,
    )

    options = ArxivZhOptions(
        output=output,
        config=config,
        concurrency=resolved_config.translation.concurrency,
        api_key=api_key or "",
        model=resolved_config.llm.get_model(),
        endpoint=resolved_config.llm.endpoint or "https://api.deepseek.com",
        compile_pdf=resolved_config.compilation.enabled,
        max_chunks=resolved_config.translation.max_chunks,
    )
    return options, resolved_config


def _copy_source_tree_to_translated(source_dir: Path, translated_dir: Path) -> None:
    for item in source_dir.iterdir():
        target = translated_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _arxiv_zh_stage(status: str = "pending") -> dict[str, Any]:
    return {
        "status": status,
        "completed": status == "completed",
        "started_at": None,
        "completed_at": None,
        "error": None,
    }


def _arxiv_zh_metadata_template(
    *,
    arxiv_input: str,
    arxiv_id: str,
    layout: ArxivZhOutputLayout,
) -> dict[str, Any]:
    base_id, version = _split_arxiv_id_version(arxiv_id)
    return {
        "schema_version": 1,
        "arxiv": {
            "id": arxiv_id,
            "base_id": base_id,
            "version": version,
            "input": arxiv_input,
            "last_input": arxiv_input,
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            "source_url": f"https://arxiv.org/e-print/{arxiv_id}",
            "output_dir": str(layout.root),
            "output_dir_name": layout.root.name,
        },
        "run": {
            "status": "pending",
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "download": _arxiv_zh_stage(),
            "translation": _arxiv_zh_stage(),
            "compilation": _arxiv_zh_stage(),
        },
    }


def _load_arxiv_zh_metadata(layout: ArxivZhOutputLayout) -> dict[str, Any]:
    return _load_json_file(layout.metadata)


def _write_arxiv_zh_metadata(
    layout: ArxivZhOutputLayout,
    metadata_data: dict[str, Any],
) -> None:
    metadata_data.setdefault("schema_version", 1)
    run_data = metadata_data.setdefault("run", {})
    run_data["updated_at"] = _utc_now_iso()
    layout.metadata.write_text(
        json.dumps(metadata_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _ensure_arxiv_zh_metadata(
    layout: ArxivZhOutputLayout,
    *,
    arxiv_input: str,
    arxiv_id: str,
) -> dict[str, Any]:
    metadata_data = _load_arxiv_zh_metadata(layout)
    if not metadata_data:
        root_id = layout.root.name.removeprefix("arxiv-").replace("_", "/")
        root_base_id, root_version = _split_arxiv_id_version(root_id)
        if _arxiv_identity_compatible(
            requested_id=arxiv_id,
            existing_id=root_id,
            existing_base_id=root_base_id,
            existing_version=root_version,
        ):
            arxiv_id = root_id
        metadata_data = _arxiv_zh_metadata_template(
            arxiv_input=arxiv_input,
            arxiv_id=arxiv_id,
            layout=layout,
        )
        _write_arxiv_zh_metadata(layout, metadata_data)
        return metadata_data

    existing_id, existing_base_id, existing_version = _metadata_arxiv_identity(
        metadata_data
    )
    if not _arxiv_identity_compatible(
        requested_id=arxiv_id,
        existing_id=existing_id,
        existing_base_id=existing_base_id,
        existing_version=existing_version,
    ):
        raise ValueError(
            "Existing metadata belongs to a different arXiv paper: "
            f"{existing_id or 'unknown'}"
        )

    metadata_data.setdefault("schema_version", 1)
    arxiv_data = metadata_data.setdefault("arxiv", {})
    arxiv_data.setdefault("id", arxiv_id)
    base_id, version = _split_arxiv_id_version(str(arxiv_data.get("id") or arxiv_id))
    arxiv_data.setdefault("base_id", base_id)
    arxiv_data.setdefault("version", version)
    arxiv_data.setdefault("input", arxiv_input)
    arxiv_data["last_input"] = arxiv_input
    arxiv_data["output_dir"] = str(layout.root)
    arxiv_data["output_dir_name"] = layout.root.name
    run_data = metadata_data.setdefault("run", {})
    run_data.setdefault("status", "pending")
    run_data.setdefault("created_at", _utc_now_iso())
    for stage in ("download", "translation", "compilation"):
        run_data.setdefault(stage, _arxiv_zh_stage())
    _write_arxiv_zh_metadata(layout, metadata_data)
    return metadata_data


def _metadata_stage(metadata_data: dict[str, Any], stage: str) -> dict[str, Any]:
    run_data = metadata_data.setdefault("run", {})
    return run_data.setdefault(stage, _arxiv_zh_stage())


def _metadata_stage_completed(metadata_data: dict[str, Any], stage: str) -> bool:
    stage_data = _metadata_stage(metadata_data, stage)
    return bool(stage_data.get("completed")) and stage_data.get("status") == "completed"


def _set_metadata_stage_status(
    metadata_data: dict[str, Any],
    stage: str,
    status: str,
    **fields: Any,
) -> None:
    stage_data = _metadata_stage(metadata_data, stage)
    stage_data["status"] = status
    stage_data["completed"] = status == "completed"
    if status == "running":
        stage_data["started_at"] = _utc_now_iso()
        stage_data["error"] = None
    if status in {"completed", "failed", "skipped"}:
        stage_data["completed_at"] = _utc_now_iso()
    if status == "completed":
        stage_data["error"] = None
    if status == "failed" and "error" not in fields:
        stage_data["error"] = "unknown"
    stage_data.update(fields)


def _set_metadata_run_status(metadata_data: dict[str, Any], status: str) -> None:
    metadata_data.setdefault("run", {})["status"] = status


def _existing_source_download_result(
    layout: ArxivZhOutputLayout,
    metadata_data: dict[str, Any],
) -> Optional[DownloadResult]:
    if not _metadata_stage_completed(metadata_data, "download"):
        return None
    if not layout.source_dir.exists():
        return None

    files = [path for path in layout.source_dir.rglob("*") if path.is_file()]
    if not files:
        return None

    main_tex_value = _metadata_stage(metadata_data, "download").get("main_tex")
    main_tex = Path(str(main_tex_value)) if main_tex_value else None
    if main_tex is None or not main_tex.exists():
        try:
            main_tex = ArxivDownloader(
                cache_dir=layout.cache_dir / "downloads"
            ).find_main_tex(files)
        except Exception:
            return None

    arxiv_data = metadata_data.get("arxiv", {})
    arxiv_id = str(arxiv_data.get("id") or layout.root.name.removeprefix("arxiv-"))
    return DownloadResult(
        arxiv_id=arxiv_id,
        source_dir=layout.source_dir,
        main_tex=main_tex,
        downloaded_files=files,
    )


def _mark_existing_translation_if_present(
    layout: ArxivZhOutputLayout,
    metadata_data: dict[str, Any],
) -> None:
    translated_tex = layout.translated_dir / "main_zh.tex"
    if translated_tex.exists() and not _metadata_stage_completed(
        metadata_data,
        "translation",
    ):
        _set_metadata_stage_status(
            metadata_data,
            "translation",
            "completed",
            translated_tex=str(translated_tex),
            status_source="existing_file",
        )


def _compile_arxiv_zh_translated_tex(
    *,
    layout: ArxivZhOutputLayout,
    config: Config,
    metadata_data: dict[str, Any],
) -> Optional[Path]:
    translated_tex_path = layout.translated_dir / "main_zh.tex"
    if not translated_tex_path.exists():
        _set_metadata_stage_status(
            metadata_data,
            "compilation",
            "failed",
            error=f"Translated TeX not found: {translated_tex_path}",
        )
        _set_metadata_run_status(metadata_data, "compile_failed")
        _write_arxiv_zh_metadata(layout, metadata_data)
        raise ValueError(f"Translated TeX not found: {translated_tex_path}")

    _set_metadata_stage_status(
        metadata_data,
        "compilation",
        "running",
        translated_tex=str(translated_tex_path),
    )
    _write_arxiv_zh_metadata(layout, metadata_data)

    compiler = _arxiv_zh_compiler_from_config(config)
    result = compiler.compile_file(
        translated_tex_path,
        layout.pdf_dir / "main_zh.pdf",
        logs_dir=layout.logs_dir,
        build_dir=layout.root / "build",
        prefer_latexmk=config.compilation.prefer_latexmk,
        engine_policy=config.compilation.engine_policy,
        fallback_engines=config.compilation.fallback_engines,
        allow_pdflatex_cjk=config.compilation.allow_pdflatex_cjk,
        allow_shell_escape=config.compilation.allow_shell_escape,
        max_repair_rounds=config.compilation.max_repair_rounds,
        chinese_package=config.compilation.chinese_package,
        font_config=config.fonts,
    )
    if not result.success:
        error = result.error_message or "Compilation failed"
        _set_metadata_stage_status(
            metadata_data,
            "compilation",
            "failed",
            error=error,
            attempts_path=str(result.diagnostic_path) if result.diagnostic_path else None,
        )
        _set_metadata_run_status(metadata_data, "compile_failed")
        _write_arxiv_zh_metadata(layout, metadata_data)
        raise RuntimeError(error)

    pdf_path = result.pdf_path
    _set_metadata_stage_status(
        metadata_data,
        "compilation",
        "completed",
        pdf_path=str(pdf_path) if pdf_path else None,
        engine_used=result.engine_used,
        attempts_path=str(result.diagnostic_path) if result.diagnostic_path else None,
        repaired_tex_path=(
            str(result.repaired_tex_path) if result.repaired_tex_path else None
        ),
    )
    _set_metadata_run_status(metadata_data, "success")
    _write_arxiv_zh_metadata(layout, metadata_data)
    return pdf_path


def _run_arxiv_zh_compile_only(
    *,
    arxiv_id: str,
    options: ArxivZhOptions,
    config: Config,
) -> None:
    layout = _prepare_arxiv_zh_output_dirs(options.output)
    parsed_id = ArxivDownloader.parse_id(arxiv_id)
    metadata_data = _ensure_arxiv_zh_metadata(
        layout,
        arxiv_input=arxiv_id,
        arxiv_id=parsed_id,
    )
    _mark_existing_translation_if_present(layout, metadata_data)

    if not _metadata_stage_completed(metadata_data, "translation"):
        raise ValueError(
            "Cannot compile only before translation is completed. "
            f"Missing metadata state for {layout.translated_dir / 'main_zh.tex'}."
        )

    layout.translate_log.parent.mkdir(parents=True, exist_ok=True)
    _append_text_log(layout.translate_log, f"Starting compile-only job for {arxiv_id}")
    _compile_arxiv_zh_translated_tex(
        layout=layout,
        config=config,
        metadata_data=metadata_data,
    )
    _append_text_log(layout.translate_log, "Compile-only job completed")


def _arxiv_zh_compiler_from_config(config: Config) -> LaTeXCompiler:
    return LaTeXCompiler(
        timeout=config.compilation.timeout,
        fonts_dir=config.fonts.dir,
        use_tinytex=config.compilation.use_tinytex,
        tinytex_driver=config.compilation.tinytex_driver,
        tinytex_paths=config.compilation.tinytex_paths,
        install_missing_packages=config.compilation.install_missing_packages,
        install_timeout=config.compilation.install_timeout,
        total_timeout=config.compilation.total_timeout,
        max_package_install_rounds=config.compilation.max_package_install_rounds,
    )


def _doctor_check(name: str, status: str, message: str) -> ArxivZhDoctorCheck:
    return ArxivZhDoctorCheck(name=name, status=status, message=message)


RECOMMENDED_CONDA_ENV = "arxiv-zh"


def _conda_env_name(env: dict[str, str]) -> Optional[str]:
    conda_name = env.get("CONDA_DEFAULT_ENV")
    if conda_name:
        return Path(conda_name).name
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        return Path(conda_prefix).name
    return None


def _check_conda_environment(env: dict[str, str]) -> ArxivZhDoctorCheck:
    conda_prefix = env.get("CONDA_PREFIX")
    if not conda_prefix:
        return _doctor_check(
            "conda 环境",
            "FAIL",
            "未检测到 CONDA_PREFIX；请先运行 conda activate arxiv-zh",
        )

    env_name = _conda_env_name(env)
    if env_name != RECOMMENDED_CONDA_ENV:
        return _doctor_check(
            "conda 环境",
            "FAIL",
            f"当前环境是 {env_name or 'unknown'}；请切换到 conda activate arxiv-zh",
        )

    return _doctor_check(
        "conda 环境",
        "PASS",
        f"{env_name} ({conda_prefix})",
    )


def _probe_writable_directory(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".arxiv-zh-doctor-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True, str(path)
    except Exception as exc:
        return False, str(exc)


def _font_name_available(font_name: Optional[str], available_fonts: list[str]) -> bool:
    if not font_name:
        return False
    font_key = font_name.lower()
    return any(
        font_key == available.lower()
        or font_key in available.lower()
        or font_key in Path(available).name.lower()
        for available in available_fonts
    )


def _probe_arxiv_reachable(timeout: int = 8) -> tuple[bool, str]:
    request = urllib.request.Request(
        "https://arxiv.org",
        headers={"User-Agent": "arxiv-zh-doctor"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", None) or response.getcode()
            response.read(1)
        return True, f"HTTP {status}"
    except Exception as exc:
        return False, str(exc)


def _format_proxy_env(env: dict[str, str]) -> str:
    proxy_names = (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    )
    configured = [name for name in proxy_names if env.get(name)]
    if not configured:
        return "未设置 HTTP_PROXY / HTTPS_PROXY / ALL_PROXY"
    return "已设置：" + ", ".join(configured)


def _run_doctor_command(
    cmd: list[str],
    *,
    env: dict[str, str],
    timeout: int,
) -> tuple[bool, str]:
    try:
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
    except subprocess.TimeoutExpired:
        return False, f"命令超时（{timeout}s）：{' '.join(cmd)}"
    except Exception as exc:
        return False, str(exc)

    output = (process.stdout + "\n" + process.stderr).strip()
    if process.returncode == 0:
        return True, output or "OK"
    return False, output or f"exit code {process.returncode}"


def _probe_tlmgr_repository(
    tlmgr: str,
    *,
    env: dict[str, str],
    timeout: int = 30,
) -> tuple[bool, str]:
    ok, detail = _run_doctor_command(
        [tlmgr, "option", "repository"],
        env=env,
        timeout=timeout,
    )
    if not ok:
        return False, detail
    lines = [line.strip() for line in detail.splitlines() if line.strip()]
    return True, lines[-1] if lines else detail


def _probe_tlmgr_tgpagella_search(
    tlmgr: str,
    *,
    env: dict[str, str],
    timeout: int = 60,
) -> tuple[bool, str]:
    ok, detail = _run_doctor_command(
        [tlmgr, "search", "--global", "--file", "/tgpagella.sty"],
        env=env,
        timeout=timeout,
    )
    if ok and ("tex-gyre" in detail.lower() or "tgpagella.sty" in detail.lower()):
        return True, "可搜索 /tgpagella.sty（通常由 tex-gyre 提供）"
    hint = (
        "可能需要配置代理，或手动将 tlmgr repository 换到可访问的 CTAN 镜像。"
    )
    return False, f"{detail}; {hint}"


async def _ping_arxiv_zh_llm(config: Config, api_key: str) -> str:
    provider = get_sdk_client(
        config.llm.sdk,
        model=config.llm.get_model(),
        key=api_key,
        endpoint=config.llm.endpoint or "https://api.deepseek.com",
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )
    return await provider.ping()


def _collect_arxiv_zh_doctor_checks(config_path: Optional[Path]) -> list[ArxivZhDoctorCheck]:
    checks: list[ArxivZhDoctorCheck] = []

    try:
        config = _load_config_for_arxiv_zh(config_path)
    except Exception as exc:
        checks.append(_doctor_check("配置文件", "FAIL", str(exc)))
        return checks

    config_source = str(config_path) if config_path else "默认配置"
    checks.append(_doctor_check("配置文件", "PASS", f"已加载 {config_source}"))
    checks.append(_check_conda_environment(os.environ))

    if config.llm.sdk == "deepseek":
        checks.append(_doctor_check("LLM SDK", "PASS", "deepseek"))
    else:
        checks.append(
            _doctor_check(
                "LLM SDK",
                "FAIL",
                "arxiv-zh 当前只支持 llm.sdk: deepseek",
            )
        )

    api_key, api_key_env = _resolve_arxiv_zh_api_key(config)
    if api_key:
        checks.append(
            _doctor_check(
                "API Key",
                "PASS",
                f"已通过 llm.key 或 {api_key_env}/.env 读取",
            )
        )
    else:
        checks.append(
            _doctor_check(
                "API Key",
                "FAIL",
                f"未找到 {api_key_env}，请配置 shell 环境变量、.env 或 llm.key",
            )
        )

    if config.llm.sdk == "deepseek" and api_key:
        try:
            result = asyncio.run(_ping_arxiv_zh_llm(config, api_key))
            preview = result.strip().replace("\n", " ")[:80] or "empty response"
            checks.append(_doctor_check("DeepSeek 连通性", "PASS", preview))
        except Exception as exc:
            checks.append(_doctor_check("DeepSeek 连通性", "FAIL", str(exc)))
    else:
        checks.append(
            _doctor_check(
                "DeepSeek 连通性",
                "WARN",
                "跳过：LLM SDK 或 API Key 未通过检查",
            )
        )

    output_root = _resolve_path_from_config(config.paths.output_dir, config_path)
    ok, detail = _probe_writable_directory(output_root)
    checks.append(
        _doctor_check(
            "输出目录",
            "PASS" if ok else "FAIL",
            f"可写：{detail}" if ok else f"不可写：{detail}",
        )
    )

    cache_root = output_root / ".doctor-cache"
    ok, detail = _probe_writable_directory(cache_root)
    checks.append(
        _doctor_check(
            "缓存目录",
            "PASS" if ok else "FAIL",
            f"可写：{detail}" if ok else f"不可写：{detail}",
        )
    )
    if ok:
        try:
            cache_root.rmdir()
        except OSError:
            pass

    font_dir = Path(config.fonts.dir).expanduser() if config.fonts.dir else None
    local_font_files = find_font_files(font_dir)
    available_fonts = get_available_fonts(font_dir=font_dir)
    configured_fonts = [config.fonts.main, config.fonts.sans, config.fonts.mono]
    missing_fonts = [
        font
        for font in configured_fonts
        if font and not _font_name_available(font, available_fonts)
    ]
    if font_dir and not font_dir.exists():
        checks.append(_doctor_check("中文字体", "WARN", f"字体目录不存在：{font_dir}"))
    elif missing_fonts:
        checks.append(
            _doctor_check(
                "中文字体",
                "WARN",
                "以下配置字体未在本机检测到："
                + ", ".join(str(font) for font in missing_fonts),
            )
        )
    elif local_font_files and not shutil.which("fc-scan"):
        checks.append(
            _doctor_check(
                "中文字体",
                "PASS",
                "fontconfig: missing; "
                f"local fonts: found {len(local_font_files)} files; "
                "fallback to local font files",
            )
        )
    elif all(configured_fonts):
        checks.append(
            _doctor_check(
                "中文字体",
                "PASS",
                f"{config.fonts.main} / {config.fonts.sans} / {config.fonts.mono}",
            )
        )
    else:
        checks.append(_doctor_check("中文字体", "WARN", "未配置完整 CJK 字体"))

    if not config.compilation.enabled:
        checks.append(_doctor_check("LaTeX 编译", "WARN", "配置中已关闭 compilation.enabled"))
    else:
        compiler = _arxiv_zh_compiler_from_config(config)
        env = compiler._build_env()
        tinytex_paths = compiler._resolve_tinytex_paths()
        if config.compilation.use_tinytex:
            if tinytex_paths:
                checks.append(
                    _doctor_check(
                        "TinyTeX 路径",
                        "PASS",
                        ", ".join(str(path) for path in tinytex_paths),
                    )
                )
            else:
                checks.append(
                    _doctor_check(
                        "TinyTeX 路径",
                        "WARN",
                        "未找到常见 TinyTeX 路径，将依赖系统 PATH；"
                        "如未初始化 TinyTeX，请运行 "
                        "Rscript -e 'tinytex::install_tinytex()'",
                    )
                )

            rscript = compiler._rscript_path(env)
            r_required = config.compilation.tinytex_driver == "r_tinytex"
            checks.append(
                _doctor_check(
                    "Rscript",
                    "PASS" if rscript else ("FAIL" if r_required else "WARN"),
                    rscript
                    or "未找到；auto 会降级到 latexmk，r_tinytex 强制模式不可用",
                )
            )
            r_tinytex_ok, r_tinytex_detail = compiler._r_tinytex_available(env)
            if r_tinytex_ok:
                checks.append(
                    _doctor_check(
                        "R tinytex",
                        "PASS",
                        "官方 tinytex wrapper 可用，支持自动安装缺失包",
                    )
                )
                checks.append(
                    _doctor_check(
                        "TinyTeX 自动补包",
                        "PASS",
                        "将优先使用 tinytex::latexmk(..., install_packages = TRUE)",
                    )
                )
            else:
                fallback_message = (
                    f"{r_tinytex_detail}；auto 会降级到 latexmk + tlmgr hook。"
                )
                checks.append(
                    _doctor_check(
                        "R tinytex",
                        "FAIL" if r_required else "WARN",
                        fallback_message,
                    )
                )
                checks.append(
                    _doctor_check(
                        "TinyTeX 自动补包",
                        "FAIL" if r_required else "WARN",
                        "未使用官方 R tinytex wrapper；"
                        "请安装 R 包 tinytex 以启用推荐补包路径",
                    )
                )

        latexmk = compiler._which("latexmk", env)
        if config.compilation.prefer_latexmk:
            checks.append(
                _doctor_check(
                    "latexmk",
                    "PASS" if latexmk else "WARN",
                    latexmk or "未找到；会尝试直接调用 LaTeX 引擎",
                )
            )

        requested_engines = (
            [config.compilation.engine_policy]
            if config.compilation.engine_policy != "auto"
            else list(config.compilation.fallback_engines)
        )
        available_engines = [
            engine
            for engine in requested_engines
            if engine != "pdflatex" or config.compilation.allow_pdflatex_cjk
            if compiler._which(engine, env)
        ]
        checks.append(
            _doctor_check(
                "LaTeX 引擎",
                "PASS" if available_engines else "FAIL",
                ", ".join(available_engines)
                if available_engines
                else "未找到可用的 xelatex/lualatex",
            )
        )

        if config.compilation.install_missing_packages:
            tlmgr = compiler._which("tlmgr", env)
            checks.append(
                _doctor_check(
                    "tlmgr",
                    "PASS" if tlmgr else "WARN",
                    tlmgr
                    or "未找到；TinyTeX 缺包自动安装将不可用。"
                    "如未初始化 TinyTeX，请运行 "
                    "Rscript -e 'tinytex::install_tinytex()'",
                )
            )
            if tlmgr:
                repo_ok, repo_detail = _probe_tlmgr_repository(tlmgr, env=env)
                checks.append(
                    _doctor_check(
                        "tlmgr repository",
                        "PASS" if repo_ok else "FAIL",
                        repo_detail,
                    )
                )
                search_ok, search_detail = _probe_tlmgr_tgpagella_search(
                    tlmgr,
                    env=env,
                )
                checks.append(
                    _doctor_check(
                        "tlmgr 缺包搜索",
                        "PASS" if search_ok else "FAIL",
                        search_detail,
                    )
                )

        if config.compilation.use_tinytex:
            checks.append(_doctor_check("代理环境", "PASS", _format_proxy_env(env)))

    ok, detail = _probe_arxiv_reachable()
    checks.append(
        _doctor_check(
            "arXiv 网络",
            "PASS" if ok else "FAIL",
            detail,
        )
    )

    return checks


def _print_arxiv_zh_doctor_report(checks: list[ArxivZhDoctorCheck]) -> None:
    table = Table(title="arxiv-zh 环境体检")
    table.add_column("项目", style="cyan", no_wrap=True)
    table.add_column("状态", no_wrap=True)
    table.add_column("说明")

    style_by_status = {
        "PASS": "green",
        "WARN": "yellow",
        "FAIL": "red",
    }
    for check in checks:
        style = style_by_status.get(check.status, "white")
        table.add_row(check.name, f"[{style}]{check.status}[/{style}]", check.message)

    console.print(table)


def _run_arxiv_zh_doctor(config_path: Optional[Path]) -> bool:
    checks = _collect_arxiv_zh_doctor_checks(config_path)
    _print_arxiv_zh_doctor_report(checks)
    return not any(check.status == "FAIL" for check in checks)


def _resolve_cli_version() -> str:
    for package_name in ("arxiv-zh", "arxiv-translate"):
        try:
            return metadata.version(package_name)
        except metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    return "unknown"


def _version_callback(value: bool) -> None:
    if not value:
        return
    console.print(_resolve_cli_version(), markup=False)
    raise typer.Exit(code=0)


@app.callback(invoke_without_command=True)
def root_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
) -> None:
    if ctx.invoked_subcommand is None and not version:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=0)


def _print_provider_cache_summary(provider: Any) -> None:
    get_summary = getattr(provider, "get_cache_stats_summary", None)
    if not callable(get_summary):
        return

    try:
        summary = get_summary()
    except Exception as e:
        console.print(f"[yellow]Cache summary unavailable: {e}[/yellow]")
        return

    if not isinstance(summary, dict):
        return
    if int(summary.get("request_count", 0) or 0) <= 0:
        return

    formatter = getattr(provider, "format_cache_stats_summary", None)
    lines: list[str] = []
    if callable(formatter):
        try:
            formatted = formatter()
            if isinstance(formatted, str):
                lines = [formatted]
            elif isinstance(formatted, list):
                lines = [str(line) for line in formatted if str(line).strip()]
        except Exception as e:
            console.print(f"[yellow]Cache summary format failed: {e}[/yellow]")

    if not lines:
        lines = [
            "[CACHE SUMMARY] "
            f"requests={summary.get('request_count', 0)} "
            f"hit={summary.get('cache_hit_count', 0)} "
            f"miss={summary.get('cache_miss_count', 0)} "
            f"cached_tokens={summary.get('cached_tokens_total', 0)} "
            f"total_tokens={summary.get('total_tokens_total', 0)}"
        ]

    for line in lines:
        console.print(line, style="cyan", markup=False)


def _validate_provider_args(
    *,
    sdk_name: Optional[str],
    key_val: Optional[str],
    endpoint_val: Optional[str],
) -> None:
    if sdk_name == "ark":
        console.print(
            "[bold red]Error:[/bold red] sdk=ark has been removed. "
            "Please use openai-style config with an Ark endpoint "
            "(ark.*.volces.com)."
        )
        raise typer.Exit(code=1)

    ark_autoroute = should_use_ark_autoroute(sdk_name, endpoint_val)
    requires_key = sdk_name is not None or ark_autoroute
    if requires_key and not key_val:
        if ark_autoroute:
            console.print(
                "[bold red]Error:[/bold red] API key is required for Ark endpoint. "
                "Please set llm.key in config or use --key."
            )
        else:
            console.print(
                "[bold red]Error:[/bold red] API key not found. "
                "Please set llm.key in config or use --key."
            )
        raise typer.Exit(code=1)


def _build_local_cache(config: Any, disabled: bool) -> Optional[LocalTranslationCache]:
    if disabled or not getattr(config.cache, "enabled", True):
        return None
    cache_dir = LocalTranslationCache.resolve_cache_dir(config.paths.cache_dir)
    return LocalTranslationCache(
        cache_dir=cache_dir,
        max_size_mb=config.cache.max_size_mb,
        ttl_days=config.cache.ttl_days,
        compression=config.cache.compression,
        key_mode=config.cache.key_mode,
    )


def _write_local_cache_after_quality_gate(
    *,
    local_cache: Optional[LocalTranslationCache],
    translated_chunks: list[TranslatedChunk],
    missing_fallback_ids: set[str],
) -> tuple[int, int]:
    if local_cache is None:
        return (0, 0)

    cache_written = 0
    cache_skipped = 0

    for chunk in translated_chunks:
        metadata = chunk.metadata
        metadata.setdefault("local_cache_written", False)
        metadata.setdefault("local_cache_skip_reason", None)

        if metadata.get("local_cache_hit"):
            metadata["local_cache_skip_reason"] = "cache_hit"
            cache_skipped += 1
            continue

        key_hash = str(metadata.get("local_cache_key_hash") or "")
        if not key_hash:
            metadata["local_cache_skip_reason"] = "missing_cache_key"
            cache_skipped += 1
            continue

        if metadata.get("skipped") or metadata.get("skipped_placeholder"):
            metadata["local_cache_skip_reason"] = "skipped_chunk"
            cache_skipped += 1
            continue

        if chunk.chunk_id in missing_fallback_ids:
            metadata["local_cache_skip_reason"] = "missing_fallback"
            cache_skipped += 1
            continue

        if not bool(metadata.get("placeholder_audit_passed")):
            metadata["local_cache_skip_reason"] = "placeholder_audit_failed"
            cache_skipped += 1
            continue

        if not bool(metadata.get("brace_audit_passed")):
            metadata["local_cache_skip_reason"] = "brace_audit_failed"
            cache_skipped += 1
            continue

        if bool(metadata.get("brace_fallback_applied")):
            metadata["local_cache_skip_reason"] = "brace_fallback"
            cache_skipped += 1
            continue

        if not bool(metadata.get("line_end_audit_passed", True)):
            metadata["local_cache_skip_reason"] = "line_end_audit_failed"
            cache_skipped += 1
            continue

        if bool(metadata.get("line_end_fallback_applied")):
            metadata["local_cache_skip_reason"] = "line_end_fallback"
            cache_skipped += 1
            continue

        if BuiltInRules.collect_chunk_quality_warning_types(metadata):
            metadata["local_cache_skip_reason"] = "quality_warning"
            cache_skipped += 1
            continue

        if not chunk.translation.strip():
            metadata["local_cache_skip_reason"] = "empty_translation"
            cache_skipped += 1
            continue

        try:
            wrote = local_cache.put_by_hash(key_hash, chunk.translation)
        except Exception as e:
            metadata["local_cache_skip_reason"] = f"cache_write_error:{e}"
            cache_skipped += 1
            continue

        if wrote:
            metadata["local_cache_written"] = True
            metadata["local_cache_skip_reason"] = None
            cache_written += 1
        else:
            metadata["local_cache_skip_reason"] = "cache_write_rejected"
            cache_skipped += 1

    return (cache_written, cache_skipped)


def _print_validation_result(result: Any) -> None:
    errors = [error for error in result.errors if error.severity == "error"]
    warnings = [error for error in result.errors if error.severity == "warning"]
    infos = [error for error in result.errors if error.severity == "info"]

    if errors:
        console.print(
            "[yellow]Validation Issues "
            f"({len(errors)} error(s), {len(warnings)} warning(s)):[/yellow]"
        )
        for err in errors + warnings + infos:
            color = "red" if err.severity == "error" else "yellow"
            console.print(f"[{color}]- {err.message}[/{color}]")
            if err.suggestion:
                console.print(f"  Suggestion: {err.suggestion}", style="dim")
        return

    if warnings or infos:
        console.print(
            "[yellow]Validation Passed with Warnings "
            f"({len(warnings) + len(infos)}):[/yellow]"
        )
        for err in warnings + infos:
            console.print(f"[yellow]- {err.message}[/yellow]")
            if err.suggestion:
                console.print(f"  Suggestion: {err.suggestion}", style="dim")
        return

    console.print("[green]Validation Passed[/green]")


@app.command()
def translate(
    arxiv_url: str = typer.Argument(..., help="arXiv ID or URL to translate"),
    output_dir: Path = typer.Option(
        Path("output"), "-o", "--output-dir", help="Directory to save results"
    ),
    sdk: Optional[str] = typer.Option(
        None,
        help="SDK to use (openai, openai-coding, anthropic, anthropic-coding, bailian, or None for direct HTTP)",
    ),
    model: Optional[str] = typer.Option(None, help="Model name to use"),
    key: Optional[str] = typer.Option(None, help="API Key"),
    endpoint: Optional[str] = typer.Option(None, help="API endpoint URL"),
    no_compile: bool = typer.Option(False, help="Skip PDF compilation"),
    keep_source: bool = typer.Option(False, help="Keep downloaded source files"),
    concurrency: int = typer.Option(
        50,
        "-c",
        "--concurrency",
        help="Max concurrent API requests (lower = safer for rate limits)",
    ),
    high_quality: bool = typer.Option(
        False,
        "--high-quality",
        "-hq",
        help="启用高质量翻译模式，为每个 chunk 提供摘要上下文",
    ),
    abstract: Optional[str] = typer.Option(
        None, "--abstract", help="手动提供摘要文本（覆盖自动提取）"
    ),
    no_local_cache: bool = typer.Option(
        False,
        "--no-local-cache",
        help="Disable local persistent translation cache for this run",
    ),
):
    """
    Translate an arXiv paper to Chinese.
    """
    # Load configuration
    config = load_config()

    # Overrides
    sdk_name = sdk or config.llm.sdk
    model_name = model or config.llm.get_model()
    key_val = key or config.llm.key
    endpoint_val = endpoint or config.llm.endpoint

    _validate_provider_args(
        sdk_name=sdk_name,
        key_val=key_val,
        endpoint_val=endpoint_val,
    )

    console.print(
        Panel.fit(
            f"[bold blue]arxiv-translate Pipeline[/bold blue]\n"
            f"Target: [cyan]{arxiv_url}[/cyan]\n"
            f"SDK: [green]{sdk_name or 'HTTP'}[/green] ({model_name})\n"
            f"Output: [yellow]{output_dir}[/yellow]",
            title="Starting Job",
        )
    )

    async def run_pipeline():
        local_cache: Optional[LocalTranslationCache] = None
        try:
            # 1. Download
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Downloading source...", total=None)
                downloader = ArxivDownloader()
                try:
                    download_result = downloader.download(arxiv_url, output_dir)
                    progress.update(
                        task, description=f"Downloaded: {download_result.arxiv_id}"
                    )
                except Exception as e:
                    progress.update(
                        task, description=f"[red]Download failed: {e}[/red]"
                    )
                    raise

            # 2. Parse
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Parsing LaTeX...", total=None)
                parser = LaTeXParser(
                    extra_protected_envs=config.parser.extra_protected_environments,
                    font_config=config.fonts,
                )
                try:
                    doc = parser.parse_file(str(download_result.main_tex))
                    progress.update(
                        task, description=f"Parsed {len(doc.chunks)} chunks"
                    )
                except Exception as e:
                    progress.update(task, description=f"[red]Parsing failed: {e}[/red]")
                    raise

            # Save parser state for placeholder validation
            parser_state_path = (
                output_dir / download_result.arxiv_id / "parser_state.json"
            )
            doc.save_parser_state(parser_state_path)

            # 3. Translate
            console.print("\n[bold]Translating...[/bold]")
            glossary = load_glossary()
            provider_kwargs: dict[str, Any] = {
                "temperature": config.llm.temperature,
                "max_tokens": config.llm.max_tokens,
            }
            if sdk_name in ("openai-coding", "anthropic-coding"):
                provider_kwargs["full_glossary"] = glossary

            provider = get_sdk_client(
                sdk_name,
                model=model_name,
                key=key_val,
                endpoint=endpoint_val,
                **provider_kwargs,
            )
            try:
                local_cache = _build_local_cache(config, no_local_cache)
                if local_cache is not None:
                    console.print(
                        f"[cyan]Local cache enabled: {local_cache.db_path}[/cyan]"
                    )
            except Exception as e:
                local_cache = None
                console.print(
                    f"[yellow]Local cache unavailable, continuing without it: {e}[/yellow]"
                )
            reset_cache_stats = getattr(provider, "reset_cache_stats", None)
            if callable(reset_cache_stats):
                try:
                    reset_cache_stats()
                except Exception as e:
                    console.print(
                        f"[yellow]Cache stats reset skipped: {e}[/yellow]"
                    )

            # Prepare high-quality mode parameters
            abstract_text = None
            examples = []
            if high_quality:
                # Get abstract: CLI argument > extracted abstract > fallback
                abstract_text = abstract or getattr(doc, "abstract", "") or ""
                # Load few-shot examples
                examples_path = getattr(config.translation, "examples_path", None)
                examples = (
                    load_examples(examples_path) if examples_path else load_examples()
                )
                console.print(
                    f"[cyan]High-quality mode enabled: {len(examples)} examples loaded[/cyan]"
                )

            pipeline = TranslationPipeline(
                provider=provider,
                glossary=glossary,
                state_file=output_dir
                / download_result.arxiv_id
                / "translation_state.json",
                few_shot_examples=examples,
                abstract_context=abstract_text,
                custom_system_prompt=config.translation.custom_system_prompt,
                model_name=model_name,
                hq_mode=high_quality,
                batch_short_threshold=config.translation.batch_short_threshold,
                batch_max_chars=config.translation.batch_max_chars,
                sequential_mode=(sdk_name in ("openai-coding", "anthropic-coding")),
                local_cache=local_cache,
                cache_key_mode=config.cache.key_mode,
            )

            chunk_data = [{"chunk_id": c.id, "content": c.content} for c in doc.chunks]

            console.print(
                f"[bold]Translating {len(chunk_data)} chunks (max {concurrency} concurrent)...[/bold]"
            )

            batch_stats = {"batches": 0, "long_chunks": 0, "total_calls": 0}

            def on_batch_stats(num_batches: int, num_long: int, total_calls: int):
                batch_stats["batches"] = num_batches
                batch_stats["long_chunks"] = num_long
                batch_stats["total_calls"] = total_calls

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task(
                    "Translating chunks...", total=len(chunk_data)
                )

                def update_progress(completed: int, total: int):
                    progress.update(task_id, completed=completed, total=total)

                translated_chunks = await pipeline.translate_document(
                    chunks=chunk_data,
                    context="Academic Paper",
                    max_concurrent=concurrency,
                    progress_callback=update_progress,
                    batch_stats_callback=on_batch_stats,
                )

            if batch_stats["total_calls"] > 0:
                console.print(
                    f"[cyan]Batch optimization: {len(chunk_data)} chunks → "
                    f"{batch_stats['total_calls']} API calls "
                    f"({batch_stats['batches']} batches + {batch_stats['long_chunks']} long chunks)[/cyan]"
                )

            results = [tc.model_dump() for tc in translated_chunks]
            console.print(f"[green]Translation complete: {len(results)} chunks[/green]")
            _print_provider_cache_summary(provider)

            # Reconstruct
            translated_map = {r["chunk_id"]: r["translation"] for r in results}
            translated_chunk_map = {chunk.chunk_id: chunk for chunk in translated_chunks}
            source_chunk_map = {chunk.id: chunk.content for chunk in doc.chunks}

            markdown_changed_chunks = 0
            markdown_converted_spans = 0
            markdown_skipped_spans = 0
            for chunk_id, translation_text in list(translated_map.items()):
                source_text = source_chunk_map.get(chunk_id, "")
                fixed_text, md_audit = sanitize_markdown_bold_safe(
                    source=source_text,
                    translation=translation_text,
                )
                translated_map[chunk_id] = fixed_text
                if chunk_id in translated_chunk_map:
                    translated_chunk_map[chunk_id].metadata[
                        "markdown_bold_postprocess"
                    ] = md_audit
                    translated_chunk_map[chunk_id].translation = fixed_text
                if md_audit.get("changed"):
                    markdown_changed_chunks += 1
                    markdown_converted_spans += int(md_audit.get("converted_count", 0))
                markdown_skipped_spans += int(md_audit.get("skipped_count", 0))

            if markdown_changed_chunks > 0:
                console.print(
                    f"[cyan]Markdown bold sanitized: "
                    f"{markdown_changed_chunks} chunk(s), "
                    f"{markdown_converted_spans} span(s) converted[/cyan]"
                )
            elif markdown_skipped_spans > 0:
                console.print(
                    f"[cyan]Markdown bold audit skipped {markdown_skipped_spans} candidate span(s) for safety[/cyan]"
                )

            disable_missing_fallback_ids = {
                chunk.chunk_id
                for chunk in translated_chunks
                if bool(chunk.metadata.get("placeholder_retry_exhausted"))
            }

            translated_map, ph_issues = validate_translated_placeholders(
                translated_map,
                doc,
                disable_missing_fallback_ids=disable_missing_fallback_ids,
            )
            missing_fallback_ids = {
                issue["chunk_id"]
                for issue in ph_issues
                if issue.get("type") == "missing_fallback"
            }

            if ph_issues:
                console.print(
                    f"\n[yellow]Placeholder Issues ({len(ph_issues)}):[/yellow]"
                )
                for issue in ph_issues:
                    if issue["type"] == "typo_fixed":
                        console.print(
                            f"[yellow]  TYPO FIXED: chunk {issue['chunk_id'][:8]}..., "
                            f"{issue['bad']} → {issue['fixed_to']}[/yellow]"
                        )
                    elif issue["type"] == "hallucination":
                        console.print(
                            f"[yellow]  HALLUCINATION REMOVED: chunk {issue['chunk_id'][:8]}..., "
                            f"{issue['bad']} deleted[/yellow]"
                        )
                    elif issue["type"] == "missing":
                        console.print(
                            f"[red]  MISSING: chunk {issue['chunk_id'][:8]}..., "
                            f"{issue['bad']} lost in translation[/red]"
                        )
                    elif issue["type"] == "missing_fallback":
                        console.print(
                            f"[red]  MISSING FALLBACK: chunk {issue['chunk_id'][:8]}..., "
                            f"missing={issue['bad']} -> reverted to source[/red]"
                        )

            translated_tex, translated_chunk_start_lines = (
                doc.reconstruct_with_chunk_start_lines(translated_map)
            )

            translated_chunks_for_validation = [
                TranslatedChunk(
                    source=chunk.source,
                    translation=translated_map.get(chunk.chunk_id, chunk.translation),
                    chunk_id=chunk.chunk_id,
                    metadata=dict(chunk.metadata or {}),
                )
                for chunk in translated_chunks
            ]

            if local_cache is not None:
                cache_written, cache_skipped = _write_local_cache_after_quality_gate(
                    local_cache=local_cache,
                    translated_chunks=translated_chunks_for_validation,
                    missing_fallback_ids=missing_fallback_ids,
                )
                if cache_written > 0 or cache_skipped > 0:
                    console.print(
                        f"[cyan]Local cache write: written={cache_written}, skipped={cache_skipped}[/cyan]"
                    )

            # Save
            out_file = download_result.main_tex.parent / "main_translated.tex"
            out_file.write_text(translated_tex, encoding="utf-8")
            console.print(f"[green]Translation saved to {out_file}[/green]")

            # 4. Validate
            console.print("\n[bold]Validating...[/bold]")
            validator = ValidationEngine()

            # Extract original text for validation
            original_full, source_chunk_start_lines = (
                doc.reconstruct_with_chunk_start_lines()
            )

            val_result = validator.validate(
                translated_tex,
                original_full,
                translated_chunks=translated_chunks_for_validation,
                source_chunk_start_lines=source_chunk_start_lines,
                translation_chunk_start_lines=translated_chunk_start_lines,
            )
            _print_validation_result(val_result)

            # 5. Compile
            if not no_compile:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Compiling PDF...", total=None)
                    compiler = LaTeXCompiler(
                        timeout=config.compilation.timeout,
                        fonts_dir=config.fonts.dir,
                        use_tinytex=config.compilation.use_tinytex,
                        tinytex_driver=config.compilation.tinytex_driver,
                        tinytex_paths=config.compilation.tinytex_paths,
                        install_missing_packages=(
                            config.compilation.install_missing_packages
                        ),
                        install_timeout=config.compilation.install_timeout,
                        total_timeout=config.compilation.total_timeout,
                        max_package_install_rounds=(
                            config.compilation.max_package_install_rounds
                        ),
                    )
                    compile_error: Optional[str] = None
                    try:
                        latex_source = out_file.read_text(encoding="utf-8")
                        # Save the final version that will be compiled (for debugging)
                        out_file.write_text(latex_source, encoding="utf-8")
                        pdf_path = (
                            output_dir
                            / download_result.arxiv_id
                            / f"{download_result.arxiv_id}.pdf"
                        )
                        result = compiler.compile(
                            latex_source,
                            pdf_path,
                            working_dir=download_result.main_tex.parent,
                        )
                        if result.success:
                            progress.update(
                                task, description=f"Compiled: {result.pdf_path}"
                            )
                            console.print(
                                Panel(
                                    f"[bold green]Success![/bold green]\nPDF: {result.pdf_path}"
                                )
                            )
                        else:
                            compile_error = result.error_message or "Unknown error"
                    except Exception as e:
                        compile_error = str(e)

                    if compile_error:
                        progress.update(task, description="[red]Compilation failed[/red]")
                        console.print(f"[yellow]Error: {compile_error}[/yellow]")
                        console.print(
                            "[yellow]Generated .tex file is saved. You may try compiling it manually.[/yellow]"
                        )
                        raise RuntimeError(f"Compilation failed: {compile_error}")

        except Exception as e:
            console.print(f"[bold red]Pipeline failed:[/bold red] {e}")
            raise typer.Exit(code=1)
        finally:
            if local_cache is not None:
                try:
                    local_cache.close()
                except Exception:
                    pass

    asyncio.run(run_pipeline())


def _run_arxiv_zh_pipeline(
    *,
    arxiv_id: str,
    options: ArxivZhOptions,
    config: Config,
) -> None:
    layout = _prepare_arxiv_zh_output_dirs(options.output)
    parsed_id = ArxivDownloader.parse_id(arxiv_id)
    metadata_data = _ensure_arxiv_zh_metadata(
        layout,
        arxiv_input=arxiv_id,
        arxiv_id=parsed_id,
    )
    _mark_existing_translation_if_present(layout, metadata_data)
    layout.translate_log.write_text("", encoding="utf-8")
    _append_text_log(layout.translate_log, f"Starting arxiv-zh job for {arxiv_id}")

    if _metadata_stage_completed(metadata_data, "compilation"):
        pdf_path = _metadata_stage(metadata_data, "compilation").get("pdf_path")
        if pdf_path and Path(str(pdf_path)).exists():
            _append_text_log(
                layout.translate_log,
                f"Compilation already completed; skipping: {pdf_path}",
            )
            _set_metadata_run_status(metadata_data, "success")
            _write_arxiv_zh_metadata(layout, metadata_data)
            return

    if _metadata_stage_completed(metadata_data, "translation"):
        translated_tex = layout.translated_dir / "main_zh.tex"
        if translated_tex.exists():
            _append_text_log(
                layout.translate_log,
                "Translation already completed; skipping download and translation",
            )
            if options.compile_pdf:
                _compile_arxiv_zh_translated_tex(
                    layout=layout,
                    config=config,
                    metadata_data=metadata_data,
                )
            else:
                _set_metadata_stage_status(
                    metadata_data,
                    "compilation",
                    "skipped",
                    reason="compilation.disabled",
                )
                _set_metadata_run_status(metadata_data, "success")
                _write_arxiv_zh_metadata(layout, metadata_data)
            return

    async def run_pipeline() -> None:
        local_cache: Optional[LocalTranslationCache] = None
        download_result = None
        translated_count = 0
        total_chunks = 0
        try:
            config.llm.key = options.api_key
            config.llm.endpoint = options.endpoint
            config.paths.cache_dir = str(layout.cache_dir)
            config.cache.enabled = True

            download_result = _existing_source_download_result(layout, metadata_data)
            if download_result is not None:
                _append_text_log(
                    layout.translate_log,
                    f"Download already completed; reusing {layout.source_dir}",
                )
            else:
                _append_text_log(layout.translate_log, "Downloading source")
                _set_metadata_stage_status(metadata_data, "download", "running")
                _write_arxiv_zh_metadata(layout, metadata_data)
                downloader = ArxivDownloader(cache_dir=layout.cache_dir / "downloads")
                download_result = downloader.download(
                    arxiv_id,
                    layout.root,
                    extract_dir=layout.source_dir,
                )
                _set_metadata_stage_status(
                    metadata_data,
                    "download",
                    "completed",
                    source_dir=str(layout.source_dir),
                    main_tex=str(download_result.main_tex),
                    archive_path=str(
                        layout.cache_dir
                        / "downloads"
                        / f"{download_result.arxiv_id}.tar.gz"
                    ),
                    files_count=len(download_result.downloaded_files),
                )
                _write_arxiv_zh_metadata(layout, metadata_data)
                _append_text_log(
                    layout.translate_log,
                    f"Downloaded {download_result.arxiv_id} to {layout.source_dir}",
                )

            _append_text_log(layout.translate_log, "Parsing LaTeX")
            parser = LaTeXParser(
                extra_protected_envs=config.parser.extra_protected_environments,
                font_config=config.fonts,
            )
            doc = parser.parse_file(str(download_result.main_tex))
            total_chunks = len(doc.chunks)
            doc.save_parser_state(layout.cache_dir / "parser_state.json")
            _append_text_log(layout.translate_log, f"Parsed {total_chunks} chunks")

            glossary = load_glossary()
            provider = get_sdk_client(
                "deepseek",
                model=options.model,
                key=options.api_key,
                endpoint=options.endpoint,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens,
            )

            try:
                local_cache = _build_local_cache(config, disabled=False)
                if local_cache is not None:
                    _append_text_log(
                        layout.translate_log,
                        f"Local cache enabled: {local_cache.db_path}",
                    )
            except Exception as exc:
                local_cache = None
                _append_text_log(
                    layout.translate_log,
                    f"Local cache unavailable; continuing without it: {exc}",
                )

            examples_path = getattr(config.translation, "examples_path", None)
            examples = load_examples(examples_path) if examples_path else load_examples()
            pipeline = TranslationPipeline(
                provider=provider,
                glossary=glossary,
                state_file=layout.cache_dir / "translation_state.json",
                few_shot_examples=examples,
                abstract_context=getattr(doc, "abstract", "") or "",
                custom_system_prompt=config.translation.custom_system_prompt,
                model_name=options.model,
                hq_mode=False,
                batch_short_threshold=config.translation.batch_short_threshold,
                batch_max_chars=config.translation.batch_max_chars,
                sequential_mode=False,
                local_cache=local_cache,
                cache_key_mode=config.cache.key_mode,
            )

            chunk_data = [{"chunk_id": c.id, "content": c.content} for c in doc.chunks]
            if options.max_chunks is not None:
                chunk_data = chunk_data[: options.max_chunks]
            _append_text_log(
                layout.translate_log,
                f"Translating {len(chunk_data)} of {total_chunks} chunks",
            )
            _set_metadata_stage_status(
                metadata_data,
                "translation",
                "running",
                total_chunks=total_chunks,
            )
            _write_arxiv_zh_metadata(layout, metadata_data)

            translated_chunks = await pipeline.translate_document(
                chunks=chunk_data,
                context="Academic Paper",
                max_concurrent=options.concurrency,
                progress_callback=None,
                batch_stats_callback=None,
            )
            translated_count = len(translated_chunks)

            translated_map = {
                chunk.chunk_id: chunk.translation for chunk in translated_chunks
            }
            source_chunk_map = {chunk.id: chunk.content for chunk in doc.chunks}
            for chunk_id, translation_text in list(translated_map.items()):
                fixed_text, _md_audit = sanitize_markdown_bold_safe(
                    source=source_chunk_map.get(chunk_id, ""),
                    translation=translation_text,
                )
                translated_map[chunk_id] = fixed_text

            translated_map, ph_issues = validate_translated_placeholders(
                translated_map,
                doc,
                disable_missing_fallback_ids=set(),
            )
            if ph_issues:
                _append_text_log(
                    layout.translate_log,
                    f"Placeholder audit produced {len(ph_issues)} issue(s)",
                )

            translated_tex, translated_chunk_start_lines = (
                doc.reconstruct_with_chunk_start_lines(translated_map)
            )

            translated_chunks_for_validation = [
                TranslatedChunk(
                    source=chunk.source,
                    translation=translated_map.get(chunk.chunk_id, chunk.translation),
                    chunk_id=chunk.chunk_id,
                    metadata=dict(chunk.metadata or {}),
                )
                for chunk in translated_chunks
            ]

            if local_cache is not None:
                _write_local_cache_after_quality_gate(
                    local_cache=local_cache,
                    translated_chunks=translated_chunks_for_validation,
                    missing_fallback_ids=set(),
                )

            if layout.translated_dir.exists():
                shutil.rmtree(layout.translated_dir)
            layout.translated_dir.mkdir(parents=True, exist_ok=True)
            _copy_source_tree_to_translated(layout.source_dir, layout.translated_dir)
            translated_tex_path = layout.translated_dir / "main_zh.tex"
            translated_tex_path.write_text(translated_tex, encoding="utf-8")
            _set_metadata_stage_status(
                metadata_data,
                "translation",
                "completed",
                translated_tex=str(translated_tex_path),
                translated_chunks=translated_count,
                total_chunks=total_chunks,
                parser_state=str(layout.cache_dir / "parser_state.json"),
                translation_state=str(layout.cache_dir / "translation_state.json"),
            )
            _write_arxiv_zh_metadata(layout, metadata_data)
            _append_text_log(
                layout.translate_log,
                f"Wrote translated TeX to {translated_tex_path}",
            )

            original_full, source_chunk_start_lines = (
                doc.reconstruct_with_chunk_start_lines()
            )
            validator = ValidationEngine()
            val_result = validator.validate(
                translated_tex,
                original_full,
                translated_chunks=translated_chunks_for_validation,
                source_chunk_start_lines=source_chunk_start_lines,
                translation_chunk_start_lines=translated_chunk_start_lines,
            )
            _append_text_log(
                layout.translate_log,
                f"Validation completed with {len(val_result.errors)} issue(s)",
            )

            if options.compile_pdf:
                _append_text_log(layout.translate_log, "Compiling PDF")
                _compile_arxiv_zh_translated_tex(
                    layout=layout,
                    config=config,
                    metadata_data=metadata_data,
                )
            else:
                _set_metadata_stage_status(
                    metadata_data,
                    "compilation",
                    "skipped",
                    reason="compilation.disabled",
                )
                _set_metadata_run_status(metadata_data, "success")
                _write_arxiv_zh_metadata(layout, metadata_data)

            _append_text_log(layout.translate_log, "Job completed")

        except Exception as exc:
            _append_text_log(layout.translate_log, f"Job failed: {exc}")
            if download_result is None:
                _set_metadata_stage_status(
                    metadata_data,
                    "download",
                    "failed",
                    error=str(exc),
                )
            elif not _metadata_stage_completed(metadata_data, "translation"):
                _set_metadata_stage_status(
                    metadata_data,
                    "translation",
                    "failed",
                    error=str(exc),
                    translated_chunks=translated_count,
                    total_chunks=total_chunks,
                )
            if _metadata_stage(metadata_data, "compilation").get("status") == "failed":
                _set_metadata_run_status(metadata_data, "compile_failed")
            else:
                _set_metadata_run_status(metadata_data, "failed")
            _write_arxiv_zh_metadata(layout, metadata_data)
            raise
        finally:
            if local_cache is not None:
                try:
                    local_cache.close()
                except Exception:
                    pass

    asyncio.run(run_pipeline())


@zh_app.command()
def arxiv_zh_root(
    ctx: typer.Context,
    arxiv_id: Optional[str] = typer.Argument(None, help="arXiv ID or URL"),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Config YAML path. Defaults are used when omitted.",
    ),
    doctor: bool = typer.Option(
        False,
        "--doctor",
        help="Run environment checks and exit.",
    ),
    compile_only: bool = typer.Option(
        False,
        "--compile-only",
        help="Compile existing translated/main_zh.tex and exit.",
    ),
) -> None:
    if doctor:
        ok = _run_arxiv_zh_doctor(config)
        raise typer.Exit(code=0 if ok else 1)

    if not arxiv_id:
        console.print(ctx.get_help())
        raise typer.Exit(code=0)

    try:
        options, resolved_config = _resolve_arxiv_zh_options(
            arxiv_id=arxiv_id,
            config=config,
            require_api_key=not compile_only,
        )
        if compile_only:
            _run_arxiv_zh_compile_only(
                arxiv_id=arxiv_id,
                options=options,
                config=resolved_config,
            )
        else:
            _run_arxiv_zh_pipeline(
                arxiv_id=arxiv_id,
                options=options,
                config=resolved_config,
            )
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[bold red]arxiv-zh failed:[/bold red] {exc}")
        raise typer.Exit(code=1)


@config_app.command("show")
def config_show():
    """Show current configuration."""
    config = load_config()
    console.print(config.model_dump())


@config_app.command("set")
def config_set(key: str, value: str):
    """
    Set a configuration value (dot-separated).
    Example: arx config set llm.models deepseek-v4-flash
    """
    config_file = ensure_config_dir() / "config.yaml"

    # Load raw yaml to preserve structure if possible, or just dict
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    # Update nested key
    keys = key.split(".")
    current = data
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
        if not isinstance(current, dict):
            console.print(f"[red]Error: {k} is not a dictionary[/red]")
            raise typer.Exit(1)

    # Attempt type conversion
    val = value
    if value.lower() == "true":
        val = True
    elif value.lower() == "false":
        val = False
    elif value.isdigit():
        val = int(value)
    else:
        try:
            val = float(value)
        except ValueError:
            pass

    current[keys[-1]] = val

    try:
        Config(**deep_merge(load_defaults(), data))
    except Exception as exc:
        console.print(f"[red]Invalid config value for {key}:[/red] {exc}")
        raise typer.Exit(1)

    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

    console.print(f"[green]Updated {key} = {val}[/green]")


@glossary_app.command("add")
def glossary_add(
    term: str = typer.Argument(..., help="Term to add"),
    translation: str = typer.Argument(..., help="Translation of the term"),
    domain: Optional[str] = typer.Option(None, help="Domain context"),
    notes: Optional[str] = typer.Option(None, help="Additional notes"),
):
    """Add a term to the glossary."""
    glossary_file = ensure_config_dir() / "glossary.yaml"

    if glossary_file.exists():
        with open(glossary_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    data[term] = {"target": translation, "domain": domain, "notes": notes}

    with open(glossary_file, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)

    console.print(f"[green]Added term:[/green] {term} -> {translation}")


@app.command()
def validate(
    tex_file: Path = typer.Argument(..., exists=True, help="Path to .tex file"),
    original_file: Optional[Path] = typer.Option(
        None, help="Original .tex file for comparison"
    ),
):
    """
    Validate a LaTeX file.
    """
    with open(tex_file, "r", encoding="utf-8") as f:
        content = f.read()

    original = ""
    if original_file and original_file.exists():
        with open(original_file, "r", encoding="utf-8") as f:
            original = f.read()

    validator = ValidationEngine()
    result = validator.validate(content, original)

    if result.valid and not result.errors:
        console.print("[green]File is valid![/green]")
        return

    _print_validation_result(result)


@cache_app.command("stats")
def cache_stats():
    """Show local translation cache stats."""
    config = load_config()
    try:
        cache = _build_local_cache(config, disabled=False)
    except Exception as e:
        console.print(f"[red]Local cache unavailable:[/red] {e}")
        raise typer.Exit(code=1)

    if cache is None:
        console.print("[yellow]Local cache is disabled in config.[/yellow]")
        raise typer.Exit(code=0)

    try:
        stats = cache.stats()
    finally:
        cache.close()

    table = Table(title="Local Translation Cache Stats")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("DB Path", stats["db_path"])
    table.add_row("Entries", str(stats["entry_count"]))
    table.add_row(
        "Size",
        f"{stats['total_size_mb']:.2f} MB / {stats['max_size_mb']:.2f} MB",
    )
    table.add_row("Usage", f"{stats['usage_ratio'] * 100:.2f}%")
    table.add_row("Compression", str(stats["compression"]))
    table.add_row("TTL", f"{stats['ttl_days']} days")
    table.add_row("Total Hits", str(stats["total_hits"]))
    table.add_row("Total Misses", str(stats["total_misses"]))
    table.add_row("Hit Rate", f"{stats['hit_rate'] * 100:.2f}%")
    table.add_row("Total Writes", str(stats["total_writes"]))
    table.add_row("Expired Purged", str(stats["total_expired_purged"]))
    table.add_row("LRU Evicted", str(stats["total_evicted_lru"]))
    console.print(table)


@cache_app.command("clear")
def cache_clear(
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Skip confirmation prompt and clear local cache directly",
    )
):
    """Clear local translation cache."""
    config = load_config()
    try:
        cache = _build_local_cache(config, disabled=False)
    except Exception as e:
        console.print(f"[red]Local cache unavailable:[/red] {e}")
        raise typer.Exit(code=1)

    if cache is None:
        console.print("[yellow]Local cache is disabled in config.[/yellow]")
        raise typer.Exit(code=0)

    try:
        if not yes:
            confirmed = typer.confirm(
                f"Clear all local cache entries at {cache.db_path}?",
                default=False,
            )
            if not confirmed:
                console.print("[yellow]Cancelled.[/yellow]")
                raise typer.Exit(code=0)

        deleted = cache.clear()
    finally:
        cache.close()

    console.print(f"[green]Cleared local cache entries:[/green] {deleted}")


def main():
    app()


def zh_main():
    zh_app()


if __name__ == "__main__":
    main()

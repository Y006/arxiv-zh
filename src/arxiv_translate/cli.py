import asyncio
import importlib.metadata as metadata
import time
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
from arxiv_translate.downloader.arxiv import ArxivDownloader
from arxiv_translate.parser.latex_parser import LaTeXParser
from arxiv_translate.rules.config import load_config
from arxiv_translate.rules.glossary import load_glossary
from arxiv_translate.rules.examples import load_examples
from arxiv_translate.rules.user_paths import (
    ensure_config_dir,
    migrate_legacy_files,
)
from arxiv_translate.translator import get_sdk_client, should_use_ark_autoroute
from arxiv_translate.translator.pipeline import TranslationPipeline, TranslatedChunk
from arxiv_translate.translator.postprocess import sanitize_markdown_bold_safe
from arxiv_translate.parser.structure import validate_translated_placeholders
from arxiv_translate.validator.engine import ValidationEngine

app = typer.Typer(
    name="arx",
    help="arxiv-translate - arXiv Paper Translator",
    add_completion=False,
    no_args_is_help=False,
)
config_app = typer.Typer(help="Manage configuration")
glossary_app = typer.Typer(help="Manage glossary terms")
cache_app = typer.Typer(help="Manage local translation cache")
app.add_typer(config_app, name="config")
app.add_typer(glossary_app, name="glossary")
app.add_typer(cache_app, name="cache")

console = Console()


def _resolve_cli_version() -> str:
    for package_name in ("arxiv-translate", "ieeA"):
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
            provider_kwargs: dict[str, Any] = {"temperature": config.llm.temperature}
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

            if val_result.valid:
                console.print("[green]Validation Passed[/green]")
            else:
                console.print(
                    f"[yellow]Validation Issues ({len(val_result.errors)}):[/yellow]"
                )
                for err in val_result.errors:
                    color = "red" if err.severity == "error" else "yellow"
                    console.print(f"[{color}]- {err.message}[/{color}]")
                    if err.suggestion:
                        console.print(f"  Suggestion: {err.suggestion}", style="dim")

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


@config_app.command("show")
def config_show():
    """Show current configuration."""
    config = load_config()
    console.print(config.model_dump())


@config_app.command("set")
def config_set(key: str, value: str):
    """
    Set a configuration value (dot-separated).
    Example: arx config set llm.model gpt-4
    """
    migrate_legacy_files()
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
    migrate_legacy_files()
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
def ping(
    sdk: Optional[str] = typer.Option(
        None,
        help="SDK to use (openai, openai-coding, anthropic, anthropic-coding, bailian, or None for direct HTTP)",
    ),
    model: Optional[str] = typer.Option(None, help="Model name to use"),
    key: Optional[str] = typer.Option(None, help="API Key"),
    endpoint: Optional[str] = typer.Option(None, help="API endpoint URL"),
):
    """
    Test LLM connectivity. Sends a minimal request to verify the configured LLM is reachable.
    """
    config = load_config()

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
            f"[bold blue]arxiv-translate Ping[/bold blue]\n"
            f"SDK: [green]{sdk_name or 'HTTP'}[/green]\n"
            f"Model: [cyan]{model_name}[/cyan]\n"
            f"Endpoint: [yellow]{endpoint_val or 'default'}[/yellow]",
            title="Testing LLM Connectivity",
        )
    )

    async def do_ping():
        try:
            provider = get_sdk_client(
                sdk_name,
                model=model_name,
                key=key_val,
                endpoint=endpoint_val,
                temperature=config.llm.temperature,
            )
            start = time.perf_counter()
            result = await provider.ping()
            elapsed = time.perf_counter() - start

            console.print(
                f"\n[bold green]✅ 连通成功[/bold green]  "
                f"耗时 [cyan]{elapsed:.2f}s[/cyan]\n"
                f"  模型回复: [dim]{result[:120]}{'...' if len(result) > 120 else ''}[/dim]"
            )
        except Exception as e:
            console.print(f"\n[bold red]❌ 连通失败[/bold red]\n  {e}")
            raise typer.Exit(code=1)

    asyncio.run(do_ping())


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

    if result.valid:
        console.print("[green]File is valid![/green]")
    else:
        console.print(f"[red]Found {len(result.errors)} errors[/red]")
        for err in result.errors:
            console.print(f"- {err.message} ({err.severity})")


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


if __name__ == "__main__":
    main()

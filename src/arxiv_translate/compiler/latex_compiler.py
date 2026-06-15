import json
import glob
import os
import shutil
import subprocess
import tempfile
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .chinese_support import (
    ENGINE_CONFLICT_PRIMITIVES,
    _strip_engine_conflict_primitives,
    contains_cjk_text,
    inject_chinese_support,
    inject_chinese_support_for_engine,
    normalize_unicode_engine_source,
)

SAFE_REDEFINED_COMMANDS = {
    "red",
    "todo",
    "TODO",
    "cmark",
    "xmark",
    "Cmark",
    "Xmark",
    "markover",
    "greencircle",
}


def _first_log_context(
    lines: List[str],
    pattern: str,
    *,
    flags: int = re.IGNORECASE,
    context_lines: int = 3,
) -> Optional[str]:
    regex = re.compile(pattern, flags)
    for index, line in enumerate(lines):
        if regex.search(line):
            return "\n".join(lines[index : min(index + context_lines, len(lines))])
    return None


def build_compile_error_summary(log_content: str) -> str:
    """Build a concise markdown summary from a LaTeX compile log."""
    lines = (log_content or "").replace("\r\n", "\n").replace("\r", "\n").splitlines()

    sections: List[Tuple[str, Optional[str]]] = [
        (
            "First LaTeX Error",
            _first_log_context(lines, r"(?:^!\s*)?LaTeX\s+Error:|^!\s*LaTeX\s+Error:"),
        ),
        (
            "First Undefined control sequence",
            _first_log_context(lines, r"Undefined control sequence\."),
        ),
        (
            "First Missing $ inserted",
            _first_log_context(lines, r"Missing\s+\$\s+inserted\."),
        ),
        (
            "First Missing file",
            _first_log_context(lines, r"File\s+[`'][^`']+[`']\s+not\s+found"),
        ),
        (
            "First Font Error",
            _first_log_context(lines, r"fontspec\s+Error|fontsp\s*ec\s+Error"),
        ),
        (
            "First Package Error",
            _first_log_context(lines, r"Package\s+[^\s:]+\s+Error:"),
        ),
        (
            "First Unicode/CJK Error",
            _first_log_context(lines, r"Unicode\s+character|CJK|xeCJK|luatexja"),
        ),
        (
            "First Bibliography Error",
            _first_log_context(lines, r"BibTeX|Biber|biblatex|I found no"),
        ),
        (
            "First Engine Mismatch",
            _first_log_context(lines, r"requires\s+(?:XeTeX|LuaTeX)|change your typesetting engine"),
        ),
        (
            "Shell Escape Required",
            _first_log_context(lines, r"shell-escape|write18|minted"),
        ),
    ]

    output = ["# Compile Error Summary", ""]
    for title, content in sections:
        output.append(f"## {title}")
        output.append("")
        if content:
            output.append("```text")
            output.append(content)
            output.append("```")
        else:
            output.append("Not found.")
        output.append("")

    output.append("## Last 80 log lines")
    output.append("")
    output.append("```text")
    output.extend(lines[-80:] if lines else ["<empty log>"])
    output.append("```")
    output.append("")
    return "\n".join(output)


def write_compile_error_summary(log_path: Path, summary_path: Path) -> Path:
    log_content = ""
    if log_path.exists():
        log_content = log_path.read_text(encoding="utf-8", errors="replace")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        build_compile_error_summary(log_content),
        encoding="utf-8",
    )
    return summary_path


@dataclass
class CompileAttempt:
    engine: str
    command: List[str]
    tex_path: Optional[Path]
    success: bool
    returncode: Optional[int] = None
    error: Optional[str] = None
    category: str = "unknown"
    repairs: List[str] = field(default_factory=list)
    engine_log_path: Optional[Path] = None
    missing_file: Optional[str] = None
    first_error: Optional[str] = None
    driver: Optional[str] = None
    driver_detail: Optional[str] = None


@dataclass
class CompilationResult:
    success: bool
    pdf_path: Optional[Path] = None
    log_content: Optional[str] = None
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    engine_used: Optional[str] = None
    attempts: List[CompileAttempt] = field(default_factory=list)
    repaired_tex_path: Optional[Path] = None
    diagnostic_path: Optional[Path] = None
    driver: Optional[str] = None
    driver_detail: Optional[str] = None


class LaTeXCompiler:
    DEFAULT_TINYTEX_PATHS = [
        "~/Library/TinyTeX/bin/universal-darwin",
        "~/.TinyTeX/bin/*",
        "~/TinyTeX/bin/*",
        "/Library/TinyTeX/bin/universal-darwin",
        "/Library/TeX/texbin",
    ]

    def __init__(
        self,
        timeout: int = 600,
        fonts_dir: Optional[Union[str, Path]] = None,
        *,
        use_tinytex: bool = True,
        tinytex_driver: str = "auto",
        tinytex_paths: Optional[List[str]] = None,
        install_missing_packages: bool = True,
        total_timeout: int = 7200,
    ):
        self.timeout = timeout
        # Priority: xelatex (best CJK), lualatex (good CJK), pdflatex (fallback)
        self.engines = ["xelatex", "lualatex", "pdflatex"]
        self.fonts_dir = Path(fonts_dir).resolve() if fonts_dir else None
        self.use_tinytex = use_tinytex
        self.tinytex_driver = tinytex_driver
        self.tinytex_paths = list(tinytex_paths or self.DEFAULT_TINYTEX_PATHS)
        self.install_missing_packages = install_missing_packages
        self.total_timeout = total_timeout

    def inject_chinese_support(self, latex_source: str) -> str:
        """Wrapper around the injection logic."""
        return inject_chinese_support(latex_source)

    def _build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        tinytex_paths = self._resolve_tinytex_paths()
        if tinytex_paths:
            existing_path = env.get("PATH", "")
            env["PATH"] = os.pathsep.join(
                [str(path) for path in tinytex_paths] + ([existing_path] if existing_path else [])
            )
        if self.fonts_dir and self.fonts_dir.exists():
            existing_font_dirs = env.get("OSFONTDIR", "").strip()
            if existing_font_dirs:
                env["OSFONTDIR"] = (
                    f"{str(self.fonts_dir)}{os.pathsep}{existing_font_dirs}"
                )
            else:
                env["OSFONTDIR"] = str(self.fonts_dir)
        return env

    @staticmethod
    def _with_texinputs(
        env: Dict[str, str],
        *,
        build_path: Path,
        tex_dir: Path,
    ) -> Dict[str, str]:
        patched = dict(env)
        source_dir = tex_dir.parent / "source"
        candidates = [build_path, tex_dir]
        if source_dir.exists():
            candidates.append(source_dir)

        entries: List[str] = []
        seen: set[str] = set()
        for path in candidates:
            resolved = str(path.resolve())
            for entry in (resolved, f"{resolved}//"):
                if entry in seen:
                    continue
                seen.add(entry)
                entries.append(entry)

        existing = patched.get("TEXINPUTS", "")
        if existing:
            entries.append(existing)
        entries.append("")
        patched["TEXINPUTS"] = os.pathsep.join(entries)
        return patched

    def _resolve_tinytex_paths(self) -> List[Path]:
        if not self.use_tinytex:
            return []

        resolved: List[Path] = []
        seen: set[str] = set()
        for configured in self.tinytex_paths:
            expanded = os.path.expandvars(os.path.expanduser(str(configured)))
            matches = glob.glob(expanded) if any(ch in expanded for ch in "*?[]") else [expanded]
            for match in matches:
                path = Path(match)
                if not path.is_dir():
                    continue
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                resolved.append(path.resolve())
        return resolved

    def _which(self, command: str, env: Optional[Dict[str, str]] = None) -> Optional[str]:
        path = (env or self._build_env()).get("PATH")
        return shutil.which(command, path=path)

    def _rscript_path(self, env: Optional[Dict[str, str]] = None) -> Optional[str]:
        return self._which("Rscript", env)

    def _r_tinytex_available(
        self,
        env: Optional[Dict[str, str]] = None,
        *,
        timeout: int = 30,
    ) -> Tuple[bool, str]:
        rscript = self._rscript_path(env)
        if not rscript:
            return False, "Rscript not found"
        try:
            process = subprocess.run(
                [
                    rscript,
                    "--vanilla",
                    "-e",
                    (
                        "quit(status = if "
                        "(requireNamespace('tinytex', quietly = TRUE)) 0 else 1)"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
                env=env or self._build_env(),
            )
        except subprocess.TimeoutExpired:
            return False, f"R tinytex probe timed out after {timeout}s"
        except Exception as exc:
            return False, str(exc)

        detail = (process.stdout + "\n" + process.stderr).strip()
        if process.returncode == 0:
            return True, rscript
        return False, detail or "R package tinytex not installed"

    def _resolve_compile_driver(self, env: Dict[str, str]) -> Tuple[str, str]:
        if self.tinytex_driver == "latexmk" or not self.use_tinytex:
            return "latexmk", "configured"

        r_available, detail = self._r_tinytex_available(env)
        if self.tinytex_driver == "r_tinytex":
            if not r_available:
                return "missing_r_tinytex", detail
            return "r_tinytex", detail

        if r_available:
            return "r_tinytex", detail
        return "latexmk", detail

    def _build_r_tinytex_command(
        self,
        *,
        rscript: str,
        engine: str,
        candidate: Path,
        build_path: Path,
        allow_shell_escape: bool,
        install_packages: bool,
    ) -> List[str]:
        install_packages_value = "TRUE" if install_packages else "FALSE"
        tlmgr_repository_setup = (
            "tlmgr_repo <- Sys.getenv('ARXIV_ZH_TLMGR_REPOSITORY', unset = ''); "
            "tlmgr_path <- Sys.getenv('ARXIV_ZH_TLMGR_PATH', unset = ''); "
            "tlmgr_proxy <- Sys.getenv('ARXIV_ZH_TLMGR_PROXY', unset = ''); "
            "if (nzchar(tlmgr_repo)) { "
            "if (!nzchar(tlmgr_path)) tlmgr_path <- Sys.which('tlmgr'); "
            "if (nzchar(tlmgr_path)) { "
            "tlmgr_wrapper_dir <- tempfile('arxiv_zh_tlmgr_'); "
            "dir.create(tlmgr_wrapper_dir, recursive = TRUE, showWarnings = FALSE); "
            "tlmgr_wrapper <- file.path(tlmgr_wrapper_dir, 'tlmgr'); "
            "proxy_lines <- character(); "
            "if (nzchar(tlmgr_proxy)) proxy_lines <- c("
            "paste('export all_proxy=', shQuote(tlmgr_proxy), sep = ''), "
            "paste('export http_proxy=', shQuote(tlmgr_proxy), sep = ''), "
            "paste('export https_proxy=', shQuote(tlmgr_proxy), sep = ''), "
            "paste('export ALL_PROXY=', shQuote(tlmgr_proxy), sep = ''), "
            "paste('export HTTP_PROXY=', shQuote(tlmgr_proxy), sep = ''), "
            "paste('export HTTPS_PROXY=', shQuote(tlmgr_proxy), sep = '')); "
            "writeLines(c('#!/bin/sh', proxy_lines, "
            "paste('exec', shQuote(tlmgr_path), '--repository', "
            "shQuote(tlmgr_repo), '\"$@\"')), tlmgr_wrapper); "
            "Sys.chmod(tlmgr_wrapper, '0755'); "
            "options(tinytex.tlmgr.path = tlmgr_wrapper); "
            "} "
            "}; "
        )
        expression = (
            tlmgr_repository_setup
            +
            "args <- commandArgs(trailingOnly = TRUE); "
            "file <- args[[1]]; "
            "engine <- args[[2]]; "
            "output_dir <- args[[3]]; "
            "shell_escape <- identical(args[[4]], 'true'); "
            "engine_args <- c('-interaction=nonstopmode', '-file-line-error', "
            "paste0('-output-directory=', output_dir)); "
            "if (shell_escape) engine_args <- c(engine_args, '-shell-escape'); "
            "tinytex::latexmk(file, engine = engine, engine_args = engine_args, "
            "emulation = TRUE, clean = FALSE, "
            f"install_packages = {install_packages_value})"
        )
        return [
            rscript,
            "--vanilla",
            "-e",
            expression,
            str(candidate),
            engine,
            str(build_path),
            "true" if allow_shell_escape else "false",
        ]

    def _run_r_tinytex_command(
        self,
        cmd: List[str],
        *,
        cwd: Path,
        log_path: Path,
        engine_log_path: Path,
        env: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, str, Optional[str], Optional[int]]:
        header = f"$ {' '.join(cmd)}\n"
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(header)
        try:
            process = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.total_timeout,
                encoding="utf-8",
                errors="replace",
                env=env or self._build_env(),
            )
            combined_log = process.stdout + "\n" + process.stderr
            engine_log = self._read_engine_log(engine_log_path)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(combined_log)
                log_file.write("\n")
                if engine_log:
                    log_file.write(
                        "\n"
                        f"===== TeX engine log: {engine_log_path} =====\n"
                    )
                    log_file.write(engine_log)
                    if not engine_log.endswith("\n"):
                        log_file.write("\n")
            full_log = log_path.read_text(encoding="utf-8", errors="replace")
            diagnostic_log = combined_log + "\n" + engine_log
            if process.returncode != 0:
                error_detail = self._compile_error_detail(
                    diagnostic_log,
                    driver="r_tinytex",
                )
                return (
                    False,
                    full_log,
                    f"R tinytex exited with code {process.returncode}. "
                    f"{error_detail}",
                    process.returncode,
                )
            return True, full_log, None, process.returncode
        except subprocess.TimeoutExpired:
            message = f"R tinytex compilation timed out after {self.total_timeout}s"
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(message + "\n")
            return (
                False,
                log_path.read_text(encoding="utf-8", errors="replace"),
                message,
                None,
            )
        except Exception as exc:
            message = str(exc)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(message + "\n")
            return (
                False,
                log_path.read_text(encoding="utf-8", errors="replace"),
                message,
                None,
            )

    @staticmethod
    def _read_engine_log(engine_log_path: Path) -> str:
        if not engine_log_path.exists():
            return ""
        try:
            return engine_log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    def _compile_file_with_r_tinytex(
        self,
        *,
        tex_file: Path,
        output_path: Path,
        build_path: Path,
        log_path: Path,
        diagnostic_path: Path,
        original_source: str,
        prepared_source: str,
        available_engines: List[str],
        allow_pdflatex_cjk: bool,
        allow_shell_escape: bool,
        max_repair_rounds: int,
        chinese_package: str,
        font_config: Optional[Any],
        env: Dict[str, str],
    ) -> CompilationResult:
        rscript = self._rscript_path(env)
        if not rscript:
            message = "R tinytex driver selected but Rscript was not found."
            log_path.write_text(message, encoding="utf-8")
            write_compile_error_summary(
                log_path,
                log_path.parent / "compile_error_summary.md",
            )
            return CompilationResult(
                success=False,
                log_content=message,
                error_message=message,
                diagnostic_path=diagnostic_path,
            )

        attempts: List[CompileAttempt] = []
        last_log = ""
        last_error: Optional[str] = None
        for engine in available_engines:
            engine_source = inject_chinese_support_for_engine(
                prepared_source,
                engine=engine,
                font_config=font_config,
                allow_pdflatex_cjk=allow_pdflatex_cjk,
                chinese_package=chinese_package,
                font_dir=self.fonts_dir,
            )
            applied_repairs: List[str] = []
            source_repair_count = 0
            base_repairs = [
                "r_tinytex_auto_install"
                if self.install_missing_packages
                else "r_tinytex"
            ]

            for round_index in range(max_repair_rounds + 1):
                if round_index == 0:
                    candidate = build_path / f"{tex_file.stem}.{engine}.r_tinytex.tex"
                else:
                    candidate = build_path / (
                        f"{tex_file.stem}.{engine}.r_tinytex.round{round_index + 1}.tex"
                    )
                candidate.write_text(engine_source, encoding="utf-8")
                engine_log_path = build_path / f"{candidate.stem}.log"
                cmd = self._build_r_tinytex_command(
                    rscript=rscript,
                    engine=engine,
                    candidate=candidate,
                    build_path=build_path,
                    allow_shell_escape=allow_shell_escape,
                    install_packages=self.install_missing_packages,
                )
                success, log_content, error, returncode = self._run_r_tinytex_command(
                    cmd,
                    cwd=tex_file.parent,
                    log_path=log_path,
                    engine_log_path=engine_log_path,
                    env=env,
                )
                last_log = log_content
                last_error = error
                attempt_log = self._read_engine_log(engine_log_path) or log_content
                missing_file = self._extract_missing_latex_file(attempt_log)
                first_error = self._extract_error(attempt_log)
                built_pdf = self._find_built_pdf(
                    build_path,
                    candidate_stem=candidate.stem,
                    tex_stem=tex_file.stem,
                )
                has_healthy_pdf = bool(
                    built_pdf and self._is_pdf_healthy(built_pdf)
                )
                category = (
                    "success"
                    if success and has_healthy_pdf
                    else self._classify_compile_log(attempt_log, error)
                )
                if has_healthy_pdf and not success:
                    category = "success_with_wrapper_warning"
                attempt = CompileAttempt(
                    engine=engine,
                    command=cmd,
                    tex_path=candidate,
                    success=success or has_healthy_pdf,
                    returncode=returncode,
                    error=error,
                    category=category,
                    repairs=[*base_repairs, *applied_repairs],
                    engine_log_path=engine_log_path if engine_log_path.exists() else None,
                    missing_file=missing_file,
                    first_error=first_error,
                    driver="r_tinytex",
                    driver_detail=rscript,
                )
                attempts.append(attempt)

                if has_healthy_pdf and built_pdf:
                    shutil.copy2(built_pdf, output_path)
                    if self._is_pdf_healthy(output_path):
                        self._sync_repaired_tex_on_success(
                            tex_file,
                            original_source,
                            engine_source,
                        )
                        self._write_compile_attempts(diagnostic_path, attempts)
                        warning_message = None
                        if not success:
                            warning_message = (
                                "R tinytex returned a non-zero exit code, "
                                "but a healthy PDF was produced and copied."
                            )
                        return CompilationResult(
                            success=True,
                            pdf_path=output_path,
                            log_content=log_content,
                            error_message=None,
                            warning_message=warning_message,
                            engine_used=engine,
                            attempts=attempts,
                            repaired_tex_path=candidate,
                            diagnostic_path=diagnostic_path,
                            driver="r_tinytex",
                            driver_detail=rscript,
                        )
                    last_error = "Generated PDF failed integrity check after copy."
                elif success:
                    last_error = "Generated PDF failed integrity check (%PDF/%%EOF)."

                patched_source, fallback_reason, changed = (
                    self._apply_compile_file_log_fallbacks(
                        engine_source,
                        log_content,
                        build_path,
                    )
                )
                if (
                    not changed
                    or fallback_reason in applied_repairs
                    or source_repair_count >= max_repair_rounds
                ):
                    break
                engine_source = patched_source
                applied_repairs.append(fallback_reason)
                source_repair_count += 1

        self._write_compile_attempts(diagnostic_path, attempts)
        log_content = log_path.read_text(encoding="utf-8", errors="replace")
        write_compile_error_summary(log_path, log_path.parent / "compile_error_summary.md")
        return CompilationResult(
            success=False,
            log_content=log_content or last_log,
            error_message=last_error or "R tinytex compilation failed.",
            attempts=attempts,
            diagnostic_path=diagnostic_path,
            driver="r_tinytex",
            driver_detail=rscript,
        )

    def compile_file(
        self,
        tex_file: Union[str, Path],
        output_path: Union[str, Path],
        logs_dir: Union[str, Path],
        build_dir: Optional[Union[str, Path]] = None,
        prefer_latexmk: bool = True,
        engine_policy: str = "auto",
        fallback_engines: Optional[List[str]] = None,
        allow_pdflatex_cjk: bool = False,
        allow_shell_escape: bool = False,
        max_repair_rounds: int = 3,
        chinese_package: str = "auto",
        font_config: Optional[Any] = None,
    ) -> CompilationResult:
        """Compile an existing TeX file while preserving logs and build artifacts."""
        tex_file = Path(tex_file).resolve()
        output_path = Path(output_path).resolve()
        logs_dir = Path(logs_dir).resolve()
        build_path = Path(build_dir).resolve() if build_dir else logs_dir / "build"
        log_path = logs_dir / "compile.log"
        summary_path = logs_dir / "compile_error_summary.md"
        diagnostic_path = logs_dir / "compile_attempts.json"

        logs_dir.mkdir(parents=True, exist_ok=True)
        build_path.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        env = self._build_env()
        env = self._with_texinputs(
            env,
            build_path=build_path,
            tex_dir=tex_file.parent,
        )

        if not tex_file.exists():
            message = f"TeX file not found: {tex_file}"
            log_path.write_text(message, encoding="utf-8")
            write_compile_error_summary(log_path, summary_path)
            return CompilationResult(
                success=False,
                log_content=message,
                error_message=message,
                diagnostic_path=diagnostic_path,
            )

        try:
            original_source = tex_file.read_text(encoding="utf-8")
        except Exception as exc:
            message = f"Unable to read TeX file: {exc}"
            log_path.write_text(message, encoding="utf-8")
            write_compile_error_summary(log_path, summary_path)
            return CompilationResult(
                success=False,
                log_content=message,
                error_message=message,
                diagnostic_path=diagnostic_path,
            )

        prepared_source, uses_precompiled_bbl = self._prepare_compile_source(
            tex_file,
            build_path,
            original_source=original_source,
        )
        attempts: List[CompileAttempt] = []
        available_engines = self._select_compile_file_engines(
            engine_policy=engine_policy,
            fallback_engines=fallback_engines,
            latex_source=prepared_source,
            allow_pdflatex_cjk=allow_pdflatex_cjk,
        )
        compile_driver, driver_detail = self._resolve_compile_driver(env)
        if compile_driver == "missing_r_tinytex":
            message = (
                "R tinytex driver requested but unavailable: "
                f"{driver_detail or 'unknown reason'}"
            )
            log_path.write_text(message, encoding="utf-8")
            self._write_compile_attempts(diagnostic_path, attempts)
            write_compile_error_summary(log_path, summary_path)
            return CompilationResult(
                success=False,
                log_content=message,
                error_message=message,
                attempts=attempts,
                diagnostic_path=diagnostic_path,
                driver="r_tinytex",
                driver_detail=driver_detail,
            )
        if compile_driver == "r_tinytex":
            self._append_compile_log(
                log_path,
                "Using R tinytex driver"
                + (
                    " for automatic package installation."
                    if self.install_missing_packages
                    else " without automatic package installation."
                ),
            )
            return self._compile_file_with_r_tinytex(
                tex_file=tex_file,
                output_path=output_path,
                build_path=build_path,
                log_path=log_path,
                diagnostic_path=diagnostic_path,
                original_source=original_source,
                prepared_source=prepared_source,
                available_engines=available_engines,
                allow_pdflatex_cjk=allow_pdflatex_cjk,
                allow_shell_escape=allow_shell_escape,
                max_repair_rounds=max_repair_rounds,
                chinese_package=chinese_package,
                font_config=font_config,
                env=env,
            )

        if prefer_latexmk and self._which("latexmk", env):
            command_mode = "latexmk"
        else:
            command_mode = "direct"

        if command_mode == "direct":
            available_engines = [
                engine for engine in available_engines if self._which(engine, env)
            ]
        if not available_engines:
            message = "No usable LaTeX engine was found on PATH."
            log_path.write_text(message, encoding="utf-8")
            self._write_compile_attempts(diagnostic_path, attempts)
            write_compile_error_summary(log_path, summary_path)
            return CompilationResult(
                success=False,
                log_content=message,
                error_message=message,
                attempts=attempts,
                diagnostic_path=diagnostic_path,
            )

        if self.tinytex_driver == "auto" and self.use_tinytex and driver_detail:
            self._append_compile_log(
                log_path,
                "R tinytex driver unavailable; falling back to latexmk: "
                f"{driver_detail}",
            )

        last_log = ""
        last_error: Optional[str] = None
        successful_source: Optional[str] = None
        successful_candidate: Optional[Path] = None
        successful_engine: Optional[str] = None

        for engine in available_engines:
            engine_source = inject_chinese_support_for_engine(
                prepared_source,
                engine=engine,
                font_config=font_config,
                allow_pdflatex_cjk=allow_pdflatex_cjk,
                chinese_package=chinese_package,
                font_dir=self.fonts_dir,
            )
            applied_repairs: List[str] = []
            source_repair_count = 0

            for round_index in range(max_repair_rounds + 1):
                candidate = build_path / (
                    f"{tex_file.stem}.{engine}.round{round_index + 1}.tex"
                )
                candidate.write_text(engine_source, encoding="utf-8")
                engine_log_path = build_path / f"{candidate.stem}.log"
                cmd = self._build_compile_file_command(
                    engine=engine,
                    candidate=candidate,
                    build_path=build_path,
                    uses_precompiled_bbl=uses_precompiled_bbl,
                    prefer_latexmk=(command_mode == "latexmk"),
                    allow_shell_escape=allow_shell_escape,
                )
                (
                    success,
                    log_content,
                    error,
                    returncode,
                ) = self._run_compile_file_command(
                    cmd,
                    cwd=tex_file.parent,
                    log_path=log_path,
                    engine_log_path=engine_log_path,
                    env=env,
                )
                last_log = log_content
                last_error = error
                attempt_log = self._read_engine_log(engine_log_path) or log_content
                missing_file = self._extract_missing_latex_file(attempt_log)
                first_error = self._extract_error(attempt_log)
                built_pdf = self._find_built_pdf(
                    build_path,
                    candidate_stem=candidate.stem,
                    tex_stem=tex_file.stem,
                )
                has_healthy_pdf = bool(
                    built_pdf and self._is_pdf_healthy(built_pdf)
                )
                category = (
                    "success"
                    if success and has_healthy_pdf
                    else self._classify_compile_log(attempt_log, error)
                )
                if has_healthy_pdf and not success:
                    category = "success_with_latexmk_warning"
                attempt = CompileAttempt(
                    engine=engine,
                    command=cmd,
                    tex_path=candidate,
                    success=success or has_healthy_pdf,
                    returncode=returncode,
                    error=error,
                    category=category,
                    repairs=list(applied_repairs),
                    engine_log_path=engine_log_path if engine_log_path.exists() else None,
                    missing_file=missing_file,
                    first_error=first_error,
                    driver=command_mode,
                    driver_detail=driver_detail or "configured",
                )
                attempts.append(attempt)

                if has_healthy_pdf and built_pdf:
                    shutil.copy2(built_pdf, output_path)
                    if self._is_pdf_healthy(output_path):
                        successful_source = engine_source
                        successful_candidate = candidate
                        successful_engine = engine
                        self._sync_repaired_tex_on_success(
                            tex_file,
                            original_source,
                            successful_source,
                        )
                        self._write_compile_attempts(diagnostic_path, attempts)
                        return CompilationResult(
                            success=True,
                            pdf_path=output_path,
                            log_content=log_content,
                            engine_used=successful_engine,
                            attempts=attempts,
                            repaired_tex_path=successful_candidate,
                            diagnostic_path=diagnostic_path,
                            warning_message=(
                                "latexmk returned a non-zero exit code, "
                                "but a healthy PDF was produced and copied."
                                if not success
                                else None
                            ),
                            driver=command_mode,
                            driver_detail=driver_detail or "configured",
                        )
                    last_error = "Generated PDF failed integrity check after copy."

                if success:
                    last_error = "Generated PDF failed integrity check (%PDF/%%EOF)."

                patched_source, fallback_reason, changed = (
                    self._apply_compile_file_log_fallbacks(
                        engine_source,
                        attempt_log,
                        build_path,
                    )
                )
                if (
                    not changed
                    or fallback_reason in applied_repairs
                    or source_repair_count >= max_repair_rounds
                ):
                    break
                engine_source = patched_source
                applied_repairs.append(fallback_reason)
                source_repair_count += 1

        self._write_compile_attempts(diagnostic_path, attempts)
        log_content = log_path.read_text(encoding="utf-8", errors="replace")
        write_compile_error_summary(log_path, summary_path)
        return CompilationResult(
            success=False,
            log_content=log_content or last_log,
            error_message=last_error or "Compilation failed.",
            attempts=attempts,
            diagnostic_path=diagnostic_path,
            driver=command_mode if "command_mode" in locals() else compile_driver,
            driver_detail=driver_detail or "configured",
        )

    def compile(
        self,
        latex_source: str,
        output_path: Union[str, Path],
        working_dir: Optional[Union[str, Path]] = None,
    ) -> CompilationResult:
        """
        Compiles LaTeX source to PDF using multiple engines with fallback.

        Args:
            latex_source: The LaTeX code to compile.
            output_path: Where to save the generated PDF.
            working_dir: Optional directory containing resources (images, etc.).
                         If provided, contents are copied to the temp compile dir.
        """
        output_path = Path(output_path).resolve()
        compile_source = normalize_unicode_engine_source(latex_source)

        # Create a temporary directory for compilation to keep things clean
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # If working_dir is provided, copy its contents to temp_dir
            if working_dir:
                working_dir_path = Path(working_dir)
                if working_dir_path.exists():
                    self._copy_resources(working_dir_path, temp_path)

            source_file = temp_path / "main.tex"
            env = self._build_env()
            last_error = None
            last_log = None
            applied_fallback_reasons = set()
            engine_failures: List[Tuple[str, Optional[str], str]] = []
            last_round_failures: List[Tuple[str, Optional[str], str]] = []
            failure_rank: Dict[str, int] = {
                "fontspec": 0,
                "package": 1,
                "latex": 2,
                "unknown": 3,
            }

            for _round in range(6):
                source_file.write_text(compile_source, encoding="utf-8")
                missing_font: Optional[str] = None
                missing_file: Optional[str] = None
                has_microtype_tracking_error = False
                conflict_primitive: Optional[str] = None
                round_failures: List[Tuple[str, Optional[str], str]] = []

                for engine in self.engines:
                    # Skip engines that are not installed
                    if not self._which(engine, env):
                        continue

                    success, log, error = self._run_engine(
                        engine, source_file, temp_path, compile_source
                    )

                    if success:
                        # Move generated PDF to output_path
                        pdf_file = temp_path / "main.pdf"
                        if pdf_file.exists() and self._is_pdf_healthy(pdf_file):
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(pdf_file, output_path)
                            if self._is_pdf_healthy(output_path):
                                return CompilationResult(
                                    success=True,
                                    pdf_path=output_path,
                                    log_content=log,
                                    engine_used=engine,
                                )
                            last_log = log
                            last_error = (
                                "Generated PDF failed integrity check after copy."
                            )
                            continue
                        last_log = log
                        last_error = "Generated PDF failed integrity check."
                        continue

                    last_log = log
                    last_error = error
                    if error:
                        error_tag = self._classify_error_text(error)
                        failure_record = (engine, error, error_tag)
                        engine_failures.append(failure_record)
                        round_failures.append(failure_record)
                    if missing_font is None:
                        missing_font = self._extract_missing_font_name(log)
                    if missing_file is None:
                        missing_file = self._extract_missing_latex_file(log)
                    if not has_microtype_tracking_error:
                        has_microtype_tracking_error = self._has_microtype_tracking_error(
                            log
                        )
                    if conflict_primitive is None:
                        conflict_primitive = self._extract_engine_conflict_primitive(
                            log
                        )

                fallback_applied = False
                if missing_font:
                    (
                        patched_source,
                        fallback_reason,
                    ) = self._apply_missing_font_fallback(
                        compile_source,
                        missing_font,
                    )
                    if (
                        patched_source != compile_source
                        and fallback_reason not in applied_fallback_reasons
                    ):
                        compile_source = patched_source
                        applied_fallback_reasons.add(fallback_reason)
                        fallback_applied = True

                if not fallback_applied and missing_file:
                    (
                        patched_source,
                        fallback_reason,
                        fallback_changed,
                    ) = self._apply_missing_file_fallback(
                        compile_source,
                        missing_file,
                        workspace_dir=temp_path,
                    )
                    if (
                        fallback_changed
                        and fallback_reason not in applied_fallback_reasons
                    ):
                        compile_source = patched_source
                        applied_fallback_reasons.add(fallback_reason)
                        fallback_applied = True

                if not fallback_applied and conflict_primitive:
                    (
                        patched_source,
                        fallback_reason,
                        fallback_changed,
                    ) = self._apply_pdftex_primitive_fallback(
                        compile_source,
                        workspace_dir=temp_path,
                    )
                    if (
                        fallback_changed
                        and fallback_reason not in applied_fallback_reasons
                    ):
                        compile_source = patched_source
                        applied_fallback_reasons.add(fallback_reason)
                        fallback_applied = True

                if not fallback_applied and has_microtype_tracking_error:
                    (
                        patched_source,
                        fallback_reason,
                        fallback_changed,
                    ) = self._apply_microtype_tracking_fallback(
                        compile_source,
                        workspace_dir=temp_path,
                    )
                    if (
                        fallback_changed
                        and fallback_reason not in applied_fallback_reasons
                    ):
                        compile_source = patched_source
                        applied_fallback_reasons.add(fallback_reason)
                        fallback_applied = True

                if round_failures:
                    last_round_failures = round_failures
                if fallback_applied:
                    continue
                break

            # If we reach here, all engines failed
            failures_for_report = last_round_failures or engine_failures
            if (
                failures_for_report
                and all(tag == "unknown" for _, _, tag in failures_for_report)
                and any(tag != "unknown" for _, _, tag in engine_failures)
            ):
                failures_for_report = engine_failures

            if failures_for_report:
                best_engine, best_error, _ = sorted(
                    failures_for_report,
                    key=lambda item: failure_rank.get(item[2], 99),
                )[0]
                last_error = f"[{best_engine}] {best_error}"

            return CompilationResult(
                success=False,
                log_content=last_log,
                error_message=f"All engines failed. Last error: {last_error}",
                engine_used=None,
            )

    def _copy_resources(self, src: Path, dst: Path):
        """Copies resource files from src to dst, ignoring hidden files."""
        try:
            for item in src.iterdir():
                if item.name.startswith("."):
                    continue

                target = dst / item.name
                if item.is_dir():
                    shutil.copytree(item, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, target)
        except Exception:
            # Ignore copy errors (e.g. permission issues), compilation might still work
            pass

    def _prepare_compile_source(
        self,
        tex_file: Path,
        build_path: Path,
        *,
        original_source: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """Normalize TeX source and align bibliography artifacts without mutating TeX."""
        if original_source is None:
            try:
                original_source = tex_file.read_text(encoding="utf-8")
            except Exception:
                return "", False

        original_source = self._repair_repeated_translated_preamble(
            tex_file,
            original_source,
        )
        normalized = normalize_unicode_engine_source(original_source)
        rewritten, bbl_aliases = self._rewrite_bibliography_to_precompiled_bbl(
            normalized,
            tex_file,
        )
        self._copy_bbl_aliases(tex_file, build_path, bbl_aliases)
        self._copy_build_local_inputs(tex_file, build_path)
        return rewritten, bool(bbl_aliases)

    @staticmethod
    def _repair_repeated_translated_preamble(tex_file: Path, source: str) -> str:
        """Recover from translation outputs that duplicate local preamble content."""
        begin_match = re.search(r"\\begin\{document\}", source)
        if not begin_match:
            return source

        preamble = source[: begin_match.start()]
        if not LaTeXCompiler._looks_like_repeated_translated_preamble(preamble):
            return source

        sibling_candidates: List[Path] = []
        if tex_file.name == "main_zh.tex":
            sibling_candidates.append(tex_file.with_name("main.tex"))
        if tex_file.stem.endswith("_zh"):
            sibling_candidates.append(tex_file.with_name(f"{tex_file.stem[:-3]}.tex"))

        for sibling in dict.fromkeys(sibling_candidates):
            if sibling == tex_file or not sibling.exists():
                continue
            try:
                sibling_source = sibling.read_text(encoding="utf-8")
            except Exception:
                continue
            sibling_begin = re.search(r"\\begin\{document\}", sibling_source)
            if not sibling_begin:
                continue
            sibling_preamble = sibling_source[: sibling_begin.start()]
            if len(sibling_preamble) >= len(preamble) / 2:
                continue
            return sibling_preamble.rstrip() + "\n\n" + source[begin_match.start() :]

        return source

    @staticmethod
    def _looks_like_repeated_translated_preamble(preamble: str) -> bool:
        preamble_lines = preamble.count("\n") + 1
        if len(preamble) < 120_000 and preamble_lines < 2_000:
            return False

        repeated_markers = (
            preamble.count(r"\newcommand{\red}")
            + preamble.count(r"\newcommand{\todo}")
            + preamble.count(r"\newcommand{\greencircle}")
            + preamble.count("之后即可")
        )
        return repeated_markers >= 20

    def _copy_build_local_inputs(self, tex_file: Path, build_path: Path) -> None:
        """Copy mutable local TeX support files to build/ for compile fallbacks."""
        source_dir = tex_file.parent
        if not source_dir.exists():
            return
        build_path.mkdir(parents=True, exist_ok=True)
        for path in source_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".cls", ".sty", ".def"}:
                continue
            try:
                if build_path in path.resolve().parents:
                    continue
                relative = path.relative_to(source_dir)
            except Exception:
                continue
            target = build_path / relative
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            except Exception:
                continue

    def _prepare_compile_inputs(self, tex_file: Path, build_path: Path) -> bool:
        """Normalize TeX source and align precompiled bibliography artifacts."""
        try:
            source = tex_file.read_text(encoding="utf-8")
        except Exception:
            return False

        rewritten, uses_precompiled_bbl = self._prepare_compile_source(
            tex_file,
            build_path,
            original_source=source,
        )
        if rewritten != source:
            tex_file.write_text(rewritten, encoding="utf-8")

        return uses_precompiled_bbl

    def _select_compile_file_engines(
        self,
        *,
        engine_policy: str,
        fallback_engines: Optional[List[str]],
        latex_source: str,
        allow_pdflatex_cjk: bool,
    ) -> List[str]:
        supported = {"xelatex", "lualatex", "pdflatex"}
        if engine_policy != "auto":
            requested = [engine_policy]
        else:
            requested = list(fallback_engines or ["xelatex", "lualatex"])
            if not contains_cjk_text(latex_source) and "pdflatex" not in requested:
                requested.append("pdflatex")

        engines: List[str] = []
        for engine in requested:
            engine = engine.lower()
            if engine not in supported:
                continue
            if engine == "pdflatex" and contains_cjk_text(latex_source):
                if not allow_pdflatex_cjk:
                    continue
            if engine not in engines:
                engines.append(engine)
        return engines

    def _build_compile_file_command(
        self,
        *,
        engine: str,
        candidate: Path,
        build_path: Path,
        uses_precompiled_bbl: bool,
        prefer_latexmk: bool,
        allow_shell_escape: bool,
    ) -> List[str]:
        if prefer_latexmk:
            engine_flag = "-pdf" if engine == "pdflatex" else f"-{engine}"
            cmd = [
                "latexmk",
                engine_flag,
                "-interaction=nonstopmode",
                "-file-line-error",
                f"-output-directory={build_path}",
            ]
            if allow_shell_escape:
                cmd.append("-shell-escape")
            if uses_precompiled_bbl:
                cmd.append("-bibtex-")
            cmd.append(str(candidate))
            return cmd

        cmd = [
            engine,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-output-directory={build_path}",
        ]
        if allow_shell_escape:
            cmd.append("-shell-escape")
        cmd.append(str(candidate))
        return cmd

    def _run_compile_file_command(
        self,
        cmd: List[str],
        *,
        cwd: Path,
        log_path: Path,
        engine_log_path: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, str, Optional[str], Optional[int]]:
        header = f"$ {' '.join(cmd)}\n"
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(header)
        try:
            process = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding="utf-8",
                errors="replace",
                env=env or self._build_env(),
            )
            combined_log = process.stdout + "\n" + process.stderr
            engine_log = (
                self._read_engine_log(engine_log_path) if engine_log_path else ""
            )
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(combined_log)
                log_file.write("\n")
                if engine_log and engine_log_path:
                    log_file.write(
                        "\n"
                        f"===== TeX engine log: {engine_log_path} =====\n"
                    )
                    log_file.write(engine_log)
                    if not engine_log.endswith("\n"):
                        log_file.write("\n")
            full_log = log_path.read_text(encoding="utf-8", errors="replace")
            diagnostic_log = combined_log + "\n" + engine_log
            if process.returncode != 0:
                error_detail = self._compile_error_detail(
                    diagnostic_log,
                    driver="latexmk",
                )
                return (
                    False,
                    full_log,
                    f"Compilation command exited with code {process.returncode}. "
                    f"{error_detail}",
                    process.returncode,
                )
            return True, full_log, None, process.returncode
        except subprocess.TimeoutExpired:
            message = "Compilation timed out"
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(message + "\n")
            return (
                False,
                log_path.read_text(encoding="utf-8", errors="replace"),
                message,
                None,
            )
        except Exception as exc:
            message = str(exc)
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(message + "\n")
            return (
                False,
                log_path.read_text(encoding="utf-8", errors="replace"),
                message,
                None,
            )

    @staticmethod
    def _append_compile_log(log_path: Path, message: str) -> None:
        if not message:
            return
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(message.rstrip() + "\n")

    @staticmethod
    def _find_built_pdf(
        build_path: Path,
        *,
        candidate_stem: str,
        tex_stem: str,
    ) -> Optional[Path]:
        for pdf_path in (
            build_path / f"{candidate_stem}.pdf",
            build_path / f"{tex_stem}.pdf",
        ):
            if pdf_path.exists():
                return pdf_path
        return None

    @staticmethod
    def _sync_repaired_tex_on_success(
        tex_file: Path,
        original_source: str,
        repaired_source: str,
    ) -> None:
        if repaired_source == original_source:
            return
        backup = tex_file.with_name(f"{tex_file.stem}.before_compile{tex_file.suffix}")
        if not backup.exists():
            backup.write_text(original_source, encoding="utf-8")
        tex_file.write_text(repaired_source, encoding="utf-8")

    def _apply_compile_file_log_fallbacks(
        self,
        latex_source: str,
        log_content: str,
        workspace_dir: Path,
    ) -> Tuple[str, str, bool]:
        missing_font = self._extract_missing_font_name(log_content)
        if missing_font:
            patched, reason = self._apply_missing_font_fallback(
                latex_source,
                missing_font,
            )
            if patched != latex_source:
                return patched, reason, True

        missing_file = self._extract_missing_latex_file(log_content)
        if missing_file:
            patched, reason, changed = self._apply_missing_file_fallback(
                latex_source,
                missing_file,
                workspace_dir=workspace_dir,
            )
            if changed:
                return patched, reason, True

        if self._has_axessibility_pdftex_primitive_error(log_content):
            patched, reason, changed = self._apply_axessibility_fallback(
                latex_source,
                workspace_dir=workspace_dir,
            )
            if changed:
                return patched, reason, True

        if self._extract_engine_conflict_primitive(log_content):
            patched, reason, changed = self._apply_pdftex_primitive_fallback(
                latex_source,
                workspace_dir=workspace_dir,
            )
            if changed:
                return patched, reason, True

        if self._has_microtype_tracking_error(log_content):
            patched, reason, changed = self._apply_microtype_tracking_fallback(
                latex_source,
                workspace_dir=workspace_dir,
            )
            if changed:
                return patched, reason, True

        if self._has_microtype_disable_ligatures_error(log_content):
            patched, reason, changed = self._apply_microtype_disable_ligatures_fallback(
                latex_source,
                workspace_dir=workspace_dir,
            )
            if changed:
                return patched, reason, True

        redefined_commands = self._extract_redefined_commands(log_content)
        if redefined_commands:
            patched, reason, changed = self._apply_redefined_command_fallback(
                latex_source,
                redefined_commands,
                workspace_dir=workspace_dir,
            )
            if changed:
                return patched, reason, True

        if self._has_noopndent_error(log_content):
            patched, reason, changed = self._apply_noopndent_fallback(
                latex_source,
                workspace_dir=workspace_dir,
            )
            if changed:
                return patched, reason, True

        return latex_source, "no_fallback", False

    @staticmethod
    def _has_axessibility_pdftex_primitive_error(log: str) -> bool:
        lowered = (log or "").lower()
        if "axessibility" not in lowered or "undefined control sequence" not in lowered:
            return False
        return any(
            f"\\{primitive}" in lowered
            for primitive in (
                "pdfcompresslevel",
                "pdfobjcompresslevel",
                "pdfoutput",
                "pdfminorversion",
                "pdfmapline",
            )
        )

    @staticmethod
    def _rewrite_remove_axessibility(text: str) -> Tuple[str, bool]:
        pattern = re.compile(
            r"\\(?P<cmd>RequirePackage|usepackage)"
            r"(?P<opts>\[[^\]]*\])?\{(?P<pkgs>[^}]*)\}"
        )
        changed = False

        def replace(match: re.Match[str]) -> str:
            nonlocal changed
            packages = [pkg.strip() for pkg in match.group("pkgs").split(",")]
            if not any(pkg.lower() == "axessibility" for pkg in packages):
                return match.group(0)

            changed = True
            kept = [pkg for pkg in packages if pkg and pkg.lower() != "axessibility"]
            if not kept:
                return ""
            return (
                f"\\{match.group('cmd')}"
                f"{match.group('opts') or ''}"
                + "{"
                + ",".join(kept)
                + "}"
            )

        patched = pattern.sub(replace, text)
        return patched, changed

    def _apply_axessibility_fallback(
        self,
        latex_source: str,
        workspace_dir: Optional[Path] = None,
    ) -> Tuple[str, str, bool]:
        patched_source, source_changed = self._rewrite_remove_axessibility(
            latex_source
        )
        workspace_changed = False
        if workspace_dir and workspace_dir.exists():
            workspace_changed = self._patch_workspace_text_files(
                workspace_dir,
                self._rewrite_remove_axessibility,
                skip_main_tex=True,
            )
        changed = source_changed or workspace_changed
        if changed:
            return patched_source, "fallback_remove_axessibility", True
        return latex_source, "no_fallback", False

    @staticmethod
    def _classify_compile_log(log_content: str, error: Optional[str] = None) -> str:
        lowered = f"{log_content or ''}\n{error or ''}".lower()
        if re.search(
            r"(?:(?:must|requires?|needed?|disabled)\W+(?:.{0,80})"
            r"(?:shell-escape|write18)|(?:shell-escape|write18)\W+(?:.{0,80})"
            r"(?:required|requires?|needed?|disabled|must))",
            lowered,
        ):
            return "shell_escape"
        if re.search(
            r"(?:package\s+fontspec\s+error|fontspec\s+error|the font\s+\"[^\"]+\"\s+cannot\s+be)",
            lowered,
        ):
            return "fontspec"
        if re.search(r"file\s+[`'][^`']+[`']\s+not\s+found", lowered):
            return "missing_file"
        if "bibtex" in lowered or "biber" in lowered:
            return "bibliography"
        if "requires xetex" in lowered or "requires luatex" in lowered:
            return "engine_mismatch"
        if "unicode character" in lowered or "cjk" in lowered:
            return "unicode"
        if "package" in lowered:
            return "package"
        if "undefined control sequence" in lowered or "latex error" in lowered:
            return "latex"
        return "unknown"

    @staticmethod
    def _attempt_to_json(attempt: CompileAttempt) -> Dict[str, Any]:
        return {
            "engine": attempt.engine,
            "command": attempt.command,
            "tex_path": str(attempt.tex_path) if attempt.tex_path else None,
            "success": attempt.success,
            "returncode": attempt.returncode,
            "error": attempt.error,
            "category": attempt.category,
            "repairs": attempt.repairs,
            "engine_log_path": (
                str(attempt.engine_log_path) if attempt.engine_log_path else None
            ),
            "missing_file": attempt.missing_file,
            "first_error": attempt.first_error,
            "driver": attempt.driver,
            "driver_detail": attempt.driver_detail,
        }

    def _write_compile_attempts(
        self,
        diagnostic_path: Path,
        attempts: List[CompileAttempt],
    ) -> None:
        diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"attempts": [self._attempt_to_json(item) for item in attempts]}
        diagnostic_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _rewrite_bibliography_to_precompiled_bbl(
        self,
        latex_source: str,
        tex_file: Path,
    ) -> Tuple[str, List[Tuple[Path, str]]]:
        aliases: List[Tuple[Path, str]] = []
        base_dir = tex_file.parent

        def select_bbl(names: List[str]) -> Optional[List[Tuple[Path, str]]]:
            if any((base_dir / f"{name}.bib").exists() for name in names):
                return None

            named_bbls = [(base_dir / f"{name}.bbl", f"{name}.bbl") for name in names]
            if named_bbls and all(path.exists() for path, _ in named_bbls):
                return named_bbls

            stem_bbl = base_dir / f"{tex_file.stem}.bbl"
            if stem_bbl.exists():
                return [(stem_bbl, stem_bbl.name)]

            main_bbl = base_dir / "main.bbl"
            if main_bbl.exists():
                return [(main_bbl, f"{tex_file.stem}.bbl")]

            bbl_files = sorted(base_dir.glob("*.bbl"))
            if len(bbl_files) == 1:
                return [(bbl_files[0], f"{tex_file.stem}.bbl")]

            return None

        def replace(match: re.Match[str]) -> str:
            names = [name.strip() for name in match.group(1).split(",") if name.strip()]
            selected = select_bbl(names)
            if not selected:
                return match.group(0)
            aliases.extend(selected)
            return "\n".join(f"\\input{{{alias_name}}}" for _, alias_name in selected)

        rewritten = re.sub(r"\\bibliography\{([^}]+)\}", replace, latex_source)
        if rewritten != latex_source:
            rewritten = re.sub(r"\\bibliographystyle\{[^}]+\}\n?", "", rewritten)
        return rewritten, aliases

    def _copy_bbl_aliases(
        self,
        tex_file: Path,
        build_path: Path,
        aliases: List[Tuple[Path, str]],
    ) -> None:
        if not aliases:
            stem_bbl = tex_file.parent / f"{tex_file.stem}.bbl"
            if stem_bbl.exists():
                aliases = [(stem_bbl, stem_bbl.name)]
            else:
                return

        build_path.mkdir(parents=True, exist_ok=True)
        for source_path, alias_name in aliases:
            if not source_path.exists():
                continue
            target_path = tex_file.parent / alias_name
            try:
                if source_path.resolve() != target_path.resolve():
                    shutil.copy2(source_path, target_path)
                shutil.copy2(target_path, build_path / alias_name)
            except Exception:
                continue

    def _run_engine(
        self, engine: str, source_file: Path, cwd: Path, latex_source: str
    ) -> Tuple[bool, str, Optional[str]]:
        """Runs the full compilation cycle: (xelatex + bibtex) × 2."""
        env = self._build_env()

        # 1. First xelatex pass (generate .aux for bibtex)
        success, log, error = self._run_single_pass(engine, source_file, cwd, env=env)
        # Don't fail on first pass - references will be unresolved

        # 2. Check for existing .bbl file (pre-compiled bibliography)
        main_bbl = cwd / "main.bbl"
        if not main_bbl.exists():
            bbl_files = list(cwd.glob("*.bbl"))
            if bbl_files:
                shutil.copy2(bbl_files[0], main_bbl)

        # 3. Run bibtex (always try if bibliography command exists)
        bib_tool = self._detect_bibliography_tool(latex_source)
        if bib_tool and self._which(bib_tool, env):
            self._run_bibliography_tool(bib_tool, cwd, env=env)

        # 4. Second xelatex pass (incorporate bibliography)
        self._run_single_pass(engine, source_file, cwd, env=env)

        # 5. Run bibtex again (resolve any new citations)
        if bib_tool and self._which(bib_tool, env):
            self._run_bibliography_tool(bib_tool, cwd, env=env)

        # 6. Third xelatex pass (resolve all cross-references)
        self._run_single_pass(engine, source_file, cwd, env=env)

        # 7. Fourth xelatex pass (final - ensure all references resolved)
        success, log, error = self._run_single_pass(engine, source_file, cwd, env=env)

        return success, log, error

    def _detect_bibliography_tool(self, latex_source: str) -> Optional[str]:
        # Check for biblatex -> biber
        if re.search(r"\\usepackage(\[.*\])?\{biblatex\}", latex_source):
            return "biber"
        # Check for standard bibliography -> bibtex
        if re.search(r"\\bibliography\{", latex_source):
            return "bibtex"
        return None

    def _extract_missing_font_name(self, log: str) -> Optional[str]:
        if not log:
            return None
        match = re.search(
            r'fontsp\s*ec Error:\s*The font "([^"]+)" cannot be(?:\s*\n.*)?\s*found',
            log,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        return None

    def _apply_missing_font_fallback(
        self,
        latex_source: str,
        missing_font: str,
    ) -> Tuple[str, str]:
        missing_key = missing_font.strip().lower()

        if missing_key in {"fontawesome", "font awesome"}:
            if re.search(
                r"\\usepackage(?:\[[^\]]*\])?\{fontawesome\}",
                latex_source,
            ) and not re.search(r"\\fa[A-Za-z]+", latex_source):
                patched = re.sub(
                    r"^[ \t]*\\usepackage(?:\[[^\]]*\])?\{fontawesome\}[ \t]*\n?",
                    "",
                    latex_source,
                    flags=re.MULTILINE,
                )
                if patched != latex_source:
                    return patched, "remove_unused_fontawesome"
            return latex_source, "no_fallback"

        main_match = re.search(
            r"\\setCJKmainfont(?:\[[^\]]*\])?\{([^}]*)\}",
            latex_source,
        )
        if not main_match:
            return latex_source, "no_fallback"

        main_font = main_match.group(1).strip()
        if not main_font:
            return latex_source, "no_fallback"

        if main_font.lower() == missing_key:
            patched_source = latex_source
            patched_source = re.sub(
                r"^[ \t]*\\setCJKmainfont(?:\[[^\]]*\])?\{[^}]*\}[ \t]*\n?",
                "",
                patched_source,
                flags=re.MULTILINE,
            )
            patched_source = re.sub(
                r"^[ \t]*\\setCJKsansfont(?:\[[^\]]*\])?\{[^}]*\}[ \t]*\n?",
                "",
                patched_source,
                flags=re.MULTILINE,
            )
            patched_source = re.sub(
                r"^[ \t]*\\setCJKmonofont(?:\[[^\]]*\])?\{[^}]*\}[ \t]*\n?",
                "",
                patched_source,
                flags=re.MULTILINE,
            )
            if patched_source != latex_source:
                return patched_source, "fallback_remove_explicit_cjk_fonts"
            return latex_source, "no_fallback"

        sans_match = re.search(
            r"\\setCJKsansfont(?:\[[^\]]*\])?\{([^}]*)\}",
            latex_source,
        )
        mono_match = re.search(
            r"\\setCJKmonofont(?:\[[^\]]*\])?\{([^}]*)\}",
            latex_source,
        )
        sans_font = sans_match.group(1).strip() if sans_match else None
        mono_font = mono_match.group(1).strip() if mono_match else None

        should_patch = False
        if sans_font and sans_font.lower() == missing_key:
            should_patch = True
        if mono_font and mono_font.lower() == missing_key:
            should_patch = True
        if not should_patch:
            return latex_source, "no_fallback"

        patched_source = latex_source
        patched_source = re.sub(
            r"(\\setCJKsansfont(?:\[[^\]]*\])?\{)[^}]*\}",
            rf"\g<1>{main_font}" + "}",
            patched_source,
            count=1,
        )
        patched_source = re.sub(
            r"(\\setCJKmonofont(?:\[[^\]]*\])?\{)[^}]*\}",
            rf"\g<1>{main_font}" + "}",
            patched_source,
            count=1,
        )
        if patched_source != latex_source:
            return patched_source, "fallback_cjk_aux_fonts_to_main"

        return latex_source, "no_fallback"

    def _extract_missing_latex_file(self, log: str) -> Optional[str]:
        if not log:
            return None
        match = re.search(
            r"File [`']([^`']+)[`'] not found",
            log,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _rewrite_bxcoloremoji_names_false(text: str) -> Tuple[str, bool]:
        pattern = (
            r"\\(?P<cmd>RequirePackage|usepackage)"
            r"(?:\[(?P<opts>[^\]]*)\])?\{bxcoloremoji\}"
        )

        def _replace(match: re.Match[str]) -> str:
            cmd = match.group("cmd")
            opts = match.group("opts")
            if opts is None:
                return f"\\{cmd}[names=false]" + "{bxcoloremoji}"

            parts = [part.strip() for part in opts.split(",") if part.strip()]
            new_parts: List[str] = []
            has_names = False
            for part in parts:
                if re.fullmatch(r"names(?:\s*=\s*true)?", part, flags=re.IGNORECASE):
                    new_parts.append("names=false")
                    has_names = True
                elif re.fullmatch(
                    r"names\s*=\s*false", part, flags=re.IGNORECASE
                ):
                    new_parts.append("names=false")
                    has_names = True
                else:
                    new_parts.append(part)
            if not has_names:
                new_parts.append("names=false")
            return f"\\{cmd}[{','.join(new_parts)}]" + "{bxcoloremoji}"

        patched = re.sub(pattern, _replace, text)
        return patched, patched != text

    def _apply_missing_file_fallback(
        self,
        latex_source: str,
        missing_file: str,
        workspace_dir: Optional[Path] = None,
    ) -> Tuple[str, str, bool]:
        file_key = missing_file.strip().lower()
        if file_key != "bxcoloremoji-names.def":
            return latex_source, "no_fallback", False

        patched_source, source_changed = self._rewrite_bxcoloremoji_names_false(
            latex_source
        )
        workspace_changed = False
        if workspace_dir and workspace_dir.exists():
            workspace_changed = self._patch_workspace_text_files(
                workspace_dir,
                self._rewrite_bxcoloremoji_names_false,
                skip_main_tex=True,
            )

        changed = source_changed or workspace_changed
        if changed:
            return patched_source, "fallback_bxcoloremoji_names_false", True
        return latex_source, "no_fallback", False

    @staticmethod
    def _extract_engine_conflict_primitive(log: str) -> Optional[str]:
        if not log:
            return None

        lowered = log.lower()
        if "undefined control sequence" not in lowered:
            return None

        primitive_pattern = "|".join(re.escape(cmd) for cmd in ENGINE_CONFLICT_PRIMITIVES)
        detailed_match = re.search(
            rf"Undefined control sequence\.(?:.|\n){{0,300}}?l\.\d+\s*\\(?P<cmd>{primitive_pattern})\b",
            log,
            re.IGNORECASE,
        )
        if detailed_match:
            return detailed_match.group("cmd").lower()

        for cmd in ENGINE_CONFLICT_PRIMITIVES:
            if f"\\{cmd}" in lowered:
                return cmd

        return None

    @staticmethod
    def _rewrite_strip_engine_conflict_primitives(text: str) -> Tuple[str, bool]:
        patched = _strip_engine_conflict_primitives(text)
        return patched, patched != text

    def _apply_pdftex_primitive_fallback(
        self,
        latex_source: str,
        workspace_dir: Optional[Path] = None,
    ) -> Tuple[str, str, bool]:
        patched_source, source_changed = self._rewrite_strip_engine_conflict_primitives(
            latex_source
        )
        workspace_changed = False
        if workspace_dir and workspace_dir.exists():
            workspace_changed = self._patch_workspace_text_files(
                workspace_dir,
                self._rewrite_strip_engine_conflict_primitives,
                skip_main_tex=True,
            )
        changed = source_changed or workspace_changed
        if changed:
            return patched_source, "fallback_remove_pdftex_primitives", True
        return latex_source, "no_fallback", False

    @staticmethod
    def _has_microtype_tracking_error(log: str) -> bool:
        lowered = (log or "").lower().replace("\r", "\n")
        return bool(
            re.search(r"package\s+microtype\s+error", lowered)
            and re.search(
                r"tracking\s+feature\s+only\s+works\s+with\s+p(?:\s|\(microtype\))*dftex",
                lowered,
            )
        )

    @staticmethod
    def _has_microtype_disable_ligatures_error(log: str) -> bool:
        lowered = (log or "").lower().replace("\r", "\n")
        return bool(
            re.search(r"package\s+microtype\s+error", lowered)
            and "disabling ligatures" in lowered
            and "only possible" in lowered
        )

    @staticmethod
    def _rewrite_microtype_tracking(text: str) -> Tuple[str, bool]:
        pattern = re.compile(
            r"\\(?P<cmd>RequirePackage|usepackage)\[(?P<opts>[^\]]*)\]\{microtype\}"
        )
        changed = False

        def _replace(match: re.Match[str]) -> str:
            nonlocal changed
            cmd = match.group("cmd")
            opts = match.group("opts")
            parts = [part.strip() for part in opts.split(",") if part.strip()]
            kept_parts: List[str] = []
            removed = False
            for part in parts:
                if re.fullmatch(
                    r"tracking(?:\s*=\s*smallcaps)?", part, flags=re.IGNORECASE
                ):
                    removed = True
                    continue
                kept_parts.append(part)

            if not removed:
                return match.group(0)

            changed = True
            if kept_parts:
                return f"\\{cmd}[{','.join(kept_parts)}]" + "{microtype}"
            return f"\\{cmd}" + "{microtype}"

        patched = pattern.sub(_replace, text)
        return patched, changed

    @staticmethod
    def _rewrite_microtype_disable_ligatures(text: str) -> Tuple[str, bool]:
        patched = re.sub(
            r"^[ \t]*\\DisableLigatures(?:\[[^\]]*\])?\{[^}]*\}[ \t]*\n?",
            "",
            text,
            flags=re.MULTILINE,
        )
        return patched, patched != text

    def _patch_workspace_text_files(
        self,
        workspace_dir: Path,
        rewriter,
        *,
        skip_main_tex: bool,
    ) -> bool:
        changed_any = False
        for path in workspace_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".tex", ".cls", ".sty", ".def"}:
                continue
            if skip_main_tex and path.name == "main.tex":
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            patched, changed = rewriter(text)
            if not changed:
                continue
            path.write_text(patched, encoding="utf-8")
            changed_any = True
        return changed_any

    def _apply_microtype_tracking_fallback(
        self,
        latex_source: str,
        workspace_dir: Optional[Path] = None,
    ) -> Tuple[str, str, bool]:
        patched_source, source_changed = self._rewrite_microtype_tracking(latex_source)
        workspace_changed = False
        if workspace_dir and workspace_dir.exists():
            workspace_changed = self._patch_workspace_text_files(
                workspace_dir,
                self._rewrite_microtype_tracking,
                skip_main_tex=True,
            )
        changed = source_changed or workspace_changed
        if changed:
            return patched_source, "fallback_disable_microtype_tracking", True
        return latex_source, "no_fallback", False

    def _apply_microtype_disable_ligatures_fallback(
        self,
        latex_source: str,
        workspace_dir: Optional[Path] = None,
    ) -> Tuple[str, str, bool]:
        patched_source, source_changed = self._rewrite_microtype_disable_ligatures(
            latex_source
        )
        workspace_changed = False
        if workspace_dir and workspace_dir.exists():
            workspace_changed = self._patch_workspace_text_files(
                workspace_dir,
                self._rewrite_microtype_disable_ligatures,
                skip_main_tex=True,
            )
        changed = source_changed or workspace_changed
        if changed:
            return patched_source, "fallback_disable_microtype_ligatures", True
        return latex_source, "no_fallback", False

    @staticmethod
    def _extract_redefined_commands(log: str) -> List[str]:
        if not log:
            return []
        commands: List[str] = []
        normalized = re.sub(r"defi\s+ned", "defined", log, flags=re.IGNORECASE)
        for match in re.finditer(
            r"LaTeX Error:\s+Command\s+\\([A-Za-z@]+)\s+already defined",
            normalized,
            flags=re.IGNORECASE,
        ):
            command = match.group(1)
            if command in SAFE_REDEFINED_COMMANDS and command not in commands:
                commands.append(command)
        return commands

    @staticmethod
    def _has_noopndent_error(log: str) -> bool:
        lowered = (log or "").lower()
        return "undefined control sequence" in lowered and "\\noopndent" in lowered

    @staticmethod
    def _rewrite_noopndent(text: str) -> Tuple[str, bool]:
        patched = text.replace("\\noopndent", "\\noindent")
        return patched, patched != text

    def _apply_noopndent_fallback(
        self,
        latex_source: str,
        workspace_dir: Optional[Path] = None,
    ) -> Tuple[str, str, bool]:
        patched_source, source_changed = self._rewrite_noopndent(latex_source)
        workspace_changed = False
        if workspace_dir and workspace_dir.exists():
            workspace_changed = self._patch_workspace_text_files(
                workspace_dir,
                self._rewrite_noopndent,
                skip_main_tex=True,
            )
        changed = source_changed or workspace_changed
        if changed:
            return patched_source, "fallback_fix_noopndent", True
        return latex_source, "no_fallback", False

    @staticmethod
    def _rewrite_redefined_newcommands(
        text: str,
        commands: List[str],
    ) -> Tuple[str, bool]:
        if not commands:
            return text, False

        changed = False
        patched = text
        commands_to_patch = list(dict.fromkeys([*commands, *SAFE_REDEFINED_COMMANDS]))
        for command in commands_to_patch:
            pattern = re.compile(
                rf"\\newcommand\s*(?P<star>\*)?\s*"
                rf"(?P<braced>\{{\\{re.escape(command)}\}}|\\{re.escape(command)})",
            )

            def replace(match: re.Match[str]) -> str:
                nonlocal changed
                changed = True
                return (
                    "\\providecommand"
                    + (match.group("star") or "")
                    + match.group("braced")
                )

            patched = pattern.sub(replace, patched)
        return patched, changed

    def _apply_redefined_command_fallback(
        self,
        latex_source: str,
        commands: List[str],
        workspace_dir: Optional[Path] = None,
    ) -> Tuple[str, str, bool]:
        def rewriter(text: str) -> Tuple[str, bool]:
            return self._rewrite_redefined_newcommands(text, commands)

        patched_source, source_changed = rewriter(latex_source)
        workspace_changed = False
        if workspace_dir and workspace_dir.exists():
            workspace_changed = self._patch_workspace_text_files(
                workspace_dir,
                rewriter,
                skip_main_tex=True,
            )
        changed = source_changed or workspace_changed
        if changed:
            return patched_source, "fallback_provide_redefined_commands", True
        return latex_source, "no_fallback", False

    def _run_bibliography_tool(
        self,
        tool: str,
        cwd: Path,
        *,
        env: Optional[Dict[str, str]] = None,
    ) -> bool:
        cmd = [tool, "main"]
        try:
            subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                timeout=60,
                encoding="utf-8",
                errors="replace",
                env=env or self._build_env(),
            )
            return True
        except Exception:
            return False

    def _run_single_pass(
        self,
        engine: str,
        source_file: Path,
        cwd: Path,
        *,
        env: Optional[Dict[str, str]] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """Runs a single pass of the latex engine."""
        cmd = [
            engine,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            source_file.name,
        ]

        try:
            pdf_file = cwd / "main.pdf"
            try:
                if pdf_file.exists():
                    pdf_file.unlink()
            except Exception:
                pass

            process = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding="utf-8",
                errors="replace",
                env=env or self._build_env(),
            )

            # Read log file if it exists, as it's more complete than stdout
            log_content = process.stdout + "\n" + process.stderr
            log_file = cwd / "main.log"
            if log_file.exists():
                try:
                    file_log = log_file.read_text(encoding="utf-8", errors="replace")
                    if file_log.strip():
                        log_content = file_log
                except Exception:
                    pass

            if process.returncode != 0:
                return (
                    False,
                    log_content,
                    f"Compilation command exited with code {process.returncode}. "
                    f"{self._extract_error(log_content)}",
                )

            if not self._is_pdf_healthy(pdf_file):
                return (
                    False,
                    log_content,
                    "Generated PDF failed integrity check (%PDF/%%EOF).",
                )
            return True, log_content, None

        except subprocess.TimeoutExpired:
            return False, "Timeout expired", "Compilation timed out"
        except Exception as e:
            return False, str(e), str(e)

    def _extract_error(self, log: str) -> str:
        """Extracts the first meaningful error from the log."""
        if not log:
            return "No log content"

        lines = log.splitlines()
        package_or_latex_error = re.compile(
            r"(Package\s+[^\s:]+\s+Error:|LaTeX\s+Error:)",
            re.IGNORECASE,
        )
        undefined_control_error = re.compile(
            r"Undefined control sequence\.",
            re.IGNORECASE,
        )
        misplaced_noalign_error = re.compile(
            r"Misplaced\s+\\noalign\.",
            re.IGNORECASE,
        )
        for i, line in enumerate(lines):
            # LaTeX errors usually start with !
            if line.strip().startswith("!"):
                # capture context (up to 5 lines)
                return "\n".join(lines[i : min(i + 5, len(lines))])
            if package_or_latex_error.search(line):
                return "\n".join(lines[i : min(i + 5, len(lines))])
            if undefined_control_error.search(line):
                return "\n".join(lines[i : min(i + 5, len(lines))])
            if misplaced_noalign_error.search(line):
                return "\n".join(lines[i : min(i + 5, len(lines))])

        # Fallback: check for common error patterns if no ! found
        if "Fatal error" in log:
            return "Fatal error detected in logs"

        return "Unknown error (check full logs)"

    def _compile_error_detail(self, log: str, *, driver: str) -> str:
        detail = self._extract_error(log)
        missing_file = self._extract_missing_latex_file(log)
        if not missing_file:
            return detail

        if driver == "r_tinytex":
            hint = (
                "官方 R tinytex 自动补包未完成；请检查 TinyTeX/tlmgr "
                "repository、网络或代理后重试。"
            )
        else:
            hint = (
                "当前使用 latexmk driver，不会自动安装缺失 TeX 包；"
                "推荐使用 tinytex_driver: r_tinytex，或手动通过 TinyTeX/tlmgr "
                "安装缺失包后重试。"
            )
        return f"{detail}\nMissing TeX file: {missing_file}. {hint}"

    @staticmethod
    def _classify_error_text(error_text: str) -> str:
        lowered = str(error_text).lower()
        if "fontspec" in lowered:
            return "fontspec"
        if "package" in lowered:
            return "package"
        if "undefined control sequence" in lowered:
            return "latex"
        if "misplaced \\noalign" in lowered:
            return "latex"
        if "latex error" in lowered or lowered.startswith("!"):
            return "latex"
        return "unknown"

    def _is_pdf_healthy(self, pdf_file: Path) -> bool:
        """Basic PDF health check: file exists, has %PDF header, and contains %%EOF trailer."""
        try:
            if not pdf_file.exists():
                return False
            if pdf_file.stat().st_size < 8:
                return False

            with pdf_file.open("rb") as f:
                header = f.read(5)
                if not header.startswith(b"%PDF"):
                    return False

                tail_window = min(pdf_file.stat().st_size, 2048)
                f.seek(-tail_window, os.SEEK_END)
                tail = f.read()
                if b"%%EOF" not in tail:
                    return False

            return True
        except Exception:
            return False

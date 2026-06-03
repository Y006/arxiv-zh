import os
import shutil
import subprocess
import tempfile
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .chinese_support import (
    ENGINE_CONFLICT_PRIMITIVES,
    _strip_engine_conflict_primitives,
    inject_chinese_support,
)


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
class CompilationResult:
    success: bool
    pdf_path: Optional[Path] = None
    log_content: Optional[str] = None
    error_message: Optional[str] = None
    engine_used: Optional[str] = None


class LaTeXCompiler:
    def __init__(self, timeout: int = 120, fonts_dir: Optional[Union[str, Path]] = None):
        self.timeout = timeout
        # Priority: xelatex (best CJK), lualatex (good CJK), pdflatex (fallback)
        self.engines = ["xelatex", "lualatex", "pdflatex"]
        self.fonts_dir = Path(fonts_dir).resolve() if fonts_dir else None

    def inject_chinese_support(self, latex_source: str) -> str:
        """Wrapper around the injection logic."""
        return inject_chinese_support(latex_source)

    def _build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        if self.fonts_dir and self.fonts_dir.exists():
            existing_font_dirs = env.get("OSFONTDIR", "").strip()
            if existing_font_dirs:
                env["OSFONTDIR"] = (
                    f"{str(self.fonts_dir)}{os.pathsep}{existing_font_dirs}"
                )
            else:
                env["OSFONTDIR"] = str(self.fonts_dir)
        return env

    def compile_file(
        self,
        tex_file: Union[str, Path],
        output_path: Union[str, Path],
        logs_dir: Union[str, Path],
        build_dir: Optional[Union[str, Path]] = None,
        prefer_latexmk: bool = True,
    ) -> CompilationResult:
        """Compile an existing TeX file while preserving logs and build artifacts."""
        tex_file = Path(tex_file).resolve()
        output_path = Path(output_path).resolve()
        logs_dir = Path(logs_dir).resolve()
        build_path = Path(build_dir).resolve() if build_dir else logs_dir / "build"
        log_path = logs_dir / "compile.log"
        summary_path = logs_dir / "compile_error_summary.md"

        logs_dir.mkdir(parents=True, exist_ok=True)
        build_path.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")

        if not tex_file.exists():
            message = f"TeX file not found: {tex_file}"
            log_path.write_text(message, encoding="utf-8")
            write_compile_error_summary(log_path, summary_path)
            return CompilationResult(success=False, log_content=message, error_message=message)

        def run_command(cmd: List[str]) -> Tuple[bool, str, Optional[str]]:
            header = f"$ {' '.join(cmd)}\n"
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(header)
            try:
                process = subprocess.run(
                    cmd,
                    cwd=tex_file.parent,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    encoding="utf-8",
                    errors="replace",
                    env=self._build_env(),
                )
                combined_log = process.stdout + "\n" + process.stderr
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(combined_log)
                    log_file.write("\n")
                if process.returncode != 0:
                    return (
                        False,
                        log_path.read_text(encoding="utf-8", errors="replace"),
                        f"Compilation command exited with code {process.returncode}.",
                    )
                return True, log_path.read_text(encoding="utf-8", errors="replace"), None
            except subprocess.TimeoutExpired:
                message = "Compilation timed out"
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(message + "\n")
                return False, log_path.read_text(encoding="utf-8", errors="replace"), message
            except Exception as exc:
                message = str(exc)
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(message + "\n")
                return False, log_path.read_text(encoding="utf-8", errors="replace"), message

        engine_used: Optional[str] = None
        success = False
        log_content = ""
        error: Optional[str] = None

        if prefer_latexmk and shutil.which("latexmk"):
            engine_used = "latexmk"
            success, log_content, error = run_command(
                [
                    "latexmk",
                    "-xelatex",
                    "-interaction=nonstopmode",
                    "-file-line-error",
                    f"-output-directory={build_path}",
                    tex_file.name,
                ]
            )
        else:
            if not shutil.which("xelatex"):
                message = "Neither latexmk nor xelatex was found on PATH."
                log_path.write_text(message, encoding="utf-8")
                write_compile_error_summary(log_path, summary_path)
                return CompilationResult(
                    success=False,
                    log_content=message,
                    error_message=message,
                    engine_used=None,
                )
            engine_used = "xelatex"
            for _ in range(4):
                success, log_content, error = run_command(
                    [
                        "xelatex",
                        "-interaction=nonstopmode",
                        "-file-line-error",
                        f"-output-directory={build_path}",
                        tex_file.name,
                    ]
                )
                if not success:
                    break

        built_pdf = build_path / f"{tex_file.stem}.pdf"
        if success and self._is_pdf_healthy(built_pdf):
            shutil.copy2(built_pdf, output_path)
            if self._is_pdf_healthy(output_path):
                return CompilationResult(
                    success=True,
                    pdf_path=output_path,
                    log_content=log_content,
                    engine_used=engine_used,
                )
            error = "Generated PDF failed integrity check after copy."
        elif success:
            error = "Generated PDF failed integrity check (%PDF/%%EOF)."

        log_content = log_path.read_text(encoding="utf-8", errors="replace")
        write_compile_error_summary(log_path, summary_path)
        return CompilationResult(
            success=False,
            log_content=log_content,
            error_message=error or "Compilation failed.",
            engine_used=engine_used,
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
        compile_source = latex_source

        # Create a temporary directory for compilation to keep things clean
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # If working_dir is provided, copy its contents to temp_dir
            if working_dir:
                working_dir_path = Path(working_dir)
                if working_dir_path.exists():
                    self._copy_resources(working_dir_path, temp_path)

            source_file = temp_path / "main.tex"
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
                    if not shutil.which(engine):
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

    def _run_engine(
        self, engine: str, source_file: Path, cwd: Path, latex_source: str
    ) -> Tuple[bool, str, Optional[str]]:
        """Runs the full compilation cycle: (xelatex + bibtex) × 2."""

        # 1. First xelatex pass (generate .aux for bibtex)
        success, log, error = self._run_single_pass(engine, source_file, cwd)
        # Don't fail on first pass - references will be unresolved

        # 2. Check for existing .bbl file (pre-compiled bibliography)
        main_bbl = cwd / "main.bbl"
        if not main_bbl.exists():
            bbl_files = list(cwd.glob("*.bbl"))
            if bbl_files:
                shutil.copy2(bbl_files[0], main_bbl)

        # 3. Run bibtex (always try if bibliography command exists)
        bib_tool = self._detect_bibliography_tool(latex_source)
        if bib_tool and shutil.which(bib_tool):
            self._run_bibliography_tool(bib_tool, cwd)

        # 4. Second xelatex pass (incorporate bibliography)
        self._run_single_pass(engine, source_file, cwd)

        # 5. Run bibtex again (resolve any new citations)
        if bib_tool and shutil.which(bib_tool):
            self._run_bibliography_tool(bib_tool, cwd)

        # 6. Third xelatex pass (resolve all cross-references)
        self._run_single_pass(engine, source_file, cwd)

        # 7. Fourth xelatex pass (final - ensure all references resolved)
        success, log, error = self._run_single_pass(engine, source_file, cwd)

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
            if path.suffix.lower() not in {".tex", ".cls", ".sty"}:
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

    def _run_bibliography_tool(self, tool: str, cwd: Path) -> bool:
        cmd = [tool, "main"]
        try:
            subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                timeout=60,
                encoding="utf-8",
                errors="replace",
            )
            return True
        except Exception:
            return False

    def _run_single_pass(
        self, engine: str, source_file: Path, cwd: Path
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
                env=self._build_env(),
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

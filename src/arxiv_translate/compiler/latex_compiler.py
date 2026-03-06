import os
import shutil
import subprocess
import tempfile
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

from .chinese_support import inject_chinese_support


@dataclass
class CompilationResult:
    success: bool
    pdf_path: Optional[Path] = None
    log_content: Optional[str] = None
    error_message: Optional[str] = None
    engine_used: Optional[str] = None


class LaTeXCompiler:
    def __init__(self, timeout: int = 120):
        self.timeout = timeout
        # Priority: xelatex (best CJK), lualatex (good CJK), pdflatex (fallback)
        self.engines = ["xelatex", "lualatex", "pdflatex"]

    def inject_chinese_support(self, latex_source: str) -> str:
        """Wrapper around the injection logic."""
        return inject_chinese_support(latex_source)

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

            for _round in range(3):
                source_file.write_text(compile_source, encoding="utf-8")
                missing_font: Optional[str] = None

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
                    if missing_font is None:
                        missing_font = self._extract_missing_font_name(log)

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
                        continue
                break

            # If we reach here, all engines failed
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
        for i, line in enumerate(lines):
            # LaTeX errors usually start with !
            if line.strip().startswith("!"):
                # capture context (up to 5 lines)
                return "\n".join(lines[i : min(i + 5, len(lines))])

        # Fallback: check for common error patterns if no ! found
        if "Fatal error" in log:
            return "Fatal error detected in logs"

        return "Unknown error (check full logs)"

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

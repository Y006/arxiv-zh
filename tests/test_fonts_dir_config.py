import ast
import inspect

import arxiv_translate.cli as cli_module
from arxiv_translate.rules.config import Config, load_defaults
from arxiv_translate.compiler.chinese_support import get_available_fonts


def test_default_config_includes_fonts_dir():
    defaults = load_defaults()
    assert defaults["fonts"]["dir"] == "fonts"


def test_font_config_accepts_fonts_dir_field():
    config = Config(fonts={"dir": "project_fonts"})
    assert config.fonts.dir == "project_fonts"


def test_cli_passes_fonts_dir_into_latex_compiler():
    source = inspect.getsource(cli_module)
    tree = ast.parse(source)

    class CompilerCallVisitor(ast.NodeVisitor):
        def __init__(self):
            self.found = False

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == "LaTeXCompiler":
                for keyword in node.keywords:
                    if keyword.arg == "fonts_dir":
                        self.found = True
            self.generic_visit(node)

    visitor = CompilerCallVisitor()
    visitor.visit(tree)
    assert visitor.found is True


def test_local_font_dir_scan_finds_sample_cjk_families():
    font_dir = cli_module._project_font_dir()

    families = set(get_available_fonts(font_dir=font_dir, include_system=False))

    assert "STSong" in families
    assert "STXihei" in families
    assert "STKaiti" in families

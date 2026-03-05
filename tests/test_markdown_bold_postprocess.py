"""Tests for conservative markdown bold sanitization."""

from arxiv_translate.translator.postprocess import sanitize_markdown_bold_safe


def test_markdown_bold_basic_conversion():
    source = "这是重点说明。"
    translation = "这是**重点**说明。"

    fixed, audit = sanitize_markdown_bold_safe(source, translation)

    assert fixed == r"这是\textbf{重点}说明。"
    assert audit["changed"] is True
    assert audit["converted_count"] == 1
    assert audit["skipped_count"] == 0


def test_markdown_bold_skip_on_latex_sensitive_payload():
    source = "安全优先。"
    translation = r"Unsafe **\alpha** and **$x$** and **a{b}**"

    fixed, audit = sanitize_markdown_bold_safe(source, translation)

    assert fixed == translation
    assert audit["changed"] is False
    assert audit["converted_count"] == 0
    assert audit["skipped_count"] == 3


def test_markdown_bold_skip_when_source_contains_double_asterisk():
    source = "原文里本来就有 **literal** 星号语义。"
    translation = "译文出现**重点**。"

    fixed, audit = sanitize_markdown_bold_safe(source, translation)

    assert fixed == translation
    assert audit["changed"] is False
    assert audit["converted_count"] == 0
    assert audit["skipped_count"] == 1
    assert "source_contains_double_asterisk" in audit["skipped_reasons"]

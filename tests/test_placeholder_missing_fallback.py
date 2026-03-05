"""Tests for missing placeholder fallback behavior."""

from arxiv_translate.parser.structure import (
    Chunk,
    LaTeXDocument,
    validate_translated_placeholders,
)


def test_missing_placeholder_triggers_chunk_source_fallback():
    chunk = Chunk(
        id="chunk-1",
        content="This has [[MATH_1]] placeholder.",
        latex_wrapper="%s",
        context="paragraph",
        preserved_elements={"[[MATH_1]]": r"$x^2$"},
    )
    doc = LaTeXDocument(preamble="", chunks=[chunk], body_template="{{CHUNK_chunk-1}}")

    translated_map = {"chunk-1": "这里丢失了占位符。"}
    fixed_map, issues = validate_translated_placeholders(translated_map, doc)

    assert fixed_map["chunk-1"] == chunk.content
    assert any(issue["type"] == "missing" for issue in issues)
    fallback_issue = next(
        (issue for issue in issues if issue["type"] == "missing_fallback"), None
    )
    assert fallback_issue is not None
    assert fallback_issue["chunk_id"] == "chunk-1"
    assert "[[MATH_1]]" in fallback_issue["bad"]


def test_missing_placeholder_fallback_can_be_disabled_for_chunk():
    chunk = Chunk(
        id="chunk-1",
        content="This has [[MATH_1]] placeholder.",
        latex_wrapper="%s",
        context="paragraph",
        preserved_elements={"[[MATH_1]]": r"$x^2$"},
    )
    doc = LaTeXDocument(preamble="", chunks=[chunk], body_template="{{CHUNK_chunk-1}}")

    translated_map = {"chunk-1": "这里丢失了占位符。"}
    fixed_map, issues = validate_translated_placeholders(
        translated_map,
        doc,
        disable_missing_fallback_ids={"chunk-1"},
    )

    assert fixed_map["chunk-1"] == "这里丢失了占位符。"
    assert any(issue["type"] == "missing" for issue in issues)
    assert not any(issue["type"] == "missing_fallback" for issue in issues)

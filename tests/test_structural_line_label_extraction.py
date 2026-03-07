"""Tests for structural-line label extraction and nested author thanks chunks."""

import warnings

from arxiv_translate.parser.latex_parser import LaTeXParser


def _process_structural_lines(text: str):
    parser = LaTeXParser()
    parser.chunks = []
    parser.protected_counter = 0
    parser.placeholder_map = {}
    processed = parser._chunk_paragraphs(text)
    return parser, processed


def test_forest_labels_are_chunked_from_structural_lines():
    text = "\n".join(
        [
            r"[Advanced \\ Detector \\ Research\\(Sec. 5), text width=3.2em, fill=blue!10,",
            r"    [{\includegraphics[width=4.3cm]{figures/human_assist.png}} \\ Human-Assisted Methods, text width=12em",
            r"        [\emph{Imperceptible Features:} \cite{refA} / \cite{refB}, text width=19em]",
        ]
    )

    parser, processed = _process_structural_lines(text)

    contents = {chunk.content for chunk in parser.chunks}
    assert "Advanced \\\\ Detector \\\\ Research\\\\(Sec. 5)" in contents
    assert "Human-Assisted Methods" in contents
    assert "Imperceptible Features:" in contents
    assert processed.count("{{CHUNK_") == 3


def test_author_thanks_content_is_extracted_before_author_protection():
    parser = LaTeXParser()
    parser.chunks = []
    parser.protected_counter = 0
    parser.placeholder_map = {}

    author_text = (
        r"\author{Derek Fai Wong\thanks{Yulin Yuan and Derek Fai Wong are "
        r"co-coresponding authors.}}"
    )

    protected = parser._protect_author_block(author_text)

    assert protected == "[[AUTHOR_1]]"

    thanks_chunks = [chunk for chunk in parser.chunks if chunk.context == "thanks"]
    assert len(thanks_chunks) == 1
    assert (
        thanks_chunks[0].content
        == "Yulin Yuan and Derek Fai Wong are co-coresponding authors."
    )

    protected_chunks = [chunk for chunk in parser.chunks if chunk.context == "protected"]
    assert len(protected_chunks) == 1
    preserved_values = list(protected_chunks[0].preserved_elements.values())
    assert len(preserved_values) == 1
    assert "{{CHUNK_" in preserved_values[0]


def test_parse_file_does_not_warn_for_nested_author_thanks_chunk(tmp_path):
    parser = LaTeXParser()
    tex_path = tmp_path / "author_thanks.tex"
    tex_path.write_text(
        "\n".join(
            [
                r"\documentclass{article}",
                r"\author{Derek Fai Wong\thanks{Yulin Yuan and Derek Fai Wong are co-coresponding authors.}}",
                r"\begin{document}",
                r"\maketitle",
                r"\end{document}",
            ]
        ),
        encoding="utf-8",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parser.parse_file(str(tex_path))

    orphan_warnings = [
        warning
        for warning in caught
        if "created without placeholders" in str(warning.message)
    ]
    assert orphan_warnings == []

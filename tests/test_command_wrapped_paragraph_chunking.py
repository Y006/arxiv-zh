"""Tests for command-wrapped paragraph body extraction."""

from arxiv_translate.parser.latex_parser import LaTeXParser


def _chunk_paragraph_wrapped_text(text: str):
    parser = LaTeXParser()
    parser.chunks = []
    parser.protected_counter = 0
    parser.placeholder_map = {}
    processed = parser._chunk_paragraphs(text)
    paragraph_chunks = [chunk for chunk in parser.chunks if chunk.context == "paragraph"]
    return processed, paragraph_chunks


def test_textcolor_wrapped_body_is_chunked():
    text = r"\textcolor{black}{This body text should be translated instead of staying in English.}"

    processed, paragraph_chunks = _chunk_paragraph_wrapped_text(text)

    assert "{{CHUNK_" in processed
    assert len(paragraph_chunks) == 1
    assert paragraph_chunks[0].content == text


def test_textcolor_wrapped_body_with_citation_is_chunked():
    text = (
        r"\textcolor{black}{Another recent work is SemStamp~\cite{semstamp}, "
        r"which should still be chunked for translation.}"
    )

    processed, paragraph_chunks = _chunk_paragraph_wrapped_text(text)

    assert "{{CHUNK_" in processed
    assert len(paragraph_chunks) == 1
    assert "Another recent work is SemStamp" in paragraph_chunks[0].content


def test_short_command_wrapped_label_is_not_chunked():
    text = r"\textbf{Title}"

    processed, paragraph_chunks = _chunk_paragraph_wrapped_text(text)

    assert processed == text
    assert paragraph_chunks == []

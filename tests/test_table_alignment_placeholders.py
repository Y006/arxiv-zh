"""Tests for table alignment ampersand placeholder protection."""

import re

from arxiv_translate.parser.latex_parser import LaTeXParser


def test_table_alignment_ampersands_are_preserved_as_raw_ampersands():
    parser = LaTeXParser()
    row = r"{} & Ratings 5\&4 & {$97\%$} & {$96\%$} \\"

    placeholder = parser._maybe_chunk_paragraph(row)
    assert placeholder.startswith("{{CHUNK_")
    assert len(parser.chunks) == 1

    chunk = parser.chunks[0]
    assert "[[AMP_" in chunk.content
    assert "5\\&4" in chunk.content
    assert any(v == "&" for v in chunk.preserved_elements.values())

    reconstructed = chunk.reconstruct(chunk.content)

    # Table column separators stay as raw '&' after reconstruction.
    assert len(re.findall(r"(?<!\\)&", reconstructed)) == 3
    # Textual escaped ampersand inside cell content is preserved.
    assert "5\\&4" in reconstructed

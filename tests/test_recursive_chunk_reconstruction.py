from arxiv_translate.parser.structure import Chunk, LaTeXDocument


def test_reconstruct_recursively_expands_nested_chunk_placeholders():
    child = Chunk(
        id="child",
        content="Nested Heading",
        latex_wrapper="%s",
        context="subsubsection",
    )
    parent = Chunk(
        id="parent",
        content=r"\textcolor{black}{\subsubsection{{{CHUNK_child}}} Body text.}",
        latex_wrapper="%s",
        context="paragraph",
    )
    doc = LaTeXDocument(
        preamble="",
        chunks=[child, parent],
        body_template="{{CHUNK_parent}}",
    )

    reconstructed = doc.reconstruct(
        {
            "child": "模型驱动水印",
            "parent": parent.content,
        }
    )

    assert "{{CHUNK_child}}" not in reconstructed
    assert "模型驱动水印" in reconstructed


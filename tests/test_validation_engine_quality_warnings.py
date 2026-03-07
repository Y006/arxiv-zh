from arxiv_translate.translator.pipeline import TranslatedChunk
from arxiv_translate.validator.engine import ValidationEngine


def test_validator_warns_on_unreplaced_chunk_placeholder():
    engine = ValidationEngine()

    result = engine.validate(
        translated=r"\textcolor{black}{\subsubsection{{{CHUNK_child}}} Body text.}",
        original=r"\textcolor{black}{\subsubsection{Body text.}}",
    )

    assert result.valid is True
    assert any(
        "chunk placeholder" in error.message.lower() for error in result.errors
    )
    assert any(error.severity == "warning" for error in result.errors)


def test_validator_warns_on_chunk_quality_metadata():
    engine = ValidationEngine()

    result = engine.validate(
        translated="This detector should translate the English body text.",
        original="This detector should translate the English body text.",
        translated_chunks=[
            TranslatedChunk(
                source="This detector should translate the English body text.",
                translation="This detector should translate the English body text.",
                chunk_id="c1",
                metadata={
                    "quality_warning_types": ["untranslated_retry_exhausted"],
                    "untranslated_audit_passed": False,
                },
            )
        ],
    )

    assert result.valid is True
    assert any("untranslated" in error.message.lower() for error in result.errors)
    assert any(error.severity == "warning" for error in result.errors)

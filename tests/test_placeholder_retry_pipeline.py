"""Tests for placeholder audit retry rounds in translation pipeline."""

from unittest.mock import AsyncMock, patch

import pytest

from arxiv_translate.translator.pipeline import TranslationPipeline, TranslatedChunk


def _build_pipeline(**kwargs) -> TranslationPipeline:
    provider = AsyncMock()
    provider.prepare_prompt_cache_variants = AsyncMock(return_value=None)
    return TranslationPipeline(provider=provider, **kwargs)


@pytest.mark.asyncio
async def test_placeholder_retry_succeeds_on_second_attempt():
    pipeline = _build_pipeline(batch_short_threshold=0)
    call_count = {"c1": 0, "c2": 0}

    async def fake_translate_chunk(chunk: str, chunk_id: str, context=None):
        call_count[chunk_id] += 1
        if chunk_id == "c1" and call_count[chunk_id] == 1:
            translation = "第一次漏占位符"
        elif chunk_id == "c1":
            translation = "第二次修复 [[MATH_1]]"
        else:
            translation = "普通文本翻译"
        return TranslatedChunk(
            source=chunk,
            translation=translation,
            chunk_id=chunk_id,
            metadata={},
        )

    chunks = [
        {"chunk_id": "c1", "content": "source [[MATH_1]] chunk"},
        {"chunk_id": "c2", "content": "plain source chunk"},
    ]

    with patch.object(pipeline, "translate_chunk", side_effect=fake_translate_chunk):
        results = await pipeline.translate_document(chunks, max_concurrent=4)

    by_id = {chunk.chunk_id: chunk for chunk in results}
    assert call_count["c1"] == 2
    assert call_count["c2"] == 1
    assert by_id["c1"].metadata["placeholder_attempt"] == 2
    assert by_id["c1"].metadata["placeholder_audit_passed"] is True
    assert by_id["c1"].metadata["placeholder_retry_exhausted"] is False
    assert by_id["c2"].metadata["placeholder_attempt"] == 1


@pytest.mark.asyncio
async def test_placeholder_retry_exhausted_keeps_third_translation():
    pipeline = _build_pipeline(batch_short_threshold=0)
    call_count = {"c1": 0}

    async def fake_translate_chunk(chunk: str, chunk_id: str, context=None):
        call_count[chunk_id] += 1
        return TranslatedChunk(
            source=chunk,
            translation=f"第{call_count[chunk_id]}次仍缺失",
            chunk_id=chunk_id,
            metadata={},
        )

    chunks = [{"chunk_id": "c1", "content": "source [[MATH_1]] chunk"}]

    with patch.object(pipeline, "translate_chunk", side_effect=fake_translate_chunk):
        results = await pipeline.translate_document(chunks, max_concurrent=2)

    chunk = results[0]
    assert call_count["c1"] == 3
    assert chunk.translation == "第3次仍缺失"
    assert chunk.metadata["placeholder_attempt"] == 3
    assert chunk.metadata["placeholder_audit_passed"] is False
    assert chunk.metadata["placeholder_retry_exhausted"] is True
    assert chunk.metadata["placeholder_warning_emitted"] is True
    assert chunk.metadata["placeholder_missing"] == ["[[MATH_1]]"]


@pytest.mark.asyncio
async def test_placeholder_retry_detects_duplicate_placeholder_counts():
    pipeline = _build_pipeline(batch_short_threshold=0)
    call_count = {"c1": 0}

    async def fake_translate_chunk(chunk: str, chunk_id: str, context=None):
        call_count[chunk_id] += 1
        translation = (
            "第一次重复 [[MATH_1]] [[MATH_1]]"
            if call_count[chunk_id] == 1
            else "第二次修复 [[MATH_1]]"
        )
        return TranslatedChunk(
            source=chunk,
            translation=translation,
            chunk_id=chunk_id,
            metadata={},
        )

    chunks = [{"chunk_id": "c1", "content": "source [[MATH_1]] chunk"}]

    with patch.object(pipeline, "translate_chunk", side_effect=fake_translate_chunk):
        results = await pipeline.translate_document(chunks, max_concurrent=2)

    chunk = results[0]
    assert call_count["c1"] == 2
    assert chunk.translation == "第二次修复 [[MATH_1]]"
    assert chunk.metadata["placeholder_audit_passed"] is True
    assert chunk.metadata["placeholder_retry_exhausted"] is False


@pytest.mark.asyncio
async def test_partial_batch_retry_only_failed_chunks():
    pipeline = _build_pipeline(batch_short_threshold=10_000, batch_max_chars=10_000)
    batch_calls: list[list[str]] = []

    async def fake_translate_batch(chunks, context=None):
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        batch_calls.append(chunk_ids)
        round_no = len(batch_calls)

        results = []
        for chunk in chunks:
            if chunk["chunk_id"] == "c1" and round_no == 1:
                translation = "第一次漏占位符"
            elif chunk["chunk_id"] == "c1":
                translation = "重试修复 [[MATH_1]]"
            else:
                translation = "正常块翻译"
            results.append(
                TranslatedChunk(
                    source=chunk["content"],
                    translation=translation,
                    chunk_id=chunk["chunk_id"],
                    metadata={},
                )
            )
        return results

    async def fail_translate_chunk(*args, **kwargs):
        raise AssertionError("This test should stay on batch path")

    chunks = [
        {"chunk_id": "c1", "content": "source [[MATH_1]] chunk"},
        {"chunk_id": "c2", "content": "plain source chunk"},
    ]

    with (
        patch.object(pipeline, "translate_batch", side_effect=fake_translate_batch),
        patch.object(pipeline, "translate_chunk", side_effect=fail_translate_chunk),
    ):
        results = await pipeline.translate_document(chunks, max_concurrent=4)

    assert batch_calls[0] == ["c1", "c2"]
    assert batch_calls[1] == ["c1"]
    by_id = {chunk.chunk_id: chunk for chunk in results}
    assert by_id["c1"].metadata["placeholder_attempt"] == 2
    assert by_id["c2"].metadata["placeholder_attempt"] == 1


@pytest.mark.asyncio
async def test_brace_escape_drift_fixed_without_retry():
    pipeline = _build_pipeline(batch_short_threshold=0)
    call_count = {"c1": 0}

    source = r'Tool call: \{"name": "search", "args": \{"q": "paper"\}\}'
    drift_translation = r'工具调用：{"name": "search", "args": {"q": "paper"}}'

    async def fake_translate_chunk(chunk: str, chunk_id: str, context=None):
        call_count[chunk_id] += 1
        return TranslatedChunk(
            source=chunk,
            translation=drift_translation,
            chunk_id=chunk_id,
            metadata={},
        )

    chunks = [{"chunk_id": "c1", "content": source}]
    with patch.object(pipeline, "translate_chunk", side_effect=fake_translate_chunk):
        results = await pipeline.translate_document(chunks, max_concurrent=2)

    chunk = results[0]
    assert call_count["c1"] == 1
    assert chunk.metadata["brace_audit_passed"] is True
    assert chunk.metadata["brace_fix_applied"] is True
    assert chunk.metadata["brace_fix_edit_count"] > 0
    assert chunk.metadata["brace_retry_exhausted"] is False
    assert chunk.metadata["brace_fallback_applied"] is False
    assert r'\{"name": "search"' in chunk.translation


@pytest.mark.asyncio
async def test_brace_retry_exhausted_keeps_last_translation_with_warning():
    pipeline = _build_pipeline(batch_short_threshold=0)
    call_count = {"c1": 0}

    source = r'Schema: \{"a": \{"b": 1\}\}'
    last_translation = '模式："a": {"b": 1'

    async def fake_translate_chunk(chunk: str, chunk_id: str, context=None):
        call_count[chunk_id] += 1
        return TranslatedChunk(
            source=chunk,
            translation=last_translation,
            chunk_id=chunk_id,
            metadata={},
        )

    chunks = [{"chunk_id": "c1", "content": source}]
    with patch.object(pipeline, "translate_chunk", side_effect=fake_translate_chunk):
        results = await pipeline.translate_document(chunks, max_concurrent=2)

    chunk = results[0]
    assert call_count["c1"] == 3
    assert chunk.translation == last_translation
    assert chunk.metadata["brace_retry_exhausted"] is True
    assert chunk.metadata["brace_fallback_applied"] is False
    assert chunk.metadata["brace_audit_passed"] is False
    assert chunk.metadata["kept_last_translation_on_fallback"] is True
    assert "brace_retry_exhausted" in chunk.metadata["quality_warning_types"]


@pytest.mark.asyncio
async def test_brace_wrapper_suffix_is_repaired_without_retry():
    pipeline = _build_pipeline(batch_short_threshold=0)
    call_count = {"c1": 0}

    source = (
        "\\textcolor{black}{\\paragraph{{{CHUNK_head}}}\n"
        "Model-driven methods embed watermarks into the LLM.}"
    )
    missing_suffix_translation = (
        "\\textcolor{black}{\\paragraph{{{CHUNK_head}}}\n"
        "模型驱动方法将水印嵌入大语言模型。"
    )

    async def fake_translate_chunk(chunk: str, chunk_id: str, context=None):
        call_count[chunk_id] += 1
        return TranslatedChunk(
            source=chunk,
            translation=missing_suffix_translation,
            chunk_id=chunk_id,
            metadata={},
        )

    chunks = [{"chunk_id": "c1", "content": source}]
    with patch.object(pipeline, "translate_chunk", side_effect=fake_translate_chunk):
        results = await pipeline.translate_document(chunks, max_concurrent=2)

    chunk = results[0]
    assert call_count["c1"] == 1
    assert chunk.translation.endswith("。}")
    assert chunk.metadata["brace_audit_passed"] is True
    assert chunk.metadata["brace_fix_applied"] is True
    assert chunk.metadata["brace_retry_exhausted"] is False


@pytest.mark.asyncio
async def test_added_safe_inline_formatting_command_is_unwrapped_without_retry():
    pipeline = _build_pipeline(batch_short_threshold=0)
    call_count = {"c1": 0}

    source = (
        "\\textcolor{black}{LLMDet computes perplexity scores for selected "
        "\\textit{n}-grams.}"
    )
    translation_with_extra_formatting = (
        "\\textcolor{black}{LLMDet会为选定\\textit{n}元语法计算困惑度得分，并统计这些"
        "\\textit{n}元语法的结果。}"
    )

    async def fake_translate_chunk(chunk: str, chunk_id: str, context=None):
        call_count[chunk_id] += 1
        return TranslatedChunk(
            source=chunk,
            translation=translation_with_extra_formatting,
            chunk_id=chunk_id,
            metadata={},
        )

    chunks = [{"chunk_id": "c1", "content": source}]
    with patch.object(pipeline, "translate_chunk", side_effect=fake_translate_chunk):
        results = await pipeline.translate_document(chunks, max_concurrent=2)

    chunk = results[0]
    assert call_count["c1"] == 1
    assert chunk.translation.count("\\textit{") == 1
    assert "这些n元语法" in chunk.translation
    assert chunk.metadata["brace_audit_passed"] is True
    assert chunk.metadata["brace_fix_applied"] is True
    assert chunk.metadata["brace_retry_exhausted"] is False


@pytest.mark.asyncio
async def test_line_end_missing_marker_fixed_without_retry():
    pipeline = _build_pipeline(batch_short_threshold=0)
    call_count = {"c1": 0}

    source = (
        r"\texttt{row-a} [[AMP_1]] desc \\" + "\n" + r"\texttt{row-b} [[AMP_2]] desc \\"
    )
    drift_translation = (
        r"\texttt{row-a} [[AMP_1]] 描述 \\" + "\n" + r"\texttt{row-b} [[AMP_2]] 描述"
    )

    async def fake_translate_chunk(chunk: str, chunk_id: str, context=None):
        call_count[chunk_id] += 1
        return TranslatedChunk(
            source=chunk,
            translation=drift_translation,
            chunk_id=chunk_id,
            metadata={},
        )

    chunks = [{"chunk_id": "c1", "content": source}]
    with patch.object(pipeline, "translate_chunk", side_effect=fake_translate_chunk):
        results = await pipeline.translate_document(chunks, max_concurrent=2)

    chunk = results[0]
    assert call_count["c1"] == 1
    assert chunk.metadata["line_end_audit_passed"] is True
    assert chunk.metadata["line_end_fix_applied"] is True
    assert chunk.metadata["line_end_retry_exhausted"] is False
    assert chunk.metadata["line_end_fallback_applied"] is False
    assert chunk.translation.endswith(r"\\")


@pytest.mark.asyncio
async def test_line_end_retry_exhausted_keeps_last_translation_with_warning():
    pipeline = _build_pipeline(batch_short_threshold=0)
    call_count = {"c1": 0}
    source = r"\texttt{task-a} [[AMP_1]] desc \\" + "\n" + r"\texttt{task-b} [[AMP_2]] desc \\"
    last_translation = r"\texttt{task-a} [[AMP_1]] 描述 \texttt{task-b} [[AMP_2]] 描述"

    async def fake_translate_chunk(chunk: str, chunk_id: str, context=None):
        call_count[chunk_id] += 1
        return TranslatedChunk(
            source=chunk,
            translation=last_translation,
            chunk_id=chunk_id,
            metadata={},
        )

    chunks = [{"chunk_id": "c1", "content": source}]
    with patch.object(pipeline, "translate_chunk", side_effect=fake_translate_chunk):
        results = await pipeline.translate_document(chunks, max_concurrent=2)

    chunk = results[0]
    assert call_count["c1"] == 3
    assert chunk.translation == last_translation
    assert chunk.metadata["line_end_retry_exhausted"] is True
    assert chunk.metadata["line_end_fallback_applied"] is False
    assert chunk.metadata["line_end_audit_passed"] is False
    assert chunk.metadata["kept_last_translation_on_fallback"] is True
    assert "line_end_retry_exhausted" in chunk.metadata["quality_warning_types"]


@pytest.mark.asyncio
async def test_untranslated_retry_succeeds_on_second_attempt():
    pipeline = _build_pipeline(batch_short_threshold=0)
    call_count = {"c1": 0}

    source = "Model-driven methods embed watermarks into the LLM."

    async def fake_translate_chunk(chunk: str, chunk_id: str, context=None):
        call_count[chunk_id] += 1
        translation = source if call_count[chunk_id] == 1 else "模型驱动方法将水印嵌入大语言模型。"
        return TranslatedChunk(
            source=chunk,
            translation=translation,
            chunk_id=chunk_id,
            metadata={},
        )

    chunks = [{"chunk_id": "c1", "content": source}]
    with patch.object(pipeline, "translate_chunk", side_effect=fake_translate_chunk):
        results = await pipeline.translate_document(chunks, max_concurrent=2)

    chunk = results[0]
    assert call_count["c1"] == 2
    assert chunk.translation == "模型驱动方法将水印嵌入大语言模型。"
    assert chunk.metadata["untranslated_audit_passed"] is True
    assert chunk.metadata["untranslated_retry_exhausted"] is False
    assert chunk.metadata["placeholder_attempt"] == 2


@pytest.mark.asyncio
async def test_untranslated_retry_exhausted_keeps_last_translation():
    pipeline = _build_pipeline(batch_short_threshold=0)
    call_count = {"c1": 0}

    source = "This detector should translate the English body text."

    async def fake_translate_chunk(chunk: str, chunk_id: str, context=None):
        call_count[chunk_id] += 1
        return TranslatedChunk(
            source=chunk,
            translation=source,
            chunk_id=chunk_id,
            metadata={},
        )

    chunks = [{"chunk_id": "c1", "content": source}]
    with patch.object(pipeline, "translate_chunk", side_effect=fake_translate_chunk):
        results = await pipeline.translate_document(chunks, max_concurrent=2)

    chunk = results[0]
    assert call_count["c1"] == 3
    assert chunk.translation == source
    assert chunk.metadata["untranslated_audit_passed"] is False
    assert chunk.metadata["untranslated_retry_exhausted"] is True
    assert chunk.metadata["kept_last_translation_on_fallback"] is True
    assert "untranslated_retry_exhausted" in chunk.metadata["quality_warning_types"]


@pytest.mark.asyncio
async def test_untranslated_audit_skips_configuration_like_blocks():
    pipeline = _build_pipeline(batch_short_threshold=0)
    call_count = {"c1": 0}

    source = (
        "for tree={\n"
        "  grow=east,\n"
        "  reversed=true,\n"
        "  anchor=base west,\n"
        "  parent anchor=east,\n"
        "  child anchor=west,\n"
        "  },\n"
        "  where level=1{font=\\scriptsize,fill=pink!5}{},\n"
    )

    async def fake_translate_chunk(chunk: str, chunk_id: str, context=None):
        call_count[chunk_id] += 1
        return TranslatedChunk(
            source=chunk,
            translation=source,
            chunk_id=chunk_id,
            metadata={},
        )

    chunks = [{"chunk_id": "c1", "content": source}]
    with patch.object(pipeline, "translate_chunk", side_effect=fake_translate_chunk):
        results = await pipeline.translate_document(chunks, max_concurrent=2)

    chunk = results[0]
    assert call_count["c1"] == 1
    assert chunk.translation == source
    assert chunk.metadata["untranslated_audit_passed"] is True
    assert chunk.metadata["untranslated_retry_exhausted"] is False


@pytest.mark.asyncio
async def test_untranslated_audit_skips_command_dominated_blocks():
    pipeline = _build_pipeline(batch_short_threshold=0)
    call_count = {"c1": 0}

    source = (
        "\\pgfplotsset{compat=1.16}\n"
        "\\definecolor{colorA}{RGB}{230, 230, 250}\n"
        "\\definecolor{colorB}{RGB}{255, 218, 185}\n"
        "[[ENV_21]]\n"
    )

    async def fake_translate_chunk(chunk: str, chunk_id: str, context=None):
        call_count[chunk_id] += 1
        return TranslatedChunk(
            source=chunk,
            translation=source,
            chunk_id=chunk_id,
            metadata={},
        )

    chunks = [{"chunk_id": "c1", "content": source}]
    with patch.object(pipeline, "translate_chunk", side_effect=fake_translate_chunk):
        results = await pipeline.translate_document(chunks, max_concurrent=2)

    chunk = results[0]
    assert call_count["c1"] == 1
    assert chunk.translation == source
    assert chunk.metadata["untranslated_audit_passed"] is True
    assert chunk.metadata["untranslated_retry_exhausted"] is False


@pytest.mark.asyncio
async def test_retry_union_covers_placeholder_and_line_end_failures():
    pipeline = _build_pipeline(batch_short_threshold=10_000, batch_max_chars=10_000)
    batch_calls: list[list[str]] = []

    async def fake_translate_batch(chunks, context=None):
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        batch_calls.append(chunk_ids)
        round_no = len(batch_calls)

        results = []
        for chunk in chunks:
            if chunk["chunk_id"] == "c1":
                translation = (
                    "第一轮缺占位符"
                    if round_no == 1
                    else "第二轮修复 [[MATH_1]]"
                )
            else:
                translation = (
                    r"\texttt{row-a} [[AMP_1]] 描述 \texttt{row-b} [[AMP_2]] 描述"
                    if round_no == 1
                    else (
                        r"\texttt{row-a} [[AMP_1]] 描述 \\"
                        + "\n"
                        + r"\texttt{row-b} [[AMP_2]] 描述 \\"
                    )
                )
            results.append(
                TranslatedChunk(
                    source=chunk["content"],
                    translation=translation,
                    chunk_id=chunk["chunk_id"],
                    metadata={},
                )
            )
        return results

    async def fail_translate_chunk(*args, **kwargs):
        raise AssertionError("This test should stay on batch path")

    chunks = [
        {"chunk_id": "c1", "content": "source [[MATH_1]] chunk"},
        {
            "chunk_id": "c2",
            "content": (
                r"\texttt{row-a} [[AMP_1]] desc \\" + "\n" + r"\texttt{row-b} [[AMP_2]] desc \\"
            ),
        },
    ]

    with (
        patch.object(pipeline, "translate_batch", side_effect=fake_translate_batch),
        patch.object(pipeline, "translate_chunk", side_effect=fail_translate_chunk),
    ):
        results = await pipeline.translate_document(chunks, max_concurrent=4)

    assert batch_calls[0] == ["c1", "c2"]
    assert batch_calls[1] == ["c1", "c2"]
    by_id = {chunk.chunk_id: chunk for chunk in results}
    assert by_id["c1"].metadata["placeholder_attempt"] == 2
    assert by_id["c2"].metadata["placeholder_attempt"] == 2
    assert by_id["c2"].metadata["line_end_audit_passed"] is True

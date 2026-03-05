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

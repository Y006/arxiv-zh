from unittest.mock import AsyncMock, patch

import pytest

from arxiv_translate.cache.key_builder import CacheKeyBuilder
from arxiv_translate.cache.local_translation_cache import LocalTranslationCache
from arxiv_translate.translator.pipeline import TranslationPipeline, TranslatedChunk


def _make_pipeline(cache: LocalTranslationCache, **kwargs) -> TranslationPipeline:
    provider = AsyncMock()
    provider.prepare_prompt_cache_variants = AsyncMock(return_value=None)
    return TranslationPipeline(
        provider=provider,
        local_cache=cache,
        cache_key_mode="relaxed_chunk",
        **kwargs,
    )


def _payload_for(builder: CacheKeyBuilder, source_text: str) -> dict:
    return builder.build_payload(
        source_text=source_text,
        prompt_variant_semantic="relaxed_chunk",
        glossary_hints={},
        context=None,
        few_shot_examples=[],
        custom_system_prompt=None,
        key_mode="relaxed_chunk",
    )


@pytest.mark.asyncio
async def test_pipeline_uses_local_cache_hit_without_provider_call(tmp_path):
    cache = LocalTranslationCache(cache_dir=tmp_path, max_size_mb=64, ttl_days=30)
    try:
        builder = CacheKeyBuilder(key_mode="relaxed_chunk")
        payload = _payload_for(builder, "hello world")
        key_hash = builder.hash_payload_hex(payload)
        cache.put_by_hash(key_hash, "缓存命中译文")

        pipeline = _make_pipeline(cache)
        chunks = [{"chunk_id": "c1", "content": "hello world"}]
        results = await pipeline.translate_document(chunks, max_concurrent=2)

        assert results[0].translation == "缓存命中译文"
        assert results[0].metadata["local_cache_hit"] is True
        assert results[0].metadata["local_cache_key"] == key_hash[:12]
        pipeline.provider.prepare_prompt_cache_variants.assert_not_called()
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_pipeline_partial_cache_hit_only_requests_misses(tmp_path):
    cache = LocalTranslationCache(cache_dir=tmp_path, max_size_mb=64, ttl_days=30)
    try:
        builder = CacheKeyBuilder(key_mode="relaxed_chunk")
        cached_payload = _payload_for(builder, "cached chunk")
        cached_hash = builder.hash_payload_hex(cached_payload)
        cache.put_by_hash(cached_hash, "缓存译文")

        pipeline = _make_pipeline(cache, batch_short_threshold=0)
        seen_chunk_ids: list[str] = []

        async def fake_translate_chunk(chunk: str, chunk_id: str, context=None):
            seen_chunk_ids.append(chunk_id)
            return TranslatedChunk(
                source=chunk,
                translation="在线译文",
                chunk_id=chunk_id,
                metadata={},
            )

        chunks = [
            {"chunk_id": "c1", "content": "cached chunk"},
            {"chunk_id": "c2", "content": "miss chunk"},
        ]

        with patch.object(pipeline, "translate_chunk", side_effect=fake_translate_chunk):
            results = await pipeline.translate_document(chunks, max_concurrent=2)

        by_id = {chunk.chunk_id: chunk for chunk in results}
        assert seen_chunk_ids == ["c2"]
        assert by_id["c1"].metadata["local_cache_hit"] is True
        assert by_id["c2"].metadata["local_cache_hit"] is False
        assert "local_cache_key" in by_id["c2"].metadata
    finally:
        cache.close()


@pytest.mark.asyncio
async def test_pipeline_preserves_batch_for_miss_chunks(tmp_path):
    cache = LocalTranslationCache(cache_dir=tmp_path, max_size_mb=64, ttl_days=30)
    try:
        builder = CacheKeyBuilder(key_mode="relaxed_chunk")
        cached_hash = builder.hash_payload_hex(_payload_for(builder, "cached chunk"))
        cache.put_by_hash(cached_hash, "缓存译文")

        pipeline = _make_pipeline(cache, batch_short_threshold=10_000, batch_max_chars=10_000)
        batch_calls: list[list[str]] = []

        async def fake_translate_batch(chunks, context=None):
            batch_calls.append([chunk["chunk_id"] for chunk in chunks])
            return [
                TranslatedChunk(
                    source=chunk["content"],
                    translation="批量译文",
                    chunk_id=chunk["chunk_id"],
                    metadata={},
                )
                for chunk in chunks
            ]

        async def fail_translate_chunk(*args, **kwargs):
            raise AssertionError("miss chunks should stay on batch path")

        chunks = [
            {"chunk_id": "c1", "content": "cached chunk"},
            {"chunk_id": "c2", "content": "miss chunk"},
        ]

        with (
            patch.object(pipeline, "translate_batch", side_effect=fake_translate_batch),
            patch.object(pipeline, "translate_chunk", side_effect=fail_translate_chunk),
        ):
            results = await pipeline.translate_document(chunks, max_concurrent=2)

        assert batch_calls == [["c2"]]
        by_id = {chunk.chunk_id: chunk for chunk in results}
        assert by_id["c1"].metadata["local_cache_hit"] is True
        assert by_id["c2"].metadata["local_cache_hit"] is False
    finally:
        cache.close()

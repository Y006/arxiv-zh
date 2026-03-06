import random
import string

from arxiv_translate.cache.key_builder import CacheKeyBuilder
from arxiv_translate.cache.local_translation_cache import LocalTranslationCache


def _random_text(length: int, seed: int) -> str:
    rnd = random.Random(seed)
    alphabet = string.ascii_letters + string.digits + string.punctuation
    return "".join(rnd.choice(alphabet) for _ in range(length))


def test_cache_put_get_roundtrip_zstd(tmp_path):
    cache = LocalTranslationCache(cache_dir=tmp_path, max_size_mb=64, ttl_days=30)
    try:
        key = "a1" * 16
        text = "这是缓存内容 [[MATH_1]]"
        assert cache.put_by_hash(key, text) is True
        assert cache.get_by_hash(key) == text
    finally:
        cache.close()


def test_cache_ttl_expiry_30_days(tmp_path):
    cache = LocalTranslationCache(cache_dir=tmp_path, max_size_mb=64, ttl_days=30)
    try:
        key = "b2" * 16
        assert cache.put_by_hash(key, "will expire")
        cache._conn.execute(
            "UPDATE entries SET expires_at = ? WHERE key_hash = ?",
            (cache._now() - 1, bytes.fromhex(key)),
        )
        cache._conn.commit()
        assert cache.get_by_hash(key) is None
    finally:
        cache.close()


def test_cache_lru_evict_by_compressed_bytes(tmp_path):
    cache = LocalTranslationCache(cache_dir=tmp_path, max_size_mb=1, ttl_days=30)
    try:
        key1 = "c3" * 16
        key2 = "d4" * 16
        text1 = _random_text(900_000, seed=1)
        text2 = _random_text(900_000, seed=2)
        cache.put_by_hash(key1, text1)
        cache.put_by_hash(key2, text2)

        stats = cache.stats()
        assert stats["total_size_bytes"] <= stats["max_size_bytes"]
        assert cache.get_by_hash(key1) is None
        assert cache.get_by_hash(key2) is not None
    finally:
        cache.close()


def test_cache_stats_hit_miss_write_counters(tmp_path):
    cache = LocalTranslationCache(cache_dir=tmp_path, max_size_mb=64, ttl_days=30)
    try:
        key = "e5" * 16
        cache.put_by_hash(key, "cached")
        assert cache.get_by_hash(key) == "cached"
        assert cache.get_by_hash("ff" * 16) is None

        stats = cache.stats()
        assert stats["total_writes"] >= 1
        assert stats["total_hits"] >= 1
        assert stats["total_misses"] >= 1
    finally:
        cache.close()


def test_key_builder_stable_and_order_independent_for_glossary():
    builder = CacheKeyBuilder(key_mode="relaxed_chunk")
    payload_a = builder.build_payload(
        source_text="source text",
        prompt_variant_semantic="relaxed_chunk",
        glossary_hints={"B": "乙", "A": "甲"},
        context="ctx",
        few_shot_examples=[{"source": "s", "target": "t"}],
    )
    payload_b = builder.build_payload(
        source_text="source text",
        prompt_variant_semantic="relaxed_chunk",
        glossary_hints={"A": "甲", "B": "乙"},
        context="ctx",
        few_shot_examples=[{"source": "s", "target": "t"}],
    )

    assert builder.hash_payload_hex(payload_a) == builder.hash_payload_hex(payload_b)

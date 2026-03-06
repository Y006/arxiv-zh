from typer.testing import CliRunner

from arxiv_translate.cache.local_translation_cache import LocalTranslationCache
from arxiv_translate.cli import (
    _write_local_cache_after_quality_gate,
    app,
)
from arxiv_translate.rules.config import Config
from arxiv_translate.translator.pipeline import TranslatedChunk


def _make_config(tmp_path) -> Config:
    config = Config()
    config.paths.cache_dir = str(tmp_path / "cache")
    config.cache.enabled = True
    config.cache.max_size_mb = 64
    config.cache.ttl_days = 30
    config.cache.compression = "zstd"
    config.cache.key_mode = "relaxed_chunk"
    return config


def test_cli_only_writes_cache_when_placeholder_audit_passed(tmp_path):
    cache = LocalTranslationCache(cache_dir=tmp_path / "cache")
    try:
        good_key = "11" * 16
        bad_key = "22" * 16
        fallback_key = "33" * 16

        chunks = [
            TranslatedChunk(
                source="s1",
                translation="通过审计的译文",
                chunk_id="c1",
                metadata={
                    "local_cache_key_hash": good_key,
                    "placeholder_audit_passed": True,
                    "brace_audit_passed": True,
                    "brace_fallback_applied": False,
                },
            ),
            TranslatedChunk(
                source="s2",
                translation="未通过审计的译文",
                chunk_id="c2",
                metadata={
                    "local_cache_key_hash": bad_key,
                    "placeholder_audit_passed": False,
                    "brace_audit_passed": True,
                    "brace_fallback_applied": False,
                },
            ),
            TranslatedChunk(
                source="s3",
                translation="回退译文",
                chunk_id="c3",
                metadata={
                    "local_cache_key_hash": fallback_key,
                    "placeholder_audit_passed": True,
                    "brace_audit_passed": True,
                    "brace_fallback_applied": False,
                },
            ),
        ]

        written, skipped = _write_local_cache_after_quality_gate(
            local_cache=cache,
            translated_chunks=chunks,
            missing_fallback_ids={"c3"},
        )
        assert written == 1
        assert skipped == 2
        assert cache.get_by_hash(good_key) == "通过审计的译文"
        assert cache.get_by_hash(bad_key) is None
        assert cache.get_by_hash(fallback_key) is None
    finally:
        cache.close()


def test_cli_skips_cache_write_when_brace_audit_fails(tmp_path):
    cache = LocalTranslationCache(cache_dir=tmp_path / "cache")
    try:
        bad_key = "44" * 16
        chunk = TranslatedChunk(
            source="s",
            translation="brace fail",
            chunk_id="c1",
            metadata={
                "local_cache_key_hash": bad_key,
                "placeholder_audit_passed": True,
                "brace_audit_passed": False,
                "brace_fallback_applied": False,
            },
        )

        written, skipped = _write_local_cache_after_quality_gate(
            local_cache=cache,
            translated_chunks=[chunk],
            missing_fallback_ids=set(),
        )
        assert written == 0
        assert skipped == 1
        assert chunk.metadata["local_cache_skip_reason"] == "brace_audit_failed"
        assert cache.get_by_hash(bad_key) is None
    finally:
        cache.close()


def test_cli_skips_cache_write_when_brace_fallback_applied(tmp_path):
    cache = LocalTranslationCache(cache_dir=tmp_path / "cache")
    try:
        bad_key = "55" * 16
        chunk = TranslatedChunk(
            source="s",
            translation="fallback",
            chunk_id="c1",
            metadata={
                "local_cache_key_hash": bad_key,
                "placeholder_audit_passed": True,
                "brace_audit_passed": True,
                "brace_fallback_applied": True,
            },
        )

        written, skipped = _write_local_cache_after_quality_gate(
            local_cache=cache,
            translated_chunks=[chunk],
            missing_fallback_ids=set(),
        )
        assert written == 0
        assert skipped == 1
        assert chunk.metadata["local_cache_skip_reason"] == "brace_fallback"
        assert cache.get_by_hash(bad_key) is None
    finally:
        cache.close()


def test_cli_skips_cache_write_when_line_end_audit_fails(tmp_path):
    cache = LocalTranslationCache(cache_dir=tmp_path / "cache")
    try:
        bad_key = "66" * 16
        chunk = TranslatedChunk(
            source="s",
            translation="line end fail",
            chunk_id="c1",
            metadata={
                "local_cache_key_hash": bad_key,
                "placeholder_audit_passed": True,
                "brace_audit_passed": True,
                "brace_fallback_applied": False,
                "line_end_audit_passed": False,
                "line_end_fallback_applied": False,
            },
        )

        written, skipped = _write_local_cache_after_quality_gate(
            local_cache=cache,
            translated_chunks=[chunk],
            missing_fallback_ids=set(),
        )
        assert written == 0
        assert skipped == 1
        assert chunk.metadata["local_cache_skip_reason"] == "line_end_audit_failed"
        assert cache.get_by_hash(bad_key) is None
    finally:
        cache.close()


def test_cli_skips_cache_write_when_line_end_fallback_applied(tmp_path):
    cache = LocalTranslationCache(cache_dir=tmp_path / "cache")
    try:
        bad_key = "77" * 16
        chunk = TranslatedChunk(
            source="s",
            translation="line end fallback",
            chunk_id="c1",
            metadata={
                "local_cache_key_hash": bad_key,
                "placeholder_audit_passed": True,
                "brace_audit_passed": True,
                "brace_fallback_applied": False,
                "line_end_audit_passed": True,
                "line_end_fallback_applied": True,
            },
        )

        written, skipped = _write_local_cache_after_quality_gate(
            local_cache=cache,
            translated_chunks=[chunk],
            missing_fallback_ids=set(),
        )
        assert written == 0
        assert skipped == 1
        assert chunk.metadata["local_cache_skip_reason"] == "line_end_fallback"
        assert cache.get_by_hash(bad_key) is None
    finally:
        cache.close()


def test_cache_stats_command_outputs_size_and_hit_rate(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    cache = LocalTranslationCache(cache_dir=tmp_path / "cache")
    try:
        cache.put_by_hash("aa" * 16, "cached-text")
        cache.get_by_hash("aa" * 16)
    finally:
        cache.close()

    monkeypatch.setattr("arxiv_translate.cli.load_config", lambda: config)
    runner = CliRunner()
    result = runner.invoke(app, ["cache", "stats"])

    assert result.exit_code == 0
    assert "Entries" in result.stdout
    assert "Hit Rate" in result.stdout


def test_cache_clear_command_clears_entries(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    cache = LocalTranslationCache(cache_dir=tmp_path / "cache")
    try:
        cache.put_by_hash("bb" * 16, "cached-text")
    finally:
        cache.close()

    monkeypatch.setattr("arxiv_translate.cli.load_config", lambda: config)
    runner = CliRunner()
    result = runner.invoke(app, ["cache", "clear", "--yes"])
    assert result.exit_code == 0

    cache_after = LocalTranslationCache(cache_dir=tmp_path / "cache")
    try:
        stats = cache_after.stats()
        assert stats["entry_count"] == 0
    finally:
        cache_after.close()

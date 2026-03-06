from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

from arxiv_translate.cache.key_builder import CacheKeyBuilder
from arxiv_translate.rules.user_paths import get_config_dir

try:
    import zstandard as zstd
except ModuleNotFoundError:  # pragma: no cover - exercised in runtime environments only
    zstd = None  # type: ignore[assignment]


class LocalTranslationCache:
    """SQLite-backed local translation cache with TTL + LRU eviction."""

    SCHEMA_VERSION = 1
    DB_FILENAME = "translation_cache.sqlite3"

    def __init__(
        self,
        *,
        cache_dir: Path,
        max_size_mb: int = 2048,
        ttl_days: int = 30,
        compression: str = "zstd",
        key_mode: str = "relaxed_chunk",
    ):
        if compression.lower() != "zstd":
            raise ValueError("Only zstd compression is supported")
        if zstd is None:
            raise RuntimeError(
                "zstandard is required for local cache. Please install dependency: zstandard"
            )

        self.cache_dir = cache_dir.expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / self.DB_FILENAME
        self.max_size_bytes = max(1, int(max_size_mb)) * 1024 * 1024
        self.ttl_days = max(1, int(ttl_days))
        self.ttl_seconds = self.ttl_days * 24 * 60 * 60
        self.compression = compression.lower()
        self.key_builder = CacheKeyBuilder(key_mode=key_mode)
        self._compressor = zstd.ZstdCompressor(level=12)
        self._decompressor = zstd.ZstdDecompressor()

        self._conn = sqlite3.connect(str(self.db_path), timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._ensure_schema()
        self.purge_expired()
        self.evict_if_needed()

    @staticmethod
    def resolve_cache_dir(cache_dir_config: str) -> Path:
        configured = Path(cache_dir_config).expanduser()
        if configured.is_absolute():
            return configured
        return get_config_dir() / configured

    def close(self) -> None:
        self._conn.close()

    def _now(self) -> int:
        return int(time.time())

    def _get_meta(self, key: str, default: str = "0") -> str:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?",
            (key,),
        ).fetchone()
        return row[0] if row else default

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def _inc_meta(self, key: str, amount: int = 1) -> None:
        current = int(self._get_meta(key, "0"))
        self._set_meta(key, str(current + amount))

    def _ensure_schema(self) -> None:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS meta ("
            "  key TEXT PRIMARY KEY,"
            "  value TEXT NOT NULL"
            ")"
        )
        version = self._get_meta("schema_version", "")
        if version and version != str(self.SCHEMA_VERSION):
            self._rebuild_schema()
            return

        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS entries ("
            "  key_hash BLOB PRIMARY KEY,"
            "  key_hex TEXT NOT NULL,"
            "  value_zstd BLOB NOT NULL,"
            "  compressed_bytes INTEGER NOT NULL,"
            "  created_at INTEGER NOT NULL,"
            "  accessed_at INTEGER NOT NULL,"
            "  expires_at INTEGER NOT NULL,"
            "  hit_count INTEGER NOT NULL DEFAULT 0"
            ")"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entries_expires_at ON entries(expires_at)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_entries_accessed_at ON entries(accessed_at)"
        )
        self._set_meta("schema_version", str(self.SCHEMA_VERSION))
        self._conn.commit()

    def _rebuild_schema(self) -> None:
        self._conn.execute("DROP TABLE IF EXISTS entries")
        self._conn.execute("DELETE FROM meta")
        self._conn.commit()
        self._ensure_schema()

    def _compress(self, text: str) -> bytes:
        return self._compressor.compress(text.encode("utf-8"))

    def _decompress(self, payload: bytes) -> str:
        return self._decompressor.decompress(payload).decode("utf-8")

    def _payload_hash_hex(self, key_payload: Dict[str, Any]) -> str:
        return self.key_builder.hash_payload_hex(key_payload)

    def get(
        self, key_payload: Dict[str, Any], *, key_hash_hex: Optional[str] = None
    ) -> Optional[str]:
        return self.get_by_hash(key_hash_hex or self._payload_hash_hex(key_payload))

    def get_by_hash(self, key_hash_hex: str) -> Optional[str]:
        key_hash = bytes.fromhex(key_hash_hex)
        row = self._conn.execute(
            "SELECT value_zstd, expires_at, hit_count FROM entries WHERE key_hash = ?",
            (key_hash,),
        ).fetchone()
        now = self._now()
        if row is None:
            self._inc_meta("total_misses", 1)
            self._conn.commit()
            return None

        value_zstd, expires_at, hit_count = row
        if int(expires_at) < now:
            self._conn.execute("DELETE FROM entries WHERE key_hash = ?", (key_hash,))
            self._inc_meta("total_misses", 1)
            self._inc_meta("total_expired_purged", 1)
            self._conn.commit()
            return None

        try:
            text = self._decompress(value_zstd)
        except Exception:
            self._conn.execute("DELETE FROM entries WHERE key_hash = ?", (key_hash,))
            self._inc_meta("total_misses", 1)
            self._inc_meta("total_corrupt_removed", 1)
            self._conn.commit()
            return None

        self._conn.execute(
            "UPDATE entries SET accessed_at = ?, hit_count = ? WHERE key_hash = ?",
            (now, int(hit_count) + 1, key_hash),
        )
        self._inc_meta("total_hits", 1)
        self._conn.commit()
        return text

    def put(
        self,
        key_payload: Dict[str, Any],
        translation: str,
        *,
        key_hash_hex: Optional[str] = None,
    ) -> bool:
        return self.put_by_hash(
            key_hash_hex or self._payload_hash_hex(key_payload), translation
        )

    def put_by_hash(self, key_hash_hex: str, translation: str) -> bool:
        if not translation:
            return False

        key_hash = bytes.fromhex(key_hash_hex)
        now = self._now()
        expires_at = now + self.ttl_seconds
        payload = self._compress(translation)
        compressed_bytes = len(payload)

        self._conn.execute(
            "INSERT INTO entries("
            "  key_hash, key_hex, value_zstd, compressed_bytes, created_at, accessed_at, expires_at, hit_count"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, 0) "
            "ON CONFLICT(key_hash) DO UPDATE SET "
            "  key_hex = excluded.key_hex,"
            "  value_zstd = excluded.value_zstd,"
            "  compressed_bytes = excluded.compressed_bytes,"
            "  created_at = excluded.created_at,"
            "  accessed_at = excluded.accessed_at,"
            "  expires_at = excluded.expires_at",
            (
                key_hash,
                key_hash_hex,
                payload,
                compressed_bytes,
                now,
                now,
                expires_at,
            ),
        )
        self._inc_meta("total_writes", 1)
        self._conn.commit()
        self.purge_expired()
        self.evict_if_needed()
        return True

    def purge_expired(self) -> int:
        now = self._now()
        row = self._conn.execute(
            "SELECT COUNT(*) FROM entries WHERE expires_at < ?",
            (now,),
        ).fetchone()
        expired_count = int(row[0]) if row else 0
        if expired_count <= 0:
            return 0

        self._conn.execute("DELETE FROM entries WHERE expires_at < ?", (now,))
        self._inc_meta("total_expired_purged", expired_count)
        self._conn.commit()
        return expired_count

    def evict_if_needed(self) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(SUM(compressed_bytes), 0) FROM entries"
        ).fetchone()
        current_size = int(row[0]) if row else 0
        if current_size <= self.max_size_bytes:
            return 0

        to_remove: list[bytes] = []
        reclaimed = 0
        for key_hash, compressed_bytes in self._conn.execute(
            "SELECT key_hash, compressed_bytes FROM entries ORDER BY accessed_at ASC"
        ):
            to_remove.append(key_hash)
            reclaimed += int(compressed_bytes)
            if current_size - reclaimed <= self.max_size_bytes:
                break

        if not to_remove:
            return 0

        self._conn.executemany(
            "DELETE FROM entries WHERE key_hash = ?",
            [(key_hash,) for key_hash in to_remove],
        )
        self._inc_meta("total_evicted_lru", len(to_remove))
        self._conn.commit()
        return len(to_remove)

    def clear(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()
        deleted = int(row[0]) if row else 0
        self._conn.execute("DELETE FROM entries")
        self._set_meta("total_hits", "0")
        self._set_meta("total_misses", "0")
        self._set_meta("total_writes", "0")
        self._set_meta("total_expired_purged", "0")
        self._set_meta("total_evicted_lru", "0")
        self._set_meta("total_corrupt_removed", "0")
        self._conn.commit()
        return deleted

    def stats(self) -> Dict[str, Any]:
        self.purge_expired()
        self.evict_if_needed()
        row = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(compressed_bytes), 0) FROM entries"
        ).fetchone()
        entry_count = int(row[0]) if row else 0
        total_bytes = int(row[1]) if row else 0
        total_hits = int(self._get_meta("total_hits", "0"))
        total_misses = int(self._get_meta("total_misses", "0"))
        total_writes = int(self._get_meta("total_writes", "0"))
        total_queries = total_hits + total_misses
        hit_rate = (total_hits / total_queries) if total_queries else 0.0

        return {
            "db_path": str(self.db_path),
            "entry_count": entry_count,
            "total_size_bytes": total_bytes,
            "total_size_mb": total_bytes / (1024 * 1024),
            "max_size_bytes": self.max_size_bytes,
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "usage_ratio": (total_bytes / self.max_size_bytes)
            if self.max_size_bytes
            else 0.0,
            "ttl_days": self.ttl_days,
            "compression": self.compression,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "total_writes": total_writes,
            "total_expired_purged": int(self._get_meta("total_expired_purged", "0")),
            "total_evicted_lru": int(self._get_meta("total_evicted_lru", "0")),
            "total_corrupt_removed": int(self._get_meta("total_corrupt_removed", "0")),
            "hit_rate": hit_rate,
        }

"""Translation pipeline with dynamic glossary hints."""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, Callable, Set, cast

from pydantic import BaseModel, Field

from ..cache.key_builder import CacheKeyBuilder
from ..cache.local_translation_cache import LocalTranslationCache
from ..rules.glossary import Glossary
from .llm_base import LLMProvider
from .prompts import build_batch_translation_text, build_system_prompt


class TranslatedChunk(BaseModel):
    """A translated chunk with source, translation, and metadata."""

    source: str
    translation: str
    chunk_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TranslationPipeline:
    """
    Translation pipeline that orchestrates chunk translation with dynamic
    glossary hints.
    """

    NEWLINE_SOFT_TOKEN = "[[SL]]"
    NEWLINE_PARA_TOKEN = "[[PL]]"
    NEWLINE_SOFT_RAW_TOKEN = "[[SL_RAW]]"
    NEWLINE_PARA_RAW_TOKEN = "[[PL_RAW]]"
    NEWLINE_SOFT_RAW_SENTINEL = "[[__ARXIV_TRANSLATE_SL_RAW__]]"
    NEWLINE_PARA_RAW_SENTINEL = "[[__ARXIV_TRANSLATE_PL_RAW__]]"
    PLACEHOLDER_RETRY_MAX_ATTEMPTS = 3
    PLACEHOLDER_AUDIT_PATTERN = re.compile(r"\[\[[A-Z_]+_\d+\]\]")
    PLACEHOLDER_AUDIT_WHITELIST = {
        NEWLINE_SOFT_TOKEN,
        NEWLINE_PARA_TOKEN,
        NEWLINE_SOFT_RAW_TOKEN,
        NEWLINE_PARA_RAW_TOKEN,
    }

    def __init__(
        self,
        provider: LLMProvider,
        glossary: Optional[Glossary] = None,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        rate_limit_delay: float = 0.0,
        state_file: Optional[Union[str, Path]] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        abstract_context: Optional[str] = None,
        custom_system_prompt: Optional[str] = None,
        model_name: str = "unknown",
        hq_mode: bool = False,
        batch_short_threshold: int = 300,
        batch_max_chars: int = 2000,
        sequential_mode: bool = False,
        request_timeout: float = 120.0,
        per_call_timeout: float = 150.0,
        local_cache: Optional[LocalTranslationCache] = None,
        cache_key_builder: Optional[CacheKeyBuilder] = None,
        cache_key_mode: str = "relaxed_chunk",
    ):
        self.provider = provider
        self.glossary = glossary or Glossary()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.rate_limit_delay = rate_limit_delay
        self.state_file = Path(state_file) if state_file else None
        self.few_shot_examples = few_shot_examples or []
        self.abstract_context = abstract_context
        self.custom_system_prompt = custom_system_prompt
        self.model_name = model_name
        self.hq_mode = hq_mode
        self.batch_short_threshold = batch_short_threshold
        self.batch_max_chars = batch_max_chars
        self.sequential_mode = sequential_mode
        self.request_timeout = request_timeout
        self.per_call_timeout = per_call_timeout
        self.local_cache = local_cache
        self.cache_key_builder = cache_key_builder or CacheKeyBuilder(
            key_mode=cache_key_mode
        )
        self.cache_key_mode = cache_key_mode
        self._started_at: Optional[str] = None
        self._last_provider_cache_meta: Optional[Dict[str, Any]] = None

    def _build_glossary_hints(self, text: str) -> Dict[str, str]:
        """Build glossary hints filtered by case-insensitive term matching."""
        if not text:
            return {}

        return {
            term: entry.target
            for term, entry in self.glossary.terms.items()
            if re.search(
                r"(?<!\w)" + re.escape(term) + r"(?!\w)",
                text,
                re.IGNORECASE | re.ASCII,
            )
        }

    def _assert_no_token_collision(self, text: str) -> None:
        """Ensure newline control tokens do not exist before encoding."""
        if self.NEWLINE_SOFT_TOKEN in text or self.NEWLINE_PARA_TOKEN in text:
            raise ValueError(
                "Newline token collision unresolved before encoding: "
                f"{self.NEWLINE_SOFT_TOKEN}/{self.NEWLINE_PARA_TOKEN}"
            )

    def _escape_newline_token_literals(self, text: str) -> str:
        """Escape literal [[SL]]/[[PL]] in source text before newline encoding."""
        escaped = text.replace(
            self.NEWLINE_SOFT_RAW_TOKEN, self.NEWLINE_SOFT_RAW_SENTINEL
        )
        escaped = escaped.replace(
            self.NEWLINE_PARA_RAW_TOKEN, self.NEWLINE_PARA_RAW_SENTINEL
        )
        escaped = escaped.replace(self.NEWLINE_SOFT_TOKEN, self.NEWLINE_SOFT_RAW_TOKEN)
        escaped = escaped.replace(self.NEWLINE_PARA_TOKEN, self.NEWLINE_PARA_RAW_TOKEN)
        return escaped

    def _restore_escaped_newline_token_literals(self, text: str) -> str:
        """Restore literal [[SL]]/[[PL]] after decoding newline tokens."""
        restored = text.replace(self.NEWLINE_SOFT_RAW_TOKEN, self.NEWLINE_SOFT_TOKEN)
        restored = restored.replace(
            self.NEWLINE_PARA_RAW_TOKEN, self.NEWLINE_PARA_TOKEN
        )
        restored = restored.replace(
            self.NEWLINE_SOFT_RAW_SENTINEL, self.NEWLINE_SOFT_RAW_TOKEN
        )
        restored = restored.replace(
            self.NEWLINE_PARA_RAW_SENTINEL, self.NEWLINE_PARA_RAW_TOKEN
        )
        return restored

    def _count_newline_breaks(self, text: str) -> tuple[int, int]:
        """Count newline breaks using greedy paragraph-first matching."""
        sl_count = 0
        pl_count = 0
        i = 0
        while i < len(text):
            if text.startswith("\n\n", i):
                pl_count += 1
                i += 2
                continue
            if text[i] == "\n":
                sl_count += 1
            i += 1
        return sl_count, pl_count

    def _encode_newlines_for_llm(self, text: str) -> tuple[str, Dict[str, int]]:
        """Encode newlines to stable control tokens before LLM translation."""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        escaped = self._escape_newline_token_literals(normalized)
        self._assert_no_token_collision(escaped)

        sl_count, pl_count = self._count_newline_breaks(escaped)
        encoded = []
        i = 0
        while i < len(escaped):
            if escaped.startswith("\n\n", i):
                encoded.append(self.NEWLINE_PARA_TOKEN)
                i += 2
                continue
            if escaped[i] == "\n":
                encoded.append(self.NEWLINE_SOFT_TOKEN)
                i += 1
                continue
            encoded.append(escaped[i])
            i += 1

        return "".join(encoded), {
            "source_sl_count": sl_count,
            "source_pl_count": pl_count,
        }

    def _decode_newlines_from_llm(self, text: str) -> str:
        """Decode control tokens back to newlines after LLM translation."""
        decoded = text.replace(self.NEWLINE_PARA_TOKEN, "\n\n")
        decoded = decoded.replace(self.NEWLINE_SOFT_TOKEN, "\n")
        return self._restore_escaped_newline_token_literals(decoded)

    def _postprocess_llm_newlines(
        self, llm_text: str, source_sl_count: int, source_pl_count: int
    ) -> tuple[str, Dict[str, Any]]:
        """Remove raw newlines, decode tokens, and return newline diagnostics."""
        normalized = llm_text.replace("\r\n", "\n").replace("\r", "\n")
        raw_newline_removed_count = normalized.count("\n")
        sanitized = normalized.replace("\n", "")

        llm_sl_token_count = sanitized.count(self.NEWLINE_SOFT_TOKEN)
        llm_pl_token_count = sanitized.count(self.NEWLINE_PARA_TOKEN)
        newline_token_mismatch = (
            llm_sl_token_count != source_sl_count
            or llm_pl_token_count != source_pl_count
        )

        warning_parts = []
        if raw_newline_removed_count > 0:
            warning_parts.append(
                f"removed {raw_newline_removed_count} raw newline(s) from LLM output"
            )
        if newline_token_mismatch:
            warning_parts.append(
                "newline token mismatch: "
                f"source(sl={source_sl_count}, pl={source_pl_count}) "
                f"vs llm(sl={llm_sl_token_count}, pl={llm_pl_token_count})"
            )

        final_translation = self._decode_newlines_from_llm(sanitized)

        return final_translation, {
            "raw_newline_removed_count": raw_newline_removed_count,
            "newline_token_mismatch": newline_token_mismatch,
            "llm_sl_token_count": llm_sl_token_count,
            "llm_pl_token_count": llm_pl_token_count,
            "newline_warning": " | ".join(warning_parts),
        }

    def _extract_placeholders_for_audit(self, text: str) -> Set[str]:
        return {
            token
            for token in self.PLACEHOLDER_AUDIT_PATTERN.findall(text or "")
            if token not in self.PLACEHOLDER_AUDIT_WHITELIST
        }

    def _audit_placeholder_alignment(
        self, source: str, translation: str
    ) -> Dict[str, Any]:
        source_placeholders = self._extract_placeholders_for_audit(source)
        translated_placeholders = self._extract_placeholders_for_audit(translation)
        missing = sorted(source_placeholders - translated_placeholders)
        spurious = sorted(translated_placeholders - source_placeholders)
        return {
            "passed": not missing and not spurious,
            "missing": missing,
            "spurious": spurious,
        }

    def _build_local_cache_payload(
        self,
        *,
        source_text: str,
        glossary_hints: Dict[str, str],
        merged_context: Optional[str],
    ) -> Dict[str, Any]:
        return self.cache_key_builder.build_payload(
            source_text=source_text,
            prompt_variant_semantic=self.cache_key_mode,
            glossary_hints=glossary_hints,
            context=merged_context,
            few_shot_examples=self.few_shot_examples,
            custom_system_prompt=self.custom_system_prompt,
            key_mode=self.cache_key_mode,
        )

    @staticmethod
    def _local_cache_key_preview(key_hash_hex: str) -> str:
        return key_hash_hex[:12]

    async def _call_with_retry(
        self,
        text: str,
        context: Optional[str] = None,
        glossary_hints: Optional[Dict[str, str]] = None,
        prompt_variant: str = "individual",
    ) -> str:
        """
        Call the LLM provider with exponential backoff retry.

        Args:
            text: The text to translate.
            context: Optional context for the translation.
            glossary_hints: Optional glossary hints.

        Returns:
            The translated text.

        Raises:
            RuntimeError: If all retries are exhausted.
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                call_kwargs: Dict[str, Any] = {
                    "text": text,
                    "context": context,
                    "glossary_hints": glossary_hints,
                    "few_shot_examples": self.few_shot_examples,
                    "custom_system_prompt": self.custom_system_prompt,
                    "prompt_variant": prompt_variant,
                }

                async def _provider_call_with_compat() -> str:
                    try:
                        return await self.provider.translate(**call_kwargs)
                    except TypeError as e:
                        if "prompt_variant" not in str(e):
                            raise
                        fallback_kwargs = dict(call_kwargs)
                        fallback_kwargs.pop("prompt_variant", None)
                        return await self.provider.translate(**fallback_kwargs)

                result = await asyncio.wait_for(
                    _provider_call_with_compat(),
                    timeout=self.per_call_timeout,
                )
                self._last_provider_cache_meta = getattr(
                    self.provider, "_last_cache_meta", None
                )
                return result
            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"Request timed out after {self.per_call_timeout}s "
                    f"(attempt {attempt + 1}/{self.max_retries})"
                )
                if attempt < self.max_retries - 1:
                    delay = max(10.0, self.retry_delay * (2 ** (attempt + 1)))
                    await asyncio.sleep(delay)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    error_str = str(e).lower()
                    if "429" in error_str or "rate limit" in error_str:
                        delay = max(5.0, self.retry_delay * (3**attempt))
                    elif (
                        "timeout" in error_str
                        or "timed out" in error_str
                        or isinstance(e, TimeoutError)
                    ):
                        # Timeout class errors: wait longer before retry
                        delay = max(10.0, self.retry_delay * (2 ** (attempt + 1)))
                    else:
                        delay = self.retry_delay * (2**attempt)
                    await asyncio.sleep(delay)

        raise last_error  # type: ignore

    async def translate_chunk(
        self,
        chunk: str,
        chunk_id: str,
        context: Optional[str] = None,
    ) -> TranslatedChunk:
        """
        Translate a single chunk with dynamic glossary hints.

        Args:
            chunk: The text chunk to translate.
            chunk_id: Unique identifier for this chunk.
            context: Optional context information.

        Returns:
            TranslatedChunk with the translation result.
        """
        encoded_text, source_breaks = self._encode_newlines_for_llm(chunk)

        # Build glossary hints for the current chunk
        glossary_hints = self._build_glossary_hints(chunk)

        # Merge abstract context with provided context for high-quality mode
        merged_context = context
        if self.abstract_context:
            if context:
                merged_context = (
                    f"{context}\n\nDocument Abstract:\n{self.abstract_context}"
                )
            else:
                merged_context = f"Document Abstract:\n{self.abstract_context}"

        raw_translation = await self._call_with_retry(
            text=encoded_text,
            context=merged_context,
            glossary_hints=None,
            prompt_variant="individual",
        )
        provider_cache_meta = self._last_provider_cache_meta

        final_translation, newline_meta = self._postprocess_llm_newlines(
            raw_translation,
            source_breaks["source_sl_count"],
            source_breaks["source_pl_count"],
        )
        decoded_sl_count, decoded_pl_count = self._count_newline_breaks(
            final_translation
        )

        return TranslatedChunk(
            source=chunk,
            translation=final_translation,
            chunk_id=chunk_id,
            metadata={
                "source_length": len(chunk),
                "batched": False,
                "batch_id": None,
                "glossary_hints": glossary_hints,
                "had_glossary_terms": len(glossary_hints) > 0,
                "glossary_terms_count": len(glossary_hints),
                "skipped_placeholder": False,
                "newline_codec_applied": True,
                "source_sl_count": source_breaks["source_sl_count"],
                "source_pl_count": source_breaks["source_pl_count"],
                "decoded_sl_count": decoded_sl_count,
                "decoded_pl_count": decoded_pl_count,
                **newline_meta,
                "provider_cache_meta": provider_cache_meta,
            },
        )

    async def translate_batch(
        self,
        chunks: List[Dict[str, str]],
        context: Optional[str] = None,
    ) -> List[TranslatedChunk]:
        encoded_chunks = []
        chunk_glossary_hints = []
        batch_glossary_hints: Dict[str, str] = {}
        source_breaks = []
        for chunk_data in chunks:
            source_text = chunk_data["content"]
            encoded_text, break_meta = self._encode_newlines_for_llm(source_text)
            encoded_chunks.append(
                {"chunk_id": chunk_data["chunk_id"], "content": encoded_text}
            )
            source_breaks.append(break_meta)
            glossary_hints = self._build_glossary_hints(source_text)
            chunk_glossary_hints.append(glossary_hints)
            batch_glossary_hints.update(glossary_hints)

        batch_text = build_batch_translation_text(encoded_chunks)

        merged_context = context
        if self.abstract_context:
            if context:
                merged_context = (
                    f"{context}\n\nDocument Abstract:\n{self.abstract_context}"
                )
            else:
                merged_context = f"Document Abstract:\n{self.abstract_context}"

        batch_instruction = "请翻译以下编号内容，保持相同的编号格式返回。"
        if merged_context:
            merged_context = f"{batch_instruction}\n\n{merged_context}"
        else:
            merged_context = batch_instruction

        raw_response = await self._call_with_retry(
            text=batch_text,
            context=merged_context,
            glossary_hints=None,
            prompt_variant="batch",
        )
        provider_cache_meta = self._last_provider_cache_meta

        pattern = r"\[(\d+)\]\s*(.+?)(?=\[\d+\]|$)"
        matches = re.findall(pattern, raw_response, re.DOTALL)

        if len(matches) != len(chunks):
            return []

        results = []
        for i, (idx_str, translated_text) in enumerate(matches):
            idx = int(idx_str) - 1
            if idx < 0 or idx >= len(chunks):
                return []

            final_translation, newline_meta = self._postprocess_llm_newlines(
                translated_text.strip(),
                source_breaks[idx]["source_sl_count"],
                source_breaks[idx]["source_pl_count"],
            )
            decoded_sl_count, decoded_pl_count = self._count_newline_breaks(
                final_translation
            )

            results.append(
                TranslatedChunk(
                    source=chunks[idx]["content"],
                    translation=final_translation,
                    chunk_id=chunks[idx]["chunk_id"],
                    metadata={
                        "source_length": len(chunks[idx]["content"]),
                        "batched": True,
                        "batch_id": None,
                        "glossary_hints": chunk_glossary_hints[idx],
                        "had_glossary_terms": len(chunk_glossary_hints[idx]) > 0,
                        "glossary_terms_count": len(chunk_glossary_hints[idx]),
                        "skipped_placeholder": False,
                        "newline_codec_applied": True,
                        "source_sl_count": source_breaks[idx]["source_sl_count"],
                        "source_pl_count": source_breaks[idx]["source_pl_count"],
                        "decoded_sl_count": decoded_sl_count,
                        "decoded_pl_count": decoded_pl_count,
                        **newline_meta,
                        "provider_cache_meta": provider_cache_meta,
                    },
                )
            )

        result_map = {r.chunk_id: r for r in results}
        ordered_results = [result_map[c["chunk_id"]] for c in chunks]

        return ordered_results

    def _load_state(self) -> Dict[str, Any]:
        """Load intermediate state from file."""
        if self.state_file and self.state_file.exists():
            data = json.loads(self.state_file.read_text(encoding="utf-8"))
            state = {
                "completed": data.get("completed", []),
                "results": data.get("results", []),
            }
            # Restore message_history if provider supports it
            if hasattr(self.provider, "set_history") and "message_history" in data:
                self.provider.set_history(data["message_history"])
            return state
        return {"completed": [], "results": []}

    def _save_state(self, state: Dict[str, Any], total_chunks: int = 0) -> None:
        """Save intermediate state to file with v2.1 schema."""
        if self.state_file:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            output = {
                "version": "2.1",
                "meta": {
                    "model": self.model_name,
                    "hq_mode": self.hq_mode,
                    "total_chunks": total_chunks,
                    "started_at": self._started_at,
                    "finished_at": state.get("_finished_at"),
                    "total_seconds": state.get("_total_seconds"),
                },
                "completed": state["completed"],
                "results": state["results"],
            }
            # Persist message_history if provider supports it
            if hasattr(self.provider, "get_history"):
                output["message_history"] = self.provider.get_history()
            self.state_file.write_text(
                json.dumps(output, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    async def translate_document(
        self,
        chunks: List[Dict[str, str]],
        context: Optional[str] = None,
        max_concurrent: int = 50,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        batch_stats_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> List[TranslatedChunk]:
        """
        Translate a document consisting of multiple chunks with concurrent requests.

        Args:
            chunks: List of chunk dictionaries with 'chunk_id' and 'content' keys.
            context: Optional context for all chunks.
            max_concurrent: Maximum number of concurrent translation requests.
            progress_callback: Optional callback (completed, total) for progress updates.

        Returns:
            List of TranslatedChunk objects in the same order as input.
        """
        self._started_at = datetime.now().isoformat()

        state = self._load_state()
        completed_ids = set(state["completed"])

        results_map: Dict[str, TranslatedChunk] = {}
        for result_data in state["results"]:
            chunk = TranslatedChunk(**result_data)
            results_map[chunk.chunk_id] = chunk

        placeholder_pattern = re.compile(r"^\[\[[A-Z_]+_\d+\]\]$")
        placeholder_chunks = []
        all_translatable_chunks = []

        for c in chunks:
            if c["chunk_id"] not in completed_ids:
                if placeholder_pattern.fullmatch(c["content"].strip()):
                    placeholder_chunks.append(c)
                else:
                    all_translatable_chunks.append(c)

        for chunk_data in placeholder_chunks:
            chunk_id = chunk_data["chunk_id"]
            content = chunk_data["content"]

            placeholder_result = TranslatedChunk(
                source=content,
                translation=content,
                chunk_id=chunk_id,
                metadata={"skipped_placeholder": True},
            )
            results_map[chunk_id] = placeholder_result
            state["completed"].append(chunk_id)
            state["results"].append(placeholder_result.model_dump())

        if not all_translatable_chunks:
            self._save_state(state, total_chunks=len(chunks))
            return [results_map[chunk_data["chunk_id"]] for chunk_data in chunks]

        # --- Document-level glossary + pre-built system prompts ---
        all_text = " ".join(c["content"] for c in all_translatable_chunks)
        doc_glossary = self._build_glossary_hints(all_text)

        # Merge abstract context (same logic as translate_chunk)
        merged_context = context
        if self.abstract_context:
            if context:
                merged_context = (
                    f"{context}\n\nDocument Abstract:\n{self.abstract_context}"
                )
            else:
                merged_context = f"Document Abstract:\n{self.abstract_context}"

        local_cache_key_map: Dict[str, str] = {}
        translatable_chunks = []
        local_cache_hit_count = 0

        if self.local_cache is not None:
            try:
                for chunk_data in all_translatable_chunks:
                    key_payload = self._build_local_cache_payload(
                        source_text=chunk_data["content"],
                        glossary_hints=doc_glossary,
                        merged_context=merged_context,
                    )
                    key_hash_hex = self.cache_key_builder.hash_payload_hex(key_payload)
                    local_cache_key_map[chunk_data["chunk_id"]] = key_hash_hex
                    cached_translation = self.local_cache.get(
                        key_payload, key_hash_hex=key_hash_hex
                    )
                    if cached_translation is None:
                        translatable_chunks.append(chunk_data)
                        continue

                    local_cache_hit_count += 1
                    cached_chunk = TranslatedChunk(
                        source=chunk_data["content"],
                        translation=cached_translation,
                        chunk_id=chunk_data["chunk_id"],
                        metadata={
                            "source_length": len(chunk_data["content"]),
                            "batched": False,
                            "batch_id": None,
                            "skipped_placeholder": False,
                            "local_cache_hit": True,
                            "local_cache_key": self._local_cache_key_preview(
                                key_hash_hex
                            ),
                            "local_cache_key_hash": key_hash_hex,
                            "local_cache_written": False,
                            "local_cache_skip_reason": None,
                        },
                    )
                    results_map[cached_chunk.chunk_id] = cached_chunk
                    state["completed"].append(cached_chunk.chunk_id)
                    state["results"].append(cached_chunk.model_dump())
            except Exception as e:
                print(f"[LOCAL CACHE] disabled due to read error: {e}")
                translatable_chunks = list(all_translatable_chunks)
                local_cache_key_map = {}
        else:
            translatable_chunks = list(all_translatable_chunks)

        if local_cache_hit_count > 0:
            print(
                f"[LOCAL CACHE] hits={local_cache_hit_count} misses={len(translatable_chunks)}"
            )

        if not translatable_chunks:
            self._save_state(state, total_chunks=len(chunks))
            return [results_map[chunk_data["chunk_id"]] for chunk_data in chunks]

        # Pre-build individual system prompt (for long_chunks)
        individual_system_prompt = build_system_prompt(
            glossary_hints=doc_glossary,
            context=merged_context,
            few_shot_examples=self.few_shot_examples,
            custom_system_prompt=self.custom_system_prompt,
        )

        # Pre-build batch system prompt (includes batch_instruction in context)
        batch_instruction = "请翻译以下编号内容，保持相同的编号格式返回。"
        batch_merged_context = (
            f"{batch_instruction}\n\n{merged_context}"
            if merged_context
            else batch_instruction
        )
        batch_system_prompt = build_system_prompt(
            glossary_hints=doc_glossary,
            context=batch_merged_context,
            few_shot_examples=self.few_shot_examples,
            custom_system_prompt=self.custom_system_prompt,
        )

        self.provider._prebuilt_system_prompt = individual_system_prompt  # type: ignore[attr-defined]
        self.provider._prebuilt_batch_prompt = batch_system_prompt  # type: ignore[attr-defined]

        def _attach_local_cache_metadata(chunk_result: TranslatedChunk) -> None:
            key_hash_hex = local_cache_key_map.get(chunk_result.chunk_id)
            if key_hash_hex is None:
                return
            chunk_result.metadata.setdefault("local_cache_hit", False)
            chunk_result.metadata["local_cache_key"] = self._local_cache_key_preview(
                key_hash_hex
            )
            chunk_result.metadata["local_cache_key_hash"] = key_hash_hex
            chunk_result.metadata.setdefault("local_cache_written", False)
            chunk_result.metadata.setdefault("local_cache_skip_reason", None)

        SHORT_THRESHOLD = self.batch_short_threshold
        MAX_BATCH_CHARS = self.batch_max_chars

        short_chunks = []
        long_chunks = []
        for c in translatable_chunks:
            if len(c["content"]) < SHORT_THRESHOLD:
                short_chunks.append(c)
            else:
                long_chunks.append(c)

        batches: List[List[Dict[str, str]]] = []
        current_batch: List[Dict[str, str]] = []
        current_len = 0

        for chunk in short_chunks:
            chunk_len = len(chunk["content"])
            if current_len + chunk_len > MAX_BATCH_CHARS and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_len = 0
            current_batch.append(chunk)
            current_len += chunk_len

        if current_batch:
            batches.append(current_batch)

        prompt_variants_to_warm: List[str] = []
        if batches:
            prompt_variants_to_warm.append("batch")
        if long_chunks:
            prompt_variants_to_warm.append("individual")
        if prompt_variants_to_warm:
            try:
                await self.provider.prepare_prompt_cache_variants(
                    prompt_variants=prompt_variants_to_warm,
                    few_shot_examples=self.few_shot_examples,
                )
            except Exception as e:
                print(f"[CACHE WARMUP] skipped due to error: {e}")

        total_api_calls = len(batches) + len(long_chunks)
        if batch_stats_callback:
            batch_stats_callback(len(batches), len(long_chunks), total_api_calls)

        translatable_chunk_map = {
            chunk_data["chunk_id"]: chunk_data for chunk_data in all_translatable_chunks
        }
        translatable_ids = [
            chunk_data["chunk_id"] for chunk_data in all_translatable_chunks
        ]

        def _split_for_retry_round(
            retry_chunks: List[Dict[str, str]],
        ) -> tuple[List[List[Dict[str, str]]], List[Dict[str, str]]]:
            short_retry: List[Dict[str, str]] = []
            long_retry: List[Dict[str, str]] = []
            for chunk_data in retry_chunks:
                if len(chunk_data["content"]) < SHORT_THRESHOLD:
                    short_retry.append(chunk_data)
                else:
                    long_retry.append(chunk_data)

            retry_batches: List[List[Dict[str, str]]] = []
            current_batch: List[Dict[str, str]] = []
            current_len = 0
            for chunk_data in short_retry:
                chunk_len = len(chunk_data["content"])
                if current_len + chunk_len > MAX_BATCH_CHARS and current_batch:
                    retry_batches.append(current_batch)
                    current_batch = []
                    current_len = 0
                current_batch.append(chunk_data)
                current_len += chunk_len
            if current_batch:
                retry_batches.append(current_batch)

            return retry_batches, long_retry

        async def _translate_retry_round(
            retry_chunks: List[Dict[str, str]],
            round_index: int,
        ) -> Dict[str, TranslatedChunk]:
            if not retry_chunks:
                return {}

            retry_batches, retry_long_chunks = _split_for_retry_round(retry_chunks)
            round_results: Dict[str, TranslatedChunk] = {}

            def _make_skipped_chunk(chunk_data: Dict[str, str], error: Any) -> TranslatedChunk:
                return TranslatedChunk(
                    source=chunk_data["content"],
                    translation=chunk_data["content"],
                    chunk_id=chunk_data["chunk_id"],
                    metadata={
                        "skipped": True,
                        "skip_reason": str(error),
                        "skipped_at": datetime.now().isoformat(),
                        "retry_round": round_index,
                    },
                )

            if self.sequential_mode:
                for i, batch in enumerate(retry_batches):
                    batch_id = f"retry_r{round_index}_batch_{i}"
                    try:
                        batch_results = await self.translate_batch(
                            chunks=batch,
                            context=context,
                        )
                    except Exception:
                        batch_results = []

                    if not batch_results:
                        batch_results = []
                        for chunk_data in batch:
                            try:
                                chunk_result = await self.translate_chunk(
                                    chunk=chunk_data["content"],
                                    chunk_id=chunk_data["chunk_id"],
                                    context=context,
                                )
                            except Exception as e:
                                chunk_result = _make_skipped_chunk(chunk_data, e)
                            batch_results.append(chunk_result)

                    for chunk_result in batch_results:
                        chunk_result.metadata["batch_id"] = batch_id
                        chunk_result.metadata["retry_round"] = round_index
                        _attach_local_cache_metadata(chunk_result)
                        round_results[chunk_result.chunk_id] = chunk_result

                for chunk_data in retry_long_chunks:
                    try:
                        chunk_result = await self.translate_chunk(
                            chunk=chunk_data["content"],
                            chunk_id=chunk_data["chunk_id"],
                            context=context,
                        )
                    except Exception as e:
                        chunk_result = _make_skipped_chunk(chunk_data, e)
                    chunk_result.metadata["retry_round"] = round_index
                    _attach_local_cache_metadata(chunk_result)
                    round_results[chunk_result.chunk_id] = chunk_result

                return round_results

            semaphore = asyncio.Semaphore(max_concurrent)

            async def _translate_chunk_retry(chunk_data: Dict[str, str]) -> TranslatedChunk:
                try:
                    async with semaphore:
                        chunk_result = await self.translate_chunk(
                            chunk=chunk_data["content"],
                            chunk_id=chunk_data["chunk_id"],
                            context=context,
                        )
                        chunk_result.metadata["retry_round"] = round_index
                        return chunk_result
                except BaseException as e:
                    if isinstance(e, (KeyboardInterrupt, SystemExit)):
                        raise
                    skipped_chunk = _make_skipped_chunk(chunk_data, e)
                    skipped_chunk.metadata["retry_round"] = round_index
                    return skipped_chunk

            async def _translate_batch_retry(
                batch: List[Dict[str, str]],
                batch_index: int,
            ) -> List[TranslatedChunk]:
                batch_id = f"retry_r{round_index}_batch_{batch_index}"
                batch_results: List[TranslatedChunk] = []

                try:
                    async with semaphore:
                        batch_results = await self.translate_batch(
                            chunks=batch,
                            context=context,
                        )
                except BaseException as e:
                    if isinstance(e, (KeyboardInterrupt, SystemExit)):
                        raise
                    batch_results = []

                if not batch_results:
                    fallback_results = await asyncio.gather(
                        *[_translate_chunk_retry(chunk_data) for chunk_data in batch],
                        return_exceptions=True,
                    )
                    for i, fallback_result in enumerate(fallback_results):
                        if isinstance(fallback_result, BaseException):
                            if isinstance(fallback_result, (KeyboardInterrupt, SystemExit)):
                                raise fallback_result
                            batch_results.append(
                                _make_skipped_chunk(batch[i], fallback_result)
                            )
                        else:
                            batch_results.append(cast(TranslatedChunk, fallback_result))

                for chunk_result in batch_results:
                    chunk_result.metadata["batch_id"] = batch_id
                    chunk_result.metadata["retry_round"] = round_index
                    _attach_local_cache_metadata(chunk_result)

                return batch_results

            batch_tasks = [
                _translate_batch_retry(batch, i) for i, batch in enumerate(retry_batches)
            ]
            chunk_tasks = [
                _translate_chunk_retry(chunk_data) for chunk_data in retry_long_chunks
            ]

            combined_results = await asyncio.gather(
                *(batch_tasks + chunk_tasks),
                return_exceptions=True,
            )
            for item in combined_results:
                if isinstance(item, BaseException):
                    if isinstance(item, (KeyboardInterrupt, SystemExit)):
                        raise item
                    continue
                if isinstance(item, list):
                    for chunk_result in item:
                        _attach_local_cache_metadata(chunk_result)
                        round_results[chunk_result.chunk_id] = chunk_result
                else:
                    chunk_result = cast(TranslatedChunk, item)
                    _attach_local_cache_metadata(chunk_result)
                    round_results[chunk_result.chunk_id] = chunk_result

            return round_results

        async def _apply_placeholder_retry_rounds() -> Set[str]:
            if not translatable_ids:
                return set()

            attempts: Dict[str, int] = {chunk_id: 1 for chunk_id in translatable_ids}
            latest_audits: Dict[str, Dict[str, Any]] = {}

            failed_ids: Set[str] = set()
            for chunk_id in translatable_ids:
                audit = self._audit_placeholder_alignment(
                    translatable_chunk_map[chunk_id]["content"],
                    results_map[chunk_id].translation,
                )
                latest_audits[chunk_id] = audit
                if not audit["passed"]:
                    failed_ids.add(chunk_id)

            print(
                f"round 1 placeholder audit: failed {len(failed_ids)} / total {len(translatable_ids)}"
            )

            for round_index in range(2, self.PLACEHOLDER_RETRY_MAX_ATTEMPTS + 1):
                if not failed_ids:
                    break

                retry_inputs = [translatable_chunk_map[cid] for cid in translatable_ids if cid in failed_ids]
                retry_results = await _translate_retry_round(retry_inputs, round_index)
                for chunk_id, retry_chunk in retry_results.items():
                    results_map[chunk_id] = retry_chunk
                    attempts[chunk_id] = round_index

                next_failed: Set[str] = set()
                for chunk_id in failed_ids:
                    audit = self._audit_placeholder_alignment(
                        translatable_chunk_map[chunk_id]["content"],
                        results_map[chunk_id].translation,
                    )
                    latest_audits[chunk_id] = audit
                    if not audit["passed"]:
                        next_failed.add(chunk_id)

                failed_ids = next_failed
                print(
                    f"round {round_index} placeholder audit: failed {len(failed_ids)} / total {len(translatable_ids)}"
                )

            exhausted_ids = set(failed_ids)
            for chunk_id in translatable_ids:
                audit = latest_audits.get(chunk_id) or self._audit_placeholder_alignment(
                    translatable_chunk_map[chunk_id]["content"],
                    results_map[chunk_id].translation,
                )
                is_exhausted = chunk_id in exhausted_ids
                results_map[chunk_id].metadata.update(
                    {
                        "placeholder_attempt": attempts.get(chunk_id, 1),
                        "placeholder_audit_passed": bool(audit["passed"]),
                        "placeholder_missing": list(audit["missing"]),
                        "placeholder_spurious": list(audit["spurious"]),
                        "placeholder_retry_exhausted": is_exhausted,
                        "placeholder_warning_emitted": is_exhausted,
                    }
                )

            if exhausted_ids:
                exhausted_prefixes = ", ".join(sorted(chunk_id[:8] for chunk_id in exhausted_ids))
                print(
                    f"[WARNING] placeholder retry exhausted: {len(exhausted_ids)} chunks ({exhausted_prefixes})"
                )

            return exhausted_ids

        def _refresh_state_from_results_map() -> None:
            state["completed"] = [
                chunk_data["chunk_id"]
                for chunk_data in chunks
                if chunk_data["chunk_id"] in results_map
            ]
            state["results"] = [
                results_map[chunk_data["chunk_id"]].model_dump()
                for chunk_data in chunks
                if chunk_data["chunk_id"] in results_map
            ]

        # --- SEQUENTIAL EXECUTION PATH ---
        if self.sequential_mode:
            completed_count = local_cache_hit_count
            total_pending = len(all_translatable_chunks)
            if progress_callback and local_cache_hit_count > 0:
                try:
                    progress_callback(completed_count, total_pending)
                except Exception:
                    pass

            for i, batch in enumerate(batches):
                batch_id = f"batch_{i}"
                try:
                    batch_results = await self.translate_batch(
                        chunks=batch,
                        context=context,
                    )
                    if not batch_results:
                        batch_results = []
                        for chunk_data in batch:
                            result = await self.translate_chunk(
                                chunk=chunk_data["content"],
                                chunk_id=chunk_data["chunk_id"],
                                context=context,
                            )
                            batch_results.append(result)
                except Exception:
                    batch_results = []
                    for chunk_data in batch:
                        result = await self.translate_chunk(
                            chunk=chunk_data["content"],
                            chunk_id=chunk_data["chunk_id"],
                            context=context,
                        )
                        batch_results.append(result)

                for r in batch_results:
                    r.metadata["batch_id"] = batch_id
                    _attach_local_cache_metadata(r)
                    results_map[r.chunk_id] = r
                    state["completed"].append(r.chunk_id)
                    state["results"].append(r.model_dump())

                completed_count += len(batch)
                if progress_callback:
                    try:
                        progress_callback(completed_count, total_pending)
                    except Exception:
                        pass
                self._save_state(state, total_chunks=len(chunks))

            for chunk_data in long_chunks:
                result = await self.translate_chunk(
                    chunk=chunk_data["content"],
                    chunk_id=chunk_data["chunk_id"],
                    context=context,
                )
                _attach_local_cache_metadata(result)
                results_map[result.chunk_id] = result
                state["completed"].append(result.chunk_id)
                state["results"].append(result.model_dump())

                completed_count += 1
                if progress_callback:
                    try:
                        progress_callback(completed_count, total_pending)
                    except Exception:
                        pass
                self._save_state(state, total_chunks=len(chunks))

            await _apply_placeholder_retry_rounds()
            _refresh_state_from_results_map()

            finished_at = datetime.now().isoformat()
            started_dt = datetime.fromisoformat(self._started_at)
            finished_dt = datetime.fromisoformat(finished_at)
            state["_finished_at"] = finished_at
            state["_total_seconds"] = round(
                (finished_dt - started_dt).total_seconds(), 2
            )
            self._save_state(state, total_chunks=len(chunks))

            return [results_map[chunk_data["chunk_id"]] for chunk_data in chunks]
        # --- END SEQUENTIAL PATH ---

        # --- CONCURRENT EXECUTION PATH (existing, unchanged) ---
        semaphore = asyncio.Semaphore(max_concurrent)
        completed_count = local_cache_hit_count
        progress_lock = asyncio.Lock()
        total_pending = len(all_translatable_chunks)
        if progress_callback and local_cache_hit_count > 0:
            try:
                progress_callback(completed_count, total_pending)
            except Exception:
                pass

        def _is_fatal_base_exception(error: BaseException) -> bool:
            return isinstance(error, (KeyboardInterrupt, SystemExit))

        def _is_recoverable_task_error(result: Any) -> bool:
            return isinstance(result, BaseException) and not _is_fatal_base_exception(
                result
            )

        async def translate_batch_with_semaphore(
            batch_chunks: List[Dict[str, str]],
            batch_index: int,
        ) -> List[TranslatedChunk]:
            nonlocal completed_count
            batch_id = f"batch_{batch_index}"
            batch_results: List[TranslatedChunk] = []

            async with semaphore:
                try:
                    batch_results = await self.translate_batch(
                        chunks=batch_chunks,
                        context=context,
                    )
                except BaseException as e:
                    if _is_fatal_base_exception(e):
                        raise
                    batch_results = []

            # Exit semaphore block before fallback to allow other tasks to proceed
            if not batch_results:
                # Fallback: translate each chunk individually, each acquiring semaphore
                fallback_tasks = [
                    translate_with_semaphore(chunk_data) for chunk_data in batch_chunks
                ]
                fallback_results = await asyncio.gather(
                    *fallback_tasks, return_exceptions=True
                )
                # Convert exceptions to skipped chunks with original text
                for i, result in enumerate(fallback_results):
                    if _is_recoverable_task_error(result):
                        chunk_data = batch_chunks[i]
                        skipped_chunk = TranslatedChunk(
                            source=chunk_data["content"],
                            translation=chunk_data[
                                "content"
                            ],  # Preserve original English
                            chunk_id=chunk_data["chunk_id"],
                            metadata={
                                "skipped": True,
                                "skip_reason": str(result),
                                "skipped_at": datetime.now().isoformat(),
                                "batch_id": batch_id,
                            },
                        )
                        _attach_local_cache_metadata(skipped_chunk)
                        batch_results.append(skipped_chunk)
                    elif isinstance(result, BaseException):
                        raise result
                    else:
                        chunk_result = cast(TranslatedChunk, result)
                        _attach_local_cache_metadata(chunk_result)
                        batch_results.append(chunk_result)

            # Mark all chunks with batch_id
            for r in batch_results:
                if "batch_id" not in r.metadata or not r.metadata["batch_id"]:
                    r.metadata["batch_id"] = batch_id
                _attach_local_cache_metadata(r)

            async with progress_lock:
                completed_count += len(batch_chunks)
                if progress_callback:
                    try:
                        progress_callback(completed_count, total_pending)
                    except Exception:
                        pass

            return batch_results

        async def translate_with_semaphore(
            chunk_data: Dict[str, str],
        ) -> TranslatedChunk:
            try:
                async with semaphore:
                    result = await self.translate_chunk(
                        chunk=chunk_data["content"],
                        chunk_id=chunk_data["chunk_id"],
                        context=context,
                    )
                    _attach_local_cache_metadata(result)

                    nonlocal completed_count
                    async with progress_lock:
                        completed_count += 1
                        if progress_callback:
                            try:
                                progress_callback(completed_count, total_pending)
                            except Exception:
                                pass

                    return result
            except BaseException as e:
                if _is_fatal_base_exception(e):
                    raise
                # Return skipped chunk with original text on any error
                skipped_chunk = TranslatedChunk(
                    source=chunk_data["content"],
                    translation=chunk_data["content"],
                    chunk_id=chunk_data["chunk_id"],
                    metadata={
                        "skipped": True,
                        "skip_reason": str(e),
                        "skipped_at": datetime.now().isoformat(),
                    },
                )
                _attach_local_cache_metadata(skipped_chunk)
                async with progress_lock:
                    completed_count += 1
                    if progress_callback:
                        try:
                            progress_callback(completed_count, total_pending)
                        except Exception:
                            pass
                return skipped_chunk

        batch_task_factories = [
            (lambda b=batch, idx=i: translate_batch_with_semaphore(b, idx))
            for i, batch in enumerate(batches)
        ]
        long_task_factories = [
            (lambda chunk=c: translate_with_semaphore(chunk)) for c in long_chunks
        ]
        all_task_factories = batch_task_factories + long_task_factories

        # Cache warmup: send the first request alone to establish prefix cache
        # on the server side, then concurrently send remaining requests.
        if all_task_factories:
            try:
                first_result = await all_task_factories[0]()
            except BaseException as e:
                if _is_fatal_base_exception(e):
                    raise
                first_result = e

            if len(all_task_factories) > 1:
                remaining_tasks = [
                    task_factory() for task_factory in all_task_factories[1:]
                ]
                remaining_results = await asyncio.gather(
                    *remaining_tasks, return_exceptions=True
                )
                results = [first_result] + list(remaining_results)
            else:
                results = [first_result]
        else:
            results = []

        skipped_count = 0
        for i, result in enumerate(results):
            if _is_recoverable_task_error(result):
                # Convert exception to graceful skip with original text
                if i < len(batches):
                    batch = batches[i]
                    for chunk_data in batch:
                        skipped_chunk = TranslatedChunk(
                            source=chunk_data["content"],
                            translation=chunk_data["content"],
                            chunk_id=chunk_data["chunk_id"],
                            metadata={
                                "skipped": True,
                                "skip_reason": str(result),
                                "skipped_at": datetime.now().isoformat(),
                            },
                        )
                        _attach_local_cache_metadata(skipped_chunk)
                        results_map[chunk_data["chunk_id"]] = skipped_chunk
                        state["completed"].append(chunk_data["chunk_id"])
                        state["results"].append(skipped_chunk.model_dump())
                        skipped_count += 1
                else:
                    chunk_idx = i - len(batches)
                    chunk_data = long_chunks[chunk_idx]
                    skipped_chunk = TranslatedChunk(
                        source=chunk_data["content"],
                        translation=chunk_data["content"],
                        chunk_id=chunk_data["chunk_id"],
                        metadata={
                            "skipped": True,
                            "skip_reason": str(result),
                            "skipped_at": datetime.now().isoformat(),
                        },
                    )
                    _attach_local_cache_metadata(skipped_chunk)
                    results_map[chunk_data["chunk_id"]] = skipped_chunk
                    state["completed"].append(chunk_data["chunk_id"])
                    state["results"].append(skipped_chunk.model_dump())
                    skipped_count += 1
                continue
            if isinstance(result, BaseException):
                raise result

            if i < len(batches):
                batch_results = cast(List[TranslatedChunk], result)
                for translated_chunk in batch_results:
                    _attach_local_cache_metadata(translated_chunk)
                    results_map[translated_chunk.chunk_id] = translated_chunk
                    state["completed"].append(translated_chunk.chunk_id)
                    state["results"].append(translated_chunk.model_dump())
            else:
                success_result = cast(TranslatedChunk, result)
                _attach_local_cache_metadata(success_result)
                results_map[success_result.chunk_id] = success_result
                state["completed"].append(success_result.chunk_id)
                state["results"].append(success_result.model_dump())

        # Log warning if any chunks were skipped
        if skipped_count > 0:
            print(
                f"[WARNING] {skipped_count} chunk(s) failed to translate and were skipped (original text preserved)."
            )

        await _apply_placeholder_retry_rounds()
        _refresh_state_from_results_map()

        finished_at = datetime.now().isoformat()
        started_dt = datetime.fromisoformat(self._started_at)
        finished_dt = datetime.fromisoformat(finished_at)
        state["_finished_at"] = finished_at
        state["_total_seconds"] = round((finished_dt - started_dt).total_seconds(), 2)

        self._save_state(state, total_chunks=len(chunks))

        return [results_map[chunk_data["chunk_id"]] for chunk_data in chunks]

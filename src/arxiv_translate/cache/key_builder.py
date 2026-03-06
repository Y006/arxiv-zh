from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from arxiv_translate.translator.prompts import DEFAULT_STYLE_PROMPT, FORMAT_RULES


class CacheKeyBuilder:
    """Build stable cache keys for chunk-level local translation cache."""

    KEY_VERSION = "local-translation-cache-v1"

    def __init__(self, key_mode: str = "relaxed_chunk"):
        self.key_mode = key_mode

    @staticmethod
    def _normalize_newlines(text: str) -> str:
        return text.replace("\r\n", "\n").replace("\r", "\n")

    def _normalize_text(self, text: Optional[str], *, strip: bool = True) -> str:
        normalized = self._normalize_newlines(text or "")
        return normalized.strip() if strip else normalized

    def _normalize_glossary(
        self, glossary_hints: Optional[Dict[str, str]]
    ) -> List[List[str]]:
        if not glossary_hints:
            return []
        return [
            [self._normalize_text(term), self._normalize_text(target)]
            for term, target in sorted(glossary_hints.items(), key=lambda item: item[0])
        ]

    def _normalize_examples(
        self, few_shot_examples: Optional[List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        for example in few_shot_examples or []:
            source = self._normalize_text(example.get("source", ""), strip=False)
            target = self._normalize_text(example.get("target", ""), strip=False)
            if source or target:
                normalized.append({"source": source, "target": target})
        return normalized

    def build_payload(
        self,
        *,
        source_text: str,
        prompt_variant_semantic: str,
        glossary_hints: Optional[Dict[str, str]] = None,
        context: Optional[str] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        custom_system_prompt: Optional[str] = None,
        key_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        style_prompt = self._normalize_text(custom_system_prompt) or self._normalize_text(
            DEFAULT_STYLE_PROMPT
        )
        return {
            "key_version": self.KEY_VERSION,
            "key_mode": key_mode or self.key_mode,
            "prompt_variant_semantic": prompt_variant_semantic,
            "style_prompt": style_prompt,
            "format_rules": self._normalize_text(FORMAT_RULES),
            "context": self._normalize_text(context),
            "glossary": self._normalize_glossary(glossary_hints),
            "few_shot_examples": self._normalize_examples(few_shot_examples),
            "source_text": self._normalize_text(source_text, strip=False),
        }

    @staticmethod
    def canonical_json(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def hash_payload(self, payload: Dict[str, Any]) -> bytes:
        canonical = self.canonical_json(payload).encode("utf-8")
        return hashlib.blake2b(canonical, digest_size=16).digest()

    def hash_payload_hex(self, payload: Dict[str, Any]) -> str:
        return self.hash_payload(payload).hex()

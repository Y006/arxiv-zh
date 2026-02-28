"""Volcano Engine Ark provider with Responses API prefix caching."""

import asyncio
import json
import time
from types import SimpleNamespace
from typing import Optional, Dict, List, Any, Set, Tuple

import httpx

from .llm_base import LLMProvider
from .prompts import build_system_prompt

# Legacy compatibility flag. Ark provider is HTTP-native and always available.
HAS_ARK = True


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


class _ArkResponsesAPI:
    def __init__(self, client: "_ArkHTTPClient"):
        self._client = client

    async def create(
        self,
        *,
        model: str,
        input: List[Dict[str, Any]],
        previous_response_id: Optional[str] = None,
        caching: Optional[Dict[str, Any]] = None,
        store: Optional[bool] = None,
        expire_at: Optional[int] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Any:
        body: Dict[str, Any] = {
            "model": model,
            "input": input,
        }
        if previous_response_id:
            body["previous_response_id"] = previous_response_id
        if caching is not None:
            body["caching"] = caching
        if store is not None:
            body["store"] = store
        if expire_at is not None:
            body["expire_at"] = int(expire_at)
        if temperature is not None:
            body["temperature"] = temperature
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens
        return await self._client._post("/responses", body)


class _ArkChatCompletionsAPI:
    def __init__(self, client: "_ArkHTTPClient"):
        self._client = client

    async def create(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        return await self._client._post("/chat/completions", body)


class _ArkChatAPI:
    def __init__(self, client: "_ArkHTTPClient"):
        self.completions = _ArkChatCompletionsAPI(client)


class _ArkHTTPClient:
    def __init__(self, base_url: str, api_key: Optional[str]):
        self._base_url = base_url.rstrip("/")
        self._headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=60.0, read=300.0, write=60.0, pool=60.0)
        )
        self.responses = _ArkResponsesAPI(self)
        self.chat = _ArkChatAPI(self)

    async def _post(self, suffix: str, body: Dict[str, Any]) -> Any:
        url = f"{self._base_url}{suffix}"
        response = await self._http.post(url, json=body, headers=self._headers)
        response.raise_for_status()
        data = response.json()
        return _to_namespace(data)


class ArkProvider(LLMProvider):
    """Volcano Engine Ark provider using Responses API prefix caching."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, api_key, **kwargs)
        if not base_url:
            raise ValueError("base_url is required for ArkProvider")

        normalized_base_url = self._normalize_base_url(base_url)
        self.client = _ArkHTTPClient(base_url=normalized_base_url, api_key=api_key)
        self._response_ids: Dict[str, str] = {}
        self._response_prefix_keys: Dict[str, str] = {}
        self._response_setup_locks: Dict[str, asyncio.Lock] = {}
        self._response_fallback_warned_variants: Set[str] = set()
        self._prefix_too_short_warned_variants: Set[str] = set()
        self._prefix_cache_disabled_variants: Set[str] = set()
        self._responses_route_disabled = False
        self._route_unavailable_warned = False
        self._prebuilt_system_prompt: Optional[str] = None
        self._prebuilt_batch_prompt: Optional[str] = None
        self._cache_log_verbose = bool(self.kwargs.get("ark_cache_log_verbose", False))
        self.reset_cache_stats()

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        normalized = base_url.rstrip("/")
        for suffix in ("/responses", "/chat/completions"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break
        return normalized.rstrip("/")

    @staticmethod
    def _get_field(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _ensure_response_state(self) -> None:
        if (
            not hasattr(self, "_response_ids")
            or self._response_ids is None  # type: ignore[attr-defined]
        ):
            self._response_ids = {}
        if (
            not hasattr(self, "_response_prefix_keys")
            or self._response_prefix_keys is None  # type: ignore[attr-defined]
        ):
            self._response_prefix_keys = {}
        if (
            not hasattr(self, "_response_setup_locks")
            or self._response_setup_locks is None  # type: ignore[attr-defined]
        ):
            self._response_setup_locks = {}
        if (
            not hasattr(self, "_response_fallback_warned_variants")
            or self._response_fallback_warned_variants is None  # type: ignore[attr-defined]
        ):
            self._response_fallback_warned_variants = set()
        if (
            not hasattr(self, "_prefix_too_short_warned_variants")
            or self._prefix_too_short_warned_variants is None  # type: ignore[attr-defined]
        ):
            self._prefix_too_short_warned_variants = set()
        if (
            not hasattr(self, "_prefix_cache_disabled_variants")
            or self._prefix_cache_disabled_variants is None  # type: ignore[attr-defined]
        ):
            self._prefix_cache_disabled_variants = set()
        if not hasattr(self, "_responses_route_disabled"):
            self._responses_route_disabled = False
        if not hasattr(self, "_route_unavailable_warned"):
            self._route_unavailable_warned = False

    def _ensure_cache_stats_state(self) -> None:
        if not hasattr(self, "_cache_log_verbose"):
            self._cache_log_verbose = False
        if not hasattr(self, "_cache_stats") or not isinstance(
            self._cache_stats, dict  # type: ignore[attr-defined]
        ):
            self.reset_cache_stats()

    def reset_cache_stats(self) -> None:
        self._cache_stats = {
            "request_count": 0,
            "cache_hit_count": 0,
            "cache_miss_count": 0,
            "cached_tokens_total": 0,
            "prompt_tokens_total": 0,
            "completion_tokens_total": 0,
            "total_tokens_total": 0,
            "missing_usage_count": 0,
            "mode_counts": {"responses": 0, "chat": 0},
            "variant_counts": {},
        }

    @staticmethod
    def _stat_int(value: Any) -> int:
        try:
            if isinstance(value, bool):
                return int(value)
            if isinstance(value, (int, float, str)):
                return int(value)
        except (TypeError, ValueError):
            pass
        return 0

    def _record_cache_meta(self, cache_meta: Optional[Dict[str, Any]]) -> None:
        self._ensure_cache_stats_state()
        if cache_meta is None:
            self._cache_stats["missing_usage_count"] += 1
            return

        self._cache_stats["request_count"] += 1
        if bool(cache_meta.get("cache_hit")):
            self._cache_stats["cache_hit_count"] += 1
        else:
            self._cache_stats["cache_miss_count"] += 1

        self._cache_stats["cached_tokens_total"] += self._stat_int(
            cache_meta.get("cached_tokens", 0)
        )
        self._cache_stats["prompt_tokens_total"] += self._stat_int(
            cache_meta.get("prompt_tokens", 0)
        )
        self._cache_stats["completion_tokens_total"] += self._stat_int(
            cache_meta.get("completion_tokens", 0)
        )
        self._cache_stats["total_tokens_total"] += self._stat_int(
            cache_meta.get("total_tokens", 0)
        )

        mode = str(cache_meta.get("mode") or "")
        if mode:
            mode_counts = self._cache_stats.setdefault("mode_counts", {})
            mode_counts[mode] = self._stat_int(mode_counts.get(mode, 0)) + 1

        variant = str(cache_meta.get("variant") or "")
        if variant:
            variant_counts = self._cache_stats.setdefault("variant_counts", {})
            variant_counts[variant] = self._stat_int(variant_counts.get(variant, 0)) + 1

    def get_cache_stats_summary(self) -> Dict[str, Any]:
        self._ensure_cache_stats_state()
        request_count = self._stat_int(self._cache_stats.get("request_count", 0))
        cache_hit_count = self._stat_int(self._cache_stats.get("cache_hit_count", 0))
        cache_miss_count = self._stat_int(self._cache_stats.get("cache_miss_count", 0))
        cache_hit_rate = (
            round((cache_hit_count / request_count) * 100, 1) if request_count else 0.0
        )

        return {
            "provider": "ark",
            "request_count": request_count,
            "cache_hit_count": cache_hit_count,
            "cache_miss_count": cache_miss_count,
            "cache_hit_rate": cache_hit_rate,
            "cached_tokens_total": self._stat_int(
                self._cache_stats.get("cached_tokens_total", 0)
            ),
            "prompt_tokens_total": self._stat_int(
                self._cache_stats.get("prompt_tokens_total", 0)
            ),
            "completion_tokens_total": self._stat_int(
                self._cache_stats.get("completion_tokens_total", 0)
            ),
            "total_tokens_total": self._stat_int(
                self._cache_stats.get("total_tokens_total", 0)
            ),
            "missing_usage_count": self._stat_int(
                self._cache_stats.get("missing_usage_count", 0)
            ),
            "mode_counts": dict(self._cache_stats.get("mode_counts", {})),
            "variant_counts": dict(self._cache_stats.get("variant_counts", {})),
        }

    def format_cache_stats_summary(self) -> List[str]:
        summary = self.get_cache_stats_summary()
        request_count = self._stat_int(summary.get("request_count", 0))
        missing_usage_count = self._stat_int(summary.get("missing_usage_count", 0))
        if request_count <= 0 and missing_usage_count <= 0:
            return []

        lines = [
            "[ARK CACHE SUMMARY] "
            f"requests={summary['request_count']} "
            f"hit={summary['cache_hit_count']} "
            f"miss={summary['cache_miss_count']} "
            f"hit_rate={summary['cache_hit_rate']}% "
            f"cached_tokens={summary['cached_tokens_total']}",
            "[ARK CACHE TOKENS] "
            f"prompt={summary['prompt_tokens_total']} "
            f"completion={summary['completion_tokens_total']} "
            f"total={summary['total_tokens_total']}",
        ]

        mode_counts = summary.get("mode_counts", {})
        if isinstance(mode_counts, dict) and mode_counts:
            mode_parts = " ".join(
                f"{k}={self._stat_int(v)}" for k, v in sorted(mode_counts.items())
            )
            lines.append(f"[ARK CACHE MODES] {mode_parts}")

        variant_counts = summary.get("variant_counts", {})
        if isinstance(variant_counts, dict) and variant_counts:
            variant_parts = " ".join(
                f"{k}={self._stat_int(v)}" for k, v in sorted(variant_counts.items())
            )
            lines.append(f"[ARK CACHE VARIANTS] {variant_parts}")

        if missing_usage_count > 0:
            lines.append(f"[ARK CACHE NOTE] missing_usage={missing_usage_count}")

        return lines

    def _get_response_id_for_variant(self, prompt_variant: str) -> Optional[str]:
        self._ensure_response_state()
        return self._response_ids.get(prompt_variant)

    def _set_response_id_for_variant(
        self,
        prompt_variant: str,
        response_id: str,
        prefix_key: Optional[str] = None,
    ) -> None:
        self._ensure_response_state()
        self._response_ids[prompt_variant] = response_id
        if prefix_key is not None:
            self._response_prefix_keys[prompt_variant] = prefix_key

    def _clear_response_id_for_variant(self, prompt_variant: str) -> None:
        self._ensure_response_state()
        self._response_ids.pop(prompt_variant, None)
        self._response_prefix_keys.pop(prompt_variant, None)

    def _get_response_lock(self, prompt_variant: str) -> asyncio.Lock:
        self._ensure_response_state()
        lock = self._response_setup_locks.get(prompt_variant)
        if lock is None:
            lock = asyncio.Lock()
            self._response_setup_locks[prompt_variant] = lock
        return lock

    @staticmethod
    def _build_few_shot_messages(
        few_shot_examples: Optional[List[Dict[str, str]]],
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if few_shot_examples:
            for example in few_shot_examples:
                messages.append({"role": "user", "content": example.get("source", "")})
                messages.append(
                    {"role": "assistant", "content": example.get("target", "")}
                )
        return messages

    @staticmethod
    def _build_prefix_key(messages: List[Dict[str, str]]) -> str:
        return json.dumps(messages, ensure_ascii=False, sort_keys=True)

    def _build_context_prefix_messages(
        self,
        prompt_variant: str,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        prompt = system_prompt or self._get_prebuilt_prompt(prompt_variant)
        if not prompt:
            return []
        messages: List[Dict[str, str]] = [{"role": "system", "content": prompt}]
        messages.extend(self._build_few_shot_messages(few_shot_examples))
        return messages

    @staticmethod
    def _extract_message_content(response: Any) -> str:
        def _content_to_text(content: Any) -> str:
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                        continue
                    item_type = ArkProvider._get_field(item, "type", "")
                    if item_type in ("output_text", "text"):
                        parts.append(str(ArkProvider._get_field(item, "text", "")))
                        continue
                    text_value = ArkProvider._get_field(item, "text", None)
                    if text_value is not None:
                        parts.append(str(text_value))
                return "".join(parts)
            return str(content)

        output_text = ArkProvider._get_field(response, "output_text", None)
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        response_text = ArkProvider._get_field(response, "text", None)
        if isinstance(response_text, str) and response_text.strip():
            return response_text.strip()

        output_items = ArkProvider._get_field(response, "output", []) or []
        for item in output_items:
            role = ArkProvider._get_field(item, "role", "")
            item_type = ArkProvider._get_field(item, "type", "")
            if role and role != "assistant":
                continue
            if item_type not in ("", "message"):
                continue
            content = ArkProvider._get_field(item, "content", None)
            text = _content_to_text(content)
            if text.strip():
                return text.strip()

        choices = ArkProvider._get_field(response, "choices", []) or []
        if choices:
            message = ArkProvider._get_field(choices[0], "message", {})
            content = ArkProvider._get_field(message, "content", "")
            return _content_to_text(content).strip()

        return ""

    @staticmethod
    def _extract_error_payload(error: Any) -> Tuple[Optional[int], Dict[str, Any], str]:
        status_code: Optional[int] = None
        payload: Dict[str, Any] = {}
        if isinstance(error, httpx.HTTPStatusError) and error.response is not None:
            status_code = error.response.status_code
            try:
                decoded = error.response.json()
                if isinstance(decoded, dict):
                    payload = decoded
            except Exception:
                payload = {}
        message = " ".join(str(error).split())
        return status_code, payload, message.lower()

    @staticmethod
    def _is_invalid_previous_response_id_error(error: Any) -> bool:
        _, payload, msg = ArkProvider._extract_error_payload(error)
        err_obj = payload.get("error", {}) if isinstance(payload, dict) else {}
        if isinstance(err_obj, dict):
            param = str(err_obj.get("param", "")).lower()
            if param == "previous_response_id":
                return True
            err_msg = str(err_obj.get("message", "")).lower()
            if "previous_response_id" in err_msg and (
                "invalid" in err_msg or "expired" in err_msg or "not found" in err_msg
            ):
                return True
        return "previous_response_id" in msg and (
            "invalid" in msg or "expired" in msg or "not found" in msg
        )

    @staticmethod
    def _is_responses_route_unavailable_error(error: Any) -> bool:
        status_code, payload, msg = ArkProvider._extract_error_payload(error)
        err_obj = payload.get("error", {}) if isinstance(payload, dict) else {}
        err_param = ""
        err_msg = ""
        if isinstance(err_obj, dict):
            err_param = str(err_obj.get("param", "")).lower()
            err_msg = str(err_obj.get("message", "")).lower()
        if status_code == 404 and (
            "responses" in msg or "responses" in err_msg or "route" in msg
        ):
            return True
        return "/responses" in msg and (
            "not found" in msg or "unsupported" in msg or "not support" in msg
        ) or ("responses" in err_param and "not found" in err_msg)

    @staticmethod
    def _is_prefix_cache_token_too_short_error(error: Any) -> bool:
        _, payload, msg = ArkProvider._extract_error_payload(error)
        err_obj = payload.get("error", {}) if isinstance(payload, dict) else {}
        err_msg = ""
        if isinstance(err_obj, dict):
            err_msg = str(err_obj.get("message", "")).lower()
        merged = f"{msg} {err_msg}"
        if (
            "prefix" in merged
            and "cache" in merged
            and "256" in merged
            and ("token" in merged or "tokens" in merged)
        ):
            return True
        return (
            "input" in merged
            and "token" in merged
            and "greater than 256" in merged
        )

    def _extract_cache_meta(
        self,
        response: Any,
        mode: str,
        prompt_variant: str,
        response_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        def _to_int(value: Any) -> int:
            try:
                if isinstance(value, bool):
                    return int(value)
                if isinstance(value, (int, float, str)):
                    return int(value)
            except (TypeError, ValueError):
                pass
            return 0

        usage = self._get_field(response, "usage", None)
        if usage is None:
            return None

        prompt_tokens = _to_int(
            self._get_field(
                usage,
                "prompt_tokens",
                self._get_field(usage, "input_tokens", 0),
            )
        )
        completion_tokens = _to_int(
            self._get_field(
                usage,
                "completion_tokens",
                self._get_field(usage, "output_tokens", 0),
            )
        )
        total_tokens = _to_int(self._get_field(usage, "total_tokens", 0))

        prompt_details = self._get_field(
            usage,
            "prompt_tokens_details",
            self._get_field(usage, "input_tokens_details", None),
        )
        cached_tokens = _to_int(self._get_field(prompt_details, "cached_tokens", 0))

        return {
            "provider": "ark",
            "mode": mode,
            "variant": prompt_variant,
            "response_id": response_id,
            "cache_hit": cached_tokens > 0,
            "cached_tokens": cached_tokens,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _print_cache_meta(cache_meta: Dict[str, Any]) -> None:
        print(
            "[ARK CACHE] "
            f"variant={cache_meta.get('variant')} "
            f"mode={cache_meta.get('mode')} "
            f"hit={cache_meta.get('cache_hit')} "
            f"cached_tokens={cache_meta.get('cached_tokens')} "
            f"prompt_tokens={cache_meta.get('prompt_tokens')} "
            f"completion_tokens={cache_meta.get('completion_tokens')} "
            f"total_tokens={cache_meta.get('total_tokens')} "
            f"response_id={cache_meta.get('response_id')}"
        )

    def _warn_response_fallback_once(self, prompt_variant: str, reason: Any) -> None:
        self._ensure_response_state()
        if prompt_variant in self._response_fallback_warned_variants:
            return
        self._response_fallback_warned_variants.add(prompt_variant)
        reason_text = " ".join(str(reason).split())
        print(
            "[ARK CACHE] "
            f"variant={prompt_variant} mode=chat fallback=True reason={reason_text}"
        )

    def _warn_responses_unavailable_once(self, reason: Any) -> None:
        self._ensure_response_state()
        if self._route_unavailable_warned:
            return
        self._route_unavailable_warned = True
        reason_text = " ".join(str(reason).split())
        print(
            "[ARK CACHE] "
            f"mode=chat responses_unavailable=True reason={reason_text}"
        )

    def _warn_prefix_too_short_once(self, prompt_variant: str, reason: Any) -> None:
        self._ensure_response_state()
        if prompt_variant in self._prefix_too_short_warned_variants:
            return
        self._prefix_too_short_warned_variants.add(prompt_variant)
        reason_text = " ".join(str(reason).split())
        print(
            "[ARK CACHE] "
            f"variant={prompt_variant} mode=responses prefix_too_short=True reason={reason_text}"
        )

    def _build_chat_messages(
        self,
        *,
        text: str,
        context: Optional[str],
        glossary_hints: Optional[Dict[str, str]],
        few_shot_examples: Optional[List[Dict[str, str]]],
        custom_system_prompt: Optional[str],
        prompt_variant: str,
    ) -> List[Dict[str, Any]]:
        few_shot_messages = self._build_few_shot_messages(few_shot_examples)
        selected_prebuilt_prompt = self._get_prebuilt_prompt(prompt_variant)

        if selected_prebuilt_prompt is not None and glossary_hints is None:
            system_content = selected_prebuilt_prompt
        else:
            system_content = build_system_prompt(
                glossary_hints=glossary_hints,
                context=context,
                few_shot_examples=few_shot_examples,
                custom_system_prompt=custom_system_prompt,
            )

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_content}]
        messages.extend(few_shot_messages)
        messages.append({"role": "user", "content": text})
        return messages

    async def setup_context(
        self,
        system_prompt: Optional[str] = None,
        *,
        prompt_variant: str = "individual",
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        """Warm responses prefix cache with stable prefix (system + few-shot)."""
        self._ensure_response_state()
        if self._responses_route_disabled:
            return

        prefix_messages = self._build_context_prefix_messages(
            prompt_variant=prompt_variant,
            few_shot_examples=few_shot_examples,
            system_prompt=system_prompt,
        )
        if not prefix_messages:
            return
        if prompt_variant in self._prefix_cache_disabled_variants:
            return

        prefix_key = self._build_prefix_key(prefix_messages)
        async with self._get_response_lock(prompt_variant):
            current_response_id = self._response_ids.get(prompt_variant)
            current_prefix_key = self._response_prefix_keys.get(prompt_variant)
            if current_response_id and current_prefix_key == prefix_key:
                return

            try:
                response = await self.client.responses.create(
                    model=self.model,
                    input=prefix_messages,
                    caching={"type": "enabled", "prefix": True},
                    store=True,
                    expire_at=int(time.time()) + 3600,
                )
            except Exception as e:
                if self._is_prefix_cache_token_too_short_error(e):
                    self._prefix_cache_disabled_variants.add(prompt_variant)
                    self._warn_prefix_too_short_once(prompt_variant, e)
                    return
                if self._is_responses_route_unavailable_error(e):
                    self._responses_route_disabled = True
                    self._warn_responses_unavailable_once(e)
                    return
                raise

            response_id = self._get_field(response, "id")
            if not response_id:
                raise RuntimeError("Ark responses.create warmup missing response id")
            self._set_response_id_for_variant(
                prompt_variant=prompt_variant,
                response_id=str(response_id),
                prefix_key=prefix_key,
            )

    async def prepare_prompt_cache_variants(
        self,
        prompt_variants: List[str],
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        for prompt_variant in prompt_variants:
            try:
                await self.setup_context(
                    prompt_variant=prompt_variant,
                    few_shot_examples=few_shot_examples,
                )
            except Exception as e:
                print(
                    "[ARK CACHE] "
                    f"variant={prompt_variant} "
                    "mode=chat "
                    f"warmup_failed=True reason={e}"
                )

    async def _call_chat_completions(
        self,
        *,
        text: str,
        context: Optional[str],
        glossary_hints: Optional[Dict[str, str]],
        few_shot_examples: Optional[List[Dict[str, str]]],
        custom_system_prompt: Optional[str],
        prompt_variant: str,
    ) -> Any:
        chat_messages = self._build_chat_messages(
            text=text,
            context=context,
            glossary_hints=glossary_hints,
            few_shot_examples=few_shot_examples,
            custom_system_prompt=custom_system_prompt,
            prompt_variant=prompt_variant,
        )
        return await self.client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            temperature=self.kwargs.get("temperature", 0.3),
        )

    async def translate(
        self,
        text: str,
        context: Optional[str] = None,
        glossary_hints: Optional[Dict[str, str]] = None,
        few_shot_examples: Optional[List[Dict[str, str]]] = None,
        custom_system_prompt: Optional[str] = None,
        prompt_variant: str = "individual",
    ) -> str:
        self._ensure_response_state()
        selected_prebuilt_prompt = self._get_prebuilt_prompt(prompt_variant)
        can_use_prev_response = (
            selected_prebuilt_prompt is not None
            and glossary_hints is None
            and prompt_variant not in self._prefix_cache_disabled_variants
        )

        used_mode = "chat"
        active_response_id: Optional[str] = None

        try:
            response: Any
            if self._responses_route_disabled:
                response = await self._call_chat_completions(
                    text=text,
                    context=context,
                    glossary_hints=glossary_hints,
                    few_shot_examples=few_shot_examples,
                    custom_system_prompt=custom_system_prompt,
                    prompt_variant=prompt_variant,
                )
            else:
                responses_input = self._build_chat_messages(
                    text=text,
                    context=context,
                    glossary_hints=glossary_hints,
                    few_shot_examples=few_shot_examples,
                    custom_system_prompt=custom_system_prompt,
                    prompt_variant=prompt_variant,
                )

                request_kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "input": responses_input,
                    "temperature": self.kwargs.get("temperature", 0.3),
                }
                previous_response_id = (
                    self._get_response_id_for_variant(prompt_variant)
                    if can_use_prev_response
                    else None
                )
                if previous_response_id:
                    request_kwargs["previous_response_id"] = previous_response_id

                try:
                    response = await self.client.responses.create(**request_kwargs)
                    used_mode = "responses"
                except Exception as first_error:
                    recovered = False
                    if (
                        request_kwargs.get("previous_response_id")
                        and self._is_invalid_previous_response_id_error(first_error)
                    ):
                        self._clear_response_id_for_variant(prompt_variant)
                        retry_kwargs = dict(request_kwargs)
                        retry_kwargs.pop("previous_response_id", None)
                        try:
                            response = await self.client.responses.create(**retry_kwargs)
                            used_mode = "responses"
                            recovered = True
                        except Exception as second_error:
                            first_error = second_error

                    if not recovered:
                        if self._is_responses_route_unavailable_error(first_error):
                            self._responses_route_disabled = True
                            self._warn_responses_unavailable_once(first_error)
                        else:
                            self._warn_response_fallback_once(
                                prompt_variant, first_error
                            )
                        response = await self._call_chat_completions(
                            text=text,
                            context=context,
                            glossary_hints=glossary_hints,
                            few_shot_examples=few_shot_examples,
                            custom_system_prompt=custom_system_prompt,
                            prompt_variant=prompt_variant,
                        )

            if used_mode == "responses":
                response_id = self._get_field(response, "id")
                if response_id and can_use_prev_response:
                    self._set_response_id_for_variant(
                        prompt_variant=prompt_variant,
                        response_id=str(response_id),
                    )
                active_response_id = str(response_id) if response_id else None

            cache_meta = self._extract_cache_meta(
                response=response,
                mode=used_mode,
                prompt_variant=prompt_variant,
                response_id=active_response_id,
            )
            self._last_cache_meta = cache_meta
            self._record_cache_meta(cache_meta)
            if cache_meta is not None and self._cache_log_verbose:
                self._print_cache_meta(cache_meta)

            return self._extract_message_content(response)
        except Exception as e:
            raise RuntimeError(f"Ark API error: {str(e)}") from e

    async def ping(self) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=10,
        )
        return self._extract_message_content(response)

    def estimate_tokens(self, text: str) -> int:
        # Heuristic: mixed CJK/English ~2.5 chars per token.
        return int(len(text) / 2.5)

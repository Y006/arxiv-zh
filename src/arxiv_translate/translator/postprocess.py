"""Postprocessing helpers for conservative markdown hallucination cleanup."""

from __future__ import annotations

import re
from typing import Any, Dict, Tuple


_BOLD_PATTERN = re.compile(
    r"(?<![\\*])\*\*(?!\*)(.+?)(?<![\\*])\*\*(?!\*)",
    re.DOTALL,
)


def _contains_plain_text(content: str) -> bool:
    return any(ch.isalnum() for ch in content)


def sanitize_markdown_bold_safe(source: str, translation: str) -> Tuple[str, Dict[str, Any]]:
    """Convert safe markdown `**...**` spans to LaTeX `\\textbf{...}`.

    Safety-first policy:
    - If source already contains `**`, do not convert anything.
    - Convert only strict `**...**` pairs (non-escaped, non-star-chain).
    - Skip candidates with LaTeX-sensitive payload/content or command-dense context.
    """

    audit: Dict[str, Any] = {
        "changed": False,
        "converted_count": 0,
        "skipped_count": 0,
        "skipped_reasons": [],
    }

    if "**" not in translation:
        return translation, audit

    if "**" in source:
        audit["skipped_count"] = 1
        audit["skipped_reasons"].append("source_contains_double_asterisk")
        return translation, audit

    rebuilt: list[str] = []
    last_idx = 0
    changed = False

    for match in _BOLD_PATTERN.finditer(translation):
        start, end = match.span()
        payload = match.group(1)

        skip_reason = ""
        payload_stripped = payload.strip()
        if not payload_stripped:
            skip_reason = "empty_payload"
        elif any(token in payload for token in ("\\", "{", "}", "$", "\n")):
            skip_reason = "latex_sensitive_payload"
        elif "[[" in payload or "]]" in payload:
            skip_reason = "placeholder_payload"
        elif not _contains_plain_text(payload_stripped):
            skip_reason = "non_text_payload"
        else:
            context_window = translation[max(0, start - 24) : min(len(translation), end + 24)]
            if re.search(r"\\[A-Za-z]+", context_window):
                skip_reason = "command_dense_context"
            elif "[[" in context_window and "]]" in context_window:
                skip_reason = "placeholder_context"

        if skip_reason:
            audit["skipped_count"] += 1
            audit["skipped_reasons"].append(skip_reason)
            continue

        rebuilt.append(translation[last_idx:start])
        rebuilt.append(r"\textbf{" + payload + "}")
        last_idx = end
        changed = True
        audit["converted_count"] += 1

    if not changed:
        return translation, audit

    rebuilt.append(translation[last_idx:])
    audit["changed"] = True
    return "".join(rebuilt), audit

"""Dream v2 Fast Extraction — Regex-based candidate extraction fallback.

Used when LLM extraction is unavailable.
Simple pattern matching for obvious memory candidates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from .taxonomy import validate_memory_type


@dataclass
class CandidateMemory:
    content: str
    type: str
    tags: List[str]
    importance: float = 0.3


def extract_candidates_from_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fast regex-based extraction from messages.

    Used as fallback when LLM extraction is unavailable.
    Looks for obvious patterns like corrections, preferences, etc.
    """
    # Combine recent messages
    recent = []
    for msg in messages[-10:]:
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        if content:
            recent.append(str(content))

    combined = "\n".join(recent)

    candidates = []

    # Correction patterns
    correction_patterns = [
        r"(?i)don't do (that|this|it again)",
        r"(?i)stop (doing|using)",
        r"(?i)never (do|use|recommend)",
        r"(?i)I told you (to |not to )",
        r"(?i)you (should|shouldn't|need to|don't need to)",
        r"(?i)please (do|don't|stop)",
        r"(?i)the right way is",
        r"(?i)actually,? (I |you )",
        r"(?i)wait,? (I |you )",
    ]

    for pattern in correction_patterns:
        matches = re.finditer(pattern, combined)
        for m in matches:
            start = max(0, m.start() - 100)
            end = min(len(combined), m.end() + 100)
            snippet = combined[start:end].strip()
            if len(snippet) > 20:
                candidates.append(CandidateMemory(
                    content=snippet,
                    type="feedback",
                    tags=["correction", "regex-match"],
                    importance=0.6,
                ))

    # Preference patterns
    pref_patterns = [
        r"(?i)I prefer",
        r"(?i)I like",
        r"(?i)I hate",
        r"(?i)my preference is",
        r"(?i)always (use|do)",
        r"(?i)never (use|do)",
        r"(?i)use (this|that|it) instead",
    ]

    for pattern in pref_patterns:
        matches = re.finditer(pattern, combined)
        for m in matches:
            start = max(0, m.start() - 100)
            end = min(len(combined), m.end() + 100)
            snippet = combined[start:end].strip()
            if len(snippet) > 20:
                candidates.append(CandidateMemory(
                    content=snippet,
                    type="user",
                    tags=["preference", "regex-match"],
                    importance=0.5,
                ))

    # Decision patterns
    decision_patterns = [
        r"(?i)we (decided|agreed|will go with)",
        r"(?i)let's (do|use|go with)",
        r"(?i)the (plan|approach|decision) is",
        r"(?i)ok,? (we|I) will",
    ]

    for pattern in decision_patterns:
        matches = re.finditer(pattern, combined)
        for m in matches:
            start = max(0, m.start() - 100)
            end = min(len(combined), m.end() + 100)
            snippet = combined[start:end].strip()
            if len(snippet) > 20:
                candidates.append(CandidateMemory(
                    content=snippet,
                    type="decisions",
                    tags=["decision", "regex-match"],
                    importance=0.5,
                ))

    # Deduplicate by content (first 100 chars)
    seen = set()
    unique = []
    for c in candidates:
        key = c.content[:100].lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return [
        {"content": c.content, "type": c.type, "tags": c.tags, "importance": c.importance}
        for c in unique[:5]
    ]


def extract_candidates(
    text: str,
    max_memories: int = 3,
) -> List[Dict[str, Any]]:
    """Extract from a single text block."""
    # Simple wrapper for backward compat
    msg = [{"role": "user", "content": text}]
    return extract_candidates_from_messages(msg)[:max_memories]


def build_pre_compress_summary(messages: List[Dict[str, Any]]) -> str:
    """Build a summary of recent messages for pre-compress rescue.

    Returns a short text summary of the conversation so far.
    """
    lines = []
    for msg in messages[-5:]:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        if content and role in ("user", "assistant"):
            lines.append(f"[{role}]: {content[:200]}")

    return "\n".join(lines)
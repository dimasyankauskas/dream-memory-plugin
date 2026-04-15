"""Dream Memory Candidate Extraction — per-turn regex-based extraction.

Extracts candidate memories from conversation turns using fast regex patterns.
No LLM calls are used here; LLM consolidation is a Phase 4 feature.

CandidateMemory is a lightweight dataclass that captures:
  - type: one of user, feedback, project, reference
  - content: the extracted fact string
  - tags: auto-generated tags from the pattern category
  - relevance: heuristic confidence score (0.0–1.0)

Callers decide whether to persist candidates (e.g. relevance threshold filtering).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CandidateMemory:
    """A candidate memory extracted from a conversation turn."""

    type: str          # user | feedback | project | reference
    content: str       # extracted fact string
    tags: List[str] = field(default_factory=list)
    relevance: float = 0.5  # heuristic confidence 0.0–1.0
    importance: float = 0.5  # initial importance = extraction confidence


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

# Each pattern group maps to a CandidateMemory type, a relevance score,
# and a tag to attach.  The named group ``match`` captures the interesting
# portion, but the FULL sentence containing the match becomes the content
# (truncated to a sensible length).

_PATTERN_DEFS = {
    "user": {
        "patterns": [
            # "I prefer...", "I always...", "I never...", "my favorite..."
            re.compile(
                r"(?P<sentence>[^.!?\n]*\bI\s+prefer\b[^.!?\n]*(?:[.!?]|$))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?P<sentence>[^.!?\n]*\bI\s+always\b[^.!?\n]*(?:[.!?]|$))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?P<sentence>[^.!?\n]*\bI\s+never\b[^.!?\n]*(?:[.!?]|$))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?P<sentence>[^.!?\n]*\bmy\s+favorite\b[^.!?\n]*(?:[.!?]|$))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?P<sentence>[^.!?\n]*\bI\s+(?:like|love|use|want|need)\s+[^.!?\n]*(?:[.!?]|$))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?P<sentence>[^.!?\n]*\bmy\s+(?:preferred|default)\s+\w+\s+is\s+[^.!?\n]*(?:[.!?]|$))",
                re.IGNORECASE,
            ),
        ],
        "tag": "preference",
        "relevance": 0.7,
        "importance": 0.7,
    },
    "feedback": {
        "patterns": [
            # Corrections: "no, don't...", "actually...", "stop doing X", "I said..."
            re.compile(
                r"(?P<sentence>[^.!?\n]*\bno\b[^.!?\n]*\bdon'?t\b[^.!?\n]*(?:[.!?]|$))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?P<sentence>[^.!?\n]*\bactually\b[^.!?\n]*(?:[.!?]|$))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?P<sentence>[^.!?\n]*\bstop\s+(?:doing\s+)?\w+[^.!?\n]*(?:[.!?]|$))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?P<sentence>[^.!?\n]*\bI\s+said\b[^.!?\n]*(?:[.!?]|$))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?P<sentence>[^.!?\n]*\bthat'?s?\s+(?:wrong|incorrect|not\s+right)[^.!?\n]*(?:[.!?]|$))",
                re.IGNORECASE,
            ),
        ],
        "tag": "correction",
        "relevance": 0.8,
        "importance": 0.8,
    },
    "project": {
        "patterns": [
            # "the project uses...", "we decided...", "the codebase is..."
            re.compile(
                r"(?P<sentence>[^.!?\n]*\b(?:the\s+)?project\s+(?:uses?|needs?|requires?)\b[^.!?\n]*(?:[.!?]|$))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?P<sentence>[^.!?\n]*\bwe\s+(?:decided|agreed|chose)\b[^.!?\n]*(?:[.!?]|$))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?P<sentence>[^.!?\n]*\b(?:the\s+)?codebase\s+(?:is|uses?|has)\b[^.!?\n]*(?:[.!?]|$))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?P<sentence>[^.!?\n]*\b(?:the\s+)?(?:app|application|service|repo)\s+(?:uses?|runs?|is)\b[^.!?\n]*(?:[.!?]|$))",
                re.IGNORECASE,
            ),
        ],
        "tag": "project-fact",
        "relevance": 0.65,
        "importance": 0.65,
    },
    "reference": {
        "patterns": [
            # "the API is at...", "the path is...", "see docs at..."
            re.compile(
                r"(?P<sentence>[^.!?\n]*\b(?:the\s+)?API\s+(?:is|lives?|endpoint)\s+(?:at|on|in)\b[^.!?\n]*(?:[.!?]|$))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?P<sentence>[^.!?\n]*\bthe\s+path\s+(?:is|to)\b[^.!?\n]*(?:[.!?]|$))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?P<sentence>[^.!?\n]*\bsee\s+(?:the\s+)?docs?\s+at\b[^.!?\n]*(?:[.!?]|$))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?P<sentence>[^.!?\n]*\b(?:the\s+)?(?:URL|endpoint|host|port)\s+is\b[^.!?\n]*(?:[.!?]|$))",
                re.IGNORECASE,
            ),
        ],
        "tag": "reference-pointer",
        "relevance": 0.6,
        "importance": 0.6,
    },
}

# Maximum content length for a candidate (avoid storing huge paragraphs)
_MAX_CONTENT_LEN = 300


# ---------------------------------------------------------------------------
# Extraction function
# ---------------------------------------------------------------------------

def extract_candidates(
    user_content: str,
    assistant_content: str,
) -> List[CandidateMemory]:
    """Extract candidate memories from a conversation turn.

    Scans *user_content* (and reserves *assistant_content* for future use)
    for regex matches defined in ``_PATTERN_DEFS``.  Returns a deduplicated
    list of :class:`CandidateMemory` instances sorted by relevance (descending).

    Parameters
    ----------
    user_content:
        The user's message text for this turn.
    assistant_content:
        The assistant's response text (currently unused for extraction but
        reserved for future context-aware patterns).

    Returns
    -------
    List[CandidateMemory]
        Candidates with relevance scores.  Callers decide whether to persist
        (e.g. relevance >= 0.6).
    """
    candidates: List[CandidateMemory] = []
    seen_content: set = set()  # deduplicate by normalised content

    # We primarily extract from user messages — preferences, corrections,
    # project facts, and references are almost always user-stated.
    text = user_content
    if not text or not isinstance(text, str):
        return candidates

    for mem_type, defn in _PATTERN_DEFS.items():
        for pattern in defn["patterns"]:
            for match in pattern.finditer(text):
                # Prefer the named group "sentence", fall back to full match
                content = match.group("sentence") if "sentence" in match.groupdict() else match.group(0)
                content = content.strip()
                if not content:
                    continue
                # Truncate to a reasonable length
                content = content[:_MAX_CONTENT_LEN]
                # Normalise for dedup: lowercased, stripped
                norm = content.lower().strip()
                if norm in seen_content:
                    continue
                seen_content.add(norm)
                candidates.append(CandidateMemory(
                    type=mem_type,
                    content=content,
                    tags=[defn["tag"]],
                    relevance=defn["relevance"],
                    importance=defn.get("importance", defn["relevance"]),
                ))

    # Sort by relevance descending so most important candidates come first
    candidates.sort(key=lambda c: c.relevance, reverse=True)
    return candidates


def extract_candidates_from_messages(
    messages: List[dict],
) -> List[CandidateMemory]:
    """Extract candidates from a list of message dicts.

    Iterates over all user-role messages in *messages* and accumulates
    candidates.  Useful for session-end extraction.

    Parameters
    ----------
    messages:
        List of message dicts, each with at least ``role`` and ``content``.

    Returns
    -------
    List[CandidateMemory]
        Deduplicated candidates from all user messages.
    """
    all_candidates: List[CandidateMemory] = []
    seen: set = set()

    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str) or len(content) < 5:
            continue
        turn_candidates = extract_candidates(content, "")
        for c in turn_candidates:
            norm = c.content.lower().strip()
            if norm not in seen:
                seen.add(norm)
                all_candidates.append(c)

    return all_candidates


def build_pre_compress_summary(messages: List[dict]) -> str:
    """Build a summary string from messages about to be compressed.

    Extracts key facts, decisions, and corrections to preserve through
    context compression.  Returns a formatted string suitable for injection
    into a compression summary prompt, or empty string if nothing notable.

    Parameters
    ----------
    messages:
        The messages that are about to be compressed.

    Returns
    -------
    str
        Formatted summary string for the compressor to preserve.
    """
    candidates = extract_candidates_from_messages(messages)
    if not candidates:
        return ""

    # Group by type
    by_type: dict = {}
    for c in candidates:
        by_type.setdefault(c.type, []).append(c)

    lines = ["[Dream Memory: Key facts to preserve through compression]"]

    type_labels = {
        "user": "User Preferences",
        "feedback": "Corrections / Feedback",
        "project": "Project Context",
        "reference": "References",
    }

    for mem_type in ("user", "feedback", "project", "reference"):
        group = by_type.get(mem_type, [])
        if not group:
            continue
        label = type_labels.get(mem_type, mem_type.capitalize())
        lines.append(f"\n### {label}")
        # Take top candidates by relevance (max 5 per type to stay concise)
        for c in group[:5]:
            lines.append(f"- {c.content}")

    return "\n".join(lines)
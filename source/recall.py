"""Dream Memory Recall — manifest-based memory retrieval WITHOUT vector search.

Scans the vault manifest.json for candidate memories matching a query string,
scores them using tag matching, type filtering, stored relevance, and recency,
then returns the top-N results with full content loaded from disk.

This is the Phase 3 recall layer. It is intentionally LLM-free: all scoring
uses local heuristics (keyword / tag overlap, stored relevance, timestamps)
so prefetch stays fast even on large vaults.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .store import DreamStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ebbinghaus forgetting curve constants
# ---------------------------------------------------------------------------

FORGETTING_FACTOR_DEFAULT: float = 0.02    # Moderate decay (~35 day half-life)
FORGETTING_FACTOR_MIN: float = 0.005       # Very slow decay (~138 day half-life)
FORGETTING_FACTOR_MAX: float = 0.05        # Fast decay (~14 day half-life)
FORGETTING_REACCESS_MULTIPLIER: float = 0.9  # Decay slows 10% on each access


def retention_score(
    importance: float,
    forgetting_factor: float,
    age_days: float,
) -> float:
    """Ebbinghaus forgetting curve: retention = importance × e^(-forgetting_factor × age_days)

    Parameters
    ----------
    importance : float
        Memory importance (0.0–1.0). Higher importance = slower forgetting.
    forgetting_factor : float
        Decay rate (0.005–0.05). Higher = faster forgetting.
    age_days : float
        Days since last access (NOT creation — use updated/last_accessed).

    Returns
    -------
    float
        Retention score in [0, importance]. At age=0, retention=importance.
    """
    if age_days < 0:
        age_days = 0
    return importance * math.exp(-forgetting_factor * age_days)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RecallResult:
    """A single recalled memory with its match score."""

    memory_type: str     # user | feedback | project | reference
    filename: str         # e.g. "prefers-vim-20250413T000000Z.md"
    content: str          # full body text of the memory
    frontmatter: Dict[str, Any]  # parsed YAML frontmatter
    score: float         # combined relevance score (higher = better)
    is_related: bool = False  # True if this is a 1-hop wikilink expansion, not a primary match


# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

# Weight for tag overlap component (0–1 range after normalisation)
_TAG_WEIGHT: float = 0.35

# Weight for stored relevance field
_RELEVANCE_WEIGHT: float = 0.25

# Weight for recency component
_RECENCY_WEIGHT: float = 0.15

# Weight for access count component
_ACCESS_WEIGHT: float = 0.15

# Weight for feedback type boost
_FEEDBACK_BOOST_WEIGHT: float = 0.10

# Recency half-life in days: after this many days a memory's recency score
# drops by half.
_RECENCY_HALF_LIFE_DAYS: float = 30.0

# Minimum combined score threshold — memories with zero query relevance
# should not appear in results just because they have high stored relevance.
_MIN_COMBINED_SCORE: float = 0.1


# ---------------------------------------------------------------------------
# RecallEngine
# ---------------------------------------------------------------------------

class RecallEngine:
    """Manifest-scanning recall engine for Dream Memory.

    Usage::

        engine = RecallEngine(store)
        results = engine.recall("python preferences", limit=5)
        for r in results:
            print(r.memory_type, r.filename, r.score, r.content[:80])
    """

    def __init__(self, store: DreamStore) -> None:
        self._store = store

    # -- Public API ----------------------------------------------------------

    def recall(
        self,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 5,
    ) -> List[RecallResult]:
        """Return up to *limit* memories matching *query*.

        Parameters
        ----------
        query:
            Free-text search string.  Words are compared against memory tags
            and snippets in the manifest.
        memory_type:
            If provided, only memories of this type are returned.
        limit:
            Maximum number of results.

        Returns
        -------
        List[RecallResult]
            Sorted by descending combined score.
        """
        if not query or not query.strip():
            return []

        query_words = self._tokenise(query)
        if not query_words:
            return []

        # Reload manifest from disk for freshness
        manifest = self._store._ensure_manifest_loaded()
        if not manifest:
            return []

        candidates: List[_ScoredCandidate] = []
        now = datetime.now(timezone.utc)

        for entry in manifest:
            # --- Type filter ---
            if memory_type and entry.get("type") != memory_type:
                continue

            # --- Tag matching ---
            tags = entry.get("tags", [])
            tag_score = self._tag_overlap(query_words, tags)

            # --- Snippet matching (bonus for query words in snippet) ---
            snippet = entry.get("snippet", "")
            snippet_score = self._snippet_overlap(query_words, snippet)

            # --- Stored relevance ---
            stored_rel = float(entry.get("relevance", 0.5))
            # Clamp to [0, 1]
            stored_rel = max(0.0, min(1.0, stored_rel))

            # --- Ebbinghaus retention (importance × forgetting curve) ---
            # Importance defaults to relevance for backward compatibility
            importance = float(entry.get("importance", stored_rel))
            base_ff = float(entry.get("forgetting_factor", FORGETTING_FACTOR_DEFAULT))
            # Spacing effect: each access slows forgetting by 10%
            # effective_ff = base_ff × (0.9 ^ access_count), floored at FORGETTING_FACTOR_MIN
            access_count_val = int(entry.get("access_count", 0))
            effective_ff = max(FORGETTING_FACTOR_MIN, base_ff * (FORGETTING_REACCESS_MULTIPLIER ** access_count_val))
            updated_str = entry.get("updated") or entry.get("created") or ""
            age_days = self._age_days(updated_str, now)
            retention = retention_score(importance, effective_ff, age_days)

            # --- Access count ---
            access_count = access_count_val
            # Normalize: log(1 + count) / log(1 + max_count) to avoid unbounded scores
            # With a floor of max_count=10 (10 accesses = saturated)
            access_score = min(1.0, math.log1p(access_count) / math.log1p(10))

            # --- Feedback type boost ---
            # Feedback memories are user corrections — highest value
            feedback_boost = 0.0
            if entry.get("type") == "feedback":
                feedback_boost = 1.0

            # --- Combined score ---
            # Tag overlap and snippet overlap share the tag weight
            text_score = min(1.0, tag_score + snippet_score * 0.5)
            combined = (
                _TAG_WEIGHT * text_score
                + _RELEVANCE_WEIGHT * stored_rel
                + _RECENCY_WEIGHT * retention
                + _ACCESS_WEIGHT * access_score
                + _FEEDBACK_BOOST_WEIGHT * feedback_boost
            )

            # Even with zero text overlap we keep the candidate if relevance
            # is decent — the stored relevance already encodes usefulness.
            # But tag/snippet matches should still boost ranking significantly.

            candidates.append(_ScoredCandidate(
                entry=entry,
                score=combined,
            ))

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)

        # Filter out candidates below minimum relevance threshold
        candidates = [c for c in candidates if c.score >= _MIN_COMBINED_SCORE]

        # Take top limit and load full content
        results: List[RecallResult] = []
        for cand in candidates[:limit]:
            entry = cand.entry
            mem_type = entry.get("type", "")
            filename = entry.get("filename", "")

            # Load full memory content
            try:
                data = self._store.read_memory(mem_type, filename)
                content = data.get("body", "")
                frontmatter = data.get("meta", {})
            except FileNotFoundError:
                logger.debug("Recall: skipping missing file %s/%s", mem_type, filename)
                continue
            except Exception as exc:
                logger.warning("Recall: error reading %s/%s: %s", mem_type, filename, exc)
                continue

            results.append(RecallResult(
                memory_type=mem_type,
                filename=filename,
                content=content,
                frontmatter=frontmatter,
                score=round(cand.score, 4),
            ))

        # After building results, increment access counts for each returned memory
        for r in results:
            try:
                self._store.increment_access_count(r.memory_type, r.filename)
            except Exception as exc:
                logger.debug("Recall: failed to increment access_count for %s/%s: %s",
                             r.memory_type, r.filename, exc)

        # 1-hop wikilink expansion
        expanded: List[RecallResult] = []
        seen_filenames = {f"{r.memory_type}/{r.filename}" for r in results}
        for r in results:
            try:
                related_entries = self._store.get_related_memories(r.memory_type, r.filename)
                for entry in related_entries:
                    key = f"{entry.get('type', '')}/{entry.get('filename', '')}"
                    if key not in seen_filenames:
                        seen_filenames.add(key)
                        # Create a RecallResult for the related memory
                        try:
                            related_data = self._store.read_memory(entry.get("type", ""), entry.get("filename", ""))
                            related_content = related_data.get("body", "") if isinstance(related_data, dict) else ""
                            related_frontmatter = related_data.get("meta", {}) if isinstance(related_data, dict) else {}
                        except Exception:
                            related_content = entry.get("snippet", "")
                            related_frontmatter = {}
                        if related_content:
                            expanded.append(RecallResult(
                                score=r.score * 0.7,  # Related items score lower than primary
                                memory_type=entry.get("type", ""),
                                filename=entry.get("filename", ""),
                                content=related_content[:300],
                                frontmatter=related_frontmatter,
                                is_related=True,
                            ))
            except Exception:
                pass

        # Add expanded results (up to the limit)
        remaining = max(0, limit * 2 - len(results))  # Allow up to 2x for related
        results.extend(expanded[:remaining])

        return results

    # -- Tokenisation --------------------------------------------------------

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """Split text into lowercase alpha-numeric tokens."""
        words: List[str] = []
        for word in text.lower().split():
            # Strip non-alphanumeric chars from edges
            cleaned = "".join(ch for ch in word if ch.isalnum() or ch == "-")
            if cleaned and len(cleaned) > 1:
                words.append(cleaned)
        return words

    # -- Scoring helpers -----------------------------------------------------

    @staticmethod
    def _tag_overlap(query_words: List[str], tags: List[str]) -> float:
        """Fraction of query words that appear in any tag (0–1)."""
        if not query_words:
            return 0.0
        normalised_tags = [t.lower() for t in tags]
        matches = sum(
            1 for w in query_words
            if any(w == t or w in t.split("-") or w in t.split("_") for t in normalised_tags)
        )
        return matches / len(query_words)

    @staticmethod
    def _snippet_overlap(query_words: List[str], snippet: str) -> float:
        """Fraction of query words appearing in the snippet text."""
        if not query_words or not snippet:
            return 0.0
        snippet_lower = snippet.lower()
        matches = sum(1 for w in query_words if w in snippet_lower)
        return matches / len(query_words)

    @staticmethod
    def _recency_score(updated_str: str, now: datetime) -> float:
        """Exponential decay recency score (1.0 = just now, → 0 as age → ∞).

        .. deprecated::
            Kept for backward compatibility.  New code should use
            :func:`retention_score` with importance and forgetting factor.
        """
        if not updated_str:
            return 0.1  # unknown timestamp → low but non-zero
        try:
            then = datetime.fromisoformat(updated_str)
            if then.tzinfo is None:
                then = then.replace(tzinfo=timezone.utc)
            age_days = max(0.0, (now - then).total_seconds() / 86400.0)
        except (ValueError, TypeError):
            return 0.1

        # Exponential decay: score = 0.5^(age / half_life)
        return 0.5 ** (age_days / _RECENCY_HALF_LIFE_DAYS)

    @staticmethod
    def _age_days(updated_str: str, now: datetime) -> float:
        """Compute age in days from an ISO timestamp string.

        Returns 0.0 for missing/invalid timestamps.
        """
        if not updated_str:
            return 0.0
        try:
            then = datetime.fromisoformat(updated_str)
            if then.tzinfo is None:
                then = then.replace(tzinfo=timezone.utc)
            return max(0.0, (now - then).total_seconds() / 86400.0)
        except (ValueError, TypeError):
            return 0.0


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

@dataclass
class _ScoredCandidate:
    entry: Dict[str, Any]
    score: float
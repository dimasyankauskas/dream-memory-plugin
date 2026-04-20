"""Dream v2 Recall Engine — Manifest-based memory retrieval with Ebbinghaus decay.

On-demand retrieval only — no auto-injection per turn.
Scoring: tag overlap (35%) + relevance (25%) + recency (15%) + access_count (15%).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RecallResult:
    """A single recalled memory result."""
    memory: Dict[str, Any]
    score: float
    snippet: str


class RecallEngine:
    """Manifest-based retrieval with forgetting curve scoring."""

    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.manifest_path = self.vault_path / "manifest.json"

    def recall(
        self,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 5,
    ) -> List[RecallResult]:
        """Recall memories matching query.

        Scoring:
          - tag overlap: 35%
          - relevance/importance: 25%
          - recency: 15%
          - access_count: 15%
        """
        manifest = self._load_manifest()
        if not manifest:
            return []

        query_lower = query.lower()
        query_words = set(re.findall(r"\w+", query_lower))
        query_tags = self._extract_tags(query_lower)

        results = []
        for entry in manifest:
            # Type filter
            if memory_type and entry.get("type") != memory_type:
                continue

            # Compute score
            score = self._compute_score(entry, query_words, query_tags, manifest)

            if score > 0:
                # Load content snippet
                content = self._load_content(entry)
                snippet = content[:300] if content else entry.get("slug", "")[:200]

                results.append(RecallResult(
                    memory={
                        **entry,
                        "content": snippet,
                    },
                    score=score,
                    snippet=snippet,
                ))

                # Increment access count (best-effort)
                self._increment_access(entry)

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def _load_manifest(self) -> List[Dict[str, Any]]:
        try:
            with open(self.manifest_path) as f:
                return json.load(f)
        except Exception:
            return []

    def _compute_score(
        self,
        entry: Dict[str, Any],
        query_words: set,
        query_tags: set,
        manifest: List[Dict[str, Any]],
    ) -> float:
        """Compute recall score for a memory entry."""
        # 1. Tag overlap (35%)
        entry_tags = set(entry.get("tags", []))
        tag_score = 0.0
        if query_tags and entry_tags:
            overlap = len(query_tags & entry_tags)
            union = len(query_tags | entry_tags)
            tag_score = overlap / union if union else 0

        # 2. Importance/relevance (25%) — coerce to float
        raw_imp = entry.get("importance", entry.get("relevance", 0.5))
        try:
            importance = float(raw_imp)
        except (TypeError, ValueError):
            importance = 0.5

        # 3. Recency (15%) — exponential decay
        created = entry.get("created", "")
        recency_score = self._recency_score(created)

        # 4. Access count (15%) — logarithmic boost
        access_count = entry.get("access_count", 0)
        access_score = min(math.log1p(access_count) / 10, 1.0)

        # Weighted sum
        total = (
            0.35 * tag_score +
            0.25 * importance +
            0.15 * recency_score +
            0.15 * access_score
        )

        # Forgetting curve discount
        forgetting = entry.get("forgetting_factor", 0.02)
        days_since = self._days_since(entry.get("created", ""))
        decay = math.exp(-forgetting * days_since)
        total *= max(0.3, min(1.0, decay))

        return total

    def _recency_score(self, created: str) -> float:
        """Exponential decay score based on age. 0.5 at 35 days."""
        if not created:
            return 0.5
        days = self._days_since(created)
        return math.exp(-0.02 * days)

    def _days_since(self, iso_timestamp: str) -> float:
        try:
            created = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return (now - created).total_seconds() / 86400
        except Exception:
            return 30  # default to ~35 day half-life

    def _extract_tags(self, text: str) -> set:
        """Extract potential tags from query text."""
        # Remove common stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
                     "what", "how", "why", "when", "where", "who", "which", "about",
                     "remember", "recall", "tell", "me", "my", "i", "you", "your"}
        words = set(re.findall(r"\w+", text.lower())) - stopwords
        return words

    def _load_content(self, entry: Dict[str, Any]) -> str:
        """Load the actual memory content from disk."""
        try:
            mem_type = entry.get("type", "")
            filename = entry.get("filename", "")
            if not filename:
                return ""
            path = self.vault_path / mem_type / filename
            if path.exists():
                text = path.read_text()
                # Strip frontmatter
                if text.startswith("---"):
                    parts = text.split("---", 2)
                    if len(parts) >= 3:
                        return parts[2].strip()
                return text.strip()
        except Exception as e:
            logger.warning("[RecallEngine] Failed to load content: %s", e)
        return ""

    def _increment_access(self, entry: Dict[str, Any]) -> None:
        """Increment access count in manifest."""
        try:
            manifest = self._load_manifest()
            for m in manifest:
                if m.get("filename") == entry.get("filename") and m.get("type") == entry.get("type"):
                    m["access_count"] = m.get("access_count", 0) + 1
                    with open(self.manifest_path, "w") as f:
                        json.dump(manifest, f, indent=2)
                    break
        except Exception:
            pass  # best-effort
"""Dream Memory Consolidation Engine — 4-Phase AutoDream cycle.

Implements the Orient → Gather → Consolidate → Prune pipeline that keeps
the dream vault lean, deduplicated, and contradiction-free.

Phase 1: Orient   — Decide whether consolidation is needed.
Phase 2: Gather   — Collect and cluster related memories.
Phase 3: Consolidate — Merge, deduplicate, resolve contradictions.
                Supports two modes:
                - 'deterministic' (default): string/tag overlap heuristics
                - 'llm': LLM-powered consolidation for smarter merging
Phase 4: Prune   — Enforce caps, delete superseded, update manifest.
"""

from __future__ import annotations

import json
import logging
import os
import re
import string
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .store import DreamStore
from .taxonomy import MEMORY_TYPES, parse_frontmatter
from .recall import retention_score, FORGETTING_FACTOR_DEFAULT, FORGETTING_FACTOR_MIN, FORGETTING_REACCESS_MULTIPLIER

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_LINES: int = 200
DEFAULT_MAX_BYTES: int = 25600
CONSOLIDATION_INTERVAL_HOURS: float = 24.0
MIN_SESSIONS_FOR_CONSOLIDATION: int = 5  # Anthropic's default session gate
TAG_OVERLAP_THRESHOLD: float = 0.5       # >50% tag overlap → same topic
CONTENT_SIMILARITY_THRESHOLD: float = 0.8  # >80% word overlap → duplicate

# Forgetting pruning thresholds
PRUNE_RETENTION_THRESHOLD: float = 0.1   # Retention below this = candidate for forgetting
PRUNE_MIN_AGE_DAYS: int = 60             # Younger memories are never forgotten
PRUNE_MAX_IMPORTANCE: float = 0.3        # High-importance memories resist forgetting

# ---------------------------------------------------------------------------
# Contradiction pattern pairs
# ---------------------------------------------------------------------------

# Each pair is (affirmative_pattern, negative_pattern).  If two memories
# about the same topic contain text matching different sides of a pair,
# they contradict.

_CONTRADICTION_PAIRS: List[Tuple[re.Pattern, re.Pattern]] = [
    # prefer/like/love X vs don't prefer/dislike/hate X
    (
        re.compile(r"\b(?:prefer|like|love|enjoy|favor)\s+(\w+)", re.IGNORECASE),
        re.compile(r"\b(?:don'?t\s+prefer|dislike|hate|don'?t\s+like|don'?t\s+love|don'?t\s+enjoy)\s+(\w+)", re.IGNORECASE),
    ),
    # use X vs don't use X
    (
        re.compile(r"\b(?:use|always\s+use|should\s+use)\s+(\w+)", re.IGNORECASE),
        re.compile(r"\b(?:don'?t\s+use|never\s+use|avoid\s+ using)\s+(\w+)", re.IGNORECASE),
    ),
    # always X vs never X
    (
        re.compile(r"\balways\s+(\w+)", re.IGNORECASE),
        re.compile(r"\bnever\s+(\w+)", re.IGNORECASE),
    ),
]

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OrientResult:
    """Result of the Orient phase."""

    needs_consolidation: bool = False
    reason: str = ""
    stale_files: List[str] = field(default_factory=list)
    oversized_files: List[str] = field(default_factory=list)
    fragmented_topics: Dict[str, int] = field(default_factory=dict)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryEntry:
    """A single memory loaded from the vault, enriched for consolidation."""

    memory_type: str
    filename: str
    tags: List[str] = field(default_factory=list)
    content: str = ""
    relevance: float = 0.5
    importance: float = 0.5
    forgetting_factor: float = 0.02
    created: str = ""
    updated: str = ""
    lines: int = 0
    bytes: int = 0
    path: str = ""
    access_count: int = 0


@dataclass
class MemoryGroup:
    """A cluster of memories about the same topic."""

    group_id: str
    entries: List[MemoryEntry] = field(default_factory=list)


@dataclass
class DuplicatePair:
    """Two memories with high content similarity."""

    file_a: str   # memory_type:filename
    file_b: str
    similarity: float


@dataclass
class ContradictionPair:
    """Two memories with contradictory content about the same topic."""

    file_a: str   # newer
    file_b: str   # older / superseded
    topic: str    # the conflicting keyword


@dataclass
class GatherResult:
    """Result of the Gather phase."""

    entries: List[MemoryEntry] = field(default_factory=list)
    groups: List[MemoryGroup] = field(default_factory=list)
    duplicates: List[DuplicatePair] = field(default_factory=list)
    contradictions: List[ContradictionPair] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsolidationAction:
    """Describes one consolidation action taken."""

    action: str           # "merge", "deduplicate", "contradict", "compress", "boost_relevance"
    target_type: str      # memory type
    target_files: List[str] = field(default_factory=list)
    result_file: str = ""
    details: str = ""


@dataclass
class ConsolidateResult:
    """Result of the Consolidate phase."""

    actions: List[ConsolidationAction] = field(default_factory=list)
    merged_count: int = 0
    deduped_count: int = 0
    pruned_count: int = 0
    stats: Dict[str, Any] = field(default_factory=dict)
    # Files to delete during prune (superseded by contradictions)
    superseded: List[str] = field(default_factory=list)  # list of "type:filename"
    # Files to cap (truncated)
    capped: List[Tuple[str, str, int]] = field(default_factory=list)  # (type, filename, max_lines)
    # Merged content to write
    merges: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PruneResult:
    """Result of the Prune phase."""

    deleted_files: List[str] = field(default_factory=list)
    capped_files: List[str] = field(default_factory=list)
    manifest_updated: bool = False
    log_written: bool = False
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsolidationResult:
    """Top-level result from run_consolidation()."""

    orient: OrientResult = field(default_factory=OrientResult)
    gather: GatherResult = field(default_factory=GatherResult)
    consolidate: ConsolidateResult = field(default_factory=ConsolidateResult)
    prune: PruneResult = field(default_factory=PruneResult)
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Consolidation Lock
# ---------------------------------------------------------------------------

class ConsolidationLockError(Exception):
    """Raised when consolidation lock cannot be acquired."""
    pass


class ConsolidationLock:
    """PID-based file lock for dream consolidation.

    Lock file at vault_path/.consolidation.lock:
    - File body = PID of acquiring process
    - File mtime = timestamp of lock acquisition
    - Stale detection: lock older than STALE_THRESHOLD is auto-broken
    """

    STALE_THRESHOLD_SECONDS: float = 3600.0  # 1 hour

    def __init__(self, vault_path: Path):
        self.lock_path = vault_path / ".consolidation.lock"
        self._fd = None
        self._acquired = False

    def acquire(self, timeout: float = 5.0) -> bool:
        """Try to acquire the consolidation lock. Returns False if already locked.

        Steps:
        1. If no lock file exists, create it with our PID → return True
        2. If lock file exists:
           a. Read the PID from the file
           b. Check if that PID is still alive (os.kill(pid, 0))
           c. Check if the lock file's mtime is older than STALE_THRESHOLD
           d. If PID is dead OR lock is stale → break the lock (delete + recreate)
           e. If PID is alive and lock is fresh → return False
        3. Write our PID to the lock file, touch mtime to now
        """
        if self.lock_path.exists():
            try:
                pid_str = self.lock_path.read_text(encoding="utf-8").strip()
                pid = int(pid_str)
            except (ValueError, OSError):
                # Corrupt lock file — break it
                pid = None

            if pid is not None:
                # If we already hold this lock, return True (re-entrant)
                if pid == os.getpid() and self._acquired:
                    return True

                pid_alive = self._is_pid_alive(pid)
                lock_stale = self._is_lock_stale()

                if not pid_alive or lock_stale:
                    # Break the stale or dead lock
                    try:
                        self.lock_path.unlink()
                    except OSError:
                        pass
                else:
                    # Lock is held by a live process and is fresh
                    return False
            else:
                # Corrupt lock file — remove it
                try:
                    self.lock_path.unlink()
                except OSError:
                    pass

        # Create the lock file with our PID
        try:
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            self.lock_path.write_text(str(os.getpid()), encoding="utf-8")
            self._acquired = True
            return True
        except OSError as exc:
            logger.warning("Failed to acquire consolidation lock: %s", exc)
            return False

    def release(self) -> None:
        """Release the lock by closing fd and deleting the lock file."""
        if self._acquired:
            try:
                if self.lock_path.exists():
                    self.lock_path.unlink()
            except OSError:
                pass
            self._acquired = False

    @property
    def is_locked(self) -> bool:
        """Check if lock file exists and is held by a live process."""
        if not self.lock_path.exists():
            return False
        try:
            pid_str = self.lock_path.read_text(encoding="utf-8").strip()
            pid = int(pid_str)
            return self._is_pid_alive(pid) and not self._is_lock_stale()
        except (ValueError, OSError):
            return False

    def _is_pid_alive(self, pid: int) -> bool:
        """Check if a PID is still alive using os.kill(pid, 0)."""
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _is_lock_stale(self) -> bool:
        """Check if the lock file's mtime is older than STALE_THRESHOLD."""
        try:
            mtime = self.lock_path.stat().st_mtime
            age = time.time() - mtime
            return age > self.STALE_THRESHOLD_SECONDS
        except OSError:
            return True

    def __enter__(self):
        if not self.acquire():
            raise ConsolidationLockError("Consolidation already in progress")
        return self

    def __exit__(self, *args):
        self.release()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_words(text: str) -> Set[str]:
    """Lowercase, strip punctuation, split into unique words."""
    text = text.lower()
    # Remove punctuation except hyphens and apostrophes within words
    text = re.sub(r"[^\w\s'-]", " ", text)
    words = {w for w in text.split() if w and len(w) > 1}
    return words


def _content_similarity(a: str, b: str) -> float:
    """Word-overlap Jaccard similarity between two strings.

    Returns a float in [0, 1].  Two memories with >80% overlap are
    considered duplicates.  Two empty strings return 0.0 (not similar).
    """
    words_a = _normalise_words(a)
    words_b = _normalise_words(b)
    if not words_a and not words_b:
        return 0.0
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _tag_overlap(tags_a: List[str], tags_b: List[str]) -> float:
    """Fraction of shared tags (Jaccard on normalised tag sets)."""
    set_a = {t.lower() for t in tags_a}
    set_b = {t.lower() for t in tags_b}
    if not set_a and not set_b:
        return 0.0
    if not set_a or not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


def _is_older(entry_a: MemoryEntry, entry_b: MemoryEntry) -> bool:
    """Return True if entry_a is older (created earlier) than entry_b."""
    def _parse_ts(ts: str) -> datetime:
        if not ts:
            return datetime(2000, 1, 1, tzinfo=timezone.utc)
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return datetime(2000, 1, 1, tzinfo=timezone.utc)

    return _parse_ts(entry_a.created) < _parse_ts(entry_b.created)


def _compute_age_days(entry: MemoryEntry, now: datetime) -> float:
    """Compute age in days for a MemoryEntry using its ``updated`` or ``created`` timestamp."""
    ts_str = entry.updated or entry.created
    if not ts_str:
        return 0.0
    try:
        then = datetime.fromisoformat(ts_str)
        if then.tzinfo is None:
            then = then.replace(tzinfo=timezone.utc)
        return max(0.0, (now - then).total_seconds() / 86400.0)
    except (ValueError, TypeError):
        return 0.0


def _detect_contradictions(
    entry_a: MemoryEntry,
    entry_b: MemoryEntry,
) -> Optional[str]:
    """If two memories contradict, return the topic keyword; else None.

    Checks whether one entry matches an affirmative pattern and the other
    matches the corresponding negative pattern.
    """
    text_a = entry_a.content.lower()
    text_b = entry_b.content.lower()

    for aff_pattern, neg_pattern in _CONTRADICTION_PAIRS:
        # Check if a matches affirmative and b matches negative
        a_aff = aff_pattern.findall(text_a)
        b_neg = neg_pattern.findall(text_b)
        if a_aff and b_neg:
            # Check for shared topic words
            common = set(w.lower() for w in a_aff) & set(w.lower() for w in b_neg)
            if common:
                return ", ".join(sorted(common))

        # Check the reverse: a matches negative, b matches affirmative
        a_neg = neg_pattern.findall(text_a)
        b_aff = aff_pattern.findall(text_b)
        if a_neg and b_aff:
            common = set(w.lower() for w in a_neg) & set(w.lower() for w in b_aff)
            if common:
                return ", ".join(sorted(common))

    return None


# ---------------------------------------------------------------------------
# Phase 1: Orient
# ---------------------------------------------------------------------------


def orient(store: DreamStore) -> OrientResult:
    """Determine whether consolidation is needed.

    Checks:
    - Time since last consolidation (skip if < 24h with no significant changes)
    - Identify stale, fragmented, and oversized memory files
    - Count total memories and per-type stats
    """
    vault_path = store.dream_root
    result = OrientResult()

    # --- Check consolidation log for last run ---
    last_consolidation = _read_last_consolidation_ts(vault_path)
    now = datetime.now(timezone.utc)

    # --- Collect all memories ---
    all_entries: List[MemoryEntry] = []
    type_counts: Dict[str, int] = {t: 0 for t in MEMORY_TYPES}

    for mem_type in MEMORY_TYPES:
        type_dir = vault_path / mem_type
        if not type_dir.exists():
            continue
        for md_file in sorted(type_dir.glob("*.md")):
            try:
                text = md_file.read_text(encoding="utf-8")
                meta = parse_frontmatter(text)
                body = DreamStore._extract_body(text)

                entry = MemoryEntry(
                    memory_type=mem_type,
                    filename=md_file.name,
                    tags=meta.get("tags", []),
                    content=body,
                    relevance=float(meta.get("relevance", 0.5)),
                    importance=float(meta.get("importance", meta.get("relevance", 0.5))),
                    forgetting_factor=float(meta.get("forgetting_factor", FORGETTING_FACTOR_DEFAULT)),
                    created=meta.get("created", ""),
                    updated=meta.get("updated", ""),
                    lines=body.count("\n") + 1,
                    bytes=md_file.stat().st_size,
                    path=str(md_file),
                    access_count=int(meta.get("access_count", 0)),
                )
                all_entries.append(entry)
                type_counts[mem_type] = type_counts.get(mem_type, 0) + 1
            except Exception as exc:
                logger.warning("Orient: skipping %s (%s)", md_file, exc)

    total_memories = len(all_entries)

    # --- Identify stale files (not updated in 30 days) ---
    stale_cutoff = now - timedelta(days=30)
    for entry in all_entries:
        ts_str = entry.updated or entry.created
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < stale_cutoff:
                result.stale_files.append(f"{entry.memory_type}:{entry.filename}")
        except (ValueError, TypeError):
            pass

    # --- Identify oversized files ---
    for entry in all_entries:
        type_spec = MEMORY_TYPES.get(entry.memory_type)
        max_lines = type_spec.max_lines if type_spec else DEFAULT_MAX_LINES
        if entry.lines > max_lines:
            result.oversized_files.append(
                f"{entry.memory_type}:{entry.filename} ({entry.lines}/{max_lines} lines)"
            )

    # --- Identify fragmented topics (multiple memories with overlapping tags) ---
    tag_groups: Dict[str, int] = {}
    for entry in all_entries:
        for tag in entry.tags:
            tag_groups[tag] = tag_groups.get(tag, 0) + 1
    result.fragmented_topics = {
        tag: count for tag, count in tag_groups.items() if count >= 3
    }

    result.stats = {
        "total_memories": total_memories,
        "type_counts": type_counts,
        "stale_count": len(result.stale_files),
        "oversized_count": len(result.oversized_files),
        "fragmented_tags": len(result.fragmented_topics),
        "last_consolidation": last_consolidation,
    }

    # --- Decide if consolidation is needed ---
    hours_since = None
    if last_consolidation:
        try:
            last_dt = datetime.fromisoformat(last_consolidation)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            hours_since = (now - last_dt).total_seconds() / 3600
        except (ValueError, TypeError):
            hours_since = None

    needs_consolidation = False
    reasons = []

    # Session count gate (Anthropic's dual-gate pattern)
    min_sessions = store._config.get("min_sessions_for_consolidation", MIN_SESSIONS_FOR_CONSOLIDATION) if hasattr(store, '_config') and store._config else MIN_SESSIONS_FOR_CONSOLIDATION
    # Ensure integer (config may store as string)
    try:
        min_sessions = int(min_sessions)
    except (ValueError, TypeError):
        min_sessions = MIN_SESSIONS_FOR_CONSOLIDATION
    sessions_since = store.get_sessions_since_consolidation()

    # URGENT conditions bypass both gates (oversized files, heavy fragmentation)
    urgent = bool(result.oversized_files) or len(result.fragmented_topics) >= 2

    if hours_since is None:
        # Never consolidated before — first consolidation runs unconditionally
        # (Session gate only applies to SUBSEQUENT consolidations)
        needs_consolidation = True
        if total_memories == 0:
            reasons.append("no previous consolidation (empty vault)")
        else:
            reasons.append("no previous consolidation")
    elif urgent:
        needs_consolidation = True
        reasons.append(f"urgent: {len(result.oversized_files)} oversized or {len(result.fragmented_topics)} fragmented")
    elif hours_since >= CONSOLIDATION_INTERVAL_HOURS and sessions_since >= min_sessions:
        # Both time gate AND session gate passed
        needs_consolidation = True
        reasons.append(f"last consolidation {hours_since:.0f}h ago + {sessions_since} sessions")
    elif hours_since >= CONSOLIDATION_INTERVAL_HOURS:
        # Time gate passed but session gate not met
        needs_consolidation = False
        reasons.append(f"time gate passed ({hours_since:.0f}h) but only {sessions_since}/{min_sessions} sessions")
    else:
        # Time gate not passed
        needs_consolidation = False
        reasons.append(f"last consolidation {hours_since:.0f}h ago (need {CONSOLIDATION_INTERVAL_HOURS:.0f}h)")

    result.needs_consolidation = needs_consolidation
    result.reason = "; ".join(reasons) if reasons else "not needed"

    # Add session info to stats
    result.stats["sessions_since_consolidation"] = sessions_since
    result.stats["min_sessions"] = min_sessions

    return result


# ---------------------------------------------------------------------------
# Phase 2: Gather
# ---------------------------------------------------------------------------


def gather(store: DreamStore, orient_result: OrientResult, memory_type: Optional[str] = None) -> GatherResult:
    """Load and cluster memories for consolidation.

    Groups memories by tag overlap, detects duplicates, and finds
    contradictions.
    """
    vault_path = store.dream_root
    result = GatherResult()

    # --- Collect entries ---
    types_to_scan = [memory_type] if memory_type else list(MEMORY_TYPES.keys())
    all_entries: List[MemoryEntry] = []

    for mem_type in types_to_scan:
        type_dir = vault_path / mem_type
        if not type_dir.exists():
            continue
        for md_file in sorted(type_dir.glob("*.md")):
            try:
                text = md_file.read_text(encoding="utf-8")
                meta = parse_frontmatter(text)
                body = DreamStore._extract_body(text)

                entry = MemoryEntry(
                    memory_type=mem_type,
                    filename=md_file.name,
                    tags=meta.get("tags", []),
                    content=body,
                    relevance=float(meta.get("relevance", 0.5)),
                    importance=float(meta.get("importance", meta.get("relevance", 0.5))),
                    forgetting_factor=float(meta.get("forgetting_factor", FORGETTING_FACTOR_DEFAULT)),
                    created=meta.get("created", ""),
                    updated=meta.get("updated", ""),
                    lines=body.count("\n") + 1,
                    bytes=md_file.stat().st_size,
                    path=str(md_file),
                    access_count=int(meta.get("access_count", 0)),
                )
                result.entries.append(entry)
                all_entries.append(entry)
            except Exception as exc:
                logger.warning("Gather: skipping %s (%s)", md_file, exc)

    result.entries = all_entries

    # --- Enrich entries with manifest data (access_count, importance, forgetting_factor) ---
    try:
        manifest = store._ensure_manifest_loaded()
        manifest_lookup = {}
        for m_entry in manifest:
            key = f"{m_entry.get('type', '')}:{m_entry.get('filename', '')}"
            manifest_lookup[key] = m_entry
        for entry in all_entries:
            key = f"{entry.memory_type}:{entry.filename}"
            m_entry = manifest_lookup.get(key)
            if m_entry:
                # Use manifest access_count (the authoritative source)
                entry.access_count = int(m_entry.get("access_count", entry.access_count))
                # Manifest may have newer importance/forgetting_factor if updated
                if "importance" in m_entry and entry.importance == 0.5:
                    entry.importance = float(m_entry.get("importance", entry.importance))
                if "forgetting_factor" in m_entry and entry.forgetting_factor == 0.02:
                    entry.forgetting_factor = float(m_entry.get("forgetting_factor", entry.forgetting_factor))
    except Exception:
        pass  # Manifest enrichment is best-effort

    # --- Group by topic (tag overlap) ---
    # Use union-find style grouping: memories sharing >50% tags are in the same group
    assigned: Set[int] = set()
    groups: List[MemoryGroup] = []

    for i, entry_a in enumerate(all_entries):
        if i in assigned:
            continue
        group_entries = [entry_a]
        assigned.add(i)

        for j, entry_b in enumerate(all_entries):
            if j in assigned:
                continue
            # Must be same memory type to group
            if entry_a.memory_type != entry_b.memory_type:
                continue
            overlap = _tag_overlap(entry_a.tags, entry_b.tags)
            if overlap > TAG_OVERLAP_THRESHOLD:
                group_entries.append(entry_b)
                assigned.add(j)

        if len(group_entries) > 1:
            group_id = f"{entry_a.memory_type}:{','.join(e.filename for e in group_entries[:3])}"
            groups.append(MemoryGroup(group_id=group_id, entries=group_entries))

    result.groups = groups

    # --- Find duplicates (high content similarity) ---
    for i, entry_a in enumerate(all_entries):
        for j, entry_b in enumerate(all_entries):
            if j <= i:
                continue
            if entry_a.memory_type != entry_b.memory_type:
                continue
            sim = _content_similarity(entry_a.content, entry_b.content)
            if sim >= CONTENT_SIMILARITY_THRESHOLD:
                key_a = f"{entry_a.memory_type}:{entry_a.filename}"
                key_b = f"{entry_b.memory_type}:{entry_b.filename}"
                result.duplicates.append(DuplicatePair(
                    file_a=key_a,
                    file_b=key_b,
                    similarity=round(sim, 3),
                ))

    # --- Find contradictions ---
    for i, entry_a in enumerate(all_entries):
        for j, entry_b in enumerate(all_entries):
            if j <= i:
                continue
            if entry_a.memory_type != entry_b.memory_type:
                continue
            # Only check contradictions for memories with overlapping tags
            if _tag_overlap(entry_a.tags, entry_b.tags) < 0.3:
                continue
            topic = _detect_contradictions(entry_a, entry_b)
            if topic:
                # Newer file is kept, older is marked superseded
                if _is_older(entry_a, entry_b):
                    older_key = f"{entry_a.memory_type}:{entry_a.filename}"
                    newer_key = f"{entry_b.memory_type}:{entry_b.filename}"
                else:
                    older_key = f"{entry_b.memory_type}:{entry_b.filename}"
                    newer_key = f"{entry_a.memory_type}:{entry_a.filename}"
                result.contradictions.append(ContradictionPair(
                    file_a=newer_key,
                    file_b=older_key,  # superseded
                    topic=topic,
                ))

    result.stats = {
        "entries_loaded": len(all_entries),
        "groups_found": len(groups),
        "duplicates_found": len(result.duplicates),
        "contradictions_found": len(result.contradictions),
    }

    return result


# ---------------------------------------------------------------------------
# Wikilink insertion helpers (Heartbeat 5)
# ---------------------------------------------------------------------------


def _add_wikilinks_to_merged_content(merged_content: str, source_entries: List, store=None) -> str:
    """Add [[wikilinks]] to merged memory content pointing to original sources."""
    from .store import slug_from_filename, make_wikilink

    links = []
    for entry in source_entries:
        filename = getattr(entry, "filename", "") if not isinstance(entry, dict) else entry.get("filename", "")
        mem_type = getattr(entry, "memory_type", "") if not isinstance(entry, dict) else entry.get("type", "")
        slug = slug_from_filename(filename)
        if slug:
            links.append(make_wikilink(mem_type, slug))

    if not links:
        return merged_content

    # Don't add if already has wikilinks
    if "## Related" in merged_content or "[[" in merged_content:
        return merged_content

    link_line = "## Related\n" + " ".join(links)
    return merged_content.rstrip() + "\n\n" + link_line + "\n"


def _add_bidirectional_wikilinks(entries_in_group: List, store, max_links: int = 5) -> None:
    """Add bidirectional [[wikilinks]] between memories in the same consolidation group.

    Args:
        entries_in_group: Memories in the same consolidation group.
        store: DreamStore instance.
        max_links: Maximum wikilinks per memory (default 5). Prevents wikilink explosion.
    """
    from .store import slug_from_filename, make_wikilink

    if len(entries_in_group) < 2:
        return

    for i, entry in enumerate(entries_in_group):
        filename_i = getattr(entry, "filename", "") if not isinstance(entry, dict) else entry.get("filename", "")
        type_i = getattr(entry, "memory_type", "") if not isinstance(entry, dict) else entry.get("type", "")

        other_links = []
        for j, other in enumerate(entries_in_group):
            if i == j:
                continue
            if len(other_links) >= max_links:
                break  # Cap reached
            filename_j = getattr(other, "filename", "") if not isinstance(other, dict) else other.get("filename", "")
            type_j = getattr(other, "memory_type", "") if not isinstance(other, dict) else other.get("type", "")

            slug = slug_from_filename(filename_j)
            if slug:
                other_links.append(make_wikilink(type_j, slug))

        if other_links:
            try:
                data = store.read_memory(type_i, filename_i)
                content = data.get("body", "") if isinstance(data, dict) else str(data)
                if content and "## Related" not in content and "[[" not in content:
                    updated = content.rstrip() + "\n\n## Related\n" + " ".join(other_links) + "\n"
                    store.update_memory(type_i, filename_i, content=updated)
            except Exception:
                pass


def _add_cross_group_wikilinks(
    store: DreamStore,
    gather_result: "GatherResult",
    consolidate_result: "ConsolidateResult",
) -> None:
    """Add [[wikilinks]] from merged memories to other surviving memories with overlapping tags.

    After consolidation, merged memories may contain content from several sources
    but lack wikilinks to related surviving memories outside their group. This
    function scans all remaining vault memories, finds tag overlaps, and adds
    ``## Related`` sections with wikilinks to related memories.
    """
    from .store import slug_from_filename, make_wikilink

    # Reload the vault to get current state after pruning
    all_memories = store.list_memories()
    if len(all_memories) < 2:
        return

    # Build a tag -> memory mapping
    tag_to_memories: Dict[str, List[Dict]] = {}
    for mem in all_memories:
        for tag in mem.get("meta", {}).get("tags", []):
            tag_to_memories.setdefault(tag, []).append(mem)

    # For each memory, find related memories via shared tags
    for mem in all_memories:
        mem_type = mem.get("type", "")
        filename = mem.get("filename", "")
        tags = set(t.lower() for t in mem.get("meta", {}).get("tags", []))

        if not tags:
            continue

        # Collect related memories (those sharing at least 1 tag)
        related_links = []
        related_keys = set()
        for tag in tags:
            for related_mem in tag_to_memories.get(tag, []):
                r_type = related_mem.get("type", "")
                r_filename = related_mem.get("filename", "")
                r_key = f"{r_type}/{r_filename}"
                # Skip self
                if r_type == mem_type and r_filename == filename:
                    continue
                if r_key in related_keys:
                    continue
                related_keys.add(r_key)

                slug = slug_from_filename(r_filename)
                if slug:
                    related_links.append(make_wikilink(r_type, slug))

        if not related_links:
            continue

        # Only add if memory doesn't already have a Related section or wikilinks
        try:
            data = store.read_memory(mem_type, filename)
            content = data.get("body", "")
            if content and "## Related" not in content and "[[" not in content:
                updated = content.rstrip() + "\n\n## Related\n" + " ".join(related_links) + "\n"
                store.update_memory(mem_type, filename, content=updated)
                logger.debug("Cross-group wikilinks added to %s/%s", mem_type, filename)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Phase 3: Consolidate
# ---------------------------------------------------------------------------


def consolidate(
    store: DreamStore,
    gather_result: GatherResult,
    config: Optional[dict] = None,
    memory_type: Optional[str] = None,
) -> ConsolidateResult:
    """Apply deterministic consolidation rules.

    Actions:
    - Merge fragments in same-topic groups
    - Deduplicate identical/near-identical memories
    - Resolve contradictions (newer supersedes older)
    - Compress verbose memories exceeding max_lines
    - Boost relevance of frequently recalled memories
    """
    config = config or {}
    max_lines = config.get("max_lines", DEFAULT_MAX_LINES)
    max_bytes = config.get("max_bytes", DEFAULT_MAX_BYTES)

    result = ConsolidateResult()
    actions: List[ConsolidationAction] = []

    # Track files to delete (deduped and superseded)
    superseded: List[str] = []      # "type:filename"
    # Track merges
    merges: List[Dict[str, Any]] = []
    # Track caps
    capped: List[Tuple[str, str, int]] = []

    # --- Merge fragment groups ---
    for group in gather_result.groups:
        if memory_type and not group.group_id.startswith(memory_type + ":"):
            continue
        if len(group.entries) < 2:
            continue

        # Merge all entries' content into one
        merged_content_parts = []
        merged_tags: Set[str] = set()
        highest_relevance = 0.0
        newest_created = ""

        # Sort by creation date so newest content comes last
        sorted_entries = sorted(group.entries, key=lambda e: e.created)
        for entry in sorted_entries:
            merged_content_parts.append(entry.content.strip())
            merged_tags.update(entry.tags)
            highest_relevance = max(highest_relevance, entry.relevance)
            if not newest_created or entry.created > newest_created:
                newest_created = entry.created

        merged_content = "\n\n---\n\n".join(merged_content_parts)

        # Merge target: keep the newest file, delete others
        keep_entry = sorted_entries[-1]
        other_files = [f"{e.memory_type}:{e.filename}" for e in sorted_entries[:-1]]

        # Build wikilinks only for entries that will survive consolidation.
        # Entries that will be superseded (deleted) should not be linked,
        # as that would create dangling wikilinks after pruning.
        # For a full-group merge, only the keep_entry survives, but linking
        # to oneself is not useful — so we skip wikilinks when all sources
        # are being merged away.
        surviving_source_entries = [e for e in group.entries
                                    if e is keep_entry]
        if surviving_source_entries and len(surviving_source_entries) < len(group.entries):
            # Only the keep_entry survives; a self-wikilink is pointless.
            # We'll add cross-group wikilinks later in run_consolidation
            # via _add_bidirectional_wikilinks for related surviving memories.
            wikilinked_content = merged_content
        else:
            wikilinked_content = _add_wikilinks_to_merged_content(merged_content, surviving_source_entries, store)

        merges.append({
            "memory_type": keep_entry.memory_type,
            "filename": keep_entry.filename,
            "content": wikilinked_content,
            "tags": sorted(merged_tags),
            "relevance": min(1.0, highest_relevance + 0.05),
        })

        superseded.extend(other_files)
        result.merged_count += 1
        actions.append(ConsolidationAction(
            action="merge",
            target_type=keep_entry.memory_type,
            target_files=[f"{e.memory_type}:{e.filename}" for e in group.entries],
            result_file=f"{keep_entry.memory_type}:{keep_entry.filename}",
            details=f"Merged {len(group.entries)} fragments into {keep_entry.filename}",
        ))

    # --- Deduplicate ---
    deduped_files: Set[str] = set()
    for dup in gather_result.duplicates:
        if memory_type:
            if not dup.file_a.startswith(memory_type + ":"):
                continue
        # Determine which to keep (newer) and which to mark superseded (older)
        key_a = dup.file_a
        key_b = dup.file_b

        # Find entries
        entry_a = _find_entry(gather_result, key_a)
        entry_b = _find_entry(gather_result, key_b)

        if entry_a is None or entry_b is None:
            continue

        if _is_older(entry_a, entry_b):
            older_key = key_a
        else:
            older_key = key_b

        if older_key not in deduped_files:
            deduped_files.add(older_key)
            superseded.append(older_key)
            result.deduped_count += 1
            actions.append(ConsolidationAction(
                action="deduplicate",
                target_type=older_key.split(":")[0],
                target_files=[key_a, key_b],
                result_file=dup.file_b if older_key == key_a else dup.file_a,
                details=f"Deduplicated: {dup.similarity:.0%} similar, removed {older_key}",
            ))

    # --- Resolve contradictions ---
    for contra in gather_result.contradictions:
        if memory_type:
            if not contra.file_a.startswith(memory_type + ":"):
                continue
        # file_b (older) is superseded by file_a (newer)
        already_superseded = contra.file_b in superseded
        if not already_superseded:
            superseded.append(contra.file_b)
        result.pruned_count += 1
        actions.append(ConsolidationAction(
            action="contradict",
            target_type=contra.file_b.split(":")[0],
            target_files=[contra.file_a, contra.file_b],
            result_file=contra.file_a,
            details=f"Contradiction on '{contra.topic}': {contra.file_b} superseded by {contra.file_a}",
        ))

    # --- Compress verbose ---
    for entry in gather_result.entries:
        if memory_type and entry.memory_type != memory_type:
            continue
        type_spec = MEMORY_TYPES.get(entry.memory_type)
        entry_max_lines = type_spec.max_lines if type_spec else max_lines
        if entry.lines > entry_max_lines:
            # Queue for truncation during prune
            capped.append((entry.memory_type, entry.filename, entry_max_lines))
            actions.append(ConsolidationAction(
                action="compress",
                target_type=entry.memory_type,
                target_files=[f"{entry.memory_type}:{entry.filename}"],
                result_file=f"{entry.memory_type}:{entry.filename}",
                details=f"Compress {entry.lines} lines → {entry_max_lines}",
            ))
        elif entry.bytes > max_bytes:
            capped.append((entry.memory_type, entry.filename, max_lines))
            actions.append(ConsolidationAction(
                action="compress",
                target_type=entry.memory_type,
                target_files=[f"{entry.memory_type}:{entry.filename}"],
                result_file=f"{entry.memory_type}:{entry.filename}",
                details=f"Over byte limit: {entry.bytes} > {max_bytes}",
            ))

    # --- Boost relevance of frequently recalled ---
    # Memories with high relevance (>0.7) that appear in multiple groups
    # get a small relevance boost
    group_member_counts: Dict[str, int] = {}
    for group in gather_result.groups:
        for entry in group.entries:
            key = f"{entry.memory_type}:{entry.filename}"
            group_member_counts[key] = group_member_counts.get(key, 0) + 1

    for key, count in group_member_counts.items():
        if count >= 2:
            actions.append(ConsolidationAction(
                action="boost_relevance",
                target_type=key.split(":")[0],
                target_files=[key],
                result_file=key,
                details=f"Boost relevance: appears in {count} groups",
            ))

    # --- Forget: prune low-retention, low-importance, old memories ---
    # Ebbinghaus forgetting curve: memories with low retention, low importance,
    # and sufficient age are candidates for deletion.
    now = datetime.now(timezone.utc)
    for entry in gather_result.entries:
        if memory_type and entry.memory_type != memory_type:
            continue
        age_days = _compute_age_days(entry, now)
        ff = entry.forgetting_factor
        imp = entry.importance
        access_count = entry.access_count
        # Spacing effect: effective ff = base_ff × (0.9 ^ access_count)
        effective_ff = max(FORGETTING_FACTOR_MIN, ff * (FORGETTING_REACCESS_MULTIPLIER ** access_count))
        ret = retention_score(imp, effective_ff, age_days)

        if (
            ret < PRUNE_RETENTION_THRESHOLD
            and imp < PRUNE_MAX_IMPORTANCE
            and age_days > PRUNE_MIN_AGE_DAYS
        ):
            key = f"{entry.memory_type}:{entry.filename}"
            if key not in superseded:
                superseded.append(key)
                actions.append(ConsolidationAction(
                    action="forget",
                    target_type=entry.memory_type,
                    target_files=[key],
                    result_file="",
                    details=f"Forgotten: retention={ret:.3f}, importance={imp:.2f}, age={age_days:.0f}d",
                ))

    result.actions = actions
    result.superseded = superseded
    result.capped = capped
    result.merges = merges
    result.stats = {
        "total_actions": len(actions),
        "merged_count": result.merged_count,
        "deduped_count": result.deduped_count,
        "contradictions_resolved": result.pruned_count,
        "compressions_queued": len(capped),
    }

    return result


# ---------------------------------------------------------------------------
# Phase 3b: LLM-powered Consolidation
# ---------------------------------------------------------------------------


# Default model for LLM consolidation
DEFAULT_CONSOLIDATE_MODEL: str = "glm-5.1:cloud"


def _resolve_api_settings(config: dict) -> Tuple[str, str, str]:
    """Resolve API base_url, api_key, and model from config and env vars.

    Resolution order:
    1. Explicit keys in config: consolidate_base_url, consolidate_api_key, consolidate_model
    2. Environment variables: OPENROUTER_API_KEY / OPENAI_API_KEY for key,
       OPENAI_BASE_URL for base URL
    3. Defaults: OpenRouter base URL, model from config or DEFAULT_CONSOLIDATE_MODEL
    """
    # API key: config > OPENROUTER_API_KEY > OPENAI_API_KEY
    api_key = config.get("consolidate_api_key", "") or os.getenv("OPENROUTER_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")

    # Base URL: config > OPENAI_BASE_URL > default OpenRouter
    base_url = (
        config.get("consolidate_base_url", "")
        or os.getenv("OPENAI_BASE_URL", "")
        or "https://openrouter.ai/api/v1"
    )

    # Model: config > DEFAULT_CONSOLIDATE_MODEL
    model = config.get("consolidate_model", "") or DEFAULT_CONSOLIDATE_MODEL

    return base_url.rstrip("/"), api_key, model


def _format_memories_for_llm(entries: List[MemoryEntry]) -> str:
    """Format memory entries as a structured text block for the LLM prompt."""
    lines: List[str] = []
    for i, entry in enumerate(entries, 1):
        lines.append(f"--- Memory {i} ---")
        lines.append(f"Type: {entry.memory_type}")
        lines.append(f"Filename: {entry.filename}")
        lines.append(f"Tags: {', '.join(entry.tags) if entry.tags else '(none)'}")
        lines.append(f"Relevance: {entry.relevance:.2f}")
        lines.append(f"Created: {entry.created}")
        lines.append(f"Updated: {entry.updated}")
        lines.append(f"Content:\n{entry.content.strip()}")
        lines.append("")
    return "\n".join(lines)


def _build_llm_prompt(entries: List[MemoryEntry]) -> str:
    """Build the system + user prompt for LLM consolidation.

    The LLM is asked to analyse the memories and produce a JSON object
    describing merges, deduplications, contradiction resolutions, and
    compressions.
    """
    memories_text = _format_memories_for_llm(entries)

    system_prompt = """You are a memory consolidation engine for a CLI AI assistant. Your job is to distill noise into signal.

    The raw input includes ALL memories ever written — most are noise. A good consolidation REDUCES count significantly. 50 memories that compress to 5 is success. 50 memories that stay 50 is failure.

    You MUST respond with a JSON object (no markdown fences, just raw JSON) with
    this exact schema:

    {
      "actions": [
        {
          "action": "merge",
          "source_indices": [0, 2],
          "result_index": 0,
          "merged_content": "Consolidated content...",
          "merged_tags": ["tag1", "tag2"],
          "relevance": 0.8,
          "details": "..."
        },
        {
          "action": "deduplicate",
          "keep_index": 1,
          "remove_indices": [3],
          "details": "..."
        },
        {
          "action": "contradict",
          "newer_index": 5,
          "older_indices": [4],
          "details": "Memory 5 contradicts memory 4; newer version supersedes"
        },
        {
          "action": "compress",
          "target_index": 6,
          "compressed_content": "Shorter version...",
          "details": "Compressed verbose reference from 150 lines to 30"
        },
        {
          "action": "boost_relevance",
          "target_indices": [0],
          "new_relevance": 0.85,
          "details": "..."
        },
        {
          "action": "delete",
          "target_indices": [7, 8, 9],
          "details": "Session chronicles — no durable insight. 'worked on X' is not a memory."
        }
      ]
    }

    Rules:
    - Be AGGRESSIVE with deletion. If a memory is a session chronicle (worked on X, ran Y, created Z) with no decision or insight — DELETE it.
    - When merging, combine all unique facts. Include WHY decisions were made (decided_by relationships).
    - When resolving contradictions, keep the NEWER memory.
    - When deduplicating, keep the more detailed or recently updated version.
    - For compress, preserve key facts AND rationale. Compress the verbose, keep the signal.
    - boost_relevance only for memories that appear cross-session.
    - ONLY produce actions that add value. Empty actions array = memories were already good.
    - source_indices, keep_index, newer_index, target_indices, remove_indices, older_indices
      refer to 0-based indices in the input memory list.
    - merged_content or compressed_content must preserve ALL important information.
    - merged_tags should be the union of tags from all source memories.
    - If deleting more than half the memories, that's fine — that's the point."""

    user_prompt = f"Analyse these {len(entries)} memories and produce consolidation actions:\n\n{memories_text}"

    return system_prompt, user_prompt


def _parse_llm_response(
    response_text: str,
    entries: List[MemoryEntry],
) -> List[ConsolidationAction]:
    """Parse the LLM's JSON response into ConsolidationAction objects.

    Returns a list of ConsolidationAction objects derived from the LLM response.
    Updates the `merges`, `superseded`, and `capped` lists on the caller side.
    """
    actions: List[ConsolidationAction] = []

    # Strip markdown code fences if present
    text = response_text.strip()
    if text.startswith("```"):
        # Remove opening fence
        try:
            first_newline = text.index("\n")
            text = text[first_newline + 1:]
        except ValueError:
            # No newline in triple-backtick response; strip the fences
            text = text[3:]
        # Remove closing fence
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("LLM consolidation: failed to parse response as JSON, falling back to regex extraction")
        # Try to extract JSON from within the text
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                logger.error("LLM consolidation: could not extract valid JSON from response")
                return actions
        else:
            return actions

    raw_actions = data.get("actions", [])
    if not isinstance(raw_actions, list):
        logger.warning("LLM consolidation: 'actions' is not a list")
        return actions

    for raw in raw_actions:
        if not isinstance(raw, dict):
            continue

        action_type = raw.get("action", "")
        details = raw.get("details", "")

        try:
            if action_type == "merge":
                source_indices = raw.get("source_indices", [])
                result_index = raw.get("result_index", 0)
                merged_content = raw.get("merged_content", "")
                merged_tags = raw.get("merged_tags", [])
                relevance = float(raw.get("relevance", 0.5))

                # Resolve indices to actual entries
                source_entries = [entries[i] for i in source_indices if i < len(entries)]
                if not source_entries:
                    continue
                target = entries[result_index] if result_index < len(entries) else source_entries[0]

                target_files = [f"{e.memory_type}:{e.filename}" for e in source_entries]
                actions.append(ConsolidationAction(
                    action="merge",
                    target_type=target.memory_type,
                    target_files=target_files,
                    result_file=f"{target.memory_type}:{target.filename}",
                    details=details or f"Merged {len(source_entries)} memories",
                ))

            elif action_type == "deduplicate":
                keep_index = raw.get("keep_index", 0)
                remove_indices = raw.get("remove_indices", [])

                if keep_index >= len(entries):
                    continue

                keep_entry = entries[keep_index]
                remove_entries = [entries[i] for i in remove_indices if i < len(entries)]
                if not remove_entries:
                    continue

                target_files = [f"{keep_entry.memory_type}:{keep_entry.filename}"]
                target_files.extend(f"{e.memory_type}:{e.filename}" for e in remove_entries)

                for rem_entry in remove_entries:
                    actions.append(ConsolidationAction(
                        action="deduplicate",
                        target_type=rem_entry.memory_type,
                        target_files=[f"{keep_entry.memory_type}:{keep_entry.filename}", f"{rem_entry.memory_type}:{rem_entry.filename}"],
                        result_file=f"{keep_entry.memory_type}:{keep_entry.filename}",
                        details=details or f"Deduplicate: removed {rem_entry.filename}",
                    ))

            elif action_type == "contradict":
                newer_index = raw.get("newer_index", 0)
                older_indices = raw.get("older_indices", [])

                if newer_index >= len(entries):
                    continue

                newer_entry = entries[newer_index]
                older_entries = [entries[i] for i in older_indices if i < len(entries)]

                for old_entry in older_entries:
                    actions.append(ConsolidationAction(
                        action="contradict",
                        target_type=old_entry.memory_type,
                        target_files=[f"{newer_entry.memory_type}:{newer_entry.filename}", f"{old_entry.memory_type}:{old_entry.filename}"],
                        result_file=f"{newer_entry.memory_type}:{newer_entry.filename}",
                        details=details or f"Contradiction: {old_entry.filename} superseded by {newer_entry.filename}",
                    ))

            elif action_type == "compress":
                target_index = raw.get("target_index", 0)
                if target_index >= len(entries):
                    continue
                target = entries[target_index]
                # Compress content info is stored for later processing
                actions.append(ConsolidationAction(
                    action="compress",
                    target_type=target.memory_type,
                    target_files=[f"{target.memory_type}:{target.filename}"],
                    result_file=f"{target.memory_type}:{target.filename}",
                    details=details or f"Compress memory {target.filename}",
                ))

            elif action_type == "boost_relevance":
                target_indices = raw.get("target_indices", [])
                for idx in target_indices:
                    if idx < len(entries):
                        target = entries[idx]
                        new_relevance = float(raw.get("new_relevance", 0.7))
                        actions.append(ConsolidationAction(
                            action="boost_relevance",
                            target_type=target.memory_type,
                            target_files=[f"{target.memory_type}:{target.filename}"],
                            result_file=f"{target.memory_type}:{target.filename}",
                            details=details or f"Boost relevance to {new_relevance:.2f}",
                        ))

            elif action_type == "delete":
                target_indices = raw.get("target_indices", [])
                for idx in target_indices:
                    if idx < len(entries):
                        target = entries[idx]
                        actions.append(ConsolidationAction(
                            action="delete",
                            target_type=target.memory_type,
                            target_files=[f"{target.memory_type}:{target.filename}"],
                            result_file="",
                            details=details or f"Delete: no durable insight in {target.filename}",
                        ))

        except (IndexError, ValueError, TypeError) as exc:
            logger.warning("LLM consolidation: skipping invalid action %s: %s", raw, exc)
            continue

    return actions


def _extract_llm_merges(
    llm_actions: List[ConsolidationAction],
    entries: List[MemoryEntry],
    raw_response: dict,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Extract merge operations and superseded files from parsed LLM actions + raw response.

    For merge actions, we need the merged content/tags from the LLM response,
    which can't be reconstructed from ConsolidationAction alone.

    Returns (merges_list, superseded_list) where merges_list entries match the
    format expected by the prune phase.
    """
    merges: List[Dict[str, Any]] = []
    superseded: List[str] = []

    raw_actions = raw_response.get("actions", [])

    # Build a map from action index to raw action data
    for i, action in enumerate(llm_actions):
        if action.action != "merge":
            continue

        # Find the matching raw action data
        raw_match = None
        for raw in raw_actions:
            if raw.get("action") != "merge":
                continue
            # Match by result_file
            if raw.get("result_index") is not None and raw.get("result_index") < len(entries):
                expected_file = f"{entries[raw['result_index']].memory_type}:{entries[raw['result_index']].filename}"
                if expected_file == action.result_file:
                    raw_match = raw
                    break

        # Determine merge target
        result_key = action.result_file
        parts = result_key.split(":", 1)
        mem_type = parts[0] if len(parts) == 2 else ""
        filename = parts[1] if len(parts) == 2 else result_key

        if raw_match:
            # Use the LLM's merged content and tags
            merged_content = raw_match.get("merged_content", "")
            merged_tags = raw_match.get("merged_tags", [])
            relevance = float(raw_match.get("relevance", 0.5))
        else:
            # Fallback: concatenate source entry contents
            source_content_parts = []
            merged_tags_set: Set[str] = set()
            highest_relevance = 0.5
            for key in action.target_files:
                key_parts = key.split(":", 1)
                for entry in entries:
                    if entry.memory_type == key_parts[0] and entry.filename == key_parts[1]:
                        source_content_parts.append(entry.content.strip())
                        merged_tags_set.update(entry.tags)
                        highest_relevance = max(highest_relevance, entry.relevance)
            merged_content = "\n\n---\n\n".join(source_content_parts)
            merged_tags = sorted(merged_tags_set)
            relevance = min(1.0, highest_relevance + 0.05)

        merges.append({
            "memory_type": mem_type,
            "filename": filename,
            "content": merged_content,
            "tags": merged_tags,
            "relevance": relevance,
        })

        # All source files except the result file are superseded
        for key in action.target_files:
            if key != result_key:
                superseded.append(key)

    return merges, superseded


def llm_consolidate(
    memories: List[MemoryEntry],
    config: dict,
) -> ConsolidateResult:
    """Use an LLM to intelligently consolidate memories.

    Calls an OpenAI-compatible API to analyse memories and produce
    consolidation actions: merges, deduplications, contradiction
    resolutions, compressions, and relevance boosts.

    Parameters
    ----------
    memories:
        List of MemoryEntry objects loaded from the vault.
    config:
        Configuration dict.  Key entries:
        - consolidate_model: model identifier (default: 'glm-5.1:cloud')
        - consolidate_api_key: explicit API key (or env var fallback)
        - consolidate_base_url: explicit base URL (or env var fallback)
        - max_lines: per-type line cap for compress actions
        - max_bytes: byte cap for compress actions

    Returns
    -------
    ConsolidateResult
        Actions, merges, superseded, and capped lists — same shape as
        the deterministic consolidate() function.
    """
    result = ConsolidateResult()

    if not memories:
        result.stats = {"total_actions": 0, "mode": "llm"}
        return result

    # --- Resolve API settings ---
    base_url, api_key, model = _resolve_api_settings(config)

    if not api_key:
        logger.warning("LLM consolidation: no API key available, falling back to deterministic")
        result.stats = {"total_actions": 0, "mode": "llm_fallback_no_key"}
        return result

    # --- Build prompt ---
    system_prompt, user_prompt = _build_llm_prompt(memories)

    # --- Call the LLM ---
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        logger.info("LLM consolidation: calling model %s with %d memories", model, len(memories))

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
        )

        response_text = response.choices[0].message.content or ""

    except Exception as exc:
        logger.error("LLM consolidation API call failed: %s", exc)
        result.stats = {"total_actions": 0, "mode": "llm_error", "error": str(exc)}
        return result

    # --- Parse the response ---
    llm_actions = _parse_llm_response(response_text, memories)

    # --- Parse raw JSON for merge content extraction ---
    raw_response: dict = {}
    stripped = response_text.strip()
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        if first_nl != -1:
            stripped = stripped[first_nl + 1:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3].rstrip()
    try:
        raw_response = json.loads(stripped)
    except json.JSONDecodeError:
        json_match = re.search(r'\{[\s\S]*\}', stripped)
        if json_match:
            try:
                raw_response = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                raw_response = {}

    # --- Convert LLM actions to ConsolidateResult ---
    result.actions = llm_actions

    # Count action types
    result.merged_count = sum(1 for a in llm_actions if a.action == "merge")
    result.deduped_count = sum(1 for a in llm_actions if a.action == "deduplicate")
    result.pruned_count = sum(1 for a in llm_actions if a.action == "contradict")

    # Extract merges (with LLM-provided content)
    merges, superseded_from_merges = _extract_llm_merges(llm_actions, memories, raw_response)
    result.merges = merges

    # Build superseded list from dedup, contradict, delete, and merge actions
    superseded: List[str] = list(superseded_from_merges)
    for action in llm_actions:
        if action.action == "deduplicate":
            # The first file in target_files is the kept one; others are removed
            if len(action.target_files) > 1:
                for removed_key in action.target_files[1:]:
                    if removed_key not in superseded:
                        superseded.append(removed_key)
        elif action.action == "contradict":
            # The second file in target_files is the older/superseded one
            if len(action.target_files) > 1:
                older_key = action.target_files[1]
                if older_key not in superseded:
                    superseded.append(older_key)
        elif action.action == "delete":
            # All files in a delete action are removed
            for target_key in action.target_files:
                if target_key not in superseded:
                    superseded.append(target_key)
    result.superseded = superseded

    # Build capped list from compress actions
    max_lines = config.get("max_lines", DEFAULT_MAX_LINES)
    capped: List[Tuple[str, str, int]] = []
    for action in llm_actions:
        if action.action == "compress":
            key = action.result_file
            parts = key.split(":", 1)
            if len(parts) == 2:
                mem_type, filename = parts
                type_spec = MEMORY_TYPES.get(mem_type)
                file_max = type_spec.max_lines if type_spec else max_lines
                capped.append((mem_type, filename, file_max))
    result.capped = capped

    result.stats = {
        "total_actions": len(llm_actions),
        "merged_count": result.merged_count,
        "deduped_count": result.deduped_count,
        "contradictions_resolved": result.pruned_count,
        "deleted_count": sum(1 for a in llm_actions if a.action == "delete"),
        "compressions_queued": len(capped),
        "mode": "llm",
    }

    return result


# ---------------------------------------------------------------------------
# Phase 4: Prune
# ---------------------------------------------------------------------------


def prune(
    store: DreamStore,
    consolidate_result: ConsolidateResult,
    config: Optional[dict] = None,
    dry_run: bool = False,
) -> PruneResult:
    """Enforce file caps, delete superseded files, update manifest, write log."""
    config = config or {}
    max_lines = config.get("max_lines", DEFAULT_MAX_LINES)
    max_bytes = config.get("max_bytes", DEFAULT_MAX_BYTES)

    result = PruneResult()

    # --- Delete superseded files ---
    delete_skipped = 0
    for superseded_key in consolidate_result.superseded:
        parts = superseded_key.split(":", 1)
        if len(parts) != 2:
            continue
        mem_type, filename = parts
        if not dry_run:
            try:
                deleted = store.delete_memory(mem_type, filename)
                if deleted:
                    result.deleted_files.append(superseded_key)
                else:
                    # File already gone — likely deduplicated earlier in this run
                    delete_skipped += 1
                    logger.debug("Prune: skip delete %s — already gone", superseded_key)
            except Exception as exc:
                logger.warning("Prune: failed to delete %s: %s", superseded_key, exc)
        else:
            result.deleted_files.append(f"[dry-run] {superseded_key}")

    if delete_skipped > 0:
        logger.info("Prune: %d files already deleted (likely deduped earlier in run)", delete_skipped)

    # --- Apply merges (update merged content) ---
    for merge_info in consolidate_result.merges:
        if dry_run:
            result.capped_files.append(f"[dry-run] merge {merge_info['filename']}")
            continue
        try:
            mem_type = merge_info["memory_type"]
            filename = merge_info["filename"]
            # Guard: file may have been deleted as superseded since consolidate ran
            filepath = store.get_memory_path(mem_type, filename)
            if not filepath.exists():
                logger.debug("Prune: skip merge %s/%s — file already deleted (likely superseded)", mem_type, filename)
                continue
            store.update_memory(
                mem_type,
                filename,
                content=merge_info["content"],
                tags=merge_info["tags"],
                relevance=merge_info["relevance"],
            )
            result.capped_files.append(f"merged {mem_type}/{filename}")
        except Exception as exc:
            logger.warning("Prune: failed to merge %s: %s", merge_info["filename"], exc)

    # --- Cap oversized files ---
    for mem_type, filename, file_max_lines in consolidate_result.capped:
        if dry_run:
            result.capped_files.append(f"[dry-run] cap {mem_type}/{filename} to {file_max_lines} lines")
            continue
        try:
            data = store.read_memory(mem_type, filename)
            content = data.get("body", "")
            lines = content.split("\n")
            if len(lines) > file_max_lines:
                # Keep the FIRST max_lines — important context is at the top
                truncated = "\n".join(lines[:file_max_lines])
                store.update_memory(mem_type, filename, content=truncated)
                result.capped_files.append(f"capped {mem_type}/{filename}")
        except FileNotFoundError:
            # File was deleted as superseded since consolidate built the plan
            logger.debug("Prune: skip cap %s/%s — file already deleted", mem_type, filename)
        except Exception as exc:
            logger.warning("Prune: failed to cap %s/%s: %s", mem_type, filename, exc)

    # --- Update manifest ---
    if not dry_run:
        try:
            # Force manifest reload to pick up changes
            store._manifest = None
            store._ensure_manifest_loaded()
            result.manifest_updated = True
        except Exception as exc:
            logger.warning("Prune: manifest reload failed: %s", exc)

    # --- Write consolidation log (always, even for 0-action runs) ---
    if not dry_run:
        _write_consolidation_log(store, consolidate_result, result)

    result.log_written = True
    result.stats = {
        "deleted_count": len([f for f in result.deleted_files if not f.startswith("[dry-run]")]),
        "capped_count": len(result.capped_files),
        "manifest_updated": result.manifest_updated,
    }

    return result


# ---------------------------------------------------------------------------
# Cron integration entry point
# ---------------------------------------------------------------------------


def run_cron_consolidation(hermes_home: Optional[str] = None) -> str:
    """Entry point for scheduled (cron) consolidation runs.

    Reads the Dream plugin config from ``hermes_home/config.yaml``, creates
    a DreamStore, and runs the full 4-phase consolidation cycle.  Returns a
    human-readable summary string suitable for cron output logging.

    Parameters
    ----------
    hermes_home:
        Path to the Hermes home directory.  If *None*, falls back to
        ``hermes_constants.get_hermes_home()``.

    Returns
    -------
    str
        A summary of what happened (or why it was skipped).
    """
    from pathlib import Path as _Path

    # --- Resolve hermes_home ---
    if hermes_home is None:
        try:
            from hermes_constants import get_hermes_home
            hermes_home = str(get_hermes_home())
        except Exception as exc:
            return f"[dream cron] ERROR: cannot resolve hermes_home: {exc}"

    home = _Path(hermes_home)

    # --- Load plugin config from config.yaml ---
    config: dict = {}
    config_path = home / "config.yaml"
    if config_path.exists():
        try:
            import yaml as _yaml
            with open(config_path) as _f:
                all_config = _yaml.safe_load(_f) or {}
            config = all_config.get("plugins", {}).get("dream", {}) or {}
        except Exception as exc:
            logger.warning("run_cron_consolidation: failed to read config: %s", exc)
    else:
        logger.info("run_cron_consolidation: no config.yaml found at %s", config_path)

    # --- Resolve vault path ---
    vault_path_str = config.get("vault_path", str(home / "dream_vault"))
    vault_path_str = vault_path_str.replace("$HERMES_HOME", str(home))
    vault_path_str = vault_path_str.replace("${HERMES_HOME}", str(home))
    vault_path = _Path(vault_path_str)

    # Compute the actual dream root (vault_path / vault_subdir if set)
    vault_subdir = config.get("vault_subdir", "") or ""
    dream_root = (vault_path / vault_subdir) if vault_subdir else vault_path

    if not dream_root.exists():
        return f"[dream cron] SKIP: vault path does not exist: {vault_path}"

    # --- Create store and run consolidation ---
    store = DreamStore(vault_path, config=config)
    store.initialize()

    consolidation_config = {
        "max_lines": config.get("max_lines", 100),
        "max_bytes": config.get("max_bytes", 50000),
    }

    try:
        result = run_consolidation(
            store=store,
            config=consolidation_config,
            dry_run=False,
        )
    except Exception as exc:
        logger.error("run_cron_consolidation: consolidation failed: %s", exc)
        return f"[dream cron] ERROR: consolidation failed: {exc}"

    # --- Build summary ---
    orient = result.orient
    consolidate = result.consolidate
    prune = result.prune

    if not orient.needs_consolidation:
        return (
            f"[dream cron] SKIP: consolidation not needed "
            f"(reason: {orient.reason})"
        )

    summary_lines = [
        "[dream cron] Consolidation completed.",
        f"  Orient: {orient.reason}",
        f"  Gather: {result.gather.stats.get('entries_loaded', 0)} entries, "
        f"{result.gather.stats.get('groups_found', 0)} groups, "
        f"{result.gather.stats.get('duplicates_found', 0)} duplicates, "
        f"{result.gather.stats.get('contradictions_found', 0)} contradictions",
        f"  Consolidate: {consolidate.merged_count} merges, "
        f"{consolidate.deduped_count} dedupes, "
        f"{consolidate.pruned_count} contradictions resolved",
        f"  Prune: {len(prune.deleted_files)} deleted, "
        f"{len(prune.capped_files)} capped, "
        f"manifest_updated={prune.manifest_updated}",
    ]
    return "\n".join(summary_lines)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run_consolidation(
    store: DreamStore,
    config: Optional[dict] = None,
    dry_run: bool = False,
    memory_type: Optional[str] = None,
) -> ConsolidationResult:
    """Run the full 4-phase consolidation cycle.

    Parameters
    ----------
    store:
        The DreamStore instance to consolidate.
    config:
        Optional config dict with max_lines, max_bytes, etc.
        Supports ``consolidation_mode``: 'deterministic' (default) or 'llm'.
        When 'llm', calls an LLM for smarter merging decisions.
        Also supports ``consolidate_model``, ``consolidate_api_key``,
        and ``consolidate_base_url`` for LLM mode.
    dry_run:
        If True, report what would happen without making changes.
    memory_type:
        If specified, only consolidate this memory type.

    Returns
    -------
    ConsolidationResult
        Complete results from all 4 phases.
    """
    config = config or {}
    consolidation_mode = config.get("consolidation_mode", "deterministic")

    # Acquire consolidation lock to prevent concurrent runs
    lock = ConsolidationLock(store.dream_root)

    if not lock.acquire():
        # Already consolidating — return empty result
        result = ConsolidationResult(dry_run=dry_run)
        result.orient.reason = "consolidation already in progress"
        return result

    try:
        # Phase 1: Orient
        orient_result = orient(store)
        full_result = ConsolidationResult(orient=orient_result, dry_run=dry_run)

        if not orient_result.needs_consolidation and not dry_run:
            logger.info("Consolidation not needed: %s", orient_result.reason)
            return full_result

        # Phase 2: Gather
        gather_result = gather(store, orient_result, memory_type=memory_type)
        full_result.gather = gather_result

        if not gather_result.entries:
            logger.info("Consolidation: no entries to process")
            return full_result

        # Phase 3: Consolidate (determine actions)
        if consolidation_mode == "llm":
            logger.info("Consolidation: using LLM mode (model=%s)",
                         config.get("consolidate_model", DEFAULT_CONSOLIDATE_MODEL))
            consolidate_result = llm_consolidate(
                memories=gather_result.entries,
                config=config,
            )
            # If LLM consolidation produced no actions (error/fallback),
            # fall back to deterministic
            if not consolidate_result.actions and consolidate_result.stats.get("mode", "").startswith("llm_fallback"):
                logger.warning("LLM consolidation produced no results, falling back to deterministic")
                consolidate_result = consolidate(store, gather_result, config=config, memory_type=memory_type)
        else:
            consolidate_result = consolidate(store, gather_result, config=config, memory_type=memory_type)
        full_result.consolidate = consolidate_result

        if not consolidate_result.actions:
            # Even with no consolidation actions, still run prune to write log
            logger.info("Consolidation: no actions needed")
            prune_result = prune(store, consolidate_result, config=config, dry_run=dry_run)
            full_result.prune = prune_result
            # Reset session counter after successful consolidation (even with no actions)
            if not dry_run and prune_result.manifest_updated:
                store.reset_session_counter()
            return full_result

        # Phase 4: Prune (execute actions, or skip in dry_run)
        prune_result = prune(store, consolidate_result, config=config, dry_run=dry_run)
        full_result.prune = prune_result

        if dry_run:
            logger.info(
                "Consolidation dry-run: %d actions planned",
                len(consolidate_result.actions),
            )
        else:
            logger.info(
                "Consolidation complete: %d merges, %d dedupes, %d contradictions, %d capped",
                consolidate_result.merged_count,
                consolidate_result.deduped_count,
                consolidate_result.pruned_count,
                len(consolidate_result.capped),
            )
            # Reset session counter after successful consolidation
            if prune_result.manifest_updated:
                store.reset_session_counter()

            # Add bidirectional wikilinks for non-merged group members (capped at 5 per memory)
            # Cross-group wikilinks removed: caused exponential Related sections (80+ links/file)
            merged_keys = set()
            for merge_info in consolidate_result.merges:
                merged_keys.add(f"{merge_info.get('memory_type', '')}:{merge_info.get('filename', '')}")
            for superseded_key in consolidate_result.superseded:
                merged_keys.add(superseded_key)

            for group in gather_result.groups:
                if len(group.entries) < 2:
                    continue
                all_merged = all(
                    f"{e.memory_type}:{e.filename}" in merged_keys
                    for e in group.entries
                )
                if not all_merged:
                    try:
                        _add_bidirectional_wikilinks(group.entries, store, max_links=5)
                    except Exception as exc:
                        logger.debug("Bidirectional wikilinks failed for group %s: %s", group.group_id, exc)

        return full_result
    finally:
        lock.release()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _find_entry(gather_result: GatherResult, key: str) -> Optional[MemoryEntry]:
    """Look up an entry by 'type:filename' key in gather results."""
    parts = key.split(":", 1)
    if len(parts) != 2:
        return None
    mem_type, filename = parts
    for entry in gather_result.entries:
        if entry.memory_type == mem_type and entry.filename == filename:
            return entry
    return None


def _read_last_consolidation_ts(vault_path: Path) -> Optional[str]:
    """Read the last consolidation timestamp from the log file."""
    log_path = vault_path / "consolidation_log.json"
    if not log_path.exists():
        return None
    try:
        data = json.loads(log_path.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            return data[-1].get("timestamp", "")
        elif isinstance(data, dict):
            return data.get("timestamp", "")
    except Exception:
        pass
    return None


def _write_consolidation_log(
    store: DreamStore,
    consolidate_result: ConsolidateResult,
    prune_result: PruneResult,
) -> None:
    """Append a consolidation log entry."""
    log_path = store.dream_root / "consolidation_log.json"

    # Load existing log
    log_entries: List[Dict[str, Any]] = []
    if log_path.exists():
        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                log_entries = data
        except Exception:
            log_entries = []

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actions": [
            {
                "action": a.action,
                "target_type": a.target_type,
                "target_files": a.target_files,
                "result_file": a.result_file,
                "details": a.details,
            }
            for a in consolidate_result.actions
        ],
        "summary": {
            "merged": consolidate_result.merged_count,
            "deduped": consolidate_result.deduped_count,
            "contradictions": consolidate_result.pruned_count,
            "deleted": prune_result.stats.get("deleted_count", 0),
            "capped": prune_result.stats.get("capped_count", 0),
        },
    }

    log_entries.append(entry)

    # Keep only last 50 log entries
    log_entries = log_entries[-50:]

    log_path.write_text(
        json.dumps(log_entries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


"""Dream Memory Provider — AutoDream-inspired memory consolidation plugin.

Implements the MemoryProvider ABC to give the agent structured markdown
memories with taxonomy, manifest-based recall, and (Phase 2) per-turn
extraction, memory-write mirroring, and pre-compress rescue.

Config in $HERMES_HOME/config.yaml:
  plugins:
    dream:
      vault_path: $HERMES_HOME/dream_vault   # omit to use default
      max_lines: 100                          # per-memory line limit
      max_bytes: 50000                        # per-memory byte limit
      consolidate_model: ""                   # LLM model for Phase 4
      consolidate_cron: "0 3 * * *"           # cron for Phase 4
      taxonomy: true                          # enable taxonomy subdirs

Phase 3 adds:
  - dream_recall: manifest-based memory retrieval (no vector search)
  - dream_consolidate: stub for Phase 4 consolidation engine
  - prefetch / queue_prefetch: per-turn context injection
  - Updated system_prompt_block with vault status
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from .shared import load_dream_config as _load_plugin_config, resolve_vault_path as _resolve_vault_path
from .store import DreamStore
from .taxonomy import MEMORY_TYPES
from .extract import (
    CandidateMemory,
    extract_candidates,
    extract_candidates_from_messages,
    build_pre_compress_summary,
)
from .extract_llm import LLMExtractor, get_manifest_summary
from .recall import RecallEngine
from .consolidation import run_consolidation
from .consolidation import ConsolidationLock, ConsolidationLockError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cron job constants
# ---------------------------------------------------------------------------

_DREAM_CRON_JOB_NAME = "dream-consolidation"
_DREAM_CRON_JOB_SKILL = "dream"


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

DREAM_STATUS_SCHEMA = {
    "name": "dream_status",
    "description": (
        "Show Dream Memory vault statistics — count of memories per type, "
        "total memories, and vault path."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

DREAM_RECALL_SCHEMA = {
    "name": "dream_recall",
    "description": (
        "Recall relevant memories from the Dream vault using manifest-based "
        "selection. Scans memory frontmatter to find matches for the current "
        "query, returns top results by relevance."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for",
            },
            "memory_type": {
                "type": "string",
                "enum": ["user", "feedback", "project", "reference"],
                "description": "Filter by memory type",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default: 5)",
            },
        },
        "required": ["query"],
    },
}

DREAM_CONSOLIDATE_SCHEMA = {
    "name": "dream_consolidate",
    "description": (
        "Trigger memory consolidation (the 'dream' cycle). Runs Orient → "
        "Gather → Consolidate → Prune phases. Usually triggered by cron, "
        "but can be called on-demand to clean up fragmented or oversized "
        "memories."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "dry_run": {
                "type": "boolean",
                "description": "Preview changes without writing (default: false)",
            },
            "memory_type": {
                "type": "string",
                "enum": ["user", "feedback", "project", "reference"],
                "description": "Consolidate only this type",
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Relevance threshold for auto-store (candidates below this are logged only)
# ---------------------------------------------------------------------------

_MIN_RELEVANCE = 0.6

# Valid extraction modes for on_session_end
_VALID_EXTRACTION_MODES = {"regex", "llm", "both"}


# ---------------------------------------------------------------------------
# DreamMemoryProvider
# ---------------------------------------------------------------------------

class DreamMemoryProvider(MemoryProvider):
    """Dream Memory provider — structured markdown memories with taxonomy."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._store: DreamStore | None = None
        self._session_id: str = ""
        self._recall_engine: RecallEngine | None = None
        self._prefetch_cache: OrderedDict[str, str] = OrderedDict()
        self._prefetch_cache_max: int = 50
        self._pending_prefetch_query: str = ""
        self._auto_recall_budget: int = int(self._config.get("auto_recall_budget", 2048))
        self._auto_recall_top_k: int = int(self._config.get("auto_recall_top_k", 10))

        # Extraction mode: regex (fast), llm (quality), or both
        self._extraction_mode: str = self._config.get("extraction_mode", "llm").lower()
        if self._extraction_mode not in _VALID_EXTRACTION_MODES:
            logger.warning(
                "Invalid extraction_mode %r, defaulting to 'llm'",
                self._extraction_mode,
            )
            self._extraction_mode = "llm"

        # LLM extractor (created lazily, only when needed)
        self._llm_extractor: LLMExtractor | None = None
        if self._extraction_mode in ("llm", "both"):
            self._llm_extractor = LLMExtractor(self._config)

    @property
    def name(self) -> str:
        return "dream"

    # -- Core lifecycle ------------------------------------------------------

    def is_available(self) -> bool:
        """Dream is always available — it only needs the filesystem."""
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._store = DreamStore(_resolve_vault_path(self._config), config=self._config)
        self._store.initialize()
        self._session_id = session_id
        self._recall_engine = RecallEngine(self._store)

        # Track session count for consolidation gate
        self._store.increment_session_counter()

        logger.info("Dream Memory initialised — vault at %s", self._store.vault_path)

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        stats = self._store.stats()
        total = stats["total"]
        if total == 0:
            return (
                "Dream Memory active. Empty vault — memories will be captured "
                "from conversations. Use dream_recall for targeted queries."
            )
        counts = stats["counts"]
        u = counts.get("user", 0)
        f = counts.get("feedback", 0)
        p = counts.get("project", 0)
        r = counts.get("reference", 0)

        parts = [f"Dream Memory active. {total} memories stored ({u}U/{f}F/{p}P/{r}R)."]
        parts.append("Use dream_recall for targeted queries.")

        # If auto_recall is enabled, mention passive availability
        auto_recall = self._config_as_bool("auto_recall")
        if auto_recall:
            parts.append("Relevant memories are automatically injected each turn.")

        return " ".join(parts)

    # -- Tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [DREAM_STATUS_SCHEMA, DREAM_RECALL_SCHEMA, DREAM_CONSOLIDATE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "dream_status":
            return self._handle_dream_status(args)
        elif tool_name == "dream_recall":
            return self._handle_dream_recall(args)
        elif tool_name == "dream_consolidate":
            return self._handle_dream_consolidate(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def _handle_dream_status(self, args: dict) -> str:
        if not self._store:
            return json.dumps({"error": "Dream store not initialised"})
        stats = self._store.stats()
        return json.dumps(stats)

    def _handle_dream_recall(self, args: dict) -> str:
        """Handle dream_recall tool call."""
        if not self._store or not self._recall_engine:
            return json.dumps({"error": "Dream store not initialised"})

        query = args.get("query", "")
        memory_type = args.get("memory_type")
        limit = args.get("limit", 5)
        if isinstance(limit, str):
            try:
                limit = int(limit)
            except ValueError:
                limit = 5

        if not query or not query.strip():
            return json.dumps({"error": "query is required", "results": []})

        results = self._recall_engine.recall(
            query=query,
            memory_type=memory_type,
            limit=limit,
        )

        # Format results for tool response
        out = []
        for r in results:
            out.append({
                "memory_type": r.memory_type,
                "filename": r.filename,
                "content": r.content[:500],  # truncate for tool response
                "tags": r.frontmatter.get("tags", []),
                "relevance": r.frontmatter.get("relevance", 0.5),
                "score": r.score,
            })

        return json.dumps({"query": query, "results": out})

    def _handle_dream_consolidate(self, args: dict) -> str:
        """Run the 4-phase consolidation cycle (Orient → Gather → Consolidate → Prune)."""
        if not self._store:
            return json.dumps({"error": "Dream store not initialised"})

        dry_run = args.get("dry_run", False)
        memory_type = args.get("memory_type")

        consolidation_config = {
            "max_lines": self._config.get("max_lines", 100),
            "max_bytes": self._config.get("max_bytes", 50000),
            "consolidation_mode": self._config.get("consolidation_mode", "deterministic"),
            "consolidate_model": self._config.get("consolidate_model", ""),
            "consolidate_api_key": self._config.get("consolidate_api_key", ""),
            "consolidate_base_url": self._config.get("consolidate_base_url", ""),
        }

        try:
            result = run_consolidation(
                store=self._store,
                config=consolidation_config,
                dry_run=dry_run,
                memory_type=memory_type,
            )
        except Exception as exc:
            logger.error("Dream consolidation failed: %s", exc)
            return json.dumps({"error": str(exc), "status": "failed"})

        # Build action log for tool response
        action_log = []
        for action in result.consolidate.actions:
            action_log.append({
                "action": action.action,
                "target_type": action.target_type,
                "target_files": action.target_files,
                "result_file": action.result_file,
                "details": action.details,
            })

        response = {
            "status": "completed" if not dry_run else "dry_run",
            "dry_run": dry_run,
            "memory_type": memory_type,
            "orient": {
                "needs_consolidation": result.orient.needs_consolidation,
                "reason": result.orient.reason,
                "stale_files": len(result.orient.stale_files),
                "oversized_files": len(result.orient.oversized_files),
            },
            "gather": {
                "entries_loaded": result.gather.stats.get("entries_loaded", 0),
                "groups_found": result.gather.stats.get("groups_found", 0),
                "duplicates_found": result.gather.stats.get("duplicates_found", 0),
                "contradictions_found": result.gather.stats.get("contradictions_found", 0),
            },
            "consolidate": {
                "total_actions": len(action_log),
                "merged": result.consolidate.merged_count,
                "deduped": result.consolidate.deduped_count,
                "contradictions_resolved": result.consolidate.pruned_count,
                "actions": action_log,
            },
            "prune": {
                "deleted_files": len(result.prune.deleted_files),
                "capped_files": len(result.prune.capped_files),
                "manifest_updated": result.prune.manifest_updated,
            },
        }

        logger.info(
            "Dream consolidation: %s — %d actions (%d merges, %d dedupes, %d contradictions)",
            "dry-run" if dry_run else "completed",
            len(action_log),
            result.consolidate.merged_count,
            result.consolidate.deduped_count,
            result.consolidate.pruned_count,
        )

        return json.dumps(response, default=str)

    # -- Optional hooks (Phase 2 implementations) --------------------------

    def _config_as_bool(self, key: str) -> bool:
        """Read a config key as a boolean, handling string 'true'/'false'."""
        val = self._config.get(key, False)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        return bool(val)

    def should_auto_recall(self) -> bool:
        """Return True if auto-recall (passive memory injection) is enabled.

        When True, MemoryManager should inject prefetch results into every
        turn's context. When False (default), prefetch results are only
        available via explicit dream_recall tool calls.
        """
        return self._config_as_bool("auto_recall")

    def prefetch(self, query: str, *, session_id: str = "", budget: int = None) -> str:
        """Phase 3 manifest-based recall for context injection.

        Returns a formatted context block with relevant memories,
        capped to the configured budget (default 2048 bytes).

        This method is called by MemoryManager.prefetch_all() every turn.
        When auto_recall is disabled (default: False), this method still
        works but the results are only consumed when the agent explicitly
        calls dream_recall. MemoryManager.inject_auto_recall() handles
        the gating of whether to include prefetch results in the user message.
        When auto_recall is enabled, context is passively injected every turn.
        """
        if not self._store or not self._recall_engine:
            return ""

        budget = budget or self._auto_recall_budget

        # Check cache
        cache_key = f"{session_id}:{query}"
        if cache_key in self._prefetch_cache:
            return self._prefetch_cache[cache_key]

        try:
            # Get more candidates than we need, then trim to budget
            results = self._recall_engine.recall(query, limit=self._auto_recall_top_k)
        except Exception as exc:
            logger.warning("Dream prefetch recall failed: %s", exc)
            return ""

        if not results:
            if len(self._prefetch_cache) >= self._prefetch_cache_max:
                self._prefetch_cache.popitem(last=False)
            self._prefetch_cache[cache_key] = ""
            return ""

        # Build output, prioritizing feedback type, then by score
        # Sort: feedback first, then by score descending
        sorted_results = sorted(results, key=lambda r: (0 if r.memory_type == "feedback" else 1, -r.score))

        lines = ["## Dream Memory"]
        current_size = len("## Dream Memory\n")
        type_labels = {"user": "User", "feedback": "Feedback", "project": "Project", "reference": "Reference"}

        for r in sorted_results:
            label = type_labels.get(r.memory_type, r.memory_type.capitalize())
            prefix = "↳" if getattr(r, 'is_related', False) else "•"
            snippet = r.content[:150].replace("\n", " ").strip()
            if len(r.content) > 150:
                snippet += "…"
            line = f"{prefix} **{label}** ({r.score:.2f}): {snippet}"
            line_size = len(line) + 1  # +1 for newline

            if current_size + line_size > budget:
                break  # Budget exceeded

            lines.append(line)
            current_size += line_size

        if len(lines) == 1:  # Only the header
            if len(self._prefetch_cache) >= self._prefetch_cache_max:
                self._prefetch_cache.popitem(last=False)
            self._prefetch_cache[cache_key] = ""
            return ""

        block = "\n".join(lines) + "\n"
        if len(self._prefetch_cache) >= self._prefetch_cache_max:
            self._prefetch_cache.popitem(last=False)
        self._prefetch_cache[cache_key] = block
        return block

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Store the query for background pre-computation next turn."""
        self._pending_prefetch_query = query

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Extract candidate memories from a completed turn and store them.

        Calls :func:`extract_candidates` on the turn content, then writes
        any candidate with relevance >= 0.6 to the dream store.
        """
        if not self._store:
            return

        try:
            candidates = extract_candidates(user_content, assistant_content)
            if not candidates:
                return

            stored = 0
            for c in candidates:
                if c.relevance >= _MIN_RELEVANCE:
                    self._store.add_memory(
                        memory_type=c.type,
                        content=c.content,
                        tags=c.tags,
                        source=session_id or self._session_id,
                        relevance=c.relevance,
                        importance=c.importance,
                    )
                    stored += 1

            logger.info(
                "Dream sync_turn: extracted %d candidates, stored %d (threshold=%.1f)",
                len(candidates), stored, _MIN_RELEVANCE,
            )
        except Exception as exc:
            logger.warning("Dream sync_turn failed: %s", exc)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes to dream taxonomy format.

        Ensures the dream vault stays in sync with MEMORY.md / USER.md writes.

        Parameters
        ----------
        action:
            ``add``, ``replace``, or ``remove``.
        target:
            ``memory`` (MEMORY.md) or ``user`` (USER.md).
        content:
            The written content string.
        """
        if not self._store:
            return

        try:
            if action == "add":
                # Map target to dream memory type
                mem_type = "user" if target == "user" else "project"
                self._store.add_memory(
                    memory_type=mem_type,
                    content=content,
                    tags=["builtin-mirror"],
                    source=self._session_id,
                    relevance=0.8,
                )
                logger.debug("Dream on_memory_write: add %s → %s", target, mem_type)

            elif action == "replace":
                # Find the closest existing dream memory by content match and update it
                mem_type = "user" if target == "user" else "project"
                matching = self._store.list_memories(memory_type=mem_type)
                best_match = None
                best_similarity = 0.0
                content_norm = content.strip().lower()
                for entry in matching:
                    body_norm = entry.get("body", "").strip().lower()
                    # Exact match first
                    if content_norm == body_norm:
                        best_match = entry
                        best_similarity = 1.0
                        break
                    # Jaccard word overlap similarity for near-matches
                    content_words = set(content_norm.split())
                    body_words = set(body_norm.split())
                    if content_words and body_words:
                        intersection = content_words & body_words
                        union = content_words | body_words
                        similarity = len(intersection) / len(union)
                        if similarity > best_similarity and similarity > 0.8:
                            best_similarity = similarity
                            best_match = entry
                if best_match:
                    filename = best_match.get("filename", "")
                    if filename:
                        self._store.update_memory(
                            mem_type,
                            filename,
                            content=content,
                            tags=["builtin-mirror", "replaced"],
                        )
                        logger.debug(
                            "Dream on_memory_write: replaced %s/%s (similarity=%.2f)",
                            mem_type, filename, best_similarity,
                        )
                        return  # replace first match only
                # If no match found, add as new
                self._store.add_memory(
                    memory_type=mem_type,
                    content=content,
                    tags=["builtin-mirror"],
                    source=self._session_id,
                    relevance=0.8,
                )
                logger.debug("Dream on_memory_write: replace (no match) → add %s", mem_type)

            elif action == "remove":
                # Find and delete matching dream memories
                removed = 0
                for mem_type in MEMORY_TYPES:
                    matching = self._store.list_memories(memory_type=mem_type)
                    for entry in matching:
                        if content.strip().lower() in entry.get("body", "").strip().lower():
                            filename = entry.get("filename", "")
                            if filename:
                                self._store.delete_memory(mem_type, filename)
                                removed += 1
                if removed:
                    logger.debug("Dream on_memory_write: removed %d matching memories", removed)

        except Exception as exc:
            logger.warning("Dream on_memory_write failed: %s", exc)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Extract candidate memories from all messages at session end.

        Behaviour depends on ``extraction_mode`` config key:
          - ``regex``: Use regex-based extraction only (legacy behaviour).
          - ``llm``: Use LLM-powered extraction only (default, higher quality).
            Falls back to regex if the LLM call fails (no API key, timeout, etc).
          - ``both``: Run both extractors and merge results.
        """
        if not self._store or not messages:
            return

        try:
            all_candidates: List[CandidateMemory] = []
            mode = self._extraction_mode
            llm_candidates: List[CandidateMemory] = []

            if mode in ("regex", "both"):
                regex_candidates = extract_candidates_from_messages(messages)
                all_candidates.extend(regex_candidates)

            if mode in ("llm", "both") and self._llm_extractor is not None:
                manifest_summary = get_manifest_summary(self._store)
                llm_candidates = self._llm_extractor.extract(
                    session_id=self._session_id,
                    messages=messages,
                    manifest_summary=manifest_summary,
                )
                all_candidates.extend(llm_candidates)

                # Fallback: if LLM returned nothing and we haven't run regex,
                # fall back to regex extraction so we don't lose the session
                if mode == "llm" and not llm_candidates:
                    logger.info(
                        "Dream on_session_end: LLM extraction produced no results, "
                        "falling back to regex extraction"
                    )
                    regex_candidates = extract_candidates_from_messages(messages)
                    all_candidates.extend(regex_candidates)
                    mode = "regex"  # Update mode tagging to reflect regex source
            elif mode == "llm":
                # LLM extractor is None (shouldn't normally happen, but be safe)
                # Fall back to regex
                logger.warning(
                    "Dream on_session_end: LLM extractor not available, "
                    "falling back to regex extraction"
                )
                regex_candidates = extract_candidates_from_messages(messages)
                all_candidates.extend(regex_candidates)
                mode = "regex"

            if not all_candidates:
                return

            # Deduplicate candidates by normalised content + type
            seen = set()
            unique_candidates: List[CandidateMemory] = []
            # Track LLM-sourced content for tagging
            llm_keys = set()
            if mode == "both":
                for c in llm_candidates:
                    llm_keys.add((c.type, c.content.strip().lower()))
            elif mode == "llm":
                # All candidates are LLM-sourced
                for c in all_candidates:
                    llm_keys.add((c.type, c.content.strip().lower()))

            for c in all_candidates:
                key = (c.type, c.content.strip().lower())
                if key not in seen:
                    seen.add(key)
                    unique_candidates.append(c)

            stored = 0
            for c in unique_candidates:
                if c.relevance >= _MIN_RELEVANCE:
                    c_tags = list(c.tags)
                    key = (c.type, c.content.strip().lower())
                    if key in llm_keys:
                        c_tags.append("session-end-llm")
                    else:
                        c_tags.append("session-end")

                    self._store.add_memory(
                        memory_type=c.type,
                        content=c.content,
                        tags=c_tags,
                        source=self._session_id,
                        relevance=c.relevance,
                        importance=c.importance,
                    )
                    stored += 1

            logger.info(
                "Dream on_session_end (mode=%s): extracted %d candidates, stored %d",
                mode, len(unique_candidates), stored,
            )
        except Exception as exc:
            logger.warning("Dream on_session_end failed: %s", exc)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract insights from messages about to be compressed.

        Returns a summary string for the compressor to preserve.  Also
        writes any new candidates to the dream store.
        """
        if not messages:
            return ""

        try:
            # Build the compression-preservation summary
            summary = build_pre_compress_summary(messages)

            # Also persist high-confidence candidates to store
            if self._store:
                candidates = extract_candidates_from_messages(messages)
                stored = 0
                for c in candidates:
                    if c.relevance >= _MIN_RELEVANCE:
                        self._store.add_memory(
                            memory_type=c.type,
                            content=c.content,
                            tags=c.tags + ["pre-compress"],
                            source=self._session_id,
                            relevance=c.relevance,
                            importance=c.importance,
                        )
                        stored += 1
                if stored:
                    logger.info(
                        "Dream on_pre_compress: stored %d candidates from %d messages",
                        stored, len(messages),
                    )

            return summary
        except Exception as exc:
            logger.warning("Dream on_pre_compress failed: %s", exc)
            return ""

    def shutdown(self) -> None:
        """Clean shutdown."""
        self._store = None
        self._recall_engine = None

    # -- Cron integration ---------------------------------------------------

    def setup_cron(self) -> dict:
        """Create or update a Hermes cron job for nightly dream consolidation.

        Uses the ``consolidate_cron`` config key (default ``0 3 * * *``) to
        determine the schedule.  The job uses the ``dream`` skill with a
        self-contained prompt that triggers consolidation.

        Returns
        -------
        dict
            The created job dict (from ``cron.jobs.create_job``), or an
            error dict on failure.
        """
        from hermes_constants import get_hermes_home

        schedule = self._config.get("consolidate_cron", "0 3 * * *").strip()
        if not schedule:
            schedule = "0 3 * * *"

        try:
            from cron.jobs import create_job, list_jobs

            # Remove any existing dream consolidation job first (idempotent setup)
            remove_consolidation_cron(str(get_hermes_home()))

            job = create_job(
                prompt=(
                    "Run the nightly Dream Memory consolidation cycle. "
                    "Use the dream_consolidate tool to perform Orient → "
                    "Gather → Consolidate → Prune on all memory types. "
                    "Report a brief summary of actions taken."
                ),
                schedule=schedule,
                name=_DREAM_CRON_JOB_NAME,
                skills=[_DREAM_CRON_JOB_SKILL],
            )
            logger.info(
                "Dream cron job created: id=%s schedule='%s'",
                job.get("id", "?"),
                schedule,
            )
            return job

        except Exception as exc:
            logger.error("Dream setup_cron failed: %s", exc)
            return {"error": str(exc), "status": "failed"}

    def remove_cron(self) -> dict:
        """Remove the dream consolidation cron job if it exists.

        Returns
        -------
        dict
            ``{"removed": True}`` if a job was found and removed, or
            ``{"removed": False, "reason": "..."}`` otherwise.
        """
        from hermes_constants import get_hermes_home

        result = remove_consolidation_cron(str(get_hermes_home()))
        if result:
            logger.info("Dream cron job removed.")
            return {"removed": True}
        else:
            logger.info("No dream cron job found to remove.")
            return {"removed": False, "reason": "no matching job found"}

    def cron_status(self) -> dict:
        """Check whether a dream consolidation cron job exists.

        Returns
        -------
        dict
            Status info including ``enabled``, ``schedule``, and ``next_run_at``
            if a job is found, or ``{"exists": False}`` otherwise.
        """
        try:
            from cron.jobs import list_jobs

            for job in list_jobs(include_disabled=True):
                if job.get("name") == _DREAM_CRON_JOB_NAME:
                    return {
                        "exists": True,
                        "enabled": job.get("enabled", True),
                        "schedule": job.get("schedule_display", ""),
                        "next_run_at": job.get("next_run_at", ""),
                        "last_run_at": job.get("last_run_at", ""),
                        "last_status": job.get("last_status", ""),
                        "job_id": job.get("id", ""),
                    }
            return {"exists": False}

        except Exception as exc:
            logger.error("Dream cron_status failed: %s", exc)
            return {"exists": False, "error": str(exc)}

    # -- Config management ---------------------------------------------------

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write config to config.yaml under plugins.dream section.

        Masks consolidate_api_key with '***' before writing to disk.
        """
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing: dict = {}
            if config_path.exists():
                with open(config_path) as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            # Mask API key before writing to disk
            safe_values = dict(values)
            if safe_values.get("consolidate_api_key") and safe_values["consolidate_api_key"] != "***":
                safe_values["consolidate_api_key"] = "***"
            existing["plugins"]["dream"] = safe_values
            with open(config_path, "w") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception as exc:
            logger.warning("Dream save_config failed: %s", exc)

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Return config fields for ``hermes memory setup``."""
        from hermes_constants import display_hermes_home
        _default_vault = f"{display_hermes_home()}/dream_vault"
        return [
            {
                "key": "vault_path",
                "description": "Root directory for dream memory vault",
                "default": _default_vault,
            },
            {
                "key": "vault_subdir",
                "description": "Subdirectory within vault_path where dream memories are stored (empty = vault_path itself)",
                "default": "",
            },
            {
                "key": "max_lines",
                "description": "Maximum lines per memory document",
                "default": "100",
            },
            {
                "key": "max_bytes",
                "description": "Maximum bytes per memory document",
                "default": "50000",
            },
            {
                "key": "consolidate_model",
                "description": "LLM model for Phase 4 consolidation (empty = disabled)",
                "default": "",
            },
            {
                "key": "consolidation_mode",
                "description": "Consolidation mode: 'deterministic' (default) or 'llm'",
                "default": "deterministic",
                "choices": ["deterministic", "llm"],
            },
            {
                "key": "consolidate_api_key",
                "description": "API key for LLM consolidation (env CONSOLIDATE_API_KEY or OPENROUTER_API_KEY preferred)",
                "default": "",
                "secret": True,
            },
            {
                "key": "consolidate_base_url",
                "description": "API base URL for LLM consolidation (overrides OPENAI_BASE_URL env)",
                "default": "",
            },
            {
                "key": "consolidate_cron",
                "description": "Cron schedule for Phase 4 consolidation",
                "default": "0 3 * * *",
            },
            {
                "key": "taxonomy",
                "description": "Enable taxonomy-based subdirectories",
                "default": "true",
                "choices": ["true", "false"],
            },
            {
                "key": "min_sessions_for_consolidation",
                "description": "Minimum sessions before consolidation fires (Anthropic default: 5)",
                "default": "5",
            },
            {
                "key": "auto_recall",
                "description": "Enable per-turn automatic memory injection from Dream vault",
                "default": "false",
                "choices": ["true", "false"],
            },
            {
                "key": "auto_recall_budget",
                "description": "Max bytes injected per auto-recall turn (default: 2048)",
                "default": "2048",
            },
            {
                "key": "auto_recall_top_k",
                "description": "Max memories considered per auto-recall (default: 10)",
                "default": "10",
            },
            {
                "key": "forgetting_factor_default",
                "description": "Default forgetting factor (0.005=slow, 0.02=moderate, 0.05=fast decay)",
                "default": "0.02",
            },
            {
                "key": "prune_retention_threshold",
                "description": "Retention below which low-importance memories are forgotten (default: 0.1)",
                "default": "0.1",
            },
            {
                "key": "extraction_mode",
                "description": "Session-end extraction mode: 'regex' (fast), 'llm' (quality, default), or 'both'",
                "default": "llm",
                "choices": ["regex", "llm", "both"],
            },
            {
                "key": "extraction_model",
                "description": "LLM model for session-end extraction (empty = uses consolidate_model or default)",
                "default": "",
            },
            {
                "key": "extraction_api_key",
                "description": "API key for LLM extraction (env EXTRACTION_API_KEY or falls back to consolidate keys)",
                "default": "",
                "secret": True,
            },
            {
                "key": "extraction_base_url",
                "description": "API base URL for LLM extraction (overrides consolidate_base_url and env)",
                "default": "",
            },
        ]


# ---------------------------------------------------------------------------
# Standalone cron helpers (callable from CLI or external tools)
# ---------------------------------------------------------------------------


def setup_consolidation_cron(hermes_home: str = None) -> dict:
    """Create or update the dream consolidation cron job.

    This is a standalone function that can be called from the CLI, a script,
    or the cron system itself.  It does not require a DreamMemoryProvider
    instance.

    Parameters
    ----------
    hermes_home:
        Path to the Hermes home directory.  If *None*, falls back to
        ``hermes_constants.get_hermes_home()``.

    Returns
    -------
    dict
        The created job dict, or an error dict on failure.
    """
    from hermes_constants import get_hermes_home

    if hermes_home is None:
        hermes_home = str(get_hermes_home())

    provider = DreamMemoryProvider(config=_load_plugin_config())
    return provider.setup_cron()


def remove_consolidation_cron(hermes_home: str = None) -> bool:
    """Remove the dream consolidation cron job.

    This is a standalone function that can be called from the CLI, a script,
    or the cron system itself.

    Parameters
    ----------
    hermes_home:
        Path to the Hermes home directory.  If *None*, falls back to
        ``hermes_constants.get_hermes_home()``.

    Returns
    -------
    bool
        ``True`` if a matching job was found and removed, ``False`` otherwise.
    """
    from cron.jobs import list_jobs, remove_job

    removed = False
    for job in list_jobs(include_disabled=True):
        if job.get("name") == _DREAM_CRON_JOB_NAME:
            remove_job(job["id"])
            removed = True
    return removed


def get_consolidation_cron_status(hermes_home: str = None) -> dict:
    """Check whether a dream consolidation cron job exists.

    Parameters
    ----------
    hermes_home:
        Path to the Hermes home directory.  Unused but kept for API
        consistency with the other standalone helpers.

    Returns
    -------
    dict
        Status dict with ``exists``, ``enabled``, ``schedule``, etc.
    """
    provider = DreamMemoryProvider(config=_load_plugin_config())
    return provider.cron_status()


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the Dream memory provider with the plugin system."""
    config = _load_plugin_config()
    provider = DreamMemoryProvider(config=config)
    ctx.register_memory_provider(provider)
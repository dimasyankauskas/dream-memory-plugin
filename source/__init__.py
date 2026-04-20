"""Dream v2 Memory Provider — Agent Consciousness Plugin.

A memory provider plugin that gives the agent an accumulated self-model —
the persistent layer that makes Garuda feel continuous across sessions.

This is NOT a skill — it's firmware settings / consciousness.
Starts minimal, grows over time.

Vault structure:
  /consciousness/self/     — what the agent knows about itself
  /consciousness/relationship/  — learnings about the user
  /consciousness/work/    — project/tactical learnings
  /decisions/             — explicit agreements and decisions
  /feedback/              — corrections and directives
  /reference/             — stable facts: APIs, tools, paths

Config in $HERMES_HOME/config.yaml:
  plugins:
    dream_v2:
      vault_path: /path/to/vault
      extraction_model: glm-5.1:agentic
      consolidation_model: glm-5.1:agentic
      extraction_mode: llm
      hybrid_mode: true
      max_memories_per_session: 3
      significance_threshold: 0.7
      max_lines_per_file: 200
      auto_recall: false
      consolidation_mode: manual
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

DREAM_STATUS_SCHEMA = {
    "name": "dream_status",
    "description": (
        "Show Dream consciousness vault statistics — count of memories per type, "
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
        "Recall relevant memories from the Dream consciousness vault. "
        "Scans manifest for tag + keyword matches, returns top results scored by "
        "relevance, recency, and importance.\n\n"
        "Use when: answering questions about past agreements, looking up user "
        "preferences, finding technical decisions, or checking what the agent "
        "has learned about itself or its work.\n\n"
        "Do NOT use for: session search (use session_search tool instead)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for — keywords, tags, or concepts",
            },
            "memory_type": {
                "type": "string",
                "description": "Filter by memory type: consciousness/self, consciousness/relationship, consciousness/work, decisions, feedback, or reference",
                "enum": ["consciousness/self", "consciousness/relationship", "consciousness/work", "decisions", "feedback", "reference"],
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default: 5, max: 20)",
            },
        },
        "required": ["query"],
    },
}

DREAM_CONSOLIDATE_SCHEMA = {
    "name": "dream_consolidate",
    "description": (
        "Run Dream consolidation: audit the vault, merge duplicates, prune "
        "stale entries, and rebuild the MEMORY.md index. "
        "Manual trigger only — not needed frequently."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Memory Provider implementation
# ---------------------------------------------------------------------------

class DreamV2MemoryProvider(MemoryProvider):
    """Memory provider implementing Dream v2 consciousness architecture."""

    def __init__(self, config: Dict[str, Any]):
        # Don't call parent __init__ — ABC abstract
        self._config = config
        self._vault_path = Path(config.get("vault_path", str(Path.home() / ".hermes" / "dream_v2")))
        self._max_per_session = int(config.get("max_memories_per_session", 3))
        self._significance_threshold = float(config.get("significance_threshold", 0.7))
        self._hybrid_mode = config.get("hybrid_mode", True)
        self._extraction_model = config.get("extraction_model", "glm-5.1:agentic")
        self._extraction_mode = config.get("extraction_mode", "llm")
        self._max_lines = int(config.get("max_lines_per_file", 200))
        self._auto_recall = config.get("auto_recall", False)

        # Thread safety
        self._lock = threading.RLock()

        # Components (lazy init)
        self._store = None
        self._extractor = None
        self._recall = None
        self._staging = None

        # Per-session tracking
        self._session_extracted_count = 0
        self._session_id: Optional[str] = None
        self._initialized = False

        # Discord context (captured from gateway_session_key during initialize)
        self._discord_context: str = "unknown"
        self._platform: str = "unknown"

    # ─── MemoryProvider ABC ───────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "dream_v2"

    def is_available(self) -> bool:
        """Always available if vault is accessible."""
        return self._vault_path.exists()

    def initialize(self, session_id: str, **kwargs) -> None:
        """Called at agent startup. Initialize components."""
        self._session_id = session_id
        self._session_extracted_count = 0
        self._initialized = True

        # Capture platform for Discord context tagging
        self._platform = kwargs.get("platform", "unknown")

        # Parse gateway_session_key to extract Discord context
        # Format: agent:main:discord:{type}:{ids...}
        # Examples:
        #   agent:main:discord:thread:123:456      -> discord:thread:123:456
        #   agent:main:discord:group:123           -> discord:group:123
        #   agent:main:discord:dm:789               -> discord:dm:789
        #   agent:main:cli:default                  -> cli:default
        gsk = kwargs.get("gateway_session_key", "")
        self._discord_context = self._parse_discord_context(gsk)

        # Lazy init components (import heavy modules only when needed)
        from .store import DreamStore
        from .extract_llm import LLMExtractor
        from .recall import RecallEngine
        from .staging import StagingManager

        self._store = DreamStore(str(self._vault_path))
        self._extractor = LLMExtractor(
            model=self._extraction_model,
            api_key=self._config.get("api_key"),
            base_url=self._config.get("extraction_base_url"),
            timeout=int(self._config.get("extraction_timeout", 120)),
        )
        self._recall = RecallEngine(str(self._vault_path))
        self._staging = StagingManager(str(self._vault_path))

        # Bug #6 workaround: merge any staging from pre-compress rescue
        self._staging.merge_to_vault()

        logger.info("[Dream v2] Initialized at %s", self._vault_path)

    def system_prompt_block(self) -> str:
        """Return MEMORY.md index for system prompt injection at session start."""
        try:
            index_path = self._vault_path / "MEMORY.md"
            if index_path.exists():
                content = index_path.read_text().strip()
                if content:
                    header = "══════════════════════════════════════DREAM MEMORY INDEX══════════════════════════════════════"
                    return f"{header}\n{content}\n"
        except Exception as e:
            logger.warning("[Dream v2] Failed to read MEMORY.md: %s", e)
        return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Called after every conversation turn. Lightweight — no-op for v2.

        Extraction happens at session end, not per turn.
        """
        pass

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return relevant memory context for the upcoming turn. On-demand only."""
        if not self._auto_recall:
            return ""
        if not self._recall:
            return ""

        try:
            results = self._recall.recall(query, limit=5)
            if not results:
                return ""

            blocks = ["## Dream Memory Recall"]
            for r in results:
                mem = r.memory
                content = mem.get("content", "")[:500]
                if content:
                    blocks.append(f"**[{mem.get('type', '?')}]** {content}")
            return "\n\n".join(blocks)
        except Exception as e:
            logger.warning("[Dream v2] prefetch failed: %s", e)
            return ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas for dream tools."""
        return [DREAM_STATUS_SCHEMA, DREAM_RECALL_SCHEMA, DREAM_CONSOLIDATE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle dream_* tool calls."""
        if not self._initialized:
            self.initialize(args.get("session_id", "unknown"))

        if tool_name == "dream_status":
            return self._tool_status()
        elif tool_name == "dream_recall":
            return self._tool_recall(
                query=args.get("query", ""),
                memory_type=args.get("memory_type"),
                limit=int(args.get("limit", 5)),
            )
        elif tool_name == "dream_consolidate":
            return self._tool_consolidate()
        return f'{{"error": "Unknown dream tool: {tool_name}"}}'

    def shutdown(self) -> None:
        """Clean shutdown."""
        pass

    # ─── Optional hooks ───────────────────────────────────────────────────

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Called at session end. Extract significant memories."""
        if not messages or not self._initialized:
            return

        if self._extraction_mode == "llm":
            self._extract_llm(messages)
        elif self._extraction_mode == "regex":
            self._extract_regex(messages)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Called before context compression. Rescue unstored memories.

        Bug #6 workaround: upstream silently discards our return value.
        We write to a staging file. Next initialize() merges it.
        """
        if not self._hybrid_mode or not self._initialized:
            return ""

        if self._session_extracted_count >= self._max_per_session:
            return ""

        try:
            candidates = self._extractor.extract_from_messages(
                messages,
                max_memories=1,
                significance_threshold=self._significance_threshold,
            )
            if candidates:
                self._staging.write_candidates(candidates, self._session_id or "unknown")
                logger.info("[Dream v2] Pre-compress rescue: wrote %d candidates", len(candidates))
        except Exception as e:
            logger.warning("[Dream v2] Pre-compress rescue failed: %s", e)

        return ""

    # ─── Extraction ─────────────────────────────────────────────────────

    def _extract_llm(self, messages: List[Dict[str, Any]]) -> None:
        """LLM-powered extraction at session end."""
        if self._session_extracted_count >= self._max_per_session:
            return

        with self._lock:
            try:
                candidates = self._extractor.extract_from_messages(
                    messages,
                    max_memories=self._max_per_session - self._session_extracted_count,
                    significance_threshold=self._significance_threshold,
                )

                for candidate in candidates:
                    path = self._store.add_memory(
                        content=candidate["content"],
                        memory_type=candidate["type"],
                        tags=candidate.get("tags", []),
                        source=self._discord_context,
                        importance=candidate.get("importance", 0.5),
                    )
                    if path:
                        self._session_extracted_count += 1
                        logger.info("[Dream v2] Stored: %s", path.name)

                self._store.rebuild_index()
            except Exception as e:
                logger.error("[Dream v2] Extraction failed: %s", e)

    def _extract_regex(self, messages: List[Dict[str, Any]]) -> None:
        """Fast regex-based extraction. Fallback when LLM unavailable."""
        from .extract import extract_candidates_from_messages
        with self._lock:
            try:
                candidates = extract_candidates_from_messages(messages)
                for candidate in candidates:
                    if self._session_extracted_count >= self._max_per_session:
                        break
                    path = self._store.add_memory(
                        content=candidate["content"],
                        memory_type=candidate.get("type", "reference"),
                        tags=candidate.get("tags", []),
                        source=self._discord_context,
                        importance=candidate.get("importance", 0.3),
                    )
                    if path:
                        self._session_extracted_count += 1
                self._store.rebuild_index()
            except Exception as e:
                logger.error("[Dream v2] Regex extraction failed: %s", e)

    # ─── Tool handlers ───────────────────────────────────────────────────

    def _tool_status(self) -> str:
        """Return vault statistics."""
        try:
            stats = self._store.get_stats()
            lines = [
                "## Dream v2 — Consciousness Vault",
                f"**Vault:** `{self._vault_path}`",
                f"**Total memories:** {stats['total']}",
                "",
            ]
            for type_name, count in stats.get("by_type", {}).items():
                lines.append(f"  {type_name}: {count}")
            lines.extend([
                "",
                f"**Session extracted:** {self._session_extracted_count}/{self._max_per_session}",
                f"**Per-session cap:** {self._max_per_session}",
                f"**Auto-recall:** {'enabled' if self._auto_recall else 'disabled (on-demand)'}",
            ])
            return "\n".join(lines)
        except Exception as e:
            return f"Dream status error: {e}"

    def _tool_recall(self, query: str, memory_type: Optional[str], limit: int) -> str:
        """Recall relevant memories."""
        if not query:
            return "dream_recall requires a query string"

        limit = min(limit, 20)
        try:
            results = self._recall.recall(query, memory_type=memory_type, limit=limit)
            if not results:
                return f"No memories found for: {query}"

            lines = [f"## Dream Recall — {len(results)} results for: {query}\n"]
            for r in results:
                mem = r.memory
                content = mem.get("content", "")[:300]
                lines.append(f"**[{mem.get('type', '?')}]** {mem.get('slug', '?')}")
                lines.append(content)
                lines.append(f"_importance: {mem.get('importance', 0):.1f} | created: {mem.get('created', '?')[:10]}_\n")
            return "\n".join(lines)
        except Exception as e:
            return f"dream_recall error: {e}"

    def _tool_consolidate(self) -> str:
        """Run vault consolidation."""
        try:
            from .consolidation import run_consolidation
            with self._lock:
                result = run_consolidation(str(self._vault_path), mode="deterministic")
                lines = ["## Dream Consolidation Results", ""]
                lines.append(f"Merged: {result.get('merged', 0)}")
                lines.append(f"Deduped: {result.get('deduped', 0)}")
                lines.append(f"Pruned: {result.get('pruned', 0)}")
                lines.append(f"Errors: {result.get('errors', 0)}")
                return "\n".join(lines)
        except Exception as e:
            return f"dream_consolidate error: {e}"

    def _parse_discord_context(self, gateway_session_key: str) -> str:
        """Extract Discord context from gateway_session_key.

        Formats:
          agent:main:discord:thread:{parent}:{thread} -> discord:thread:parent:thread
          agent:main:discord:group:{channel}         -> discord:group:channel
          agent:main:discord:dm:{user}               -> discord:dm:user
          agent:main:{platform}:{session}              -> platform:session

        Returns a colon-separated string suitable for use as memory source tag.
        """
        if not gateway_session_key:
            return self._platform or "unknown"

        try:
            # Split: agent : main : platform : type : ...ids
            parts = gateway_session_key.split(":")
            if len(parts) < 3:
                return gateway_session_key

            platform = parts[2]  # 'discord', 'cli', etc.

            if platform == "discord" and len(parts) >= 5:
                disc_type = parts[3]  # 'thread', 'group', 'dm'
                ids = parts[4:]       # contextual IDs
                return f"discord:{disc_type}:{':'.join(ids)}"
            elif platform == "discord" and len(parts) == 4:
                # Edge case: just discord:type without IDs
                return f"discord:{parts[3]}"
            else:
                # Non-Discord: agent:main:cli:default -> cli:default
                rest = ":".join(parts[2:])
                return rest
        except Exception:
            return gateway_session_key or self._platform or "unknown"


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register this plugin with Hermes."""
    config = _load_config()
    provider = DreamV2MemoryProvider(config=config)
    ctx.register_memory_provider(provider)
    logger.info("[Dream v2] Registered as memory provider")


def _load_config() -> dict:
    """Load dream_v2 config from config.yaml."""
    try:
        from hermes_constants import get_hermes_home
        from pathlib import Path
        import yaml

        config_path = get_hermes_home() / "config.yaml"
        if not config_path.exists():
            return {}

        with open(config_path) as f:
            all_config = yaml.safe_load(f) or {}

        return all_config.get("plugins", {}).get("dream_v2", {}) or {}
    except Exception:
        return {}
"""Dream v2 Memory Store — Vault CRUD operations.

Handles reading/writing memory files, manifest management, and index building.
Thread-safe with file locking.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import string
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from .taxonomy import MEMORY_TYPES, make_memory_document, parse_frontmatter, validate_memory_type

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
INDEX_NAME = "MEMORY.md"
COUNTER_NAME = ".session_counter.json"


class DreamStore:
    """Manages the Dream vault filesystem."""

    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self._lock = Lock()
        self._ensure_vault()

    def _ensure_vault(self) -> None:
        """Ensure vault and all subdirectories exist."""
        subdirs = ["consciousness/self", "consciousness/relationship", "consciousness/work",
                   "decisions", "feedback", "reference", "sessions"]
        for d in subdirs:
            (self.vault_path / d).mkdir(parents=True, exist_ok=True)

        # Ensure manifest exists
        manifest_path = self.vault_path / MANIFEST_NAME
        if not manifest_path.exists():
            with open(manifest_path, "w") as f:
                json.dump([], f)

        # Ensure counter exists
        counter_path = self.vault_path / COUNTER_NAME
        if not counter_path.exists():
            self._save_counter({"sessions_since_consolidation": 0, "last_consolidated_at": None})

    # ─── Counter ─────────────────────────────────────────────────────────

    def _load_counter(self) -> Dict[str, Any]:
        try:
            with open(self.vault_path / COUNTER_NAME) as f:
                return json.load(f)
        except Exception:
            return {"sessions_since_consolidation": 0, "last_consolidated_at": None}

    def _save_counter(self, counter: Dict[str, Any]) -> None:
        with open(self.vault_path / COUNTER_NAME, "w") as f:
            json.dump(counter, f, indent=2)

    def increment_session(self) -> None:
        with self._lock:
            counter = self._load_counter()
            counter["sessions_since_consolidation"] = counter.get("sessions_since_consolidation", 0) + 1
            self._save_counter(counter)

    # ─── Manifest ────────────────────────────────────────────────────────

    def _load_manifest(self) -> List[Dict[str, Any]]:
        try:
            with open(self.vault_path / MANIFEST_NAME) as f:
                return json.load(f)
        except Exception:
            return []

    def _save_manifest(self, manifest: List[Dict[str, Any]]) -> None:
        with open(self.vault_path / MANIFEST_NAME, "w") as f:
            json.dump(manifest, f, indent=2)

    def _get_memory_path(self, memory_type: str, filename: str) -> Path:
        """Get the full path for a memory file."""
        return self.vault_path / memory_type / filename

    # ─── Add memory ─────────────────────────────────────────────────────

    def add_memory(
        self,
        content: str,
        memory_type: str,
        tags: Optional[List[str]] = None,
        source: str = "",
        importance: float = 0.5,
    ) -> Optional[Path]:
        """Add a new memory to the vault. Returns the path if successful."""
        if not validate_memory_type(memory_type):
            logger.warning("[DreamStore] Invalid memory type: %s", memory_type)
            return None

        tags = tags or []
        slug = self._make_slug(content)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
        filename = f"{slug}-{timestamp}-{rand}.md"

        doc = make_memory_document(
            content=content,
            memory_type=memory_type,
            tags=tags,
            source=source,
            relevance=importance,
            importance=importance,
        )

        mem_path = self._get_memory_path(memory_type, filename)
        with self._lock:
            try:
                with open(mem_path, "w") as f:
                    f.write(doc)

                # Update manifest
                manifest = self._load_manifest()
                entry = {
                    "type": memory_type,
                    "filename": filename,
                    "tags": tags,
                    "source": source,
                    "relevance": importance,
                    "importance": importance,
                    "created": datetime.now(timezone.utc).isoformat(),
                    "slug": slug,
                    "access_count": 0,
                    "forgetting_factor": 0.02,
                }
                manifest.append(entry)
                self._save_manifest(manifest)

                return mem_path
            except Exception as e:
                logger.error("[DreamStore] Failed to write %s: %s", mem_path, e)
                return None

    def delete_memory(self, memory_type: str, filename: str) -> bool:
        """Delete a memory file and remove from manifest."""
        mem_path = self._get_memory_path(memory_type, filename)
        with self._lock:
            try:
                if mem_path.exists():
                    mem_path.unlink()

                manifest = self._load_manifest()
                original_len = len(manifest)
                manifest = [m for m in manifest if m.get("filename") != filename]
                self._save_manifest(manifest)

                return len(manifest) < original_len
            except Exception as e:
                logger.error("[DreamStore] Delete failed: %s", e)
                return False

    def get_stats(self) -> Dict[str, Any]:
        """Return vault statistics."""
        manifest = self._load_manifest()
        by_type: Dict[str, int] = {}
        for entry in manifest:
            t = entry.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1

        return {
            "total": len(manifest),
            "by_type": by_type,
        }

    # ─── Index ──────────────────────────────────────────────────────────

    def rebuild_index(self) -> None:
        """Rebuild MEMORY.md from manifest. Called after every write."""
        manifest = self._load_manifest()
        lines = [
            "# Dream Memory Index",
            "",
            "> This is the lightweight pointer index for Dream consciousness memories.",
            "> Memories are stored in subdirectories by type. Use `dream_recall` for details.",
            "",
            "## Consciousness",
            "  - `self/` — What the agent knows about itself",
            "  - `relationship/` — Learnings about the user relationship",
            "  - `work/` — Project and technical learnings",
            "",
            "## Decisions",
            "  - `decisions/` — Explicit agreements and decisions",
            "",
            "## Feedback",
            "  - `feedback/` — Corrections and directives",
            "",
            "## Reference",
            "  - `reference/` — Stable facts: APIs, tools, paths",
            "",
            "---",
            "",
            f"**Total memories:** {len(manifest)}",
            "",
        ]

        # Group by type and show recent
        by_type: Dict[str, List[Dict]] = {}
        for entry in manifest:
            t = entry.get("type", "unknown")
            by_type.setdefault(t, []).append(entry)

        for type_name, entries in sorted(by_type.items()):
            lines.append(f"### {type_name}")
            for entry in sorted(entries, key=lambda x: x.get("created", ""), reverse=True)[:10]:
                slug = entry.get("slug", "?")[:50]
                tags = ", ".join(entry.get("tags", [])[:3])
                lines.append(f"  - **{slug}** — {tags or 'no tags'}")
            lines.append("")

        with open(self.vault_path / INDEX_NAME, "w") as f:
            f.write("\n".join(lines))

    # ─── Slug generation ────────────────────────────────────────────────

    def _make_slug(self, content: str, max_len: int = 50) -> str:
        """Create a URL-safe slug from content."""
        # Strip markdown
        text = re.sub(r"#+ ", "", content)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"[*_`>]", "", text)
        # Lowercase, replace whitespace with hyphens
        text = re.sub(r"\s+", "-", text.strip().lower())
        text = re.sub(r"[^a-z0-9\-]", "", text)
        # Truncate
        if len(text) > max_len:
            text = text[:max_len].rstrip("-")
        return text[:max_len]

    # ─── Manifest read for recall ───────────────────────────────────────

    def get_all_memories(self) -> List[Dict[str, Any]]:
        """Return full manifest entries with file paths."""
        manifest = self._load_manifest()
        for entry in manifest:
            entry["path"] = str(self._get_memory_path(entry.get("type", ""), entry.get("filename", "")))
        return manifest

    def read_memory_content(self, memory_type: str, filename: str) -> Optional[str]:
        """Read the content of a specific memory file."""
        mem_path = self._get_memory_path(memory_type, filename)
        if not mem_path.exists():
            return None
        try:
            return mem_path.read_text()
        except Exception:
            return None
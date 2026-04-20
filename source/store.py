"""Dream Memory Store — markdown file CRUD operations.

All memories are stored as ``.md`` files under a vault directory, organised
into subdirectories by memory type (user/, feedback/, project/, reference/).
Each file has YAML frontmatter followed by free-text content.

The store also maintains a ``manifest.json`` at the vault root that tracks
metadata for fast recall without scanning the entire filesystem.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .taxonomy import (
    MEMORY_TYPES,
    MemoryTypeSpec,
    make_memory_document,
    parse_frontmatter,
    render_frontmatter,
    validate_memory_type,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Wikilink helpers (Heartbeat 5)
# ---------------------------------------------------------------------------

WIKILINK_PATTERN = re.compile(r'\[\[([^\]]+)\]\]')


def extract_wikilinks(content: str) -> List[str]:
    """Extract all [[wikilink]] targets from content."""
    return WIKILINK_PATTERN.findall(content)


def slug_from_filename(filename: str) -> str:
    """Extract the slug portion from a memory filename.

    Filename format: {slug}-{timestamp}-{suffix}.md  OR  {slug}-{timestamp}.md
    Returns the slug part (everything before the timestamp).

    Examples
    --------
    "prefers-vim-20260413T120000Z-ab3f.md" → "prefers-vim"
    "my-memory-20260413T120000Z.md" → "my-memory"
    """
    name = filename.rsplit(".", 1)[0]  # remove .md
    # Timestamp format: YYYYMMDDTHHMMSSZ (with trailing Z) or YYYY-MM-DDTHHMMSS
    # Remove trailing random suffix (4 alphanumeric chars before timestamp)
    # Strategy: find the timestamp pattern and strip everything from it onward
    ts_match = re.search(r'-\d{8}T\d{6}Z?', name)
    if ts_match:
        return name[:ts_match.start()]
    # Fallback: split on dashes — timestamp has at least 4 dash-separated parts
    parts = name.split("-")
    # Try to find where the timestamp starts (8-digit date)
    for i in range(len(parts)):
        if re.match(r'^\d{8}$', parts[i]):
            return "-".join(parts[:i])
    return name


def make_wikilink(memory_type: str, slug: str) -> str:
    """Create an Obsidian wikilink: [[type/slug]]"""
    return f"[[{memory_type}/{slug}]]"


# ---------------------------------------------------------------------------
# Slug generation
# ---------------------------------------------------------------------------

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Random 4-char suffix for filename collision avoidance
import random
import string as _string_module


def slugify(text: str, max_words: int = 6) -> str:
    """Convert *text* into a filesystem-safe slug.

    Lowercases, replaces non-alphanumeric runs with ``-``, and limits to
    *max_words* words (each word truncated to 12 chars).
    """
    words = text.lower().split()[:max_words]
    slug_parts = [w[:12] for w in words]
    slug = "-".join(slug_parts)
    slug = _NON_ALNUM.sub("-", slug).strip("-")
    return slug or "memory"


def _timestamp_str() -> str:
    """Compact ISO-8601 timestamp for filenames (no colons)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

_MANIFEST_FILE = "manifest.json"


def _load_manifest(vault_path: Path) -> List[Dict[str, Any]]:
    """Load the manifest from the vault root, returning [] on error.

    Uses file locking to prevent concurrent read/write corruption.
    """
    manifest_path = vault_path / _MANIFEST_FILE
    if not manifest_path.exists():
        return []

    lock_path = vault_path / ".manifest.lock"
    try:
        lock_fd = open(lock_path, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_SH)  # shared lock for reading
            data = manifest_path.read_text(encoding="utf-8")
            return json.loads(data)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
    except Exception:
        logger.warning("Failed to read dream manifest; starting fresh")
        return []


def _save_manifest(vault_path: Path, entries: List[Dict[str, Any]]) -> None:
    """Write the manifest back to disk using atomic write + file locking.

    Writes to a temp file first, then atomically replaces the manifest
    to prevent corruption from concurrent writes. Uses exclusive file
    locking for safe concurrent access.
    """
    manifest_path = vault_path / _MANIFEST_FILE
    lock_path = vault_path / ".manifest.lock"

    # Ensure vault directory exists
    vault_path.mkdir(parents=True, exist_ok=True)

    content = json.dumps(entries, indent=2, ensure_ascii=False)

    lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)  # exclusive lock for writing

        # Write to temp file then atomically replace
        fd, tmp_path = tempfile.mkstemp(
            dir=str(vault_path), suffix=".manifest.tmp", prefix=".tmp_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                tmp_f.write(content)
                tmp_f.flush()
                os.fsync(tmp_f.fileno())
            os.replace(tmp_path, str(manifest_path))
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


# ---------------------------------------------------------------------------
# DreamStore
# ---------------------------------------------------------------------------

class DreamStore:
    """Markdown file CRUD with vault directory structure and manifest tracking."""

    # Default forgetting factor constant — moderate decay (~35-day half-life)
    FORGETTING_FACTOR_DEFAULT: float = 0.02

    def __init__(self, vault_path: Path, config: Optional[Dict[str, Any]] = None) -> None:
        self.vault_path = Path(vault_path)
        self._config = config or {}
        # The subdir within vault_path where dream memories live.
        # Configurable via config["vault_subdir"], defaulting to "" (vault_path itself).
        # When set (e.g. "dream"), memories are stored under vault_path/<subdir>/.
        vault_subdir = self._config.get("vault_subdir", "") or ""
        self._vault_subdir: str = vault_subdir
        self._dream_root: Path = (self.vault_path / vault_subdir) if vault_subdir else self.vault_path
        self._manifest: Optional[List[Dict[str, Any]]] = None

    # -- Initialisation -----------------------------------------------------

    @property
    def dream_root(self) -> Path:
        """The actual directory where dream files are stored (vault_path / vault_subdir)."""
        return self._dream_root

    def initialize(self) -> None:
        """Create vault directory structure and initialise the manifest.

        Creates subdirectories for each memory type inside the configured
        subfolder (default ``dream/``), under the vault root:

            <vault_root>/<vault_subdir>/user/, ...feedback/, ...project/, ...reference/

        This keeps Dream memories isolated from other vault content.

        In standalone mode (no vault_subdir), also creates .obsidian/
        config so the vault can be opened directly in Obsidian.
        """
        self._dream_root.mkdir(parents=True, exist_ok=True)

        for type_name in MEMORY_TYPES:
            type_dir = self._dream_root / type_name
            type_dir.mkdir(exist_ok=True)

        # Init manifest if absent
        manifest_path = self._dream_root / _MANIFEST_FILE
        if not manifest_path.exists():
            _save_manifest(self._dream_root, [])

        # Load manifest into memory
        self._manifest = _load_manifest(self._dream_root)

        # Standalone mode: create .obsidian/ config so the vault opens
        # directly in Obsidian as its own vault (not a subfolder)
        if not self._vault_subdir:
            obs_dir = self._dream_root / ".obsidian"
            obs_dir.mkdir(exist_ok=True)
            app_json = obs_dir / "app.json"
            if not app_json.exists():
                app_json.write_text(
                    '{\n  "legacyEditor": false,\n  "promptDelete": true,\n'
                    '  "showLineNumber": true,\n  "spellcheck": true,\n'
                    '  "readableLineLength": true\n}\n'
                )

    # -- Path resolution ----------------------------------------------------

    def get_memory_path(self, memory_type: str, filename: str) -> Path:
        """Resolve the full path for a memory file.

        Parameters
        ----------
        memory_type:
            One of user|feedback|project|reference.
        filename:
            The markdown filename (e.g. ``my-memory-20250101T000000Z.md``).
        """
        if not validate_memory_type(memory_type):
            raise ValueError(f"Invalid memory type: {memory_type!r}")
        if not filename.endswith(".md"):
            filename += ".md"
        return self._dream_root / memory_type / filename

    # -- Create -------------------------------------------------------------

    def add_memory(
        self,
        memory_type: str,
        content: str,
        tags: Optional[List[str]] = None,
        source: str = "",
        relevance: float = 0.5,
        importance: Optional[float] = None,
        forgetting_factor: Optional[float] = None,
    ) -> Path:
        """Create a new memory file with frontmatter.

        Returns the Path to the created file.  The filename is auto-generated
        from a content slug + timestamp.
        """
        if not validate_memory_type(memory_type):
            raise ValueError(f"Invalid memory type: {memory_type!r}")

        # Backward-compat defaults
        effective_importance = importance if importance is not None else relevance
        effective_forgetting = forgetting_factor if forgetting_factor is not None else self.FORGETTING_FACTOR_DEFAULT

        slug = slugify(content)
        ts = _timestamp_str()
        suffix = "".join(random.choices(_string_module.ascii_lowercase + _string_module.digits, k=4))
        filename = f"{slug}-{ts}-{suffix}.md"

        doc = make_memory_document(
            content=content,
            memory_type=memory_type,
            tags=tags or [],
            source=source,
            relevance=relevance,
            importance=effective_importance,
            forgetting_factor=effective_forgetting,
        )

        filepath = self.get_memory_path(memory_type, filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(doc, encoding="utf-8")

        # Update manifest
        self._add_manifest_entry(
            memory_type, filename, content, tags or [], source,
            relevance, effective_importance, effective_forgetting,
        )

        logger.debug("Dream: created %s", filepath)

        # ── LLM Wiki cross-reference hook ──────────────────────────
        # DISABLED: wiki cross-reference creates stub pollution in the vault.
        # Dream memories should stay in dream/ and be ingested into wiki
        # explicitly via the /wiki add flow, not automatically.
        # To re-enable, set plugins.dream.wiki_crossref: true in config.yaml
        if self._config.get("wiki_crossref", False):
            self._wiki_cross_reference(memory_type, content, tags or [], filepath)

        return filepath

    # -- Read ---------------------------------------------------------------

    def read_memory(self, memory_type: str, filename: str) -> Dict[str, Any]:
        """Read and parse a memory file.

        Returns a dict with ``meta`` (frontmatter dict) and ``body`` (content
        below the frontmatter).
        """
        filepath = self.get_memory_path(memory_type, filename)
        if not filepath.exists():
            raise FileNotFoundError(f"Memory not found: {filepath}")

        text = filepath.read_text(encoding="utf-8")
        meta = parse_frontmatter(text)

        # Extract body (everything after the closing --- of frontmatter)
        body = self._extract_body(text)

        return {"meta": meta, "body": body, "path": str(filepath)}

    # -- Update --------------------------------------------------------------

    def update_memory(
        self,
        memory_type: str,
        filename: str,
        content: Optional[str] = None,
        tags: Optional[List[str]] = None,
        relevance: Optional[float] = None,
        importance: Optional[float] = None,
        forgetting_factor: Optional[float] = None,
    ) -> Path:
        """Update an existing memory file.

        Only the fields that are passed as non-None are updated.  The
        ``updated`` timestamp is always refreshed.
        """
        filepath = self.get_memory_path(memory_type, filename)
        if not filepath.exists():
            raise FileNotFoundError(f"Memory not found: {filepath}")

        text = filepath.read_text(encoding="utf-8")
        meta = parse_frontmatter(text)
        body = self._extract_body(text)

        # Apply updates
        if content is not None:
            body = content
        if tags is not None:
            meta["tags"] = tags
        if relevance is not None:
            meta["relevance"] = max(0.0, min(1.0, relevance))
        if importance is not None:
            meta["importance"] = max(0.0, min(1.0, importance))
        if forgetting_factor is not None:
            meta["forgetting_factor"] = max(0.0, forgetting_factor)
        meta["updated"] = datetime.now(timezone.utc).isoformat()

        # Re-render
        doc = render_frontmatter(meta) + "\n" + body
        filepath.write_text(doc, encoding="utf-8")

        # Update manifest
        self._update_manifest_entry(memory_type, filename, meta)

        return filepath

    # -- Delete --------------------------------------------------------------

    def delete_memory(self, memory_type: str, filename: str) -> bool:
        """Remove a memory file.  Returns True if the file existed and was deleted."""
        filepath = self.get_memory_path(memory_type, filename)
        if not filepath.exists():
            return False

        filepath.unlink()
        self._remove_manifest_entry(memory_type, filename)
        logger.debug("Dream: deleted %s", filepath)
        return True

    # -- List ----------------------------------------------------------------

    def list_memories(
        self,
        memory_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List all memories, optionally filtered by type.

        Returns a list of dicts, each with: type, filename, meta (frontmatter
        dict), body (first line or snippet).
        """
        results: List[Dict[str, Any]] = []

        types_to_scan = [memory_type] if memory_type else list(MEMORY_TYPES.keys())

        for mt in types_to_scan:
            if not validate_memory_type(mt):
                continue
            type_dir = self._dream_root / mt
            if not type_dir.exists():
                continue
            for md_file in sorted(type_dir.glob("*.md")):
                try:
                    text = md_file.read_text(encoding="utf-8")
                    meta = parse_frontmatter(text)
                    body = self._extract_body(text)
                    results.append({
                        "type": mt,
                        "filename": md_file.name,
                        "meta": meta,
                        "body": body[:200],  # snippet
                    })
                except Exception as exc:
                    logger.warning("Dream: skipping %s (%s)", md_file, exc)

        return results

    # -- Vault statistics ----------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return vault statistics (counts per type, total, vault path)."""
        counts: Dict[str, int] = {}
        total = 0
        for mt in MEMORY_TYPES:
            type_dir = self._dream_root / mt
            if type_dir.exists():
                count = len(list(type_dir.glob("*.md")))
            else:
                count = 0
            counts[mt] = count
            total += count

        return {
            "vault_path": str(self.vault_path),
            "counts": counts,
            "total": total,
        }

    # -- Internal helpers ---------------------------------------------------

    @staticmethod
    def _extract_body(text: str) -> str:
        """Return the markdown body below the frontmatter block."""
        text = text.strip()
        if not text.startswith("---"):
            return text

        # Find second ---
        # Strategy: strip the first ---, then find the next ---
        after_first = text[3:].lstrip("\n")
        end_idx = after_first.find("\n---")
        if end_idx == -1:
            return after_first
        return after_first[end_idx + 4:].lstrip("\n")

    def _ensure_manifest_loaded(self) -> List[Dict[str, Any]]:
        """Load manifest from disk if not already loaded."""
        if self._manifest is None:
            self._manifest = _load_manifest(self._dream_root)
        return self._manifest

    def _add_manifest_entry(
        self,
        memory_type: str,
        filename: str,
        content: str,
        tags: List[str],
        source: str,
        relevance: float,
        importance: float = 0.5,
        forgetting_factor: float = 0.02,
    ) -> None:
        entries = self._ensure_manifest_loaded()
        entries.append({
            "type": memory_type,
            "filename": filename,
            "tags": tags,
            "source": source,
            "relevance": max(0.0, min(1.0, relevance)),
            "importance": max(0.0, min(1.0, importance)),
            "forgetting_factor": max(0.0, forgetting_factor),
            "created": datetime.now(timezone.utc).isoformat(),
            "snippet": content[:120],
            "access_count": 0,
        })
        _save_manifest(self._dream_root, entries)
        self._manifest = entries

    def _update_manifest_entry(
        self,
        memory_type: str,
        filename: str,
        meta: Dict[str, Any],
    ) -> None:
        entries = self._ensure_manifest_loaded()
        for entry in entries:
            if entry.get("type") == memory_type and entry.get("filename") == filename:
                entry["tags"] = meta.get("tags", entry.get("tags", []))
                entry["relevance"] = meta.get("relevance", entry.get("relevance", 0.5))
                # Update importance and forgetting_factor if present in meta
                if "importance" in meta:
                    entry["importance"] = max(0.0, min(1.0, meta["importance"]))
                if "forgetting_factor" in meta:
                    entry["forgetting_factor"] = max(0.0, meta["forgetting_factor"])
                entry["updated"] = meta.get("updated", "")
                # Preserve existing access_count — never overwrite on update
                if "access_count" not in entry:
                    entry["access_count"] = 0
                break
        _save_manifest(self._dream_root, entries)
        self._manifest = entries

    def _remove_manifest_entry(
        self,
        memory_type: str,
        filename: str,
    ) -> None:
        entries = self._ensure_manifest_loaded()
        entries = [
            e for e in entries
            if not (e.get("type") == memory_type and e.get("filename") == filename)
        ]
        _save_manifest(self._dream_root, entries)
        self._manifest = entries

    def increment_access_count(self, memory_type: str, filename: str) -> int:
        """Increment and persist the access count for a memory. Returns new count.

        NOTE: Currently flushes to disk on every call. For high-frequency
        recall paths this could be buffered (flush every N calls or after a
        timeout) to reduce manifest write contention. For now, immediate
        flush is acceptable.
        """
        entries = self._ensure_manifest_loaded()
        new_count = 0
        for entry in entries:
            if entry.get("type") == memory_type and entry.get("filename") == filename:
                current = entry.get("access_count", 0)
                new_count = current + 1
                entry["access_count"] = new_count
                break
        _save_manifest(self._dream_root, entries)
        self._manifest = entries
        return new_count

    def get_related_memories(self, memory_type: str, filename: str, max_hops: int = 1) -> List[Dict]:
        """Get memories linked via [[wikilinks]] from a given memory.

        Reads the memory file, extracts wikilinks, and resolves them.
        Only 1-hop expansion for now (max_hops=1).
        """
        try:
            data = self.read_memory(memory_type, filename)
            if not data:
                return []
            content = data.get("body", "") if isinstance(data, dict) else str(data)
            if not content:
                return []
        except Exception:
            return []

        links = extract_wikilinks(content)
        related = []
        seen = {f"{memory_type}/{filename}"}  # Don't include self

        for link_target in links:
            entry = self.resolve_wikilink(link_target)
            if entry:
                key = f"{entry.get('type', '')}/{entry.get('filename', '')}"
                if key not in seen:
                    seen.add(key)
                    related.append(entry)

        return related

    def resolve_wikilink(self, wikilink_target: str) -> Optional[Dict]:
        """Resolve a [[type/slug]] wikilink to an actual memory entry.

        Args:
            wikilink_target: The link target, e.g. "feedback/prefers-vim" or just "prefers-vim"

        Returns:
            Dict with manifest entry data, or None if not found.
        """
        entries = self._ensure_manifest_loaded()

        # If target contains "/", split into type + slug
        if "/" in wikilink_target:
            target_type, target_slug = wikilink_target.split("/", 1)
            for entry in entries:
                if entry.get("type") == target_type:
                    entry_slug = slug_from_filename(entry.get("filename", ""))
                    if entry_slug == target_slug:
                        return entry
        else:
            # No type prefix — search all types
            target_slug = wikilink_target
            for entry in entries:
                entry_slug = slug_from_filename(entry.get("filename", ""))
                if entry_slug == target_slug:
                    return entry

        return None

    # -- Session counter -----------------------------------------------------

    def get_sessions_since_consolidation(self) -> int:
        """Read the session counter from .session_counter.json."""
        counter_path = self._dream_root / ".session_counter.json"
        if not counter_path.exists():
            return 0
        try:
            data = json.loads(counter_path.read_text())
            return data.get("count", 0)
        except Exception:
            return 0

    def increment_session_counter(self) -> int:
        """Increment and persist the session counter. Returns new count."""
        counter_path = self._dream_root / ".session_counter.json"
        count = self.get_sessions_since_consolidation() + 1
        counter_path.write_text(json.dumps({"count": count, "last_increment": datetime.now(timezone.utc).isoformat()}))
        return count

    def reset_session_counter(self) -> None:
        """Reset session counter to 0 (called after successful consolidation)."""
        counter_path = self._dream_root / ".session_counter.json"
        counter_path.write_text(json.dumps({"count": 0, "last_increment": datetime.now(timezone.utc).isoformat()}))

    # -- LLM Wiki cross-reference ─────────────────────────────────────────

    def _wiki_cross_reference(
        self,
        memory_type: str,
        content: str,
        tags: List[str],
        dream_filepath: Path,
    ) -> None:
        """After storing a dream memory, create or update a stub wiki page.

        This keeps the ObsidianVault wiki graph connected to dream memories.
        For each dream memory with tags, we check if a wiki page exists for
        that topic. If not, we create a minimal stub page linking to the dream.
        """
        import os as _os
        vault_path = _os.environ.get(
            "OBSIDIAN_VAULT_PATH",
            str(Path.home() / "apps/Garuda_hermes/ObsidianVault"),
        )
        vault = Path(vault_path)
        if not vault.exists():
            return  # No vault, skip cross-ref

        index_path = vault / "index.md"
        log_path = vault / "log.md"

        # Only cross-ref project and feedback types (most wiki-relevant)
        if memory_type not in ("project", "feedback", "user"):
            return

        # Build a page name from the first tag or content slug
        page_name = None
        folder = "research"
        if tags:
            primary_tag = tags[0]
            page_name = primary_tag.replace("_", "-").replace(" ", "-")
        if not page_name:
            slug = slugify(content)
            page_name = slug[:40] if len(slug) > 40 else slug

        # Determine appropriate folder
        if memory_type == "project":
            folder = "projects"
        elif memory_type == "feedback":
            folder = "research"

        wiki_page = vault / folder / f"{page_name}.md"

        if wiki_page.exists():
            # Append a "Related Dream Memory" section if not already there
            existing = wiki_page.read_text(encoding="utf-8", errors="replace")
            dream_link = f"dream/{dream_filepath.name}"
            if dream_link not in existing:
                section = f"\n## Related Dream Memories\n- [[{dream_link}]]\n"
                wiki_page.write_text(existing + section, encoding="utf-8")
                logger.debug("Dream wiki cross-ref: appended to %s", wiki_page)
        else:
            # Create stub page
            now = _timestamp_str()[:10]
            dream_link = f"dream/{dream_filepath.name}"
            stub = (
                f"---\ntitle: {page_name}\ncreated: {now}\nupdated: {now}\n"
                f"type: concept\ntags: [{', '.join(tags)}]\nsources: [dream]\n---\n\n"
                f"# {page_name}\n\n> Stub page — created from Dream memory cross-reference.\n"
                f"> Ingest a source about this topic to flesh it out.\n\n"
                f"## Related Dream Memories\n- [[{dream_link}]]\n"
            )
            wiki_page.parent.mkdir(parents=True, exist_ok=True)
            wiki_page.write_text(stub, encoding="utf-8")
            logger.debug("Dream wiki cross-ref: created stub %s", wiki_page)

            # Add to index.md
            if index_path.exists():
                idx = index_path.read_text(encoding="utf-8", errors="replace")
                link_line = f"- [[{page_name}]] — Dream cross-reference stub\n"
                if link_line not in idx:
                    # Insert before the Dream Memories section
                    marker = "## Dream Memories"
                    if marker in idx:
                        idx = idx.replace(marker, link_line + marker)
                        index_path.write_text(idx, encoding="utf-8")

        # Append to wiki log
        if log_path.exists():
            now = _timestamp_str()[:10]
            entry = f"\n## [{now}] crossref | Dream → Wiki: {page_name} ({memory_type})\n"
            log_content = log_path.read_text(encoding="utf-8", errors="replace")
            log_path.write_text(log_content + entry, encoding="utf-8")
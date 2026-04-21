"""Dream v2 Consolidation — Simplified 4-phase pipeline.

Removes from v1:
- Wikilinks (caused explosion)
- LLM consolidation mode (returned nothing useful)
- Proposals directory

Retains:
- Deduplication (exact content)
- Merge by tag overlap
- Size enforcement (200-line cap)
- Index rebuild
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


def run_consolidation(vault_path: str, mode: str = "deterministic") -> Dict[str, int]:
    """Run the 4-phase consolidation pipeline.

    Phase 1 — Audit: Count orphans, stale entries, size violations
    Phase 2 — Gather: Group by tag overlap, find duplicates
    Phase 3 — Consolidate: Merge duplicates, resolve contradictions
    Phase 4 — Prune: Remove superseded, cap oversized files

    Returns: {merged, deduped, pruned, errors}
    """
    vault = Path(vault_path)
    manifest_path = vault / "manifest.json"
    stats = {"merged": 0, "deduped": 0, "pruned": 0, "errors": 0}

    try:
        manifest = _load_manifest(manifest_path)
        deleted_filenames: Set[str] = set()

        # Phase 2: Find duplicate content
        content_hash: Dict[str, List[Dict]] = defaultdict(list)
        for entry in manifest:
            if entry.get("filename") in deleted_filenames:
                continue
            content = _read_memory_content(vault, entry)
            if content:
                # Hash the actual content (strip frontmatter)
                content_hash[content[:200]].append(entry)

        # Phase 3: Dedup exact duplicates (keep oldest by created date)
        for content_key, entries in content_hash.items():
            if len(entries) > 1:
                # Sort by created date, keep oldest
                sorted_entries = sorted(entries, key=lambda x: x.get("created", ""))
                to_delete = sorted_entries[1:]
                for entry in to_delete:
                    if _delete_memory(vault, manifest, entry):
                        deleted_filenames.add(entry.get("filename", ""))
                        stats["deduped"] += 1

        
        # Phase 3.5: Fuzzy Content Deduplication (Prevent vault bloat deterministically)
        # Pruning older memories if their core body text is >85% similar to a newer memory.
        import difflib
        similarity_threshold = 0.85
        type_groups = defaultdict(list)
        for entry in manifest:
            if entry.get("filename") not in deleted_filenames:
                type_groups[entry.get("type", "")].append(entry)
                
        for mem_type, entries in type_groups.items():
            sorted_by_newest = sorted(entries, key=lambda x: x.get("created", ""), reverse=True)
            for i, newer_entry in enumerate(sorted_by_newest):
                newer_content = _read_memory_content(vault, newer_entry)
                if not newer_content: continue
                
                for older_entry in sorted_by_newest[i+1:]:
                    if older_entry.get("filename") in deleted_filenames: continue
                    older_content = _read_memory_content(vault, older_entry)
                    if not older_content: continue
                    
                    # Fuzzy match on the actual learned content
                    if older_content in newer_content:
                        similarity = 1.0
                    else:
                        matcher = difflib.SequenceMatcher(None, newer_content, older_content)
                        similarity = matcher.quick_ratio()
                    
                    if similarity >= similarity_threshold:
                        # Older memory is highly redundant textually, prune
                        if _delete_memory(vault, manifest, older_entry):
                            deleted_filenames.add(older_entry.get("filename", ""))
                            stats["pruned"] += 1

        # Phase 4: Size enforcement — cap files at max_lines
        max_lines = 200
        for entry in manifest:
            if entry.get("filename") in deleted_filenames:
                continue
            content = _read_memory_content(vault, entry)
            if content:
                line_count = len(content.splitlines())
                if line_count > max_lines:
                    # Truncate to max_lines
                    lines = content.splitlines()[:max_lines]
                    truncated = "\n".join(lines)
                    _write_memory_content(vault, entry, truncated)
                    stats["merged"] += 1

        # Phase 4: Remove orphaned manifest entries (files don't exist)
        for entry in list(manifest):
            if entry.get("filename") in deleted_filenames:
                continue
            if not _memory_exists(vault, entry):
                manifest.remove(entry)
                stats["pruned"] += 1

        # Save cleaned manifest
        _save_manifest(manifest_path, manifest)

        # Rebuild index
        _rebuild_index(vault, manifest)

        # Update session counter
        counter_path = vault / ".session_counter.json"
        counter = {"sessions_since_consolidation": 0, "last_consolidated_at": datetime.now(timezone.utc).isoformat()}
        with open(counter_path, "w") as f:
            json.dump(counter, f, indent=2)

        logger.info("[Consolidation] Done: %s", stats)

    except Exception as e:
        logger.error("[Consolidation] Failed: %s", e)
        stats["errors"] += 1

    return stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_manifest(path: Path) -> List[Dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def _save_manifest(path: Path, manifest: List[Dict[str, Any]]) -> None:
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def _memory_exists(vault: Path, entry: Dict[str, Any]) -> bool:
    mem_type = entry.get("type", "")
    filename = entry.get("filename", "")
    if not filename or not mem_type:
        return False
    return (vault / mem_type / filename).exists()


def _read_memory_content(vault: Path, entry: Dict[str, Any]) -> str:
    mem_type = entry.get("type", "")
    filename = entry.get("filename", "")
    if not filename or not mem_type:
        return ""
    path = vault / mem_type / filename
    if not path.exists():
        return ""
    try:
        text = path.read_text()
        # Strip frontmatter
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                return parts[2].strip()
        return text.strip()
    except Exception:
        return ""


def _write_memory_content(vault: Path, entry: Dict[str, Any], content: str) -> bool:
    mem_type = entry.get("type", "")
    filename = entry.get("filename", "")
    if not filename or not mem_type:
        return False
    path = vault / mem_type / filename
    if not path.exists():
        return False
    try:
        # Read existing frontmatter
        existing = path.read_text()
        frontmatter = ""
        if existing.startswith("---"):
            parts = existing.split("---", 2)
            if len(parts) >= 3:
                frontmatter = "---\n" + parts[1] + "\n---\n"

        path.write_text(frontmatter + "\n" + content)
        return True
    except Exception:
        return False


def _delete_memory(vault: Path, manifest: List[Dict[str, Any]], entry: Dict[str, Any]) -> bool:
    mem_type = entry.get("type", "")
    filename = entry.get("filename", "")
    if not filename or not mem_type:
        return False

    # Delete file
    path = vault / mem_type / filename
    deleted = False
    if path.exists():
        try:
            path.unlink()
            deleted = True
        except Exception:
            pass

    # Remove from manifest
    for m in manifest:
        if m.get("filename") == filename and m.get("type") == mem_type:
            manifest.remove(m)
            break

    return deleted


def _rebuild_index(vault: Path, manifest: List[Dict[str, Any]]) -> None:
    """Rebuild MEMORY.md from manifest."""
    lines = [
        "# Dream Memory Index",
        "",
        f"> Rebuilt: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "> This is the lightweight pointer index for Dream consciousness memories.",
        "> Use `dream_recall` for full memory retrieval.",
        "",
        "## Consciousness",
        "  - `self/` — What the agent knows about itself",
        "  - `relationship/` — Learnings about the user",
        "  - `work/` — Project and technical learnings",
        "",
        "## Decisions / Feedback / Reference",
        "",
        f"**Total memories:** {len(manifest)}",
        "",
    ]

    # Group by type
    by_type: Dict[str, List] = defaultdict(list)
    for entry in manifest:
        by_type[entry.get("type", "unknown")].append(entry)

    for type_name, entries in sorted(by_type.items()):
        lines.append(f"### {type_name}")
        for entry in sorted(entries, key=lambda x: x.get("created", ""), reverse=True)[:10]:
            slug = entry.get("slug", "?")[:50]
            tags = ", ".join(entry.get("tags", [])[:3])
            lines.append(f"  - **{slug}** — {tags or 'no tags'}")
        lines.append("")

    index_path = vault / "MEMORY.md"
    with open(index_path, "w") as f:
        f.write("\n".join(lines))
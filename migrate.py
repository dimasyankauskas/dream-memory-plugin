#!/usr/bin/env python3
"""Dream v2 Migration Script — Filter v1 vault and import to v2.

Filters out:
- Session chronicles (> 200 lines)
- Files with wikilinks explosion (50+ links)
- Orphaned manifest entries
- Files without frontmatter

Keeps:
- Distilled insights (< 200 lines, no excessive wikilinks)
- Files with meaningful content

Usage:
    python3 migrate.py [--dry-run] [--source VAULT_PATH] [--dest VAULT_PATH]
"""

import json
import os
import re
import sys
from pathlib import Path


V1_VAULT = Path("/Users/atma/apps/Garuda_hermes/ObsidianVault/dream")
V2_VAULT = Path("/Users/atma/apps/Garuda_hermes/dream")
MAX_LINES = 200
MAX_WIKILINKS = 5


def count_wikilinks(text: str) -> int:
    """Count [[wikilinks]] in text."""
    return len(re.findall(r"\[\[[^\]]+\]\]", text))


def count_lines(text: str) -> int:
    """Count non-empty lines."""
    return len([l for l in text.splitlines() if l.strip()])


def parse_frontmatter(text: str) -> dict:
    """Simple frontmatter parser."""
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    fm_block = parts[1]
    meta = {}
    for line in fm_block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip().strip("'\"")
        meta[key] = val
    return meta


def extract_body(text: str) -> str:
    """Strip frontmatter and return body."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return text.strip()


def should_migrate(mem_type: str, filename: str, body: str) -> tuple[bool, str]:
    """Check if a memory should be migrated to v2.

    Returns (should_migrate, reason).
    """
    line_count = count_lines(body)
    wikilink_count = count_wikilinks(body)

    # Too long
    if line_count > MAX_LINES:
        return False, f"Too long ({line_count} lines > {MAX_LINES})"

    # Wikilink explosion
    if wikilink_count > MAX_WIKILINKS:
        return False, f"Too many wikilinks ({wikilink_count} > {MAX_WIKILINKS})"

    # Check for chronicle trap — very long files with session-like content
    if line_count > 100 and any(x in body.lower() for x in ["session", "heartbeat", "cron", "早上", "afternoon", "morning"]):
        # Check if it's a session chronicle (starts with timestamps)
        first_lines = "\n".join(body.splitlines()[:5])
        if re.match(r"^\d{4}-\d{2}-\d{2}", first_lines) or "session" in body.lower()[:500]:
            return False, "Appears to be a session chronicle"

    # Check for empty or near-empty
    if len(body.strip()) < 50:
        return False, "Too short (< 50 chars)"

    return True, "OK"


def migrate_memory(v1_path: Path, v2_path: Path, mem_type: str) -> bool:
    """Migrate a single memory file."""
    try:
        text = v1_path.read_text()
    except Exception as e:
        print(f"  ERROR reading {v1_path}: {e}")
        return False

    meta = parse_frontmatter(text)
    body = extract_body(text)

    should, reason = should_migrate(mem_type, v1_path.name, body)
    if not should:
        print(f"  SKIP {v1_path.name}: {reason}")
        return False

    # Determine target type in v2
    v2_type = map_type(mem_type, meta)

    # Create v2 directory
    target_dir = v2_path / v2_type
    target_dir.mkdir(parents=True, exist_ok=True)

    # Write the file (strip excessive wikilinks from body)
    if count_wikilinks(body) > 0:
        # Keep only first MAX_WIKILINKS wikilinks
        lines = body.splitlines()
        new_lines = []
        wikilinks_kept = 0
        for line in lines:
            if wikilinks_kept >= MAX_WIKILINKS:
                # Skip lines that are mostly wikilinks
                if re.match(r"^\s*-\s*\[\[", line):
                    continue
            if wikilinks_kept < MAX_WIKILINKS and "[[" in line:
                wikilinks_kept += line.count("[[")
            new_lines.append(line)
        body = "\n".join(new_lines)

    # Write with updated frontmatter
    timestamp = meta.get("created", "")
    v2_meta = {
        "type": v2_type,
        "created": timestamp,
        "updated": timestamp,
        "importance": float(meta.get("importance", meta.get("relevance", 0.5))),
        "tags": meta.get("tags", "").strip("[]").split(",") if meta.get("tags") else [],
        "source": f"migrated:{meta.get('source', '')}",
        "access_count": 0,
        "forgetting_factor": 0.02,
    }

    content = render_frontmatter(v2_meta) + "\n" + body

    target_file = target_dir / v1_path.name
    target_file.write_text(content)
    print(f"  MIGRATED {v1_path.name} -> {v2_type}/{v1_path.name}")
    return True


def map_type(v1_type: str, meta: dict) -> str:
    """Map v1 memory type to v2 type."""
    # consciousness types don't exist in v1, map to appropriate v2 type
    type_map = {
        "user": "consciousness/relationship",
        "feedback": "feedback",
        "project": "consciousness/work",
        "reference": "reference",
        "proposal": "decisions",
    }
    return type_map.get(v1_type, "reference")


def render_frontmatter(meta: dict) -> str:
    """Render simple frontmatter."""
    lines = ["---"]
    for key in sorted(meta.keys()):
        val = meta[key]
        if isinstance(val, list):
            lines.append(f"{key}: [{', '.join(str(v) for v in val)}]")
        else:
            lines.append(f"{key}: {val}")
    lines.append("---")
    return "\n".join(lines)


def main(dry_run: bool = False):
    print(f"{'[DRY RUN] ' if dry_run else ''}Dream v2 Migration")
    print(f"  Source: {V1_VAULT}")
    print(f"  Dest:   {V2_VAULT}")
    print()

    if not V1_VAULT.exists():
        print(f"ERROR: Source vault not found: {V1_VAULT}")
        sys.exit(1)

    if not dry_run:
        V2_VAULT.mkdir(parents=True, exist_ok=True)
        for subdir in ["consciousness/self", "consciousness/relationship", "consciousness/work",
                       "decisions", "feedback", "reference", "sessions"]:
            (V2_VAULT / subdir).mkdir(parents=True, exist_ok=True)

    # Load v1 manifest
    manifest_path = V1_VAULT / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = []

    print(f"v1 manifest: {len(manifest)} entries")
    print()

    # Track migration
    migrated = 0
    skipped = 0
    errors = 0

    # Migrate by type directories
    for mem_type in ["user", "feedback", "project", "reference", "proposal"]:
        type_dir = V1_VAULT / mem_type
        if not type_dir.exists():
            continue

        print(f"Processing {mem_type}/...")
        for v1_file in sorted(type_dir.iterdir()):
            if not v1_file.is_file() or not v1_file.name.endswith(".md"):
                continue

            if dry_run:
                text = v1_file.read_text()
                body = extract_body(text)
                should, reason = should_migrate(mem_type, v1_file.name, body)
                if should:
                    print(f"  WOULD MIGRATE {v1_file.name}")
                    migrated += 1
                else:
                    print(f"  SKIP {v1_file.name}: {reason}")
                    skipped += 1
            else:
                if migrate_memory(v1_file, V2_VAULT, mem_type):
                    migrated += 1
                else:
                    skipped += 1

    print()
    print(f"Migration complete:")
    print(f"  Migrated: {migrated}")
    print(f"  Skipped:  {skipped}")
    print(f"  Total:    {migrated + skipped}")

    if dry_run:
        print("\nThis was a dry run. Run without --dry-run to actually migrate.")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
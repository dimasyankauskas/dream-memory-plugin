"""Dream v2 Taxonomy — memory type definitions and frontmatter utilities.

Extends v1 taxonomy with consciousness/ type hierarchy.
Consciousness types: self, relationship, work
Operational types: decisions, feedback, reference

Frontmatter schema:
  type:       str        — consciousness/self|consciousness/relationship|consciousness/work|decisions|feedback|reference
  created:    str        — ISO-8601 timestamp
  updated:    str        — ISO-8601 timestamp
  importance: float      — 0.0–1.0 importance score
  tags:       list[str]  — freeform tags
  source:     str        — session-id that created this memory
  access_count: int     — number of times recalled
  forgetting_factor: float — decay rate
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Memory type definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryTypeSpec:
    name: str
    description: str
    max_lines: int
    filename_pattern: str


MEMORY_TYPES: Dict[str, MemoryTypeSpec] = {
    # Consciousness types — the agent's accumulated self-model
    "consciousness/self": MemoryTypeSpec(
        name="consciousness/self",
        description="What the agent knows about its own capabilities, limitations, strengths, and patterns.",
        max_lines=100,
        filename_pattern="{slug}-{ts}.md",
    ),
    "consciousness/relationship": MemoryTypeSpec(
        name="consciousness/relationship",
        description="Learnings about the user — preferences, communication style, what they value.",
        max_lines=100,
        filename_pattern="{slug}-{ts}.md",
    ),
    "consciousness/work": MemoryTypeSpec(
        name="consciousness/work",
        description="Project and tactical learnings — what worked, what failed, process insights.",
        max_lines=100,
        filename_pattern="{slug}-{ts}.md",
    ),
    # Operational types
    "decisions": MemoryTypeSpec(
        name="decisions",
        description="Explicit agreements and decisions made with the user.",
        max_lines=80,
        filename_pattern="{slug}-{ts}.md",
    ),
    "feedback": MemoryTypeSpec(
        name="feedback",
        description="Corrections and directives — what the user told the agent to do differently.",
        max_lines=50,
        filename_pattern="{slug}-{ts}.md",
    ),
    "reference": MemoryTypeSpec(
        name="reference",
        description="Stable facts — APIs, tool quirks, paths, configurations.",
        max_lines=120,
        filename_pattern="{slug}-{ts}.md",
    ),
}


def validate_memory_type(type_str: str) -> bool:
    """Return True if type_str is a valid Dream v2 memory type."""
    return type_str in MEMORY_TYPES


# ---------------------------------------------------------------------------
# Frontmatter parsing / rendering
# ---------------------------------------------------------------------------

_FM_DELIM = re.compile(r"^---\s*$", re.MULTILINE)


def parse_frontmatter(text: str) -> Dict[str, Any]:
    """Parse YAML frontmatter from a markdown string."""
    text = text.strip()
    if not text.startswith("---"):
        return {}

    delim_matches = list(_FM_DELIM.finditer(text))
    if len(delim_matches) < 2:
        return {}

    start = delim_matches[0].end()
    end = delim_matches[1].start()
    fm_block = text[start:end].strip()

    try:
        import yaml
        meta = yaml.safe_load(fm_block)
        if not isinstance(meta, dict):
            return {}
    except ImportError:
        meta = _parse_frontmatter_simple(fm_block)
    except Exception:
        return {}

    if "importance" not in meta:
        meta["importance"] = meta.get("relevance", 0.5)
    if "forgetting_factor" not in meta:
        meta["forgetting_factor"] = 0.02
    if "access_count" not in meta:
        meta["access_count"] = 0

    return meta


def _parse_frontmatter_simple(fm_block: str) -> Dict[str, Any]:
    """Fallback parser when PyYAML is unavailable."""
    result: Dict[str, Any] = {}
    for line in fm_block.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

        if value.startswith("[") and value.endswith("]"):
            items = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
            result[key] = items
        else:
            try:
                result[key] = int(value)
            except ValueError:
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
    return result


def render_frontmatter(meta: Dict[str, Any]) -> str:
    """Render metadata dict to YAML frontmatter string."""
    try:
        import yaml
        body = yaml.dump(
            meta,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=True,
        ).rstrip()
    except ImportError:
        lines = []
        for key in sorted(meta.keys()):
            value = meta[key]
            if isinstance(value, list):
                lines.append(f"{key}: [{', '.join(str(v) for v in value)}]")
            else:
                lines.append(f"{key}: {value}")
        body = "\n".join(lines)

    return f"---\n{body}\n---\n"


def make_memory_document(
    content: str,
    memory_type: str,
    tags: Optional[List[str]] = None,
    source: str = "",
    relevance: float = 0.5,
    importance: Optional[float] = None,
    forgetting_factor: Optional[float] = None,
) -> str:
    """Create a complete markdown document with frontmatter."""
    if not validate_memory_type(memory_type):
        raise ValueError(f"Invalid memory type: {memory_type!r}")

    now = datetime.now(timezone.utc).isoformat()
    effective_importance = importance if importance is not None else relevance
    effective_forgetting = forgetting_factor if forgetting_factor is not None else 0.02
    meta = {
        "type": memory_type,
        "created": now,
        "updated": now,
        "importance": max(0.0, min(1.0, effective_importance)),
        "forgetting_factor": max(0.0, effective_forgetting),
        "tags": tags or [],
        "source": source,
        "access_count": 0,
    }
    return render_frontmatter(meta) + "\n" + content.lstrip("\n")
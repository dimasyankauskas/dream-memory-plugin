# Dream Memory Plugin for Hermes

> AutoDream-inspired memory consolidation plugin. Structured markdown memories with taxonomy, manifest-based recall, and scheduled 4-phase consolidation.

## Status: ✅ BUILT — 225 tests passing, plugin loads and works end-to-end

## Origin
Inspired by Claude Code's AutoDream architecture. Research source: `/Users/atma/dev/Antigravity_Expert/docs/dream/`

## Architecture Overview

### Storage Format
Each memory = `.md` file with YAML frontmatter:
```yaml
---
type: user | feedback | project | reference
created: ISO timestamp
updated: ISO timestamp
relevance: 0.0-1.0
tags: [list, of, tags]
source: session-id
---
Memory content in markdown...
```

### Memory Taxonomy (from AutoDream)
| Type | Purpose | Example |
|------|---------|---------|
| `user` | Goals, preferences, communication style | "Kedar prefers short confirmations" |
| `feedback` | Corrections with Why + How-to-apply | "Don't use markdown tables in Telegram" |
| `project` | Ongoing work facts, technical decisions | "Dream plugin lives at plugins/memory/dream/" |
| `reference` | External system pointers, paths, URLs | "Obsidian vault at ~/apps/Garuda_hermes/ObsidianVault/" |

### Lifecycle Hook Mapping
| Phase | Hermes Hook | Behavior |
|-------|-------------|----------|
| Per-turn extraction | `sync_turn()` | Extract candidate memories from each turn |
| End-of-session | `on_session_end()` | Lightweight consolidation on session facts |
| Pre-compress rescue | `on_pre_compress()` | Extract insights before context compression |
| Mirror built-in writes | `on_memory_write()` | Mirror add/replace/remove to dream taxonomy |
| Nightly consolidation | Cron job | Heavy 4-phase dream (Orient→Gather→Consolidate→Prune) |
| Recall | `prefetch()` + `queue_prefetch()` | Manifest-based selection: ≤5 relevant files |

### Plugin Directory
```
plugins/memory/dream/
├── __init__.py          # DreamMemoryProvider + register()
├── plugin.yaml          # Metadata, hooks, dependencies
├── store.py             # Markdown file management (CRUD, manifest)
├── taxonomy.py          # Memory types, frontmatter parsing
├── consolidation.py     # 4-phase dream engine
├── recall.py            # Manifest-based selection
├── extract.py           # Per-turn extraction from conversations
├── cli.py               # `hermes dream` CLI commands
└── tests/
    ├── test_store.py
    ├── test_taxonomy.py
    ├── test_consolidation.py
    ├── test_recall.py
    └── test_provider.py
```

### Config (in $HERMES_HOME/config.yaml)
```yaml
memory:
  provider: dream

plugins:
  dream:
    vault_path: $HERMES_HOME/dream
    max_lines: 200
    max_bytes: 25600
    consolidate_cron: "0 3 * * *"
    consolidate_model: glm-5.1:cloud
    taxonomy: [user, feedback, project, reference]
```

### Tools Exposed to Model
- `dream_recall` — Query manifest, retrieve relevant memories
- `dream_consolidate` — Trigger consolidation on-demand
- `dream_status` — Show memory stats (count by type, last consolidation, size)

### Update-Safety Strategy
1. Plugin is entirely self-contained in `plugins/memory/dream/`
2. Only touches MemoryProvider ABC interface (stable, well-defined)
3. Config goes through standard plugin config system
4. No monkey-patching of core Hermes code
5. CLI commands registered through `register_cli()` pattern
6. All state in `$HERMES_HOME/dream/` — separate from core state

### Reuse of Existing Infrastructure
- **System prompts**: Dream's `system_prompt_block()` returns Dream-specific context; built-in MEMORY.md context stays unchanged
- **Loops**: Dream hooks into existing MemoryManager lifecycle (initialize, prefetch, sync_turn, on_session_end, on_pre_compress, on_memory_write)
- **Cron**: Uses Hermes's existing cron system for scheduled consolidation
- **Config**: Uses standard `plugins.dream` config block

## Completed Phases

### Phase 1: Foundation ✅
- Plugin skeleton with `plugin.yaml` and `register()`
- Memory taxonomy: 4 types (user/feedback/project/reference), frontmatter parsing, validation
- DreamStore: CRUD operations on `.md` memory files with manifest tracking
- `dream_status` tool: vault statistics

### Phase 2: Capture ✅
- `extract.py`: Regex-based per-turn candidate extraction (14 patterns across 4 types)
- `sync_turn()`: Exract candidates, store those with relevance ≥ 0.6
- `on_memory_write()`: Mirror built-in memory writes to dream taxonomy
- `on_session_end()`: End-of-session extraction with `session-end` tag
- `on_pre_compress()`: Rescue insights before compression with `pre-compress` tag
- `save_config()` and `get_config_schema()`: Config management

### Phase 3: Recall ✅
- `recall.py`: Manifest-based RecallEngine (tag matching, relevance scoring, recency boost)
- `prefetch()`: Returns formatted Dream Memory context block per-turn (≤5 memories)
- `queue_prefetch()`: Stores pending query for background recall
- `dream_recall` tool: Explicit recall with type filter and limit
- `dream_consolidate` tool: On-demand consolidation trigger (stub → real in Phase 4)
- `system_prompt_block()`: Status message with memory counts

### Phase 4: Dream ✅
- `consolidation.py`: Full 4-phase consolidation engine
  - **Orient**: Read manifest, check timestamps, identify stale/oversized files
  - **Gather**: Group fragments by tag overlap, find duplicates (>80% word similarity), detect contradictions
  - **Consolidate**: Deterministic merge, deduplicate, resolve contradictions, compress, update relevance
  - **Prune**: Enforce caps, delete superseded, update manifest, write consolidation log
- `dream_consolidate` tool: Now fully functional with `dry_run` and `memory_type` filter
- Consolidation log written to vault for timestamp tracking

## Key Constraints
- Must survive Hermes agent updates (no core code modification)
- Must be installable via config (`memory.provider: dream`)
- Must reuse existing system prompts and lifecycle loops
- Must be configurable via `plugins.dream` in config.yaml
- Built-in MEMORY.md stays active alongside Dream (not replaced)
- Only one external memory provider at a time (Hermes constraint)

## References
- AutoDream source: `/Users/atma/dev/Antigravity_Expert/docs/dream/`
- MemoryProvider ABC: `/Users/atma/.hermes/hermes-agent/agent/memory_provider.py`
- MemoryManager: `/Users/atma/.hermes/hermes-agent/agent/memory_manager.py`
- Plugin discovery: `/Users/atma/.hermes/hermes-agent/plugins/memory/__init__.py`
- Holographic plugin (reference): `/Users/atma/.hermes/hermes-agent/plugins/memory/holographic/`
- Hermes agent root: `/Users/atma/.hermes/hermes-agent/`
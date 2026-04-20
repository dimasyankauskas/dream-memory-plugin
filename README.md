# 🧠 Dream Memory Plugin for Hermes

> **⚠️ STATUS: ACTIVE** — Dream v2 is running as the primary memory provider for Hermes Agent.

Dream Memory is an **Obsidian-backed session memory consolidation** system for [Hermes Agent](https://github.com/NousResearch/hermes-agent). It extracts structured memories from conversations, organizes them by taxonomy, and periodically consolidates them to prevent vault bloat.

**Live at:** [dimasyankauskas/dream-memory-plugin](https://github.com/dimasyankauskas/dream-memory-plugin)

[![Tests](https://img.shields.io/badge/tests-596%20passing-brightgreen)](./source/tests)  [![Python](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)  [![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

---

## Why Dream Memory (Paused)?

AI agents suffer from two problems:

1. **AI Amnesia** — Every session starts blank. The agent forgets your preferences, corrections, and project context.
2. **Token Bloat** — Dumping full conversation logs into context is expensive and noisy.

Dream Memory fixes both. Inspired by how human sleep consolidates memories, it runs a background consolidation pipeline that **audits, merges, deduplicates, and prunes** memories — compressing scattered notes into clean, structured, connected insights.

The result: your agent **remembers you** across sessions, stays accurate over months, and its knowledge compounds over time.

---

## Features

### 🏥 Four-Phase Consolidation Pipeline

Runs automatically on cron (default: 3am daily) or on-demand via `dream_consolidate`:

| Phase | Action | Result |
|-------|--------|--------|
| **Orient** | Audit vault for fragmentation signals | Detects when consolidation is needed |
| **Gather** | Group entries by topic, find duplicates/overlaps | Identifies merge candidates |
| **Consolidate** | Merge duplicates, resolve contradictions | Compresses 10 scattered notes → 1 clear memory |
| **Prune** | Remove superseded, stale, or redundant entries | Vault stays small and accurate |

### 🏷️ Structured Taxonomy

Four memory types with priority-weighted recall:

| Type | Purpose | Recall Priority |
|------|---------|----------------|
| **`user`** | Preferences, communication style, identity | Medium |
| **`feedback`** | Corrections, "don't do this again" directives | **Highest** |
| **`project`** | Active projects, code paths, architecture decisions | Medium |
| **`reference`** | Stable facts — APIs, tool quirks, conventions | Low |

Feedback memories always surface first. Your corrections override everything.

### 🔗 Wikilink Knowledge Graph

After consolidation, memories that share tags get connected with `[[wikilinks]]` — just like Obsidian:

```markdown
## Related

- [[my-recommendati-set-some-sort-20260414T063313Z-tjix]] — shares: preference, approach
- [[when-you-stop-working-for-the-day-20260414T030041Z-w9us]] — shares: correction
```

Links are **cross-type** — a `feedback` correction links to the `project` it applies to. This produces emergent insight: connecting a project decision to a user preference to a past correction.

**Post-pruning safety**: Wikilinks only point to *surviving* memories. No dangling references.

### ⚡ Auto-Recall (Passive Memory Injection)

Relevant memories are **automatically injected into every conversation turn** — no manual `dream_recall` calls needed.

```
config.yaml:
  plugins:
    dream:
      auto_recall: true        # enable passive injection
      auto_recall_budget: 2048  # max bytes per turn
      auto_recall_top_k: 10    # candidates considered
```

- Feedback-type memories score highest (your corrections come first)
- Recent and frequently-accessed memories get boosted scores
- Respects budget — won't flood context

### 📉 Forgetting Curves

Memories that aren't accessed gradually decay in score — just like human memory:

- Frequently-recalled memories get **stronger** (access count boosts recall score)
- One-off mentions naturally **fade** over time
- The vault self-optimizes without manual cleanup

### 🤖 LLM-Powered Extraction

Inspired by Anthropic's AutoDream, Dream uses a **forked LLM agent** to extract memories at session end — not regex pattern matching.

**How it works:**

When a session ends, the full conversation transcript is sent to an LLM with a carefully designed extraction prompt. The LLM:

1. Reads the conversation for insights worth persisting
2. Checks existing memories (manifest summary) to avoid duplicates
3. Extracts **structured memories** — not raw sentence fragments — with the right taxonomy type, tags, and relevance score
4. Returns clean, self-contained facts that are understandable without context

**Two extraction modes:**

| Mode | Per-turn (`sync_turn`) | Session-end (`on_session_end`) | Quality |
|------|----------------------|-------------------------------|---------|
| **`regex`** | Fast regex patterns | Regex scan of all messages | Low — captures raw sentences |
| **`llm`** *(default)* | No per-turn extraction | LLM analyzes full conversation | **High — captures insights** |
| **`both`** | Fast regex patterns | Both regex + LLM, merged & deduped | Highest coverage |

Regex extraction is fast and free (zero API cost, zero latency). LLM extraction costs one API call per session but produces dramatically better memories: "I love it" → "User approves of this approach and wants to continue."

**Graceful fallback**: If `extraction_mode=llm` but no API key is configured or the LLM call fails, the system silently falls back to regex extraction — no session is ever lost.

### 🔒 Consolidation Lock

Prevents race conditions — two consolidation runs can't overlap. If cron triggers while a manual run is active, it waits or skips gracefully.

### 🏠 Human-Editable Markdown Vault

Every memory is a plain markdown file with YAML frontmatter:

```markdown
---
type: feedback
tags: [correction, job-search]
access_count: 5
created: 2026-04-14T03:00:41Z
score: 0.55
---

Never mark a job CLOSED from one ATS alone. Ground truth = company's own /careers page first.
```

Open any text editor — see and edit what your agent remembers. Full transparency, full control.

**Standalone by default.** The vault is a self-contained Obsidian-compatible directory — open it as its own Obsidian vault, or integrate it into your existing vault for unified graph view.

---

## Architecture

```
dream-memory-plugin/
├── __init__.py          # DreamMemoryProvider — MemoryProvider ABC implementation
├── store.py            # DreamStore — vault filesystem operations + manifest
├── taxonomy.py         # Memory types, type validation, display names
├── extract.py          # Fast regex extraction (per-turn, sync_turn)
├── extract_llm.py      # LLM-powered extraction (session-end, quality path)
├── recall.py           # RecallEngine — manifest-based retrieval with scoring
├── consolidation.py    # 4-phase consolidation pipeline + wikilink generation
├── shared.py           # Config resolution, shared utilities
├── cli.py              # /dream CLI command (planned)
├── plugin.yaml         # Plugin metadata and hooks
└── tests/              # 596 tests across 15 test suites
    ├── test_auto_recall.py
    ├── test_candidate_extraction.py
    ├── test_consolidation.py
    ├── test_consolidation_lock.py
    ├── test_cron.py
    ├── test_extract_llm.py
    ├── test_forgetting_curves.py
    ├── test_llm_consolidation.py
    ├── test_provider.py
    ├── test_recall.py
    ├── test_regression.py
    ├── test_session_gate.py
    ├── test_store.py
    ├── test_taxonomy.py
    └── test_wikilinks.py
```

### Data Flow

```
User sends message
       │
       ▼
  ┌─────────────┐     ┌──────────────────┐
  │ Memory Write │────▶│ Candidate Extract │  (on_session_end, on_pre_compress hooks)
  └─────────────┘     └────────┬─────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │   Dream Vault    │  (markdown files + manifest.json)
                      └────────┬────────┘
                               │
                ┌──────────────┼───────────────┐
                │              │               │
                ▼              ▼               ▼
         ┌──────────┐  ┌───────────┐  ┌──────────────┐
         │ Auto-Recall│  │ dream_recall│  │ Consolidation│
         │ (passive)  │  │  (explicit) │  │   (cron/on-demand)│
         └──────────┘  └───────────┘  └──────────────┘
                │              │               │
                └──────────────┴───────────────┘
                               │
                               ▼
                      Injected into agent context
```

---

## Installation

### As a Hermes Plugin

Dream Memory is a **built-in memory provider** for [Hermes Agent](https://github.com/nichenqin/hermes-agent). It ships with the agent — no separate installation needed.

### Configuration

Add to `~/.hermes/config.yaml`:

```yaml
memory:
  provider: dream          # Enable Dream Memory (default: honcho or basic)

plugins:
  dream:
    # ─── Vault Location ───────────────────────────────────────
    # Default:  standalone vault at $HERMES_HOME/dream_vault
    #           (~/.hermes/dream_vault/) — own .obsidian/ config included
    # Obsidian: point vault_path at your existing Obsidian vault,
    #           set vault_subdir to a folder name (e.g. "dream")
    #           memories live at vault_path/vault_subdir/
    # Custom:   any absolute path you choose
    #
    # vault_path: ~/apps/Garuda_hermes/ObsidianVault   # Obsidian integration
    # vault_subdir: dream                                # Subfolder inside vault_path
    # ──────────────────────────────────────────────────────────
    max_lines: 100                       # Per-memory line limit
    max_bytes: 50000                     # Per-memory byte limit
    auto_recall: true                    # Passive memory injection every turn
    auto_recall_budget: 2048             # Max bytes injected per turn
    auto_recall_top_k: 10               # Candidates considered per recall
    taxonomy: true                       # Organize vault by type subdirs
    extraction_mode: llm                # regex | llm (default) | both
    extraction_model: ""                # LLM model for extraction (empty = uses consolidate_model or default)
    # extraction_api_key: ""            # API key (default: falls back to consolidate_api_key, then env vars)
    # extraction_base_url: ""          # Base URL (default: falls back to consolidate_base_url, then env vars)
    consolidate_cron: "0 3 * * *"       # Cron schedule for consolidation
```

#### Vault Architecture — Standalone vs. Integrated

Dream Memory supports two vault modes:

**🧊 Standalone (default — zero config)**

```
~/.hermes/dream_vault/              ← Own Obsidian vault, opened separately
├── .obsidian/                      ← Independent Obsidian config
│   └── app.json
├── user/
├── feedback/
├── project/
├── reference/
├── manifest.json
└── consolidation_log.json
```

- No configuration needed — works out of the box
- Opens as its own Obsidian vault (File → Open Vault)
- Self-contained — plugin owns the entire directory
- Portable — share or version the vault independently
- Best for: new users, standalone deployments, testing

**🔷 Obsidian-Integrated (set `vault_path`)**

```
YourObsidianVault/                  ← Your existing Obsidian vault
├── research/                       ← Your curated notes
├── projects/                       ← Your curated notes
├── frameworks/                     ← Your curated notes
└── dream/                          ← Plugin-managed (vault_subdir: dream)
    ├── user/
    ├── feedback/
    ├── project/
    ├── reference/
    ├── manifest.json
    └── consolidation_log.json
```

- Dream memories appear in your existing Obsidian graph view
- Wikilinks connect AI memories to your research and project notes
- Plugin only manages files inside `dream/` — never touches your other notes
- One Obsidian window, one knowledge base, one search
- Best for: power users who want AI memories connected to their existing knowledge graph

| Setup | `vault_path` | `vault_subdir` | Where memories live |
|-------|-------------|---------------|-------------------|
| **Standalone** (default) | *(omit)* | *(omit)* | `~/.hermes/dream_vault/` |
| **Obsidian-Integrated** | `/path/to/ObsidianVault` | `dream` | `ObsidianVault/dream/` |
| **Custom** | Any absolute path | *(optional)* | `vault_path/vault_subdir/` |

### Standalone Usage

Dream Memory can also be used independently:

```python
from dream.store import DreamStore
from dream.recall import RecallEngine
from dream.consolidation import run_consolidation

# Initialize
store = DreamStore(vault_path=Path("~/.dream_vault"))
store.initialize()

# Write a memory
store.write("user", "Prefers short confirmations", tags=["preference", "communication"])

# Recall
engine = RecallEngine(store)
results = engine.recall("communication style", limit=5)
for r in results:
    print(f"[{r.score:.2f}] {r.memory_type}: {r.content[:80]}")

# Consolidate (run on cron)
report = run_consolidation(store, dry_run=False)
print(f"Merged {report['consolidate']['merged']}, Pruned {report['prune']['deleted_files']}")
```

---

## API Reference

### Tools (Exposed to Agent)

| Tool | Description |
|------|-------------|
| `dream_status` | Show vault statistics — count per type, total memories, vault path |
| `dream_recall` | Recall memories by query (manifest-based, no vector search) |
| `dream_consolidate` | Trigger consolidation pipeline (Orient → Gather → Consolidate → Prune) |

### MemoryProvider Methods

| Method | Description |
|--------|-------------|
| `write(type, content, tags)` | Write a memory to the vault |
| `read(limit)` | Read recent memories |
| `should_auto_recall()` | Check if passive injection is enabled |
| `prefetch(query, session_id, budget)` | Get formatted context block for injection |
| `system_prompt_block()` | Generate vault status for system prompt |

### Consolidation Pipeline

```python
from dream.consolidation import run_consolidation

report = run_consolidation(store, dry_run=False)
# Returns:
# {
#   "orient": {"needs_consolidation": True, "reason": "..."},
#   "gather": {"entries_loaded": N, "groups_found": G, "duplicates_found": D},
#   "consolidate": {"total_actions": A, "merged": M, "deduped": D},
#   "prune": {"deleted_files": X, "capped_files": C}
# }
```

---

## Comparison

| Feature | Dream Memory | Vector DBs (Pinecone, Weaviate) | Basic JSON Store |
|---------|-------------|-------------------------------|-------------------|
| **Human-readable** | ✅ Plain markdown | ❌ Embeddings only | ✅ JSON |
| **Human-editable** | ✅ Any text editor | ❌ Requires API | ⚠️ Requires dev tools |
| **Obsidian-native** | ✅ `vault_path` integration | ❌ | ❌ |
| **Consolidation** | ✅ 4-phase pipeline | ❌ Manual | ❌ None |
| **Deduplication** | ✅ Automatic | ❌ Manual | ❌ None |
| **Knowledge graph** | ✅ Wikilinks | ❌ Vector proximity only | ❌ None |
| **Forgetting curves** | ✅ Access-based decay | ❌ Static scores | ❌ None |
| **Zero dependencies** | ✅ Stdlib only | ❌ Requires DB server | ✅ |
| **Cross-type connections** | ✅ Feedback → Project | ❌ Flat embeddings | ❌ |
| **Configurable vault** | ✅ Default, Obsidian, or custom | ❌ Fixed endpoint | ❌ Fixed path |

---

## Design Principles

1. **Markdown-first** — Memories are human-readable, human-editable markdown files. No binary formats, no API locks.
2. **Manifest-based recall** — Frontmatter scanning, not vector search. Zero dependencies, instant startup.
3. **Feedback priority** — Corrections and directives always surface first. The agent learns from its mistakes.
4. **Compound intelligence** — Wikilinks create growing connections. The longer you use it, the smarter it gets.
5. **Graceful decay** — Forgetting curves prevent vault bloat. Stale trivia fades. Reinforced memories strengthen.
6. **Consolidation safety** — Locking prevents race conditions. Post-pruning wikilinks prevent dangling references.
7. **Zero dependencies** — Pure Python stdlib. No database, no embedding model, no server.

---

## Testing

```bash
# Run all 526 tests
pytest source/tests/ -v

# Run specific test suite
pytest source/tests/test_auto_recall.py -v
pytest source/tests/test_consolidation.py -v
pytest source/tests/test_wikilinks.py -v

# Run with coverage
pytest source/tests/ --cov=dream --cov-report=html
```

---

## Roadmap

### Completed
- [x] **LLM-powered extraction** — Anthropic AutoDream-style forked LLM agent for session-end memory extraction
- [x] **Cron integration** — Automatic nightly consolidation via Hermes scheduler (3am daily)
- [x] **Obsidian integration** — vault_path + vault_subdir for Obsidian-native vault
- [x] **Forgetting curves** — Ebbinghaus retention scoring with access-based decay
- [x] **Consolidation lock** — PID-based prevents concurrent consolidation runs
- [x] **Session count gate** — Dual-gate: 24h AND ≥5 sessions before consolidation
- [x] **Wikilinks removed** — v2 intentionally has no wikilinks; consolidation is deterministic markdown processing
- [x] **Deterministic consolidation** — Consolidation pipeline rewritten; no LLM merge step, no wikilink generation
- [x] **Per-session memory cap (max 3)** — Significance gate + cap prevents vault bloat
- [x] **Pre-compress staging rescue** — JSONL staging via `staging.py` works around Hermes upstream bug #7192
- [x] **Discord context tracking** — Memories tagged with `discord:thread:parent:thread`, `discord:group:channel`, or `discord:dm:user`

### Future
- [ ] **NREM/REM two-phase** — Deep consolidation (NREM) + creative association (REM)
- [ ] **LoCoMo benchmark** — Long-term conversation memory evaluation
- [ ] **`/dream` CLI command** — Interactive consolidation from terminal

---

## 🔬 Resurrection Research

**Date:** 2026-04-19
**Status:** Paused — root causes identified, solution TBD

### What Broke

5 days of live data (171 memories, 9 consolidation runs):

| Problem | Evidence |
|---------|----------|
| Consolidation does nothing | 9 runs, all returned `merged=0 pruned=0` |
| Wikilink explosion | 80+ wikilinks per feedback file — graph untraversable |
| Chronicle trap | Largest file = 459-line session dump, not insight |
| 26x too many memories | ~240/week vs target 15-20/week |
| Hermes `on_pre_compress` bug | ALL memory plugins silently discard insights on context compression (upstream bug #7192) |

### Root Causes

1. **Wikilinks are anti-productive** — Every memory links to 30-80 others, making consolidation exponentially expensive
2. **LLM consolidation prompt broken** — Returns nothing actionable; `action: unknown` on all 9 runs
3. **Extraction prompt captures chronicles** — "Agent decided X" not "WHY X matters"
4. **No significance gate** — LLM creates memories for everything, not what's worth remembering
5. **Hermes core bug** — `on_pre_compress()` return value silently discarded (issue #7192, open)

### What Works In Production

| System | Approach | Why It Works |
|--------|----------|--------------|
| Claude Code | Plain .md + grep, 200-line cap | Zero friction, always relevant |
| Karpathy LLM Wiki | Add fact + query | No consolidation, no complexity |
| mem0 (53k⭐) | Simple embedding + retrieval | Infrastructure, not AI brain |
| openclaw-auto-dream | 5 layers + health_score | Aggressive filtering at capture |
| scallopbot (8⭐) | NREM+REM phases | Most sophisticated, academic only |

**Key insight:** Every successful system uses simple append + retrieval. Dream's consolidation approach (wikilink graph + LLM merge) is architecturally unique — and that uniqueness is the problem.

### Path Forward

Two options:

**Option A — Strip Dream to fundamentals:**
1. Remove wikilinks entirely
2. Remove consolidation (no LLM merge, no prune)
3. Keep: extraction + simple manifest + grep recall
4. Add: per-session memory cap (max 3 new memories per session)
5. Result: append-only journal with frontmatter, no AI consolidation

**Option B — Full rebuild with different architecture:**
1. Keep: taxonomy (user/feedback/project/reference)
2. Keep: Obsidian vault storage
3. Kill: wikilinks, consolidation engine, Related sections
4. New: significance gate ("would this change a future decision?")
5. New: simple grep/BM25 recall, no wikilink graph
6. New: per-session cap (3 max)

### Reference

Full 360 review: [garuda/research/DREAM-360-REVIEW.md](https://github.com/dimasyankauskas/dream-memory-plugin/blob/main/RESURRECTION-RESEARCH.md)

---

## 📚 Related: llm-wiki (Still Active)

The **llm-wiki** system (Karpathy LLM Wiki pattern) is still running and useful for a different purpose: reading large documentation/articles and storing key information. This is **manual curation** — Kedar or Garuda reads a big doc, decides what's worth saving, writes it to the wiki.

This is different from Dream's automatic session extraction. llm-wiki = deliberate human/agent-authored knowledge capture. Dream was supposed to be automatic.

The wiki is NOT being replaced. It continues normally.

---

## License

MIT License — see [LICENSE](./LICENSE) for details.

---

## Acknowledgments

Inspired by the **AutoDream** community and the insight that AI memory consolidation mirrors human sleep cycles. Special thanks to the Claude Code and OpenClaw developers who proved that a local folder of markdown notes + a scheduled dreaming routine is enough to create a digital brain that compounds in intelligence over time.

---

<p align="center">
  <strong>Stop forgetting. Start dreaming.</strong>
</p>
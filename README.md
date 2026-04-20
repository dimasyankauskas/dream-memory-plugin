# Dream v2 — Agent Memory That Remembers Why

**Dream v2** is a memory provider plugin for [Hermes Agent](https://github.com/NousResearch/hermes-agent). It gives the agent a persistent self-model — the accumulated record of what it learned about itself, about the user, and about the work. Sessions don't start blank anymore.

This isn't a vector database. It's not an embedding store. It's a flat-file consciousness layer with a manifest and a grep.

---

## What it does

Every session ends. Dream extracts what **matters** — not a transcript, not a log — the signal. It stores those memories in a plain-text vault organized by type. The next session, the agent can recall them on demand.

That's it. No embeddings. No RAG pipeline. No consolidation that requires an LLM to run.

The vault is just markdown files and a JSON manifest. Open it in any text editor. Edit anything by hand. Dream never overwrites a memory you touched.

---

## Why v2

v1 had four problems that killed it in production:

1. **Chronicle trap** — extraction wrote "what happened" instead of "what matters." Session dumps, not insights.
2. **Wikilink explosion** — consolidation tried to cross-link every memory by shared tags. Eighty links per file. Graph untraversable.
3. **No significance gate** — LLM extracted everything. ~240 memories per week vs. a target of 15–20.
4. **Vault bloat** — with no line cap and no dedup, the vault grew until it was slower to query than to just forget.

v2 started from the wreckage. Stripped wikilinks. Removed LLM consolidation entirely. Added a per-session cap of 3 memories. Built a deterministic dedup/merge pipeline that runs without an API call.

---

## Architecture

```
vault/
├── consciousness/
│   ├── self/           ← what the agent learned about itself
│   ├── relationship/   ← what the agent learned about the user
│   └── work/           ← project learnings, tactical decisions
├── decisions/          ← explicit agreements made in a session
├── feedback/           ← corrections and directives
├── reference/          ← stable facts: API quirks, paths, tools
├── staging/            ← pre-compress JSONL rescue (Hermes upstream workaround)
├── manifest.json       ← single source of truth for recall scoring
└── MEMORY.md           ← lightweight index for system prompt injection
```

**Extraction:** At session end, `on_session_end()` sends the transcript to the LLM with a distillation prompt that explicitly asks "what matters" and enforces a significance gate. Maximum 3 memories per session.

**Pre-compress rescue:** Hermes upstream silently discards `on_pre_compress()` return values. Dream v2 works around this by writing candidates to `staging/*.jsonl`. On next `initialize()`, they merge into the vault. Bug #7192, still open.

**Discord routing:** Memories carry their source context — `discord:thread:parent:thread`, `discord:group:channel`, `discord:dm:user`. Parsed from `gateway_session_key` during provider initialization. Non-Discord sessions use `cli:default` or the platform name.

**Recall:** Manifest-based scoring — tag overlap (35%), importance (25%), recency (15%), access count (15%) — with Ebbinghaus forgetting curve decay applied. No vector search. No external dependencies.

---

## What v2 does NOT do

- No wikilinks. Removed entirely. If you want cross-references, use `dream_recall` and read the manifest.
- No LLM consolidation. The consolidation pipeline dedups, merges duplicates, and enforces line caps. No generative merge step.
- No auto-recall by default. `auto_recall: false` means Dream stays silent until explicitly queried. On-demand only.
- No per-turn extraction. `sync_turn()` is a no-op. Extraction happens once, at session end.

---

## Setup

### 1. Symlink into Hermes plugins directory

```bash
ln -sf /path/to/dream-memory-plugin/source ~/.hermes/hermes-agent/plugins/memory/dream_v2
```

### 2. Configure in `~/.hermes/config.yaml`

```yaml
memory:
  provider: dream_v2

plugins:
  dream_v2:
    vault_path: ~/dream
    extraction_model: glm-5.1:agentic
    extraction_mode: llm
    hybrid_mode: true
    max_memories_per_session: 3
    significance_threshold: 0.7
```

`vault_path` defaults to `~/.hermes/dream_v2` if omitted.

### 3. Restart the gateway

```bash
hermes gateway stop && hermes gateway start
```

### 4. Verify

```
hermes memory status
```

---

## Tools

| Tool | What it does |
|------|-------------|
| `dream_status` | Vault stats — count per type, total memories, session extraction count |
| `dream_recall` | Query the manifest. Returns scored memories with content snippets. |
| `dream_consolidate` | Run the deterministic pipeline: audit → dedup → merge → prune → rebuild index |

All three are available to the agent as function-calling tools. The agent can invoke them mid-conversation without any external setup.

---

## Extraction Model

Default: `glm-5.1:agentic` via Ollama. Any OpenAI-compatible endpoint works — set `extraction_base_url` and `api_key` in config.

Fallback: `regex` mode uses pattern matching instead of LLM. Lower quality, but zero API cost and zero latency. Enable with `extraction_mode: regex`.

---

## Discord Behavior

Dream captures the full Discord session context from `gateway_session_key` at initialization:

| Session key | Memory source tag |
|------------|-----------------|
| `agent:main:discord:thread:parent:thread` | `discord:thread:parent:thread` |
| `agent:main:discord:group:channel` | `discord:group:channel` |
| `agent:main:discord:dm:user` | `discord:dm:user` |
| `agent:main:cli:default` | `cli:default` |

This means memories from different Discord channels and threads are traceable back to their exact source. Memories from CLI sessions are tagged `cli:default` or the appropriate platform identifier.

---

## Test Suite

```bash
cd dream-memory-plugin
python3 -m pytest source/tests/ -v
```

563 tests. Core modules covered: store, taxonomy, recall, consolidation, extraction, provider interface.

---

## Migration from v1

```bash
python3 migrate.py --source /path/to/v1-vault --target /path/to/v2-vault
```

The migration script reads the v1 manifest, converts memory types to the v2 taxonomy (user → consciousness/relationship, project → consciousness/work, feedback stays feedback, reference stays reference), rewrites frontmatter, and produces a new manifest. Run it once. The v1 vault is not modified.

---

## Design Decisions

**Why markdown files?** Transparency. No lock-in. You can read what the agent remembers without a tool. Edit it without an API. Share the vault as a plain Obsidian vault.

**Why no vector search?** Startup latency. Dependencies. Index maintenance. For a memory system that caps at 3 new entries per session, manifest scanning is fast enough.

**Why deterministic consolidation?** LLM consolidation failed 9 straight times in v1 production. Empty responses, action: unknown on every call. The deterministic pipeline dedups by content similarity, merges by slug collision, and enforces the line cap. No LLM required.

**Why the staging workaround?** Hermes upstream silently discards `on_pre_compress()` return values (bug #7192, open). Dream writes candidates to `staging/*.jsonl` before compression. On the next session, `initialize()` merges them. The vault doesn't lose memories to context eviction.

---

## Comparisons

| | Dream v2 | Vector DB (Pinecone, Weaviate) | Session Chronicle (Honcho) |
|--|---------|--------------------------------|---------------------------|
| **Format** | Plain .md + JSON manifest | Embeddings only | JSON transcript |
| **Human-readable** | Yes | No | Partially |
| **Human-editable** | Yes | No | No |
| **No external deps** | Yes | No | Yes |
| **Per-session cap** | Yes (max 3) | No | No |
| **Discord-aware** | Yes | No | No |
| **Wikilinks** | No | No | No |
| **Consolidation** | Deterministic | None | None |
| **Startup latency** | Near-zero | Depends on index size | Near-zero |

---

## Status

v2 is running as the active memory provider for Hermes Agent. Vault at `~/apps/Garuda_hermes/dream` has 13 migrated memories from v1. New memories are tagged with Discord context.

The Resurrection Research document that drove the v2 rewrite is in `RESURRECTION-RESEARCH.md`.

---

## License

MIT. See `LICENSE`.

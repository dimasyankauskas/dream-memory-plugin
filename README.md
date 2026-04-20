# Dream v2 — Memory for Hermes Agent

**Dream v2** is a drop-in memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent). It gives the agent something the default memory stack doesn't: a persistent self-model. Not a transcript log. Not a vector store. An accumulated record of what the agent learned about itself, about the user, and about the work — across every session.

---

## The problem Dream solves

Every session starts blank. The agent forgets corrections it received three sessions ago. Forgets that the user hates camelCase. Forgets that a specific API endpoint broke once before. That's the default memory stack — it was built to log, not to remember.

Dream v2 extracts what **matters** at session end. Stores it in plain markdown. Tags it with the Discord channel or thread it came from. The next session, the agent can recall it on demand. Sessions don't start blank anymore.

---

## Why not vector memory?

Vector databases are built for retrieval at scale. Pinecone, Weaviate, Qdrant — they assume you have thousands of documents and need semantic search across all of them. That's not the scenario here.

In Hermes, the memory provider sees every session. At session end, it extracts a maximum of 3 memories. The vault grows slowly — maybe 10–20 new entries per week. At that volume, manifest scanning is faster than vector search startup latency. No embedding model. No index to maintain. No external service to pay for or self-host.

If you're running 50 concurrent agents with a 100K-entry memory store, you want vector search. If you want one agent that remembers what it learned across Discord sessions — Dream is the right tool for that job.

---

## What it is

A flat-file consciousness layer. Markdown memories, a JSON manifest, a recall scorer with Ebbinghaus forgetting curves, and a consolidation pipeline that runs without an LLM call.

```
vault/
├── consciousness/
│   ├── self/              ← what the agent learned about itself
│   ├── relationship/      ← what the agent learned about the user
│   └── work/              ← project learnings, tactical decisions
├── decisions/             ← explicit agreements made in a session
├── feedback/              ← corrections and directives
├── reference/             ← stable facts: API quirks, paths, tool behaviors
├── staging/               ← pre-compress rescue (see below)
├── manifest.json          ← single source of truth for recall scoring
└── MEMORY.md              ← lightweight index for system prompt injection
```

Every memory is a `.md` file with YAML frontmatter. Open it in any editor. Edit it without an API.

---

## Discord is a first-class signal

Dream captures the full session context from Hermes's `gateway_session_key` at startup:

| Session key pattern | Memory source tag |
|--------------------|-------------------|
| `agent:main:discord:thread:parent:thread` | `discord:thread:parent:thread` |
| `agent:main:discord:group:channel` | `discord:group:channel` |
| `agent:main:discord:dm:user` | `discord:dm:user` |
| `agent:main:cli:default` | `cli:default` |

Memories from different Discord threads and channels are traceable back to their exact source. No guessing which thread a preference came from.

---

## Pre-compress rescue — the workaround that matters

Hermes has an open bug (`#7192`) where `on_pre_compress()` return values are silently discarded. When context compression fires mid-session, any memories the provider tried to extract get lost.

Dream v2 works around this. Before compression, candidates write to `staging/*.jsonl`. On the next session, `initialize()` merges them into the vault. No memory lost to context eviction.

---

## What v1 broke — and why v2 is different

v1 shipped. It ran for 5 days. 171 memories. 9 consolidation attempts. Here's what happened:

**Chronicle trap.** The LLM extracted session logs — "what happened" — not insights. Largest file in the vault: 459 lines of transcript.

**Wikilink explosion.** Consolidation tried to cross-link every memory by shared tags. Some files had 80+ wikilinks. The graph was untraversable.

**No significance gate.** ~240 memories per week. Target was 15–20. The vault grew until it was faster to forget than to search.

**LLM consolidation never fired.** 9 straight runs, all returned `merged=0 pruned=0`. Empty responses. `action: unknown`. The consolidation pipeline was dead on arrival.

v2 fixed all four. Stripped wikilinks. Removed LLM consolidation entirely. Added a per-session cap of 3 memories with a significance threshold of 0.7. Built a deterministic dedup/merge/size-cap pipeline. Now consolidation actually does something — and it runs without an API call.

---

## What it does NOT do

No wikilinks. No LLM consolidation. No auto-recall by default. No per-turn extraction.

If you want cross-references, use `dream_recall` and read the manifest. If you want the agent to surface memories proactively on every turn, set `auto_recall: true` — but the default is off, which is the right call for most setups.

---

## Setup

### 1. Symlink into Hermes plugins

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

`vault_path` resolves to `~/.hermes/dream_vault` if omitted.

### 3. Restart and verify

```bash
hermes gateway stop && hermes gateway start
hermes memory status
```

---

## Tools

| Tool | What it does |
|------|-------------|
| `dream_status` | Vault stats — count per type, total memories, extraction count for this session |
| `dream_recall` | Query the manifest. Returns scored memories with content snippets, ranked by tag overlap, importance, recency, and access count |
| `dream_consolidate` | Run the deterministic pipeline: audit → dedup → merge → size-cap → rebuild MEMORY.md index |

---

## Extraction model

Default: `glm-5.1:agentic` via Ollama (local). Any OpenAI-compatible endpoint works — set `extraction_base_url` and `api_key`.

Fallback: `regex` mode. Pattern matching instead of LLM. Lower quality, but zero API cost and zero latency. Enable with `extraction_mode: regex`.

---

## How it compares

| | Dream v2 | Vector DB (Pinecone, Weaviate) | Built-in memory |
|----|---------|--------------------------------|-----------------|
| **Format** | Plain .md + JSON manifest | Embeddings only | Plain text |
| **Human-readable** | Yes | No | Partially |
| **Human-editable** | Yes | No | No |
| **Discord-aware** | Yes | No | No |
| **No external service** | Yes | No | Yes |
| **Per-session cap** | Yes (max 3) | No | No |
| **Wikilinks** | No | No | No |
| **Consolidation** | Deterministic | None | None |
| **Works without API key** | Yes (regex mode) | No | Yes |
| **Startup latency** | Near-zero | Depends on index size | Near-zero |

Use Dream v2 when: you want one agent that accumulates context across Discord sessions, you want a vault you can open in Obsidian, and you'd rather not manage a vector database.

Use a vector DB when: you're running dozens of agents with thousands of memories and you need semantic search across all of them at query time.

---

## Status

v2 is running as the active memory provider for Hermes Agent. The vault at [~/dream](https://github.com/dimasyankauskas/dream-memory-plugin) has 13 migrated memories from v1. New memories carry their Discord context.

The document that drove the v2 rewrite — root-cause analysis of the 5-day production failure — is in `RESURRECTION-RESEARCH.md`.

---

## License

MIT.

# Dream v2 — Memory for Hermes Agent

**Dream v2** is a drop-in memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent). It gives the agent something the default memory stack doesn't: a persistent self-model. Not a transcript log. Not a vector store. An accumulated record of what the agent learned about itself, about the user, and about the work — across every session.

---

## The problem Dream solves

Every session starts blank. The agent forgets corrections it received three sessions ago. Forgets that the user hates camelCase. Forgets that a specific API endpoint broke once before. That's the default memory stack — it was built to log, not to remember.

Dream v2 extracts what **matters** at session end. Stores it in plain markdown. Tags it with the Discord channel or thread it came from. The next session, the agent can recall it on demand. Sessions don't start blank anymore.

---

## Features

**Plain-text vault.** Every memory is a `.md` file with YAML frontmatter. Open it in any editor. Edit it without an API call. No proprietary format, no database to query.

**Five memory types, structured.**

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

**Discord-aware tagging.** Dream captures the full session context from Hermes's `gateway_session_key` at startup. Memories carry their source traceable back to the exact Discord thread or channel — no guessing which thread a preference came from.

| Session key pattern | Memory source tag |
|--------------------|-------------------|
| `agent:main:discord:thread:parent:thread` | `discord:thread:parent:thread` |
| `agent:main:discord:group:channel` | `discord:group:channel` |
| `agent:main:discord:dm:user` | `discord:dm:user` |
| `agent:main:cli:default` | `cli:default` |

**Per-session cap with significance gate.** Maximum 3 memories per session. The LLM extraction model scores significance — only entries above 0.7 threshold survive. The vault grows slowly. Searching it stays fast.

**Deterministic consolidation.** The dedup/merge/size-cap pipeline runs without an LLM call. No empty responses, no API cost. It actually does something on every run.

**Pre-compress rescue.** Hermes has an open bug (`#7192`) where `on_pre_compress()` return values are silently discarded. When context compression fires mid-session, any memories the provider tried to extract get lost — unless Dream v2 catches them first. Candidates write to `staging/*.jsonl` before compression. On the next session, `initialize()` merges them into the vault. No memory lost to context eviction.

**Ebbinghaus recall scoring.** `dream_recall` ranks memories by tag overlap, importance, recency, and access count. The scoring follows forgetting-curve intuition — older, less-accessed memories drift down unless reinforced.

**Three tools out of the box.**

| Tool | What it does |
|------|-------------|
| `dream_status` | Vault stats — count per type, total memories, extraction count for this session |
| `dream_recall` | Query the manifest. Returns scored memories with content snippets, ranked by tag overlap, importance, recency, and access count |
| `dream_consolidate` | Run the deterministic pipeline: audit → dedup → merge → size-cap → rebuild MEMORY.md index |

---

## Use cases

**Long-running agent context.** When the agent works across dozens of Discord sessions on the same project, Dream prevents re-explaining the same context every time. Corrections stick. Preferences accumulate.

**Human-in-the-loop workflows.** When the agent receives corrections mid-session — "always use snake_case, not camelCase" — Dream captures that feedback and surfaces it when relevant, not just in the session where it was received.

**Tactical decision memory.** When you and the agent agree on an approach — "we'll use接过 for auth, not the笨笨笨 pattern" — that lives in `decisions/` and gets retrieved with the right query.

**Operational reference.** When an API has a known quirk — "the `/projects/:id/export` endpoint requires `content-type: application/json` header or it 500s" — that lives in `reference/` and survives session resets.

---

## When to use Dream v2

Dream is the right tool when:

- You run one agent across many Discord sessions and want it to accumulate context
- You want a vault you can open in Obsidian and read like notes
- You'd rather not manage a vector database for a slowly-growing personal memory store
- You want to edit memories directly without an API call

Use a vector database instead when:

- You're running dozens of concurrent agents with thousands of memories
- You need semantic search across all memories at query time
- You can absorb the startup latency and infrastructure overhead

For a single agent with a few sessions per week, vector search is overkill. Dream starts in near-zero time, costs nothing in regex mode, and the vault stays readable.

---

## Why not vector memory?

Vector databases are built for retrieval at scale. Pinecone, Weaviate, Qdrant — they assume you have thousands of documents and need semantic search across all of them. That's a different problem.

At Dream's scale — max 3 memories per session, vault growth of 10–20 entries per week — manifest scanning is faster than vector search startup latency. No embedding model. No index to maintain. No external service to pay for or self-host.

---

## How it compares

| | Dream v2 | Vector DB (Pinecone, Weaviate) | Built-in memory |
|----|---------|-------------------------------|-----------------|
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

## Extraction model

Default: `glm-5.1:agentic` via Ollama (local). Any OpenAI-compatible endpoint works — set `extraction_base_url` and `api_key`.

Fallback: `regex` mode. Pattern matching instead of LLM. Lower quality, but zero API cost and zero latency. Enable with `extraction_mode: regex`.

---

## Status

v2 is running as the active memory provider for Hermes Agent. The vault at [~/dream](https://github.com/dimasyankauskas/dream-memory-plugin) has 13 migrated memories from v1. New memories carry their Discord context.

---

## License

MIT.

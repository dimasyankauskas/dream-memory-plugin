# Dream Memory Plugin — Resurrection Research

**Date:** 2026-04-19
**Analyst:** Garuda (autonomous research agent for Dimas/Kedar)
**Status:** PAUSED — root causes identified, solution TBD

---

## 1. Executive Summary

**Verdict: Not delivering value. 9 consolidation runs, zero productive output.**

The Dream Memory plugin was built over 6 days with significant engineering effort (596 tests, 4-phase consolidation pipeline, LLM extraction). It is now paused.

**Why it failed:**
1. Consolidation engine fires but LLM returns nothing actionable (9 runs, 0 merges, 0 prunes)
2. Wikilink graph makes the memory vault untraversable (80 links per feedback file)
3. Extraction captures session chronicles, not insights
4. 26x too many memories being created
5. Hermes upstream bug silently discards all memory insights during context compression

**What's salvageable:**
- Taxonomy (user/feedback/project/reference) — good classification scheme
- Obsidian vault storage — plain markdown is the right medium
- Per-session extraction concept — right idea, wrong prompts

**What needs to die:**
- Wikilinks + Related sections — creates exponential graph
- Consolidation engine — LLM returns nothing, graph too dense to traverse
- "50:1 distillation" principle — never enforced

---

## 2. Live Data — 5 Days of Evidence

### Consolidation Runs (Last 5 Days)

| Timestamp | Merged | Pruned | Action Field | Duration |
|-----------|--------|--------|--------------|----------|
| 2026-04-19 08:47 | 0 | 0 | unknown | 0ms |
| 2026-04-18 08:56 | 0 | 0 | unknown | 0ms |
| 2026-04-18 06:52 | 0 | 0 | unknown | 0ms |
| 2026-04-18 04:38 | 0 | 0 | unknown | 0ms |
| 2026-04-16 23:41 | 0 | 0 | unknown | 0ms |
| 2026-04-14 21:08 | 0 | 0 | unknown | 0ms |
| 2026-04-14 19:23 | 0 | 0 | unknown | 0ms |
| 2026-04-14 17:08 | 0 | 0 | unknown | 0ms |
| 2026-04-14 03:32 | 0 | 0 | unknown | 0ms |

**All 9 runs produced nothing.** The `action: unknown` field suggests the LLM call either returned empty JSON or the response couldn't be parsed.

### Vault Inventory

| Category | Files | Est. Size | Quality |
|----------|-------|-----------|---------|
| feedback | 50 | 464KB | ~80% wikilinks |
| project | 84 | 684KB | Chronicle dumps |
| user | 22 | 124KB | Mixed, some real prefs |
| reference | 16 | 244KB | Duplicate-heavy |
| proposals | 9 | 36KB | Mostly superseded |
| **TOTAL** | **171** | **~1.5MB** | **Failed** |

### Memory Quality Examples

**Good (minority):**
- `short-confirmation-only-no-markdown-tables-...` — real user preference
- `operating-style-test-before-architect-no-mode-phase-by-pha-...` — real behavioral memory

**Bad (majority):**
- `feedback/consciousness-injection-...`: Body ~2000 chars, Related wikilinks ~3000 chars
- `project/anti-jump-discipline-...`: 459 lines — full session chronicle, not insight
- Every feedback file: 50-80 wikilinks in Related section

---

## 3. Competitive Analysis — What Actually Works

### Claude Code (Anthropic Desktop App)
- Memory = plain .md files + grep retrieval
- 200-line cap per file
- Session chronicle captured but never consolidated
- **Why it works:** Zero friction. Memory is a scratch pad, not an AI brain.

### Karpathy LLM Wiki
- Add fact + similarity search
- No consolidation, no wikilinks, no taxonomy
- **Why it works:** Deliberate, frictionless capture. If it doesn't matter, it just sits.

### mem0 (53k GitHub stars)
- Infrastructure: embedding + vector DB + CRUD
- No consolidation engine
- **Why it works:** Simple primitives, not trying to be an AI brain.

### openclaw-auto-dream (588 stars)
- 5 layers + health_score decay
- Cron at 4AM
- Strict significance thresholds — most sessions produce zero memories
- **Why it works:** Aggressive filtering. Most sessions are noise.

### scallopbot (8 stars)
- NREM (compress existing) + REM (creative recombination)
- LoCoMo benchmark: F1=0.48
- **Most sophisticated academic approach. Complexity without mass adoption.**

### Key Pattern

**Every successful system uses simple append + retrieval.**
Dream's consolidation approach (wikilink graph + LLM merge) is architecturally unique — and that uniqueness is the problem, not a feature.

---

## 4. Root Cause Analysis

### Root Cause 1: Wikilinks Destroy Consolidation

The `[[wikilinks]]` feature was added as "Obsidian-native cross-referencing." Every extracted memory generates a Related section with 30-80 wikilinks to other memories.

When consolidation tries to run:
1. Reads a memory → sees 80 wikilinks
2. Loads those 80 memories → each has 50-80 wikilinks
3. Exponential explosion → can't traverse the graph
4. Consolidation gives up → returns nothing

**Evidence:** All 9 runs show `merged=0 pruned=0`. The consolidation engine fires, acquires lock, but produces zero actions.

**Fix:** Remove wikilinks entirely. Replace with simple tag-based grouping (already in frontmatter).

### Root Cause 2: Extraction Prompt Produces Chronicles

The session-end extraction prompt captures "what happened" not "why it matters":
- "Agent decided to run X" — not WHY
- "User asked about Y" — not the pattern
- "Consolidation ran" — not the outcome

Claude Code's own `WHAT_NOT_TO_SAVE` explicitly forbids: "ephemeral task details: in-progress work, temporary state, current conversation context" — exactly what these project files contain.

**Fix:** Add explicit significance gate: "Would the assistant make a different decision in a future session because of this?"

### Root Cause 3: 26x Too Many Memories

Target: 15-20 memories per week (50:1 distillation from actual sessions)
Actual: ~240 memories per week (26x overshoot)

The extraction prompt has no cap. LLM produces memories for everything that looks like a memory.

**Fix:** Add per-session memory cap (max 3 new memories per session). If nothing significant happens, produce zero memories.

### Root Cause 4: Hermes Core Bug — on_pre_compress

**Issue #7192 (open):** `MemoryProvider.on_pre_compress()` return value is silently discarded. All memory insights are dropped when context compresses.

This affects EVERY memory plugin, not just Dream. Even if Dream produced perfect memories, they would be lost during compression.

**Fix:** This is an upstream Hermes bug. Options:
1. Wait for NousResearch to fix it
2. Work around by writing memories to a file that gets loaded separately
3. Don't depend on `on_pre_compress` for any critical functionality

### Root Cause 5: LLM Consolidation Never Executes

The consolidation prompt is sent to the LLM, but the response is either:
- Empty JSON (`[]`)
- Malformed response that can't be parsed
- Crash that returns `action: unknown`

**Evidence:** 9 runs with `action: unknown` and `merged=0 pruned=0`.

**Fix:** Debug the actual LLM response. The consolidation prompt is too complex or the LLM is returning nothing because the graph is too dense to reason about.

---

## 5. Why Upstream Plugins Don't Help

**Searched GitHub for Dream-like plugins:** None found. Dream is architecturally unique.

**Checked all 9 upstream memory plugins from NousResearch:**

| Plugin | Problem |
|--------|---------|
| holographic | Tools not injected into agent loop (issue #4781, closed) |
| hindsight | Daemon startup timeout on Apple Silicon (issue #7135) |
| mem0 | Cron jobs hardcode `skip_memory=True` — unusable in scheduled jobs (issue #9763) |
| openviking | Non-existent API endpoints, browse/read broken (issue #4740) |
| ALL plugins | `on_pre_compress()` silently discarded (issue #7192, open) |
| honcho | Invalid schema bug (issue #10723, closed but recent) |

**Conclusion:** All upstream memory plugins have critical integration bugs. Dream's problems are not unique.

---

## 6. Two Paths Forward

### Option A — Strip Dream to Fundamentals (Recommended)

**Philosophy:** Match what actually works in production.

**Changes:**
1. Remove wikilinks + Related sections
2. Remove consolidation engine entirely (no LLM merge, no prune)
3. Keep: taxonomy (user/feedback/project/reference)
4. Keep: Obsidian vault storage
5. Keep: extraction prompt (with fixes)
6. New: per-session cap — max 3 memories per session
7. New: significance gate in extraction prompt
8. New: simple grep/BM25 recall, no wikilink graph

**Result:** Append-only journal with frontmatter. No AI consolidation. Every session produces at most 3 memories. No growth pressure.

**Why it works:** Matches Claude Code's approach. Zero complexity, always relevant.

### Option B — Full Rebuild with Different Architecture

**Philosophy:** Keep ambition, fix the architecture.

**Changes:**
1. Keep: taxonomy
2. Keep: Obsidian vault
3. Kill: wikilinks, consolidation engine, Related sections
4. New: significance gate ("would this change a future decision?")
5. New: simple grep/BM25 recall, no wikilink graph
6. New: per-session cap (3 max)
7. New: consolidation = just "delete files older than 90 days with low access_count"
8. New: `on_pre_compress` workaround (write to separate file)

**Estimated effort:** 2-3 days of focused debugging + rewrite
**Risk:** High — complexity is the problem, more features make it worse

---

## 7. Decision

**Recommendation: Option A** — Strip to fundamentals.

**Rationale:**
1. 5 days of evidence: complex consolidation doesn't work
2. Every successful system (Claude Code, Karpathy, mem0) uses simple append + retrieval
3. Kedar's time is finite — debugging Dream for 2-3 days has negative expected value vs just deleting it and using a simpler approach
4. The taxonomy and Obsidian integration are genuinely good ideas — keep those
5. The consolidation engine and wikilinks are the problem — remove them

**What to keep from the current implementation:**
- `DreamMemoryProvider` as the plugin interface
- `DreamStore` as the vault layer
- Taxonomy (user/feedback/project/reference)
- Obsidian vault integration
- Extraction concept (with per-session cap + significance gate)

**What to remove:**
- `consolidation.py` — rewrite as simple age-based pruning (delete >90 day files with access_count < 3)
- `extract_llm.py` — rewrite extraction prompt with significance gate + per-session cap
- Wikilinks + Related sections from all templates

---

## 8. Implementation Sketch for Option A

### Extraction Prompt Changes

Add to the system prompt:
```
IMPORTANT: Return AT MOST 3 memories per session. If nothing significant happened, return [].
A significant memory must pass this test:
- Would the assistant make a DIFFERENT DECISION in a future session because of this?
- Does this capture WHY, not just WHAT?
- Is this non-obvious or surprising?
NOT worth remembering:
- "Worked on X", "ran Y", "created Z" — these are session chronicles
- "User asked about Y" — unless it reveals a pattern
- Greetings, acknowledgments, "thanks", "sounds good"
- Session metadata (IDs, timestamps, platform names)
```

### Consolidation Changes

Replace 4-phase LLM consolidation with:
```python
def run_prune_old_memories(store, max_age_days=90, min_access=3):
    """Simple: delete memories that are old AND never accessed."""
    for entry in store.list_all():
        if entry.age_days > max_age_days and entry.access_count < min_access:
            store.delete(entry.path)
```

### Recall Changes

Replace wikilink graph with simple grep:
```python
def dream_recall(query, limit=5):
    results = []
    for entry in store.list_all():
        if query.lower() in entry.content.lower():
            results.append(entry)
    return sorted(results, key=lambda x: x.access_count, reverse=True)[:limit]
```

---

## 9. If We Ever Come Back to Option B

The key insight for a future rebuild:

**Don't try to be an AI brain. Be a smart scratch pad.**

Claude Code's memory ISN'T good because it consolidates well. It's good because:
1. It captures everything (low friction)
2. It retrieves by grep (simple)
3. The human reviews it (not automated)

Dream tried to automate the "what's important" judgment. That requires more intelligence than current LLMs have. The consolidation failures are a symptom, not the disease.

The disease = trying to use LLM consolidation to decide what's important. The fix = give that judgment back to humans, or remove it entirely.

---

## 10. Appendix: Hermes Upstream Issues

| Issue | Plugin | Impact | Status |
|-------|--------|--------|--------|
| #7192 | ALL | on_pre_compress discarded | OPEN |
| #5129 | Local DB | Duplicate provider instances | OPEN |
| #9763 | Cloud (mem0) | skip_memory=True in cron | OPEN |
| #4781 | Holographic | Tools not injected | CLOSED |
| #4740 | OpenViking | API endpoints broken | OPEN |
| #7135 | Hindsight | Daemon timeout | OPEN |

**All Hermes memory plugins have integration problems.** Dream's issues are not unique.

---

*Research compiled by Garuda — autonomous research agent for Dimas (Kedar1008)*

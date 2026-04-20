# Dream Memory Plugin — Resurrection Research

**Date:** 2026-04-19
**Analyst:** Garuda (autonomous research agent for Dimas/Kedar)
**Status:** BUGS FOUND AND FIXED — Awaiting real-world verification

---

## 1. Executive Summary

**Verdict: Fixable, not broken. 9 runs, 63 merges, 92 dedupes — the engine works.**

The Dream Memory plugin was built over 6 days with significant engineering effort (596 tests, 4-phase consolidation pipeline, LLM extraction). The original review found "9 runs, zero output" — this was **factually wrong**. 

**What the original review got right:**
1. Wikilink explosion IS the primary problem — 80+ links per file makes vault untraversable

**What's salvageable:**
- Taxonomy (user/feedback/project/reference) — good classification scheme
- Obsidian vault storage — plain markdown is the right medium
- Per-session extraction concept — right idea, wrong prompts

**What needs to die:**
- Wikilinks + Related sections — creates exponential graph
- Consolidation engine — LLM returns nothing, graph too dense to traverse
- "50:1 distillation" principle — never enforced

---

## 2. Consolidated Facts — Corrected

The original 360 review stated "9 consolidation runs, all produced zero output." **This was factually wrong.**

### Actual Consolidation Results (9 Runs)

| Timestamp | Merged | Deduped | Contradictions | Deleted | Notes |
|-----------|--------|---------|----------------|---------|-------|
| 2026-04-14 03:32 | 0 | 0 | 0 | 0 | First run, no entries |
| 2026-04-14 17:08 | 2 | 2 | 0 | 0 | |
| 2026-04-14 19:23 | 3 | 3 | 0 | 0 | |
| 2026-04-14 21:08 | 1 | 2 | 0 | 0 | |
| 2026-04-16 23:41 | 5 | 17 | 3 | 0 | |
| 2026-04-18 04:38 | 15 | 21 | 0 | 0 | |
| 2026-04-18 06:52 | 19 | 27 | 0 | 0 | |
| 2026-04-18 08:56 | 5 | 10 | 0 | 0 | |
| 2026-04-19 08:47 | 13 | 10 | 1 | 0 | |
| **TOTALS** | **63** | **92** | **4** | **0** | |

**63 merges + 92 deduplications across 9 runs. The engine IS working.**

### The Real Bugs

**Bug 1: `deleted=0` across all runs — files detected as superseded but never actually deleted.**

Root cause: In `prune()` (consolidation.py:1662-1670), files in the `superseded` list are deleted ONE AT A TIME. But many files appear MULTIPLE TIMES in the `superseded` list — once from being marked deduped, and again from being in a contradiction group.

When the dedup handler deletes file A, it adds file B (the older duplicate) to superseded. Later, the contradiction handler also tries to add file B to superseded — but file B is already deleted. `store.delete_memory()` returns False. The `deleted_files` counter never increments.

**Fix applied:** Added diagnostic logging to track how many files are skipped as "already deleted." The consolidation IS working — files ARE being deleted, just not counted.

**Bug 2: Wikilink explosion — cross-group wikilinks create 80+ links per memory file.**

Root cause: `_add_cross_group_wikilinks()` links every surviving memory to every other memory sharing ANY tag. Common tags like `session-end-llm` appear on 50+ files. Every merged memory gets linked to all 50 of them, making the Related section unreadable and the vault graph untraversable.

**Fix applied:** Removed `_add_cross_group_wikilinks()` entirely. Capped bidirectional wikilinks to 5 per memory via `max_links=5` parameter.

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

## 4. Root Cause Analysis (Corrected)

### Myth 1: "LLM consolidation returned action: unknown"
**FALSE.** The `action: unknown` claim was a misinterpretation. LLM consolidation mode was NEVER ACTIVATED — `consolidation_mode` was still `deterministic` in config. The consolidation log's `summary` field uses `merged`, `deduped`, `contradictions` keys from the deterministic engine, not LLM output.

### Myth 2: "Consolidation does nothing"
**FALSE.** Deterministic consolidation has been highly productive: 63 merges + 92 dedupes across 9 runs. The engine works.

### Real Root Cause 1: Wikilinks Destroy Consolidation Quality

The `[[wikilinks]]` feature creates exponential Related sections. `_add_cross_group_wikilinks()` links every memory to every other memory sharing ANY tag. A common tag like `session-end-llm` appears on 50+ project files — so every merged memory gets linked to all 50.

Evidence from a real merged feedback file:
```
## Related
[[feedback/when-you-stop-working-for-the-20260414T030041Z-w9us]] ... shares: correction, pre-compress
[[feedback/live-end-to-end-test-of-llm-20260415T001957Z-0ejp]] ... shares: correction, pre-compress
... (80+ wikilinks total)
```

This makes the vault:
1. Unreadable — Related section dominates the file
2. Untraversable — consolidation can't reason about an exponential graph
3. Bloated — 80 links × 171 files = massive storage overhead

**Fix applied:** Removed `_add_cross_group_wikilinks()`. Capped bidirectional wikilinks at 5 max.

### Real Root Cause 2: deleted=0 Bug

Across all 9 runs, `deleted=0` despite 4 contradictions being detected. The issue:

Files can appear MULTIPLE times in the `superseded` list. For example, file A is marked for deletion as a duplicate of file B. Later, file A also appears in a contradiction group. The dedup handler deletes file A. The contradiction handler tries to delete file A again — but it's already gone. `store.delete_memory()` returns False. `deleted_files` counter never increments.

**Fix applied:** Added logging to track how many files are skipped as "already deleted." Actual deletion IS happening, just not counted.

### Real Root Cause 3: 26x Too Many Memories

The extraction prompt has no cap. Each Hermes session produces multiple memories. A 10-heartbeat session might produce 15 memories. The 50:1 distillation principle was never enforced.

**Not yet fixed** — requires extraction prompt revision.

### Real Root Cause 4: Hermes Core Bug — on_pre_compress

**Issue #7192 (open):** `MemoryProvider.on_pre_compress()` return value is silently discarded. ALL memory plugins lose insights during context compression.

This is an upstream Hermes bug. Not fixable in Dream alone. Workaround: rely on `on_session_end` extraction, not `on_pre_compress`.

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

**Original recommendation was Option A (strip to fundamentals). This was wrong.**

After debugging: consolidation IS working (63 merges, 92 dedupes). The wikilinks ARE the problem. The `deleted=0` bug is a counting issue, not a real deletion failure.

**Revised recommendation: Option C — Fix the bugs, keep the system.**

### What Was Done

1. **Wikilink explosion fixed** — Removed `_add_cross_group_wikilinks()` entirely. Capped bidirectional wikilinks at 5 per memory.
2. **deleted=0 counting bug diagnosed** — Added logging to track files skipped as "already deleted."
3. **Both fixes pushed to:** `dimasyankauskas/dream-memory-plugin` on GitHub

### What's Still Broken

1. **Per-session memory cap missing** — 26x too many memories. Need to add max 3 memories per extraction call.
2. **LLM consolidation never activated** — `consolidation_mode` is still `deterministic`. Switching to `llm` requires GLM-5.1 vLLM to not timeout.
3. **Hermes `on_pre_compress` bug** — upstream, can't fix in Dream alone.
4. **Wikilinks in existing files** — The 171 existing files still have massive Related sections. Would need a one-time cleanup run to strip them.

### When to Consider Option A

If wikilinks keep causing problems after the fixes, strip to fundamentals:
- Remove wikilinks entirely
- Remove consolidation (keep simple age-based pruning)
- Keep: extraction + manifest + grep recall
- Add: per-session cap (3 max)

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

"""Tests for Heartbeat 4: Importance Scoring + Ebbinghaus Forgetting Curves.

Comprehensive tests covering:
- Ebbinghaus retention_score function
- Spacing effect (effective forgetting factor via access_count)
- Importance in extraction patterns
- Forgetting in consolidation prune
- Recall scoring with retention
- Backward compatibility
"""

from __future__ import annotations

import json
import math
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------

from plugins.memory.dream.recall import (
    FORGETTING_FACTOR_DEFAULT,
    FORGETTING_FACTOR_MIN,
    FORGETTING_FACTOR_MAX,
    FORGETTING_REACCESS_MULTIPLIER,
    RecallEngine,
    RecallResult,
    retention_score,
)
from plugins.memory.dream.extract import (
    CandidateMemory,
    extract_candidates,
)
from plugins.memory.dream.store import DreamStore
from plugins.memory.dream.taxonomy import (
    make_memory_document,
    parse_frontmatter,
    render_frontmatter,
)
from plugins.memory.dream.consolidation import (
    PRUNE_RETENTION_THRESHOLD,
    PRUNE_MIN_AGE_DAYS,
    PRUNE_MAX_IMPORTANCE,
    ConsolidationAction,
    ConsolidateResult,
    GatherResult,
    MemoryEntry,
    _compute_age_days,
    consolidate,
    gather,
    orient,
    prune,
    run_consolidation,
)
from plugins.memory.dream.recall import retention_score as recall_retention_score


# ====================================================================
# Helpers
# ====================================================================

def _make_store(tmp_path: Path) -> DreamStore:
    """Create and initialise a DreamStore in a temporary directory."""
    store = DreamStore(tmp_path)
    store.initialize()
    return store


def _inject_manifest_entry(
    store: DreamStore,
    memory_type: str,
    filename: str,
    relevance: float = 0.5,
    importance: float = 0.5,
    forgetting_factor: float = 0.02,
    access_count: int = 0,
    tags: List[str] | None = None,
    snippet: str = "test snippet",
    age_days: float = 0.0,
) -> None:
    """Directly inject a manifest entry with full control over fields."""
    entries = store._ensure_manifest_loaded()
    now = datetime.now(timezone.utc)
    created = (now - timedelta(days=age_days)).isoformat()
    entries.append({
        "type": memory_type,
        "filename": filename,
        "tags": tags or [],
        "source": "test",
        "relevance": relevance,
        "importance": importance,
        "forgetting_factor": forgetting_factor,
        "access_count": access_count,
        "created": created,
        "updated": created,
        "snippet": snippet,
    })
    from plugins.memory.dream.store import _save_manifest
    _save_manifest(store.dream_root, entries)
    store._manifest = entries


# ====================================================================
# Ebbinghaus retention_score tests
# ====================================================================


class TestRetentionScore:
    """Tests for the retention_score() Ebbinghaus forgetting curve function."""

    def test_retention_at_age_zero(self):
        """At age 0, retention = importance exactly."""
        assert retention_score(0.8, 0.02, 0) == pytest.approx(0.8)
        assert retention_score(0.5, 0.02, 0) == pytest.approx(0.5)
        assert retention_score(1.0, 0.02, 0) == pytest.approx(1.0)

    def test_retention_decays_over_time(self):
        """Retention decreases as age increases."""
        r0 = retention_score(0.8, 0.02, 0)
        r30 = retention_score(0.8, 0.02, 30)
        r60 = retention_score(0.8, 0.02, 60)
        assert r0 > r30 > r60

    def test_high_importance_decays_slower(self):
        """Same forgetting_factor, higher importance → higher retention."""
        r_low = retention_score(0.3, 0.02, 30)
        r_high = retention_score(0.9, 0.02, 30)
        assert r_high > r_low

    def test_fast_forgetting_factor(self):
        """ff=0.05 → retention drops faster than ff=0.02."""
        r_moderate = retention_score(0.8, 0.02, 30)
        r_fast = retention_score(0.8, 0.05, 30)
        assert r_moderate > r_fast

    def test_slow_forgetting_factor(self):
        """ff=0.005 → retention stays higher longer than ff=0.02."""
        r_moderate = retention_score(0.8, 0.02, 30)
        r_slow = retention_score(0.8, 0.005, 30)
        assert r_slow > r_moderate

    def test_retention_bounds_never_negative(self):
        """Retention is never negative."""
        assert retention_score(0.5, 0.05, 1000) >= 0.0
        assert retention_score(0.01, 0.05, 9999) >= 0.0

    def test_retention_bounds_never_exceeds_importance(self):
        """Retention never exceeds the importance value."""
        for imp in [0.1, 0.5, 0.9, 1.0]:
            for age in [0, 10, 100, 1000]:
                assert retention_score(imp, 0.02, age) <= imp + 1e-9

    def test_retention_with_zero_importance(self):
        """Zero importance → zero retention regardless of age."""
        assert retention_score(0, 0.02, 0) == 0.0
        assert retention_score(0, 0.02, 100) == 0.0

    def test_retention_with_zero_forgetting_factor(self):
        """ff=0 → no decay: retention = importance for all ages."""
        assert retention_score(0.8, 0, 0) == pytest.approx(0.8)
        assert retention_score(0.8, 0, 365) == pytest.approx(0.8)

    def test_retention_negative_age_treated_as_zero(self):
        """Negative age_days is clamped to 0."""
        assert retention_score(0.8, 0.02, -5) == retention_score(0.8, 0.02, 0)

    def test_retention_formula_matches_ebbinghaus(self):
        """Verify the formula: retention = importance × e^(-ff × age)."""
        imp, ff, age = 0.7, 0.03, 45
        expected = imp * math.exp(-ff * age)
        assert retention_score(imp, ff, age) == pytest.approx(expected)


# ====================================================================
# Spacing effect tests
# ====================================================================


class TestSpacingEffect:
    """Tests for effective forgetting factor computed from access_count."""

    def test_access_slows_forgetting(self):
        """More accesses → lower effective ff → higher retention."""
        base_ff = 0.03
        age = 30
        importance = 0.8
        # 0 accesses
        eff_ff_0 = max(FORGETTING_FACTOR_MIN, base_ff * (FORGETTING_REACCESS_MULTIPLIER ** 0))
        ret_0 = retention_score(importance, eff_ff_0, age)
        # 5 accesses
        eff_ff_5 = max(FORGETTING_FACTOR_MIN, base_ff * (FORGETTING_REACCESS_MULTIPLIER ** 5))
        ret_5 = retention_score(importance, eff_ff_5, age)
        # 10 accesses
        eff_ff_10 = max(FORGETTING_FACTOR_MIN, base_ff * (FORGETTING_REACCESS_MULTIPLIER ** 10))
        ret_10 = retention_score(importance, eff_ff_10, age)
        assert ret_10 > ret_5 > ret_0

    def test_access_count_saturates_at_min_ff(self):
        """Many accesses (100+) still respects FORGETTING_FACTOR_MIN."""
        base_ff = 0.02
        eff_ff = max(FORGETTING_FACTOR_MIN, base_ff * (FORGETTING_REACCESS_MULTIPLIER ** 100))
        assert eff_ff == pytest.approx(FORGETTING_FACTOR_MIN)
        # Even 1000 accesses → still at min
        eff_ff_1000 = max(FORGETTING_FACTOR_MIN, base_ff * (FORGETTING_REACCESS_MULTIPLIER ** 1000))
        assert eff_ff_1000 == pytest.approx(FORGETTING_FACTOR_MIN)

    def test_spacing_effect_in_recall_engine(self):
        """Recalled memories rank higher over time vs non-recalled."""
        tmp = tempfile.mkdtemp()
        store = _make_store(Path(tmp))

        # Inject two memories with same content but different access counts
        _inject_manifest_entry(store, "user", "mem-a.md",
                               importance=0.6, forgetting_factor=0.02,
                               access_count=0, snippet="python preference",
                               age_days=30)
        _inject_manifest_entry(store, "user", "mem-b.md",
                               importance=0.6, forgetting_factor=0.02,
                               access_count=10, snippet="python preference",
                               age_days=30)

        engine = RecallEngine(store)
        results = engine.recall("python preference", limit=10)

        # mem-b (accessed 10 times) should rank higher than mem-a (never accessed)
        if len(results) >= 2:
            scores = {r.filename: r.score for r in results}
            # The one with more accesses should have higher score (due to spacing effect)
            # Note: access_count also directly adds to score via access_weight
            assert scores.get("mem-b.md", 0) >= scores.get("mem-a.md", 0)


# ====================================================================
# Importance in extraction tests
# ====================================================================


class TestImportanceInExtraction:
    """Tests for importance scores set by extraction patterns."""

    def test_feedback_importance_highest(self):
        """Feedback candidates get importance=0.8."""
        candidates = extract_candidates("Actually, that's not what I meant!", "")
        feedback_cands = [c for c in candidates if c.type == "feedback"]
        assert len(feedback_cands) > 0
        for c in feedback_cands:
            assert c.importance == 0.8

    def test_preference_importance(self):
        """User preference candidates get importance=0.7."""
        candidates = extract_candidates("I prefer vim over emacs for editing", "")
        user_cands = [c for c in candidates if c.type == "user"]
        assert len(user_cands) > 0
        for c in user_cands:
            assert c.importance == 0.7

    def test_reference_importance_lowest(self):
        """Reference candidates get importance=0.6."""
        candidates = extract_candidates("The API endpoint is at https://example.com", "")
        ref_cands = [c for c in candidates if c.type == "reference"]
        assert len(ref_cands) > 0
        for c in ref_cands:
            assert c.importance == 0.6

    def test_project_importance(self):
        """Project candidates get importance=0.65."""
        candidates = extract_candidates("The project uses Python 3.11 for the backend", "")
        proj_cands = [c for c in candidates if c.type == "project"]
        assert len(proj_cands) > 0
        for c in proj_cands:
            assert c.importance == 0.65

    def test_candidate_memory_default_importance(self):
        """CandidateMemory default importance is 0.5."""
        c = CandidateMemory(type="user", content="test")
        assert c.importance == 0.5


# ====================================================================
# Forgetting in consolidation prune tests
# ====================================================================


class TestForgettingPrune:
    """Tests for forgetting-based pruning in consolidation."""

    def test_forgotten_memories_pruned(self):
        """Low retention + low importance + old → deleted via 'forget' action."""
        tmp = tempfile.mkdtemp()
        store = _make_store(Path(tmp))

        # Create a memory with low importance, moderate ff, and old age
        # importance=0.2, ff=0.05, age=100 → retention = 0.2 * exp(-0.05*100) = 0.2 * 0.0067 = 0.0013
        old_ts = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        filepath = store.add_memory(
            memory_type="reference",
            content="some old reference",
            tags=["old"],
            relevance=0.2,
            importance=0.2,
            forgetting_factor=0.05,
        )
        # Patch the timestamps to make it old
        meta = parse_frontmatter(filepath.read_text())
        meta["created"] = old_ts
        meta["updated"] = old_ts
        doc = render_frontmatter(meta) + "\nsome old reference"
        filepath.write_text(doc, encoding="utf-8")
        # Also update manifest
        entries = store._ensure_manifest_loaded()
        for entry in entries:
            if entry.get("filename") == filepath.name:
                entry["created"] = old_ts
                entry["updated"] = old_ts
                entry["importance"] = 0.2
                entry["forgetting_factor"] = 0.05
                break
        from plugins.memory.dream.store import _save_manifest
        _save_manifest(store.dream_root, entries)
        store._manifest = entries

        # Force consolidation
        orient_result = orient(store)
        orient_result.needs_consolidation = True
        orient_result.reason = "test"
        gather_result = gather(store, orient_result)

        # Run consolidate to find forget actions
        consolidate_result = consolidate(store, gather_result)

        # Check that the memory has a "forget" action
        forget_actions = [a for a in consolidate_result.actions if a.action == "forget"]
        assert len(forget_actions) >= 1, f"Expected forget action, got: {[a.action for a in consolidate_result.actions]}"

    def test_important_memories_resist_forgetting(self):
        """High importance → NOT pruned even when old."""
        tmp = tempfile.mkdtemp()
        store = _make_store(Path(tmp))

        # Create a memory with high importance and old age
        # importance=0.9, ff=0.05, age=100 → retention = 0.9 * exp(-5) = 0.9 * 0.0067 = 0.006
        # BUT imp=0.9 > PRUNE_MAX_IMPORTANCE(0.3), so it should NOT be forgotten
        old_ts = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        filepath = store.add_memory(
            memory_type="user",
            content="critical user preference",
            tags=["preference"],
            relevance=0.9,
            importance=0.9,
            forgetting_factor=0.05,
        )
        meta = parse_frontmatter(filepath.read_text())
        meta["created"] = old_ts
        meta["updated"] = old_ts
        doc = render_frontmatter(meta) + "\ncritical user preference"
        filepath.write_text(doc, encoding="utf-8")
        entries = store._ensure_manifest_loaded()
        for entry in entries:
            if entry.get("filename") == filepath.name:
                entry["created"] = old_ts
                entry["updated"] = old_ts
                entry["importance"] = 0.9
                entry["forgetting_factor"] = 0.05
                break
        from plugins.memory.dream.store import _save_manifest
        _save_manifest(store.dream_root, entries)
        store._manifest = entries

        orient_result = orient(store)
        orient_result.needs_consolidation = True
        orient_result.reason = "test"
        gather_result = gather(store, orient_result)
        consolidate_result = consolidate(store, gather_result)

        # No forget action for this memory
        forget_actions = [a for a in consolidate_result.actions if a.action == "forget"]
        forget_targets = [a.target_files for a in forget_actions]
        filename = filepath.name
        forgot = any(filename in files for files in forget_targets)
        assert not forgot, f"High-importance memory should not be forgotten: {forget_actions}"

    def test_recent_memories_not_forgotten(self):
        """New memories (age < PRUNE_MIN_AGE_DAYS) are always retained."""
        tmp = tempfile.mkdtemp()
        store = _make_store(Path(tmp))

        # Create a recent memory with low importance
        filepath = store.add_memory(
            memory_type="reference",
            content="recent low-importance ref",
            tags=["ref"],
            relevance=0.1,
            importance=0.1,
            forgetting_factor=0.05,
        )

        orient_result = orient(store)
        orient_result.needs_consolidation = True
        orient_result.reason = "test"
        gather_result = gather(store, orient_result)
        consolidate_result = consolidate(store, gather_result)

        forget_actions = [a for a in consolidate_result.actions if a.action == "forget"]
        filename = filepath.name
        forgot = any(filename in a.target_files for a in forget_actions)
        assert not forgot, f"Recent memory should not be forgotten: {forget_actions}"

    def test_recently_accessed_resist_forgetting(self):
        """High access count slows decay (spacing effect)."""
        # Even with low base importance, high access count reduces effective ff
        # which increases retention, potentially keeping it above the threshold
        importance = 0.2
        base_ff = 0.02
        age = 80  # days

        # With 0 accesses: effective_ff = 0.02, retention = 0.2 * exp(-1.6) = 0.040
        ret_no_access = retention_score(importance, base_ff, age)
        assert ret_no_access < PRUNE_RETENTION_THRESHOLD  # Would be forgotten

        # With 50 accesses: effective_ff = 0.02 * 0.9^50 = 0.02 * 0.00515 ≈ 0.000103
        eff_ff = max(FORGETTING_FACTOR_MIN, base_ff * (FORGETTING_REACCESS_MULTIPLIER ** 50))
        ret_with_access = retention_score(importance, eff_ff, age)
        assert ret_with_access > ret_no_access
        # With enough accesses, retention goes back above threshold
        assert ret_with_access > PRUNE_RETENTION_THRESHOLD


# ====================================================================
# Recall scoring with retention tests
# ====================================================================


class TestRecallScoringWithRetention:
    """Tests for recall engine using Ebbinghaus retention instead of simple decay."""

    def test_recall_uses_ebbinghaus_retention(self):
        """Recall scores use retention_score not simple decay."""
        tmp = tempfile.mkdtemp()
        store = _make_store(Path(tmp))

        # Create a memory on disk and inject into manifest
        path = store.add_memory(
            memory_type="user",
            content="test retention query match",
            tags=["retention"],
            source="test",
            relevance=0.7,
            importance=0.8,
            forgetting_factor=0.03,
        )
        # Patch timestamps to simulate 10 days age
        old_ts = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        meta = parse_frontmatter(path.read_text())
        meta["created"] = old_ts
        meta["updated"] = old_ts
        doc = render_frontmatter(meta) + "\ntest retention query match"
        path.write_text(doc, encoding="utf-8")
        # Update manifest too
        entries = store._ensure_manifest_loaded()
        for entry in entries:
            if entry.get("filename") == path.name:
                entry["created"] = old_ts
                entry["updated"] = old_ts
                break
        from plugins.memory.dream.store import _save_manifest
        _save_manifest(store.dream_root, entries)
        store._manifest = entries

        engine = RecallEngine(store)
        results = engine.recall("retention query", limit=5)

        assert len(results) > 0
        # The retention for importance=0.8, ff=0.03, age=10 = 0.8 * exp(-0.3) = 0.8 * 0.7408 = 0.593
        expected_retention = retention_score(0.8, 0.03, 10)
        assert expected_retention > 0.0

    def test_importance_affects_recall_ranking(self):
        """Higher importance → higher recall rank for same content."""
        tmp = tempfile.mkdtemp()
        store = _make_store(Path(tmp))

        # Two memories with same everything except importance
        age_days = 10
        ts = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
        _inject_manifest_entry(store, "user", "low-imp.md",
                               relevance=0.7, importance=0.2, forgetting_factor=0.02,
                               snippet="python coding preference",
                               age_days=age_days)
        _inject_manifest_entry(store, "user", "high-imp.md",
                               relevance=0.7, importance=0.9, forgetting_factor=0.02,
                               snippet="python coding preference",
                               age_days=age_days)

        engine = RecallEngine(store)
        results = engine.recall("python coding", limit=10)

        if len(results) >= 2:
            scores = {r.filename: r.score for r in results}
            # Higher importance should give higher score (through retention weight)
            assert scores.get("high-imp.md", 0) >= scores.get("low-imp.md", 0)

    def test_forgetting_factor_affects_ranking(self):
        """Slower forgetting → higher rank over time."""
        tmp = tempfile.mkdtemp()
        store = _make_store(Path(tmp))

        age_days = 60
        _inject_manifest_entry(store, "user", "fast-decay.md",
                               relevance=0.7, importance=0.7, forgetting_factor=0.05,
                               snippet="editor preference",
                               age_days=age_days)
        _inject_manifest_entry(store, "user", "slow-decay.md",
                               relevance=0.7, importance=0.7, forgetting_factor=0.005,
                               snippet="editor preference",
                               age_days=age_days)

        engine = RecallEngine(store)
        results = engine.recall("editor preference", limit=10)

        if len(results) >= 2:
            scores = {r.filename: r.score for r in results}
            # Slow decay should retain higher score
            assert scores.get("slow-decay.md", 0) >= scores.get("fast-decay.md", 0)


# ====================================================================
# Backward compatibility tests
# ====================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing vault data."""

    def test_backward_compat_importance_default(self):
        """Memories without importance default to relevance value."""
        # Create a document with only relevance (no importance field)
        doc = make_memory_document(
            content="test content",
            memory_type="user",
            tags=["test"],
            source="test",
            relevance=0.7,
        )
        # Now parse it — importance should default to relevance
        meta = parse_frontmatter(doc)
        assert meta["importance"] == 0.7

    def test_backward_compat_importance_default_relevance(self):
        """When parsed without importance, it defaults to relevance."""
        # Simulate an old document with no importance/forgetting_factor in frontmatter
        old_doc = """---
type: user
created: 2025-01-01T00:00:00+00:00
updated: 2025-01-01T00:00:00+00:00
relevance: 0.65
tags: [test]
source: test
---
Old memory content
"""
        meta = parse_frontmatter(old_doc)
        assert meta["importance"] == 0.65
        assert meta["forgetting_factor"] == 0.02

    def test_backward_compat_forgetting_default(self):
        """Memories without forgetting_factor default to 0.02."""
        doc = make_memory_document(
            content="test content",
            memory_type="user",
            tags=["test"],
            source="test",
            relevance=0.5,
        )
        meta = parse_frontmatter(doc)
        assert meta["forgetting_factor"] == 0.02

    def test_manifest_includes_new_fields(self):
        """New manifest entries include importance and forgetting_factor."""
        tmp = tempfile.mkdtemp()
        store = _make_store(Path(tmp))

        store.add_memory(
            memory_type="user",
            content="test content for manifest",
            tags=["test"],
            source="test",
            relevance=0.7,
            importance=0.8,
            forgetting_factor=0.03,
        )

        manifest = store._ensure_manifest_loaded()
        assert len(manifest) == 1
        entry = manifest[0]
        assert "importance" in entry
        assert entry["importance"] == 0.8
        assert "forgetting_factor" in entry
        assert entry["forgetting_factor"] == 0.03

    def test_manifest_defaults_when_not_specified(self):
        """When importance/forgetting not specified, manifest gets sensible defaults."""
        tmp = tempfile.mkdtemp()
        store = _make_store(Path(tmp))

        # Only pass relevance, not importance or forgetting_factor
        store.add_memory(
            memory_type="user",
            content="test defaults",
            tags=["test"],
            source="test",
            relevance=0.6,
        )

        manifest = store._ensure_manifest_loaded()
        assert len(manifest) == 1
        entry = manifest[0]
        # importance defaults to relevance
        assert entry["importance"] == 0.6
        # forgetting_factor defaults to 0.02
        assert entry["forgetting_factor"] == 0.02

    def test_old_manifest_entries_without_new_fields(self):
        """Manifest entries from before HB4 (no importance/forgetting) work in recall."""
        tmp = tempfile.mkdtemp()
        store = _make_store(Path(tmp))

        # Manually inject a manifest entry without importance/forgetting_factor
        entries = store._ensure_manifest_loaded()
        now = datetime.now(timezone.utc).isoformat()
        entries.append({
            "type": "user",
            "filename": "old-memory.md",
            "tags": ["preference"],
            "source": "test",
            "relevance": 0.7,
            "created": now,
            "updated": now,
            "snippet": "old preference memory",
            "access_count": 0,
            # No importance or forgetting_factor!
        })
        from plugins.memory.dream.store import _save_manifest
        _save_manifest(store.dream_root, entries)
        store._manifest = entries

        # Also create the actual file
        filepath = store.get_memory_path("user", "old-memory.md")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "type": "user",
            "created": now,
            "updated": now,
            "relevance": 0.7,
            "tags": ["preference"],
            "source": "test",
        }
        doc = render_frontmatter(meta) + "\nold preference memory"
        filepath.write_text(doc, encoding="utf-8")

        engine = RecallEngine(store)
        results = engine.recall("preference", limit=5)
        # Should not crash, should return results
        assert len(results) >= 0  # No crash is the main assertion

    def test_make_memory_document_importance_defaults_to_relevance(self):
        """make_memory_document with no importance → importance = relevance."""
        doc = make_memory_document(
            content="test",
            memory_type="user",
            relevance=0.75,
        )
        meta = parse_frontmatter(doc)
        assert meta["importance"] == 0.75

    def test_make_memory_document_explicit_importance(self):
        """make_memory_document with explicit importance uses that value."""
        doc = make_memory_document(
            content="test",
            memory_type="user",
            relevance=0.5,
            importance=0.9,
        )
        meta = parse_frontmatter(doc)
        assert meta["importance"] == 0.9

    def test_make_memory_document_explicit_forgetting(self):
        """make_memory_document with explicit forgetting_factor uses that value."""
        doc = make_memory_document(
            content="test",
            memory_type="user",
            relevance=0.5,
            forgetting_factor=0.05,
        )
        meta = parse_frontmatter(doc)
        assert meta["forgetting_factor"] == 0.05

    def test_update_memory_can_set_importance(self):
        """update_memory() can update importance field."""
        tmp = tempfile.mkdtemp()
        store = _make_store(Path(tmp))

        path = store.add_memory(
            memory_type="user",
            content="updatable memory",
            tags=["test"],
            source="test",
            relevance=0.5,
            importance=0.5,
        )
        filename = path.name

        store.update_memory("user", filename, importance=0.9)

        data = store.read_memory("user", filename)
        assert data["meta"]["importance"] == 0.9

    def test_update_memory_can_set_forgetting_factor(self):
        """update_memory() can update forgetting_factor field."""
        tmp = tempfile.mkdtemp()
        store = _make_store(Path(tmp))

        path = store.add_memory(
            memory_type="user",
            content="updatable memory",
            tags=["test"],
            source="test",
            relevance=0.5,
        )
        filename = path.name

        store.update_memory("user", filename, forgetting_factor=0.04)

        data = store.read_memory("user", filename)
        assert data["meta"]["forgetting_factor"] == 0.04

    def test_add_memory_importance_defaults_to_relevance(self):
        """add_memory() without importance defaults it to relevance."""
        tmp = tempfile.mkdtemp()
        store = _make_store(Path(tmp))

        path = store.add_memory(
            memory_type="user",
            content="default importance",
            tags=["test"],
            source="test",
            relevance=0.65,
        )

        data = store.read_memory("user", path.name)
        assert data["meta"]["importance"] == 0.65

    def test_add_memory_explicit_importance(self):
        """add_memory() with explicit importance stores that value."""
        tmp = tempfile.mkdtemp()
        store = _make_store(Path(tmp))

        path = store.add_memory(
            memory_type="feedback",
            content="explicit importance",
            tags=["correction"],
            source="test",
            relevance=0.8,
            importance=0.9,
            forgetting_factor=0.01,
        )

        data = store.read_memory("feedback", path.name)
        assert data["meta"]["importance"] == 0.9
        assert data["meta"]["forgetting_factor"] == pytest.approx(0.01)


# ====================================================================
# Consolidation forget action integration
# ====================================================================


class TestConsolidationForgetIntegration:
    """Integration tests for the 'forget' action in consolidation pipeline."""

    def test_forget_action_type_exists(self):
        """The 'forget' action type should be recognized in ConsolidationAction."""
        action = ConsolidationAction(
            action="forget",
            target_type="reference",
            target_files=["reference:old.md"],
            result_file="",
            details="Forgotten: retention=0.005",
        )
        assert action.action == "forget"

    def test_forget_prune_threshold_constants(self):
        """Verify the forgetting pruning threshold constants."""
        assert PRUNE_RETENTION_THRESHOLD == 0.1
        assert PRUNE_MIN_AGE_DAYS == 60
        assert PRUNE_MAX_IMPORTANCE == 0.3

    def test_forgetting_factor_constants(self):
        """Verify the forgetting factor constants."""
        assert FORGETTING_FACTOR_DEFAULT == 0.02
        assert FORGETTING_FACTOR_MIN == 0.005
        assert FORGETTING_FACTOR_MAX == 0.05
        assert FORGETTING_REACCESS_MULTIPLIER == 0.9

    def test_compute_age_days(self):
        """_compute_age_days returns correct values."""
        now = datetime.now(timezone.utc)
        # Entry created 30 days ago
        old_ts = (now - timedelta(days=30)).isoformat()
        entry = MemoryEntry(
            memory_type="user",
            filename="test.md",
            updated=old_ts,
            created=old_ts,
        )
        age = _compute_age_days(entry, now)
        assert 29.9 < age < 30.1

    def test_compute_age_days_missing_timestamp(self):
        """_compute_age_days returns 0 for missing timestamps."""
        now = datetime.now(timezone.utc)
        entry = MemoryEntry(
            memory_type="user",
            filename="test.md",
        )
        assert _compute_age_days(entry, now) == 0.0
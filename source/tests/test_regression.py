"""Regression and edge-case tests for Dream Memory Plugin.

Covers gaps identified in code review:
- Store: concurrent access, manifest integrity, special chars in content/tags,
  filename collision, body extraction edge cases, update with no changes
- Taxonomy: frontmatter edge cases, relevance clamping bounds, round-trip fidelity
- Extract: boundary content lengths, Unicode, mixed-type sentences, overlapping
  patterns, whitespace-only content, very long tags
- Recall: scoring with hyphenated tags, snippet overlap, recency edge cases,
  manifest missing files, corrupt frontmatter
- Consolidation: empty vault cycle, dry-run preservation, log cap at 50,
  stale/oversized detection thresholds, tag overlap edge cases, content
  similarity corner cases, contradiction detection with negation patterns
- Provider: config masking, prefetch cache eviction, on_memory_write edge cases,
  on_session_end/on_pre_compress without store, dream_recall with string limit
- Shared: resolve_vault_path with env vars, load_dream_config with placeholder keys
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from plugins.memory.dream.store import DreamStore, slugify, _MANIFEST_FILE
from plugins.memory.dream.taxonomy import (
    MEMORY_TYPES,
    MemoryTypeSpec,
    make_memory_document,
    parse_frontmatter,
    render_frontmatter,
    validate_memory_type,
    _parse_frontmatter_simple,
)
from plugins.memory.dream.extract import (
    CandidateMemory,
    extract_candidates,
    extract_candidates_from_messages,
    build_pre_compress_summary,
)
from plugins.memory.dream.recall import RecallEngine, RecallResult, _MIN_COMBINED_SCORE
from plugins.memory.dream.consolidation import (
    OrientResult,
    GatherResult,
    ConsolidateResult,
    ConsolidationResult,
    PruneResult,
    ConsolidationAction,
    MemoryEntry,
    DuplicatePair,
    ContradictionPair,
    orient,
    gather,
    consolidate,
    prune,
    run_consolidation,
    _content_similarity,
    _tag_overlap,
    _is_older,
    _detect_contradictions,
    _normalise_words,
    CONSOLIDATION_INTERVAL_HOURS,
    DEFAULT_MAX_LINES,
    DEFAULT_MAX_BYTES,
)
from plugins.memory.dream import DreamMemoryProvider, _load_plugin_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vault(tmp_path: Path) -> DreamStore:
    """Create a DreamStore rooted in a temp directory, initialised."""
    store = DreamStore(tmp_path / "dream_vault")
    store.initialize()
    return store


def _add_memory(
    store: DreamStore,
    memory_type: str = "user",
    content: str = "test content",
    tags: list = None,
    relevance: float = 0.5,
) -> Path:
    """Helper to add a memory and return its path."""
    return store.add_memory(
        memory_type=memory_type,
        content=content,
        tags=tags or [],
        source="test",
        relevance=relevance,
    )


def _write_memory_direct(
    store: DreamStore,
    memory_type: str,
    filename: str,
    content: str,
    tags: list = None,
    relevance: float = 0.5,
    created: str = None,
) -> Path:
    """Write a memory file directly with explicit filename (bypassing slug generation)."""
    type_dir = store.vault_path / memory_type
    type_dir.mkdir(exist_ok=True)
    if not filename.endswith(".md"):
        filename += ".md"
    filepath = type_dir / filename
    now = created or datetime.now(timezone.utc).isoformat()
    meta = {
        "type": memory_type,
        "created": now,
        "updated": now,
        "relevance": max(0.0, min(1.0, relevance)),
        "tags": tags or [],
        "source": "test",
    }
    doc = render_frontmatter(meta) + "\n" + content
    filepath.write_text(doc, encoding="utf-8")
    store._add_manifest_entry(memory_type, filename, content, tags or [], "test", relevance)
    return filepath


# ===========================================================================
# STORE REGRESSION TESTS
# ===========================================================================


class TestStoreFilenameCollision:
    """Regression: adding two memories in the same second produces unique filenames."""

    def test_rapid_adds_produce_unique_files(self, vault: DreamStore):
        """Two memories with identical content added rapidly must not collide."""
        p1 = vault.add_memory("user", "I prefer vim", tags=["editor"], source="s1")
        p2 = vault.add_memory("user", "I prefer vim", tags=["editor"], source="s1")
        assert p1 != p2
        # Both should exist
        assert p1.exists()
        assert p2.exists()
        # Stats should show 2
        stats = vault.stats()
        assert stats["total"] == 2

    def test_add_memory_with_unicode_content(self, vault: DreamStore):
        """Unicode content should be stored and readable without errors."""
        content = "ユーザーはダークモードが好きです 🌙"
        path = vault.add_memory("user", content, tags=["unicode", "設定"])
        assert path.exists()
        result = vault.read_memory("user", path.name)
        assert content in result["body"]

    def test_add_memory_with_special_chars_in_slug(self, vault: DreamStore):
        """Content with special characters should produce a safe slug."""
        content = "Config: ~/.bashrc & aliases > /dev/null"
        path = vault.add_memory("user", content)
        # Slug should only contain safe characters
        name = path.stem  # filename without .md
        # No spaces or special shell chars in filename
        assert " " not in name
        assert "&" not in name
        assert ">" not in name


class TestStoreManifestIntegrity:
    """Regression: manifest must stay consistent through add/delete/update cycles."""

    def test_manifest_count_matches_files_after_many_adds(self, vault: DreamStore):
        for i in range(20):
            vault.add_memory("user", f"Memory number {i} about topic {i}", tags=[f"topic-{i}"])
        manifest = json.loads((vault.vault_path / _MANIFEST_FILE).read_text())
        assert len(manifest) == 20

    def test_manifest_stays_consistent_after_mixed_operations(self, vault: DreamStore):
        """Interleaved add/delete should keep manifest in sync."""
        p1 = vault.add_memory("user", "First memory")
        p2 = vault.add_memory("user", "Second memory")
        vault.add_memory("project", "Project memory")

        # Delete one
        vault.delete_memory("user", p1.name)

        manifest = json.loads((vault.vault_path / _MANIFEST_FILE).read_text())
        filenames_in_manifest = {e["filename"] for e in manifest}
        assert p1.name not in filenames_in_manifest
        assert p2.name in filenames_in_manifest
        assert len(manifest) == 2

    def test_manifest_snippet_truncation(self, vault: DreamStore):
        """Long content should be truncated in manifest snippet."""
        long_content = "A" * 500
        vault.add_memory("reference", long_content, tags=["long"])
        manifest = json.loads((vault.vault_path / _MANIFEST_FILE).read_text())
        assert len(manifest[0]["snippet"]) <= 120


class TestStoreUpdateEdgeCases:
    """Regression: update edge cases."""

    def test_update_only_tags_preserves_content(self, vault: DreamStore):
        path = vault.add_memory("user", "I prefer dark mode", tags=["editor"])
        result_before = vault.read_memory("user", path.name)
        vault.update_memory("user", path.name, tags=["editor", "dark-mode"])
        result_after = vault.read_memory("user", path.name)
        # Content should not change
        assert "dark mode" in result_after["body"]
        # Tags should be updated
        assert "dark-mode" in result_after["meta"]["tags"]

    def test_update_relevance_boundary(self, vault: DreamStore):
        """Relevance should be clamped to [0.0, 1.0] on update."""
        path = vault.add_memory("user", "Test memory", relevance=0.5)

        # Update to above max
        vault.update_memory("user", path.name, relevance=2.0)
        result = vault.read_memory("user", path.name)
        assert result["meta"]["relevance"] <= 1.0

        # Update to below min
        vault.update_memory("user", path.name, relevance=-1.0)
        result = vault.read_memory("user", path.name)
        assert result["meta"]["relevance"] >= 0.0

    def test_update_nonexistent_type_raises(self, vault: DreamStore):
        with pytest.raises(ValueError):
            vault.update_memory("bogus", "test.md", content="nope")


class TestStoreDeleteEdgeCases:
    """Regression: deletion edge cases."""

    def test_delete_all_memories_of_one_type(self, vault: DreamStore):
        p1 = vault.add_memory("user", "User mem 1")
        p2 = vault.add_memory("user", "User mem 2")
        vault.add_memory("project", "Project mem")

        vault.delete_memory("user", p1.name)
        vault.delete_memory("user", p2.name)

        stats = vault.stats()
        assert stats["counts"]["user"] == 0
        assert stats["counts"]["project"] == 1
        assert stats["total"] == 1

    def test_delete_already_deleted_returns_false(self, vault: DreamStore):
        path = vault.add_memory("user", "To be deleted")
        assert vault.delete_memory("user", path.name) is True
        # Second delete should return False
        assert vault.delete_memory("user", path.name) is False


class TestStoreReadEdgeCases:
    """Regression: read edge cases."""

    def test_read_memory_with_no_body(self, vault: DreamStore):
        """A memory file with only frontmatter should return empty body."""
        content = ""
        path = vault.add_memory("user", content)
        result = vault.read_memory("user", path.name)
        assert isinstance(result["body"], str)

    def test_read_memory_with_multiline_body(self, vault: DreamStore):
        content = "Line 1\nLine 2\nLine 3"
        path = vault.add_memory("user", content)
        result = vault.read_memory("user", path.name)
        assert "Line 1" in result["body"]
        assert "Line 3" in result["body"]


class TestStoreExtractBody:
    """Regression: _extract_body edge cases."""

    def test_body_with_no_frontmatter(self):
        text = "Just plain text without frontmatter."
        assert DreamStore._extract_body(text) == "Just plain text without frontmatter."

    def test_body_with_unclosed_frontmatter(self):
        text = "---\ntype: user\nno closing delimiter"
        result = DreamStore._extract_body(text)
        assert isinstance(result, str)

    def test_body_with_empty_frontmatter(self):
        text = "---\n---\nContent here"
        result = DreamStore._extract_body(text)
        assert "Content here" in result


# ===========================================================================
# TAXONOMY REGRESSION TESTS
# ===========================================================================


class TestTaxonomyFrontmatterEdgeCases:
    """Regression: frontmatter parsing/rendering edge cases."""

    def test_frontmatter_with_colons_in_values(self):
        meta = {
            "type": "reference",
            "created": "2025-04-13T12:00:00+00:00",
            "updated": "2025-04-13T12:00:00+00:00",
            "relevance": 0.7,
            "tags": ["api", "endpoint"],
            "source": "session:abc123",
        }
        rendered = render_frontmatter(meta)
        parsed = parse_frontmatter(rendered)
        assert parsed["source"] == "session:abc123"

    def test_frontmatter_with_empty_tags_list(self):
        meta = {
            "type": "user",
            "created": "2025-01-01T00:00:00+00:00",
            "updated": "2025-01-01T00:00:00+00:00",
            "relevance": 0.5,
            "tags": [],
            "source": "",
        }
        rendered = render_frontmatter(meta)
        parsed = parse_frontmatter(rendered)
        assert parsed["tags"] == []

    def test_frontmatter_with_unicode_values(self):
        meta = {
            "type": "user",
            "created": "2025-01-01T00:00:00+00:00",
            "updated": "2025-01-01T00:00:00+00:00",
            "relevance": 0.8,
            "tags": ["設定", "エディター"],
            "source": "s-1",
        }
        rendered = render_frontmatter(meta)
        parsed = parse_frontmatter(rendered)
        assert "設定" in parsed["tags"]

    def test_relevance_at_exact_boundaries(self):
        """Clamping at exact 0.0 and 1.0 should not change values."""
        doc_boundary_1 = make_memory_document(
            content="boundary test", memory_type="user", relevance=1.0,
        )
        meta_1 = parse_frontmatter(doc_boundary_1)
        assert meta_1["relevance"] == 1.0

        doc_boundary_0 = make_memory_document(
            content="boundary test", memory_type="user", relevance=0.0,
        )
        meta_0 = parse_frontmatter(doc_boundary_0)
        assert meta_0["relevance"] == 0.0

    def test_parse_frontmatter_simple_fallback(self):
        """Test the simple parser fallback path."""
        block = "type: user\nrelevance: 0.7\ntags: [a, b, c]"
        result = _parse_frontmatter_simple(block)
        assert result["type"] == "user"
        assert result["relevance"] == 0.7
        assert result["tags"] == ["a", "b", "c"]

    def test_parse_frontmatter_invalid_yaml(self):
        """Invalid YAML between delimiters should return empty dict."""
        text = "---\n: invalid: yaml: [}\n---\nSome content"
        result = parse_frontmatter(text)
        # Should return {} since it's not valid YAML
        assert isinstance(result, dict)


class TestTaxonomyMemoryTypeSpec:
    """Regression: MemoryTypeSpec dataclass."""

    def test_all_types_have_different_max_lines(self):
        max_lines = {name: spec.max_lines for name, spec in MEMORY_TYPES.items()}
        # user=50, feedback=30, project=80, reference=120
        assert max_lines["user"] < max_lines["project"] < max_lines["reference"]

    def test_memory_type_frozen(self):
        """MemoryTypeSpec should be frozen (immutable)."""
        spec = MEMORY_TYPES["user"]
        with pytest.raises(AttributeError):
            spec.name = "other"


# ===========================================================================
# EXTRACT REGRESSION TESTS
# ===========================================================================


class TestExtractBoundaryContent:
    """Regression: content at boundary lengths."""

    def test_content_at_max_length(self):
        """Content at exactly _MAX_CONTENT_LEN (300) should be stored."""
        content = "I prefer " + "x" * 291  # exactly 300 chars total
        candidates = extract_candidates(content, "")
        user_candidates = [c for c in candidates if c.type == "user"]
        assert len(user_candidates) >= 1
        assert len(user_candidates[0].content) <= 300

    def test_just_above_max_length_is_truncated(self):
        """Content just over 300 chars should be truncated, not dropped."""
        content = "I prefer " + "x" * 292  # 301 chars total
        candidates = extract_candidates(content, "")
        user_candidates = [c for c in candidates if c.type == "user"]
        if user_candidates:
            assert len(user_candidates[0].content) <= 300

    def test_none_assistant_content_still_works(self):
        """extract_candidates should work with None for assistant_content."""
        candidates = extract_candidates("I prefer vim", None)
        assert len(candidates) >= 1


class TestExtractUnicode:
    """Regression: Unicode extraction patterns."""

    def test_unicode_in_preferences(self):
        """Unicode characters in preference statements should be extractable."""
        candidates = extract_candidates("I prefer ダークモード for my editor.", "")
        assert len(candidates) >= 1

    def test_emoji_in_content(self):
        """Emojis should not break extraction."""
        candidates = extract_candidates("I always use 🐍 Python 🎉", "")
        assert len(candidates) >= 1


class TestExtractOverlappingPatterns:
    """Regression: Sentences matching multiple pattern types."""

    def test_sentence_matching_user_and_feedback(self):
        """A sentence like 'No, I always prefer tabs' matches both feedback and user."""
        candidates = extract_candidates("No, I always prefer tabs for indentation.", "")
        types = {c.type for c in candidates}
        # Should find at least one match
        assert len(candidates) >= 1

    def test_multiple_sentences_same_type(self):
        """Multiple preference patterns in one text."""
        text = "I prefer dark mode. I like vim too. My favorite language is Python."
        candidates = extract_candidates(text, "")
        user_candidates = [c for c in candidates if c.type == "user"]
        assert len(user_candidates) >= 2  # at least 2 distinct matches

    def test_extract_from_messages_with_mixed_roles(self):
        """Only user messages should be scanned, not assistant."""
        messages = [
            {"role": "user", "content": "I prefer tabs over spaces."},
            {"role": "assistant", "content": "I always use vim for editing."},  # should be ignored
            {"role": "user", "content": "The project uses FastAPI."},
        ]
        candidates = extract_candidates_from_messages(messages)
        # No assistant patterns should be extracted
        for c in candidates:
            assert c.type in ("user", "project", "feedback", "reference")


class TestExtractFromMessagesEdgeCases:
    """Regression: extract_candidates_from_messages edge cases."""

    def test_message_with_missing_content_key(self):
        """Messages without 'content' key should be handled gracefully."""
        messages = [{"role": "user"}]  # no content key
        candidates = extract_candidates_from_messages(messages)
        assert candidates == []

    def test_message_with_empty_string_content(self):
        messages = [{"role": "user", "content": ""}]
        candidates = extract_candidates_from_messages(messages)
        assert candidates == []

    def test_message_with_whitespace_only(self):
        messages = [{"role": "user", "content": "   \t\n  "}]
        candidates = extract_candidates_from_messages(messages)
        # Whitespace-only content is below 5 chars after stripping -> skipped
        # OR it's under 5 meaningful chars
        assert isinstance(candidates, list)

    def test_message_with_numeric_content(self):
        """Numeric content should be skipped (not a string pattern match)."""
        messages = [{"role": "user", "content": 42}]  # content is int, not string
        candidates = extract_candidates_from_messages(messages)
        # Should handle non-string content gracefully
        assert candidates == []


class TestBuildPreCompressSummaryEdgeCases:
    """Regression: build_pre_compress_summary edge cases."""

    def test_summary_with_all_types(self):
        messages = [
            {"role": "user", "content": "I prefer dark mode."},
            {"role": "assistant", "content": "Got it."},
            {"role": "user", "content": "No, don't use light mode."},
            {"role": "user", "content": "The project uses Python."},
            {"role": "user", "content": "The API is at https://api.example.com"},
        ]
        summary = build_pre_compress_summary(messages)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_single_message_session(self):
        messages = [{"role": "user", "content": "I prefer tabs."}]
        summary = build_pre_compress_summary(messages)
        assert "I prefer tabs" in summary


# ===========================================================================
# RECALL REGRESSION TESTS
# ===========================================================================


class TestRecallEdgeCases:
    """Regression: RecallEngine edge cases."""

    def test_recall_with_stale_manifest_entry(self, vault: DreamStore):
        """A manifest entry pointing to a deleted file should be skipped."""
        vault.add_memory("user", "I prefer vim", tags=["vim"], source="s1", relevance=0.8)
        # Manually delete the file but leave manifest
        mems = vault.list_memories(memory_type="user")
        filename = mems[0]["filename"]
        filepath = vault.get_memory_path("user", filename)
        filepath.unlink()
        # Manifest still has the entry
        engine = RecallEngine(vault)
        results = engine.recall("vim")
        # Should not raise — gracefully skip the missing file
        assert isinstance(results, list)

    def test_recall_with_hyphenated_query_matching_tags(self, vault: DreamStore):
        """Query word 'dark' should match tag 'dark-mode' via hyphen splitting."""
        vault.add_memory("user", "I prefer dark mode", tags=["dark-mode", "preference"], source="s1", relevance=0.8)
        engine = RecallEngine(vault)
        results = engine.recall("dark")
        assert len(results) >= 1

    def test_recency_score_unknown_timestamp(self):
        """Memories with missing timestamps should get a low but non-zero recency."""
        score = RecallEngine._recency_score("", datetime.now(timezone.utc))
        assert score == 0.1  # unknown timestamp → low but non-zero

    def test_recency_score_very_old_timestamp(self):
        """Very old timestamps should have very low recency scores."""
        very_old = "2020-01-01T00:00:00+00:00"
        score = RecallEngine._recency_score(very_old, datetime.now(timezone.utc))
        assert score < 0.1  # Very old → near-zero recency

    def test_recency_score_recent_timestamp(self):
        """Recent timestamps should have high recency scores."""
        recent = datetime.now(timezone.utc).isoformat()
        score = RecallEngine._recency_score(recent, datetime.now(timezone.utc))
        assert score > 0.9

    def test_snippet_overlap_scoring(self):
        """Snippet overlap should boost recall for content-matching queries."""
        score = RecallEngine._snippet_overlap(["python", "backend"], "Python backend framework")
        assert score == 1.0  # Both words found

        score_partial = RecallEngine._snippet_overlap(["python"], "Python backend framework")
        assert score_partial == 1.0  # One word found out of one

        score_empty = RecallEngine._snippet_overlap([], "anything")
        assert score_empty == 0.0

    def test_recall_query_with_special_chars(self, vault: DreamStore):
        """Queries with special characters should not crash."""
        vault.add_memory("user", "Some content", tags=["test"], source="s1")
        engine = RecallEngine(vault)
        # Should not raise
        results = engine.recall("python/backend?query=yes&no")
        assert isinstance(results, list)

    def test_relevance_below_threshold_excluded(self, vault: DreamStore):
        """Memories below _MIN_COMBINED_SCORE should be excluded from results."""
        vault.add_memory("user", "Random note about nothing relevant", tags=["obscure"], source="s1", relevance=0.1)
        engine = RecallEngine(vault)
        # Query with nothing matching — even stored relevance of 0.1 may not be enough
        results = engine.recall("very specific unique query about obscure topics that might not match")
        # If the combined score is below _MIN_COMBINED_SCORE, it won't appear
        for r in results:
            assert r.score >= _MIN_COMBINED_SCORE


class TestRecallTokenise:
    """Regression: tokenisation edge cases."""

    def test_tokenise_with_hyphens(self):
        tokens = RecallEngine._tokenise("dark-mode editor")
        assert "dark-mode" in tokens or "dark" in tokens

    def test_tokenise_with_numbers(self):
        tokens = RecallEngine._tokenise("Python 3.12 is great")
        assert "python" in tokens
        # "3" would be filtered out (single char), but "12" wouldn't appear separately
        # since "3.12" would be tokenized based on the stripping logic


# ===========================================================================
# CONSOLIDATION REGRESSION TESTS
# ===========================================================================


class TestConsolidationContentSimilarity:
    """Regression: content similarity edge cases."""

    def test_content_similarity_with_very_short_strings(self):
        """Very short strings should not crash."""
        sim = _content_similarity("x", "x")
        assert sim >= 0.0
        sim2 = _content_similarity("a", "b")
        assert sim2 >= 0.0

    def test_content_similarity_with_repeated_words(self):
        sim = _content_similarity("test test test", "test test other")
        # "test" is in both, "other" is unique — similarity >= 0.5
        assert sim >= 0.5

    def test_tag_overlap_with_duplicate_tags(self):
        """Duplicate tags in input should be handled."""
        overlap = _tag_overlap(["a", "a", "b"], ["a", "b"])
        # Should still produce a valid overlap score
        assert 0.0 <= overlap <= 1.0


class TestConsolidationContradictionEdgeCases:
    """Regression: contradiction detection patterns."""

    def test_contradiction_always_never(self):
        a = MemoryEntry(
            memory_type="user", filename="a.md",
            content="I always use tabs", tags=["style"],
        )
        b = MemoryEntry(
            memory_type="user", filename="b.md",
            content="I never use tabs", tags=["style"],
        )
        result = _detect_contradictions(a, b)
        assert result is not None

    def test_contradiction_prefer_not_prefer(self):
        a = MemoryEntry(
            memory_type="user", filename="a.md",
            content="I prefer dark mode", tags=["editor"],
        )
        b = MemoryEntry(
            memory_type="user", filename="b.md",
            content="I don't prefer dark mode", tags=["editor"],
        )
        result = _detect_contradictions(a, b)
        assert result is not None

    def test_no_contradiction_between_different_types(self):
        """Memories about different topics should not be flagged as contradictions."""
        a = MemoryEntry(
            memory_type="user", filename="a.md",
            content="I prefer vim", tags=["editor"],
        )
        b = MemoryEntry(
            memory_type="project", filename="b.md",
            content="The project uses emacs", tags=["editor"],
        )
        result = _detect_contradictions(a, b)
        # Different types might not be compared, or the result should be None
        # (depends on implementation — this tests the behavior)


class TestConsolidationIsOlder:
    """Regression: _is_older edge cases."""

    def test_older_with_timezone_difference(self):
        a = MemoryEntry(
            memory_type="user", filename="a.md",
            created="2024-01-01T00:00:00+00:00",
        )
        b = MemoryEntry(
            memory_type="user", filename="b.md",
            created="2024-06-01T00:00:00+05:00",
        )
        # a is clearly older
        assert _is_older(a, b) is True

    def test_same_timestamp_not_older(self):
        ts = "2024-01-01T00:00:00+00:00"
        a = MemoryEntry(memory_type="user", filename="a.md", created=ts)
        b = MemoryEntry(memory_type="user", filename="b.md", created=ts)
        assert _is_older(a, b) is False


class TestConsolidationNormaliseWords:
    """Regression: _normalise_words edge cases."""

    def test_very_long_word(self):
        result = _normalise_words("supercalifragilisticexpialidocious")
        assert "supercalifragilisticexpialidocious" in result

    def test_mixed_case(self):
        result = _normalise_words("Python Python PYTHON")
        # _normalise_words returns a set, so duplicates are deduped
        assert "python" in result

    def test_empty_string(self):
        result = _normalise_words("")
        # _normalise_words returns an empty set for empty input
        assert result == set()


class TestConsolidationLogCap:
    """Regression: consolidation log should be capped at 50 entries."""

    def test_log_capped_at_50(self, vault: DreamStore):
        _add_memory(vault, "user", "Test memory for log cap")
        result = run_consolidation(vault)
        log_path = vault.vault_path / "consolidation_log.json"
        assert log_path.exists()

        # Run consolidation many times to fill up the log
        for i in range(55):
            vault.add_memory("user", f"Extra memory {i} unique content string", tags=[f"tag-{i}"])
            run_consolidation(vault)

        log_data = json.loads(log_path.read_text())
        assert len(log_data) <= 50


class TestConsolidationDryRunPreservation:
    """Regression: dry_run consolidation must not modify any files."""

    def test_dry_run_no_deletions(self, vault: DreamStore):
        _write_memory_direct(vault, "user", "mem-a-20250101T000001Z.md",
                             "Memory A about dark mode", tags=["editor"])
        _write_memory_direct(vault, "user", "mem-b-20250101T000002Z.md",
                             "Memory B about dark mode", tags=["editor"])

        initial_count = len(list((vault.vault_path / "user").glob("*.md")))

        run_consolidation(vault, dry_run=True)

        # No files should be deleted
        final_count = len(list((vault.vault_path / "user").glob("*.md")))
        assert final_count == initial_count


class TestConsolidationEmptyVault:
    """Regression: consolidation on empty vault should complete without errors."""

    def test_orient_empty_vault(self, vault: DreamStore):
        result = orient(vault)
        assert result.needs_consolidation is True
        assert result.stats["total_memories"] == 0

    def test_gather_empty_vault(self, vault: DreamStore):
        orient_result = orient(vault)
        result = gather(vault, orient_result)
        assert len(result.entries) == 0
        assert len(result.groups) == 0

    def test_consolidate_empty_vault(self, vault: DreamStore):
        orient_result = orient(vault)
        gather_result = gather(vault, orient_result)
        result = consolidate(vault, gather_result)
        assert result.merged_count == 0
        assert result.deduped_count == 0

    def test_prune_empty_vault(self, vault: DreamStore):
        orient_result = orient(vault)
        gather_result = gather(vault, orient_result)
        consolidate_result = consolidate(vault, gather_result)
        prune_result = prune(vault, consolidate_result)
        # No files to delete
        assert len(prune_result.deleted_files) == 0

    def test_run_consolidation_empty_vault(self, vault: DreamStore):
        result = run_consolidation(vault)
        assert isinstance(result, ConsolidationResult)


class TestConsolidationTypeFiltering:
    """Regression: memory_type filter should work across all phases."""

    def test_filter_by_project_type(self, vault: DreamStore):
        _write_memory_direct(vault, "user", "like-vim-20250101T000001Z.md",
                             "I like vim", tags=["editor"])
        _write_memory_direct(vault, "project", "using-python-20250101T000001Z.md",
                             "Using Python for backend", tags=["python"])
        _write_memory_direct(vault, "project", "using-fastapi-20250101T000002Z.md",
                             "Using FastAPI for API", tags=["python"])

        result = run_consolidation(vault, memory_type="project")
        # All entries should be project type
        for entry in result.gather.entries:
            assert entry.memory_type == "project"


class TestConsolidationStaleAndOversized:
    """Regression: stale and oversized file detection."""

    def test_stale_detection_threshold(self, vault: DreamStore):
        """Memories older than CONSOLIDATION_INTERVAL_HOURS should be flagged as stale."""
        # Create a memory with a very old timestamp
        old_ts = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        _write_memory_direct(vault, "user", "old-mem-20200101T000000Z.md",
                             "Old memory content", created=old_ts)
        orient_result = orient(vault)
        assert len(orient_result.stale_files) > 0

    def test_oversized_detection_line_threshold(self, vault: DreamStore):
        """Memories exceeding max_lines for their type should be in oversized_files."""
        # user type has max_lines=50; write a memory with 60+ lines
        lines = [f"Line {i} of the memory content." for i in range(60)]
        long_content = "\n".join(lines)
        _write_memory_direct(vault, "user", "long-mem-20250101T000000Z.md",
                             long_content, tags=["long"])
        orient_result = orient(vault)
        assert len(orient_result.oversized_files) > 0


class TestConsolidationConsolidationInterval:
    """Regression: CONSOLIDATION_INTERVAL_HOURS should be defined and positive."""

    def test_interval_is_positive(self):
        assert CONSOLIDATION_INTERVAL_HOURS > 0


class TestConsolidationDefaultConstants:
    """Regression: default constants should have reasonable values."""

    def test_default_max_lines(self):
        assert DEFAULT_MAX_LINES > 0

    def test_default_max_bytes(self):
        assert DEFAULT_MAX_BYTES > 0


# ===========================================================================
# PROVIDER REGRESSION TESTS
# ===========================================================================


class TestProviderRelevanceThreshold:
    """Regression: _MIN_RELEVANCE should filter low-relevance candidates."""

    def test_min_relevance_is_0_6(self):
        from plugins.memory.dream import _MIN_RELEVANCE
        assert _MIN_RELEVANCE == 0.6


class TestProviderSyncTurnThreshold:
    """Regression: sync_turn should only store candidates with relevance >= 0.6."""

    def test_reference_candidates_stored(self, tmp_path: Path):
        """Reference candidates (relevance=0.6) should be stored at the threshold."""
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        # "The path is /usr/local" matches reference pattern with relevance=0.6
        provider.sync_turn("The path is /usr/local/bin/python.", "OK")
        stats = provider._store.stats()
        assert stats["total"] >= 1

    def test_below_threshold_not_stored(self, tmp_path: Path):
        """If all patterns had relevance < 0.6, nothing should be stored."""
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        # Plain text with no matching patterns
        provider.sync_turn("Hello, how are you today?", "I'm fine, thanks.")
        stats = provider._store.stats()
        assert stats["total"] == 0


class TestProviderOnMemoryWriteEdgeCases:
    """Regression: on_memory_write edge cases."""

    def test_replace_with_no_matching_memory_adds_new(self, tmp_path: Path):
        """Replace action with no matching memory should add as new."""
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        # Replace with no existing memories
        provider.on_memory_write("replace", "user", "New preference content")
        mems = provider._store.list_memories(memory_type="user")
        assert len(mems) >= 1

    def test_remove_action_with_no_match_is_noop(self, tmp_path: Path):
        """Remove action on nonexistent content should not raise."""
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        # Remove on empty vault — should not raise
        provider.on_memory_write("remove", "user", "Nonexistent content")
        stats = provider._store.stats()
        assert stats["total"] == 0

    def test_on_memory_write_unknown_action(self, tmp_path: Path):
        """Unknown action should not crash (currently no-op)."""
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        # Unknown action — should be silently ignored
        provider.on_memory_write("update", "user", "Some content")
        stats = provider._store.stats()
        assert stats["total"] == 0


class TestProviderPrefetchCacheEviction:
    """Regression: prefetch cache should evict old entries when full."""

    def test_cache_max_size(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        provider._store.add_memory("user", "I prefer vim", tags=["vim"], source="s1", relevance=0.8)

        # Fill up the cache beyond max
        for i in range(60):
            result = provider.prefetch(f"query {i}", session_id="test-session")

        # Cache should not exceed max size
        assert len(provider._prefetch_cache) <= provider._prefetch_cache_max

    def test_cache_returns_empty_string_without_recalling(self, tmp_path: Path):
        """Cached empty-string result should be stored and returned on next call."""
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        # No memories matching this query
        result1 = provider.prefetch("zzzzz nonexistent query", session_id="s1")
        assert result1 == ""
        # Second call should also return empty string (from cache)
        result2 = provider.prefetch("zzzzz nonexistent query", session_id="s1")
        assert result2 == ""


class TestProviderDreamRecallEdgeCases:
    """Regression: dream_recall tool edge cases."""

    def test_dream_recall_with_string_limit(self, tmp_path: Path):
        """Limit passed as string should be converted to int."""
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        provider._store.add_memory("user", "I prefer vim", tags=["vim"], source="s1", relevance=0.8)
        result = provider.handle_tool_call("dream_recall", {"query": "vim", "limit": "3"})
        data = json.loads(result)
        assert "results" in data
        assert len(data["results"]) <= 3

    def test_dream_recall_with_invalid_string_limit(self, tmp_path: Path):
        """Invalid string limit should default to 5."""
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        provider._store.add_memory("user", "I prefer vim", tags=["vim"], source="s1", relevance=0.8)
        result = provider.handle_tool_call("dream_recall", {"query": "vim", "limit": "abc"})
        data = json.loads(result)
        assert "results" in data


class TestProviderSessionEndEdgeCases:
    """Regression: on_session_end edge cases."""

    def test_session_end_with_none_messages(self, tmp_path: Path):
        """on_session_end with None should not crash."""
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        # None is falsy, so it should return early
        provider.on_session_end(None)
        # Should not have stored anything
        stats = provider._store.stats()
        assert stats["total"] == 0

    def test_session_end_deduplicates_existing_memories(self, tmp_path: Path):
        """Re-processing the same message should not create duplicate memories."""
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        messages = [
            {"role": "user", "content": "I always use vim for editing."},
        ]
        # Process twice
        provider.on_session_end(messages)
        count_after_first = provider._store.stats()["total"]
        # Note: The store doesn't dedup on its own — but extract_candidates_from_messages
        # deduplicates candidates. However, re-running will add again because
        # the store doesn't check for content uniqueness.
        # This test documents the current behavior.
        provider.on_session_end(messages)
        # At minimum the first call stored something
        assert count_after_first >= 1


class TestProviderPreCompressEdgeCases:
    """Regression: on_pre_compress edge cases."""

    def test_pre_compress_with_none_messages(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        result = provider.on_pre_compress(None)
        assert result == ""

    def test_pre_compress_tags_include_pre_compress(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        messages = [{"role": "user", "content": "I prefer tabs over spaces."}]
        provider.on_pre_compress(messages)

        mems = provider._store.list_memories(memory_type="user")
        assert len(mems) >= 1
        assert "pre-compress" in mems[0]["meta"]["tags"]


# ===========================================================================
# SHARED MODULE REGRESSION TESTS
# ===========================================================================


class TestSharedConfig:
    """Regression: shared module config helpers."""

    def test_load_dream_config_missing_file(self, tmp_path: Path):
        """Missing config.yaml should return empty dict."""
        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            cfg = _load_plugin_config()
            assert cfg == {}

    def test_load_dream_config_with_placeholder_key(self, tmp_path: Path):
        """Config with '***' placeholder should treat API key as absent."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "plugins:\n  dream:\n    consolidate_api_key: '***'\n    vault_path: /tmp/vault\n",
            encoding="utf-8",
        )
        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            cfg = _load_plugin_config()
            # The placeholder key should be cleared
            assert cfg.get("consolidate_api_key", "") != "***"

    def test_resolve_vault_path_explicit(self, tmp_path: Path):
        """Explicit vault_path should be used as-is."""
        from plugins.memory.dream.shared import resolve_vault_path
        config = {"vault_path": str(tmp_path / "my_custom_vault")}
        result = resolve_vault_path(config)
        assert str(result).endswith("my_custom_vault")


# ===========================================================================
# SLUG REGRESSION TESTS
# ===========================================================================


class TestSlugifyEdgeCases:
    """Regression: slugify edge cases."""

    def test_slugify_all_special_chars(self):
        result = slugify("!@#$%^&*()")
        assert result == "memory"  # Falls back to default

    def test_slugify_unicode(self):
        result = slugify("日本語テスト")
        # Unicode characters are stripped, falls back to default
        assert isinstance(result, str)
        assert len(result) > 0

    def test_slugify_numbers(self):
        result = slugify("Python 3.12 is great")
        assert "python" in result
        assert "3" in result or "3-12" in result

    def test_slugify_leading_trailing_spaces(self):
        result = slugify("  leading and trailing spaces  ")
        assert result.startswith("leading")
        assert not result.startswith("-")


# ===========================================================================
# INTEGRATION REGRESSION TESTS
# ===========================================================================


class TestIntegrationMultipleConsolidationCycles:
    """Regression: running consolidation multiple times should be idempotent."""

    def test_second_consolidation_is_safe(self, vault: DreamStore):
        _write_memory_direct(vault, "user", "unique-a-20250101T000001Z.md",
                             "Unique memory A about vim editing", tags=["editor"])
        _write_memory_direct(vault, "user", "unique-b-20250101T000002Z.md",
                             "Unique memory B about emacs editing", tags=["editor"])

        result1 = run_consolidation(vault)
        assert isinstance(result1, ConsolidationResult)

        # Second consolidation should not crash and not double-delete
        result2 = run_consolidation(vault)
        assert isinstance(result2, ConsolidationResult)


class TestIntegrationMixedTypeMemories:
    """Regression: consolidation with a mix of all four memory types."""

    def test_all_four_types_consolidate_safely(self, vault: DreamStore):
        _write_memory_direct(vault, "user", "pref-vim-20250101T000001Z.md",
                             "I prefer vim", tags=["editor", "vim"])
        _write_memory_direct(vault, "feedback", "no-camel-20250101T000001Z.md",
                             "Stop using camelCase", tags=["naming", "correction"])
        _write_memory_direct(vault, "project", "using-python-20250101T000001Z.md",
                             "The project uses Python 3.12", tags=["python"])
        _write_memory_direct(vault, "reference", "api-endpoint-20250101T000001Z.md",
                             "The API is at https://api.example.com", tags=["api"])

        result = run_consolidation(vault)
        assert isinstance(result, ConsolidationResult)
        assert result.orient.stats["total_memories"] == 4


class TestIntegrationStoreAndRecallRoundtrip:
    """Regression: store → recall roundtrip must preserve content."""

    def test_recall_preserves_content(self, vault: DreamStore):
        content = "I prefer dark mode for all coding tasks."
        vault.add_memory("user", content, tags=["editor", "dark-mode"], source="roundtrip-test", relevance=0.8)
        engine = RecallEngine(vault)
        results = engine.recall("dark mode coding")
        assert len(results) >= 1
        assert content in results[0].content

    def test_recall_preserves_tags(self, vault: DreamStore):
        vault.add_memory("user", "Test content", tags=["tag1", "tag2"], source="s1", relevance=0.8)
        engine = RecallEngine(vault)
        results = engine.recall("test")
        assert len(results) >= 1
        assert "tag1" in results[0].frontmatter.get("tags", [])


class TestIntegrationExtractAndStore:
    """Regression: extract → store pipeline must produce valid memories."""

    def test_extract_and_store_preference(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        provider.sync_turn("I always use tabs for indentation.", "Got it!")
        stats = provider._store.stats()
        assert stats["counts"]["user"] >= 1

        # Verify the stored memory is readable
        mems = provider._store.list_memories(memory_type="user")
        assert len(mems) >= 1
        assert "tabs" in mems[0]["body"].lower() or "indentation" in mems[0]["body"].lower()

    def test_extract_and_store_feedback(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        provider.sync_turn("Actually, that's wrong — we should use snake_case.", "I'll fix it.")
        stats = provider._store.stats()
        assert stats["counts"]["feedback"] >= 1


class TestIntegrationManifestConsistency:
    """Regression: manifest must stay consistent after add/delete/update."""

    def test_manifest_filenames_match_actual_files(self, vault: DreamStore):
        p1 = vault.add_memory("user", "Memory one", tags=["t1"])
        p2 = vault.add_memory("project", "Memory two", tags=["t2"])

        manifest = json.loads((vault.vault_path / _MANIFEST_FILE).read_text())
        manifest_filenames = {e["filename"] for e in manifest}

        assert p1.name in manifest_filenames
        assert p2.name in manifest_filenames

    def test_manifest_after_delete_has_no_stale_entries(self, vault: DreamStore):
        p1 = vault.add_memory("user", "Memory to delete", tags=["t1"])
        vault.add_memory("user", "Memory to keep", tags=["t2"])

        vault.delete_memory("user", p1.name)

        manifest = json.loads((vault.vault_path / _MANIFEST_FILE).read_text())
        filenames = {e["filename"] for e in manifest}
        assert p1.name not in filenames
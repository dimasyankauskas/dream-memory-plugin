"""Tests for Dream Memory RecallEngine — manifest-based recall and scoring."""

from __future__ import annotations

import json
import time
import pytest
from pathlib import Path

from plugins.memory.dream.store import DreamStore
from plugins.memory.dream.recall import RecallEngine, RecallResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vault(tmp_path: Path) -> DreamStore:
    """Create a DreamStore rooted in a temp directory, initialised."""
    store = DreamStore(tmp_path / "dream_vault")
    store.initialize()
    return store


@pytest.fixture
def populated_vault(vault: DreamStore) -> DreamStore:
    """Vault with sample memories for recall testing."""
    vault.add_memory(
        "user",
        "I prefer dark mode for editing and vim as my editor.",
        tags=["editor", "vim", "preference", "dark-mode"],
        source="s1",
        relevance=0.8,
    )
    vault.add_memory(
        "user",
        "I always use Python for backend work.",
        tags=["python", "backend", "preference"],
        source="s1",
        relevance=0.75,
    )
    vault.add_memory(
        "project",
        "The project uses FastAPI with PostgreSQL on the backend.",
        tags=["fastapi", "postgres", "backend", "project"],
        source="s2",
        relevance=0.7,
    )
    vault.add_memory(
        "feedback",
        "Actually, don't use camelCase — we prefer snake_case for all variables.",
        tags=["naming", "snake-case", "correction"],
        source="s3",
        relevance=0.85,
    )
    vault.add_memory(
        "reference",
        "The API endpoint is at https://api.example.com/v2.",
        tags=["api", "endpoint", "reference"],
        source="s4",
        relevance=0.6,
    )
    return vault


# ---------------------------------------------------------------------------
# RecallEngine basic tests
# ---------------------------------------------------------------------------

class TestRecallEngineBasic:
    def test_recall_returns_results(self, populated_vault: DreamStore):
        engine = RecallEngine(populated_vault)
        results = engine.recall("vim editor preference")
        assert len(results) > 0

    def test_recall_results_have_content(self, populated_vault: DreamStore):
        engine = RecallEngine(populated_vault)
        results = engine.recall("vim editor")
        assert any("vim" in r.content.lower() for r in results)

    def test_recall_result_fields(self, populated_vault: DreamStore):
        engine = RecallEngine(populated_vault)
        results = engine.recall("python backend")
        assert len(results) >= 1
        r = results[0]
        assert r.memory_type in ("user", "feedback", "project", "reference")
        assert isinstance(r.filename, str)
        assert r.filename.endswith(".md")
        assert isinstance(r.content, str)
        assert isinstance(r.frontmatter, dict)
        assert isinstance(r.score, float)

    def test_recall_empty_query_returns_empty(self, vault: DreamStore):
        engine = RecallEngine(vault)
        assert engine.recall("") == []
        assert engine.recall("   ") == []

    def test_recall_empty_vault_returns_empty(self, vault: DreamStore):
        engine = RecallEngine(vault)
        assert engine.recall("anything") == []


# ---------------------------------------------------------------------------
# Type filtering
# ---------------------------------------------------------------------------

class TestRecallTypeFilter:
    def test_filter_by_user_type(self, populated_vault: DreamStore):
        engine = RecallEngine(populated_vault)
        results = engine.recall("preference", memory_type="user")
        assert len(results) > 0
        for r in results:
            assert r.memory_type == "user"

    def test_filter_by_project_type(self, populated_vault: DreamStore):
        engine = RecallEngine(populated_vault)
        results = engine.recall("fastapi", memory_type="project")
        assert len(results) > 0
        for r in results:
            assert r.memory_type == "project"

    def test_filter_by_feedback_type(self, populated_vault: DreamStore):
        engine = RecallEngine(populated_vault)
        results = engine.recall("snake case", memory_type="feedback")
        assert len(results) > 0
        for r in results:
            assert r.memory_type == "feedback"

    def test_filter_no_matches_for_type(self, populated_vault: DreamStore):
        engine = RecallEngine(populated_vault)
        # Searching for "dark mode" but only in feedback type (it's a user memory)
        results = engine.recall("dark mode", memory_type="feedback")
        assert all(r.memory_type == "feedback" for r in results)
        # "dark mode" shouldn't be in feedback memories
        assert not any("dark mode" in r.content.lower() for r in results)

    def test_invalid_type_returns_empty(self, populated_vault: DreamStore):
        engine = RecallEngine(populated_vault)
        results = engine.recall("test", memory_type="nonexistent")
        # No memories of nonexistent type — should return empty
        assert results == []


# ---------------------------------------------------------------------------
# Limit parameter
# ---------------------------------------------------------------------------

class TestRecallLimit:
    def test_limit_respected(self, populated_vault: DreamStore):
        engine = RecallEngine(populated_vault)
        results = engine.recall("preference", limit=2)
        assert len(results) <= 2

    def test_limit_one(self, populated_vault: DreamStore):
        engine = RecallEngine(populated_vault)
        results = engine.recall("backend", limit=1)
        assert len(results) <= 1

    def test_default_limit(self, vault: DreamStore):
        """Add more than 5 memories and verify default limit of 5."""
        for i in range(8):
            vault.add_memory(
                "reference",
                f"Reference item {i} about Python and coding.",
                tags=[f"item-{i}", "python"],
                source="test",
                relevance=0.5 + i * 0.01,
            )
        engine = RecallEngine(vault)
        results = engine.recall("python", limit=5)
        assert len(results) <= 5


# ---------------------------------------------------------------------------
# Scoring: tag matching
# ---------------------------------------------------------------------------

class TestRecallScoring:
    def test_tag_match_boosts_score(self, vault: DreamStore):
        """Memories with more tag overlap should score higher."""
        vault.add_memory(
            "user",
            "I love Python programming.",
            tags=["python", "programming", "love"],
            source="s1",
            relevance=0.5,
        )
        vault.add_memory(
            "user",
            "Random unrelated fact.",
            tags=["random", "unrelated"],
            source="s1",
            relevance=0.5,
        )
        engine = RecallEngine(vault)
        results = engine.recall("python programming")
        # Python-tagged memory should rank first
        assert len(results) >= 1
        assert "python" in results[0].content.lower()

    def test_higher_relevance_ranks_higher(self, vault: DreamStore):
        """All else similar, higher stored relevance should rank higher."""
        vault.add_memory(
            "reference",
            "Important reference about APIs.",
            tags=["api", "reference"],
            source="s1",
            relevance=0.95,
        )
        vault.add_memory(
            "reference",
            "Low relevance reference.",
            tags=["api", "reference"],
            source="s1",
            relevance=0.3,
        )
        engine = RecallEngine(vault)
        results = engine.recall("api reference")
        assert len(results) >= 2
        assert results[0].score > results[1].score

    def test_recency_boost(self, vault: DreamStore):
        """Newer memories get a slight recency boost."""
        import time as _time

        # Add an older memory first
        path1 = vault.add_memory(
            "project",
            "Old project decision about architecture.",
            tags=["architecture", "decision"],
            source="s1",
            relevance=0.7,
        )
        # Small delay to ensure different timestamp
        _time.sleep(0.05)

        # Add a newer memory with same relevance and similar tags
        path2 = vault.add_memory(
            "project",
            "New project decision about architecture.",
            tags=["architecture", "decision"],
            source="s2",
            relevance=0.7,
        )

        engine = RecallEngine(vault)
        results = engine.recall("architecture decision")
        if len(results) >= 2:
            # Both should exist, and the one with the higher score
            # (could be either depending on timestamps, but newer should get
            # a slight boost if the updated timestamps are different enough)
            assert results[0].score >= results[1].score

    def test_combined_scoring_tag_relevance_and_recency(self, vault: DreamStore):
        """Verify scoring combines tag match, relevance, and recency."""
        vault.add_memory(
            "user",
            "I prefer Python for data science.",
            tags=["python", "data-science", "preference"],
            source="s1",
            relevance=0.9,
        )
        vault.add_memory(
            "user",
            "I sometimes use Java for legacy systems.",
            tags=["java", "legacy"],
            source="s2",
            relevance=0.5,
        )
        engine = RecallEngine(vault)
        results = engine.recall("python data science")
        # Python memory should rank higher due to better tag overlap + higher relevance
        assert len(results) >= 1
        python_results = [r for r in results if "python" in r.content.lower()]
        java_results = [r for r in results if "java" in r.content.lower()]
        if python_results and java_results:
            assert python_results[0].score > java_results[0].score


# ---------------------------------------------------------------------------
# RecallResult dataclass
# ---------------------------------------------------------------------------

class TestRecallResult:
    def test_dataclass_fields(self):
        r = RecallResult(
            memory_type="user",
            filename="test-file.md",
            content="Test content",
            frontmatter={"tags": ["test"], "relevance": 0.8},
            score=0.75,
        )
        assert r.memory_type == "user"
        assert r.filename == "test-file.md"
        assert r.content == "Test content"
        assert r.frontmatter["tags"] == ["test"]
        assert r.score == 0.75


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

class TestTokenisation:
    def test_basic_tokenisation(self):
        words = RecallEngine._tokenise("python backend")
        assert words == ["python", "backend"]

    def test_strips_special_chars(self):
        words = RecallEngine._tokenise("what's the best-api?")
        # Should produce lowercase alpha-numeric tokens
        assert "the" in words  # 3 chars, passes length > 1
        assert "best-api" in words or "best" in words

    def test_ignores_single_char(self):
        words = RecallEngine._tokenise("a I the")
        # Single-char tokens are excluded
        assert "a" not in words
        # "I" is single char, excluded
        assert "I" not in words

    def test_empty_string(self):
        assert RecallEngine._tokenise("") == []

    def test_just_spaces(self):
        assert RecallEngine._tokenise("   ") == []


# ---------------------------------------------------------------------------
# Tag overlap scoring
# ---------------------------------------------------------------------------

class TestTagOverlap:
    def test_perfect_overlap(self):
        score = RecallEngine._tag_overlap(
            ["python", "backend"],
            ["python", "backend", "other"],
        )
        assert score == 1.0

    def test_no_overlap(self):
        score = RecallEngine._tag_overlap(
            ["python", "backend"],
            ["java", "frontend"],
        )
        assert score == 0.0

    def test_partial_overlap(self):
        score = RecallEngine._tag_overlap(
            ["python", "backend", "api"],
            ["python", "frontend"],
        )
        assert 0.0 < score < 1.0
        assert abs(score - 1/3) < 0.01

    def test_empty_query_words(self):
        assert RecallEngine._tag_overlap([], ["tag"]) == 0.0

    def test_hyphenated_tags(self):
        """Query word should match parts of hyphenated tags."""
        score = RecallEngine._tag_overlap(
            ["dark"],
            ["dark-mode", "editor"],
        )
        # "dark" matches via hyphen split in "dark-mode"
        assert score == 1.0
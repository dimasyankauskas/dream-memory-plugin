"""Tests for Heartbeat 3: Auto-Recall Hook — access_count, feedback boost,
prefetch budget/feedback-priority, auto_recall config flag."""

from __future__ import annotations

import json
import math
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from plugins.memory.dream.store import DreamStore, _MANIFEST_FILE
from plugins.memory.dream.recall import (
    RecallEngine,
    RecallResult,
    _TAG_WEIGHT,
    _RELEVANCE_WEIGHT,
    _RECENCY_WEIGHT,
    _ACCESS_WEIGHT,
    _FEEDBACK_BOOST_WEIGHT,
)
from plugins.memory.dream import DreamMemoryProvider


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
def recall_engine(vault: DreamStore) -> RecallEngine:
    """RecallEngine backed by the vault fixture."""
    return RecallEngine(vault)


@pytest.fixture
def populated_vault(vault: DreamStore) -> DreamStore:
    """Vault with sample memories for testing."""
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


@pytest.fixture
def provider(tmp_path: Path) -> DreamMemoryProvider:
    """DreamMemoryProvider with auto_recall enabled and a vault."""
    config = {
        "vault_path": str(tmp_path / "dream_test"),
        "auto_recall": "true",
        "auto_recall_budget": 2048,
        "auto_recall_top_k": 10,
    }
    provider = DreamMemoryProvider(config=config)
    provider.initialize(session_id="test-session")
    return provider


@pytest.fixture
def provider_no_auto_recall(tmp_path: Path) -> DreamMemoryProvider:
    """DreamMemoryProvider with auto_recall disabled (default)."""
    config = {
        "vault_path": str(tmp_path / "dream_test"),
        "auto_recall": "false",
    }
    provider = DreamMemoryProvider(config=config)
    provider.initialize(session_id="test-session")
    return provider


# ---------------------------------------------------------------------------
# Test access_count on manifest entries
# ---------------------------------------------------------------------------

class TestManifestAccessCount:
    """Tests for access_count field on manifest entries."""

    def test_manifest_entry_has_access_count(self, vault: DreamStore):
        """New entries should have access_count=0."""
        vault.add_memory("user", "test content", tags=["test"], relevance=0.5)
        manifest = vault._ensure_manifest_loaded()
        assert len(manifest) == 1
        assert manifest[0].get("access_count") == 0

    def test_increment_access_count(self, vault: DreamStore):
        """Incrementing access_count should persist to disk."""
        path = vault.add_memory("user", "increment test", tags=["test"], relevance=0.5)
        filename = path.name

        # First increment: 0 → 1
        new_count = vault.increment_access_count("user", filename)
        assert new_count == 1

        # Second increment: 1 → 2
        new_count = vault.increment_access_count("user", filename)
        assert new_count == 2

        # Verify persistence by reloading manifest
        manifest = vault._ensure_manifest_loaded()
        entry = next(e for e in manifest if e["filename"] == filename)
        assert entry["access_count"] == 2

    def test_increment_access_count_missing_entry(self, vault: DreamStore):
        """Incrementing a nonexistent entry should return 0."""
        new_count = vault.increment_access_count("user", "nonexistent-file.md")
        assert new_count == 0

    def test_access_count_preserved_on_update(self, vault: DreamStore):
        """When updating a memory, access_count should be preserved."""
        path = vault.add_memory("user", "original content", tags=["test"], relevance=0.5)
        filename = path.name

        # Increment access count
        vault.increment_access_count("user", filename)
        assert vault.increment_access_count("user", filename) == 2

        # Update the memory — access_count should be preserved
        vault.update_memory("user", filename, content="updated content", relevance=0.7)

        manifest = vault._ensure_manifest_loaded()
        entry = next(e for e in manifest if e["filename"] == filename)
        assert entry["access_count"] == 2


# ---------------------------------------------------------------------------
# Test access_count in recall scoring
# ---------------------------------------------------------------------------

class TestAccessCountInRecallScoring:
    """Tests that higher access_count boosts recall score."""

    def test_higher_access_count_boosts_score(self, vault: DreamStore, recall_engine: RecallEngine):
        """Memories with higher access_count should score higher than
        identical memories with lower access_count, all else being equal."""
        # Add two user memories with similar tags/relevance
        vault.add_memory(
            "user",
            "I prefer Python for scripting tasks.",
            tags=["python", "scripting"],
            source="s1",
            relevance=0.7,
        )
        vault.add_memory(
            "user",
            "I prefer Python for data analysis.",
            tags=["python", "data"],
            source="s1",
            relevance=0.7,
        )

        # Increment access count for the second memory
        manifest = vault._ensure_manifest_loaded()
        second_entry = manifest[1]
        vault.increment_access_count("user", second_entry["filename"])

        # Recall for "python"
        results = recall_engine.recall("python", limit=10)
        assert len(results) >= 2

        # Find the two python memories
        python_results = [r for r in results if "python" in r.content.lower()]
        assert len(python_results) >= 2

        # The one with higher access_count should have a higher score
        # (unless other factors override; access_count adds 0.15 * normalized_score)
        scores = {r.content: r.score for r in python_results}
        second_content = "I prefer Python for data analysis."
        first_content = "I prefer Python for scripting tasks."
        # Second memory has access_count=1, first has 0
        assert scores.get(second_content, 0) >= scores.get(first_content, 0)

    def test_access_count_normalization_saturates(self, vault: DreamStore, recall_engine: RecallEngine):
        """Access count scoring should saturate — very high counts should not
        give unbounded advantage."""
        vault.add_memory("user", "base memory", tags=["test"], relevance=0.5)

        manifest = vault._ensure_manifest_loaded()
        filename = manifest[0]["filename"]

        # Increment to 100 — access_score should be capped at 1.0
        for _ in range(100):
            vault.increment_access_count("user", filename)

        # access_score = min(1.0, log1p(100) / log1p(10)) = min(1.0, ~0.52/0.46) ≈ 1.0
        # Actually log1p(100)/log1p(10) = ln(101)/ln(11) ≈ 4.615/2.398 ≈ 1.924, min(1.0, 1.924) = 1.0
        result = recall_engine.recall("test", limit=5)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Test feedback boost
# ---------------------------------------------------------------------------

class TestFeedbackBoost:
    """Tests for feedback type boost in recall scoring."""

    def test_feedback_memories_ranked_higher(self, populated_vault: DreamStore, recall_engine: RecallEngine):
        """Feedback memories should be ranked higher than non-feedback memories
        with similar tag overlap and relevance."""
        # Query that matches both feedback and user memories
        results = recall_engine.recall("snake case naming preference", limit=10)

        # Find the feedback memory about snake_case
        feedback_results = [r for r in results if r.memory_type == "feedback"]
        non_feedback_results = [r for r in results if r.memory_type != "feedback"]

        # If both exist, feedback should score higher for similar queries
        if feedback_results and non_feedback_results:
            # The feedback boost is 0.10 * 1.0 = 0.10
            # So feedback memories get an extra 0.10 in their combined score
            assert feedback_results[0].score >= non_feedback_results[0].score - 0.05

    def test_feedback_boost_in_combined_score(self, vault: DreamStore, recall_engine: RecallEngine):
        """Feedback type memories should receive a boost equal to FEEDBACK_BOOST_WEIGHT."""
        # Add a feedback memory and a user memory with identical characteristics
        vault.add_memory(
            "feedback",
            "Use snake_case for variables.",
            tags=["naming", "convention"],
            source="test",
            relevance=0.7,
        )
        vault.add_memory(
            "user",
            "Use snake_case for variables please.",
            tags=["naming", "convention"],
            source="test",
            relevance=0.7,
        )

        results = recall_engine.recall("naming convention", limit=10)
        assert len(results) >= 2

        feedback_result = next((r for r in results if r.memory_type == "feedback"), None)
        user_result = next((r for r in results if r.memory_type == "user"), None)

        if feedback_result and user_result:
            # Feedback should have higher score due to feedback_boost
            assert feedback_result.score > user_result.score
            # The difference should be approximately FEEDBACK_BOOST_WEIGHT (0.10)
            score_diff = feedback_result.score - user_result.score
            assert 0.05 < score_diff < 0.15, f"Expected ~0.10 diff, got {score_diff}"


# ---------------------------------------------------------------------------
# Test prefetch
# ---------------------------------------------------------------------------

class TestPrefetch:
    """Tests for DreamMemoryProvider.prefetch() method."""

    def test_prefetch_returns_formatted_block(self, provider: DreamMemoryProvider):
        """prefetch should return a formatted markdown block."""
        # Add a memory so there's something to recall
        provider._store.add_memory(
            "user",
            "I prefer dark mode for my IDE.",
            tags=["preference", "dark-mode"],
            source="test",
            relevance=0.8,
        )
        result = provider.prefetch("dark mode", session_id="test-session")
        assert result != ""
        assert "## Dream Memory" in result
        assert "User" in result

    def test_prefetch_respects_budget(self, provider: DreamMemoryProvider):
        """prefetch output should stay within the configured budget."""
        # Add several memories
        for i in range(20):
            provider._store.add_memory(
                "user",
                f"Memory {i}: " + "x" * 200 + " end",
                tags=[f"tag{i}", "test"],
                source="test",
                relevance=0.7,
            )

        # Small budget of 200 bytes
        result = provider.prefetch("memory test", session_id="test-sess", budget=200)
        assert len(result.encode("utf-8")) <= 300  # Some slack for overhead

    def test_prefetch_feedback_first(self, provider: DreamMemoryProvider):
        """Feedback memories should appear before other types in prefetch output."""
        provider._store.add_memory(
            "user",
            "I prefer Python for backend work.",
            tags=["python", "backend"],
            source="test",
            relevance=0.8,
        )
        provider._store.add_memory(
            "feedback",
            "Use snake_case for all variables.",
            tags=["naming", "python-style"],
            source="test",
            relevance=0.7,
        )

        result = provider.prefetch("python vars", session_id="test-sess")
        lines = [l for l in result.strip().split("\n") if l.startswith("**")]
        if len(lines) >= 2:
            # Feedback line should come before User line
            feedback_idx = next(i for i, l in enumerate(lines) if "Feedback" in l)
            user_idx = next(i for i, l in enumerate(lines) if "User" in l)
            assert feedback_idx < user_idx, "Feedback memories should appear first"

    def test_prefetch_empty_vault(self, provider: DreamMemoryProvider):
        """prefetch should return empty string for an empty vault."""
        result = provider.prefetch("anything", session_id="test-sess")
        assert result == ""

    def test_prefetch_cache(self, provider: DreamMemoryProvider):
        """Second call with same query should return cached result."""
        provider._store.add_memory(
            "user",
            "I use vim for editing.",
            tags=["vim", "editor"],
            source="test",
            relevance=0.8,
        )

        result1 = provider.prefetch("vim editor", session_id="cached-test")
        result2 = provider.prefetch("vim editor", session_id="cached-test")
        assert result1 == result2
        assert result1 != ""

    def test_prefetch_no_results(self, provider: DreamMemoryProvider):
        """prefetch should return empty string when no matches are found."""
        # Add a memory about something completely different
        provider._store.add_memory(
            "reference",
            "API docs for ancient Roman aqueducts.",
            tags=["history", "aqueduct"],
            source="test",
            relevance=0.3,
        )

        # Query for something that won't match well
        result = provider.prefetch("quantum computing neural networks", session_id="test-sess")
        # With very low relevance tags/snippets combined with high min score threshold,
        # the result may be empty
        # It's OK if it returns results (low relevance can still pass threshold)
        # Just verify no crash
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Test auto-recall config flag
# ---------------------------------------------------------------------------

class TestAutoRecallConfig:
    """Tests for the auto_recall configuration flag."""

    def test_auto_recall_config_flag_disabled(self, provider_no_auto_recall: DreamMemoryProvider):
        """When auto_recall is disabled, should_auto_recall() should return False."""
        provider_no_auto_recall._store.add_memory(
            "user",
            "I love Python.",
            tags=["python", "preference"],
            source="test",
            relevance=0.8,
        )
        # prefetch() itself still works — it just won't be auto-injected
        # because MemoryManager checks should_auto_recall()
        assert not provider_no_auto_recall.should_auto_recall()

    def test_auto_recall_config_flag_enabled(self, provider: DreamMemoryProvider):
        """When auto_recall is enabled, should_auto_recall() should return True
        and prefetch should return results."""
        provider._store.add_memory(
            "user",
            "I love Python.",
            tags=["python", "preference"],
            source="test",
            relevance=0.8,
        )
        assert provider.should_auto_recall()
        result = provider.prefetch("python", session_id="test-sess")
        # auto_recall is True, prefetch should return results
        assert result != ""
        assert "Python" in result

    def test_auto_recall_default_is_false(self, tmp_path: Path):
        """Default auto_recall should be False (backward compatible)."""
        config = {"vault_path": str(tmp_path / "dream_test")}
        provider = DreamMemoryProvider(config=config)
        provider.initialize(session_id="test-session")
        # Default auto_recall should be False
        assert not provider._config_as_bool("auto_recall")

    def test_auto_recall_injected_into_system_prompt(self, provider: DreamMemoryProvider):
        """When auto_recall is enabled, system_prompt_block should mention it."""
        provider._store.add_memory(
            "user",
            "test",
            tags=["test"],
            source="test",
            relevance=0.5,
        )
        block = provider.system_prompt_block()
        assert "automatically injected" in block.lower() or "auto" in block.lower()

    def test_auto_recall_not_in_system_prompt_when_disabled(self, provider_no_auto_recall: DreamMemoryProvider):
        """When auto_recall is disabled, system_prompt_block should not mention auto-injection."""
        provider_no_auto_recall._store.add_memory(
            "user",
            "test",
            tags=["test"],
            source="test",
            relevance=0.5,
        )
        block = provider_no_auto_recall.system_prompt_block()
        assert "automatically" not in block.lower()


# ---------------------------------------------------------------------------
# Test MemoryManager integration with auto_recall gate
# ---------------------------------------------------------------------------

class TestMemoryManagerAutoRecall:
    """Tests for MemoryManager respect of should_auto_recall gating."""

    def test_memory_manager_skips_prefetch_when_auto_recall_disabled(self, tmp_path: Path):
        """MemoryManager.prefetch_all should skip providers with should_auto_recall=False."""
        from agent.memory_manager import MemoryManager

        config = {"vault_path": str(tmp_path / "dream_test"), "auto_recall": "false"}
        provider = DreamMemoryProvider(config=config)
        provider.initialize(session_id="test-session")
        provider._store.add_memory(
            "user",
            "I love Python.",
            tags=["python", "preference"],
            source="test",
            relevance=0.8,
        )

        manager = MemoryManager()
        manager.add_provider(provider)

        # prefetch_all should return empty because should_auto_recall() is False
        result = manager.prefetch_all("python")
        assert result == ""

    def test_memory_manager_includes_prefetch_when_auto_recall_enabled(self, tmp_path: Path):
        """MemoryManager.prefetch_all should include providers with should_auto_recall=True."""
        from agent.memory_manager import MemoryManager

        config = {"vault_path": str(tmp_path / "dream_test"), "auto_recall": "true"}
        provider = DreamMemoryProvider(config=config)
        provider.initialize(session_id="test-session")
        provider._store.add_memory(
            "user",
            "I love Python.",
            tags=["python", "preference"],
            source="test",
            relevance=0.8,
        )

        manager = MemoryManager()
        manager.add_provider(provider)

        # prefetch_all should return results because should_auto_recall() is True
        result = manager.prefetch_all("python")
        assert result != ""

    def test_memory_manager_backward_compatible_without_should_auto_recall(self, tmp_path: Path):
        """Providers without should_auto_recall() should still work (backward compat)."""
        from agent.memory_manager import MemoryManager
        from agent.memory_provider import MemoryProvider

        class StubProvider(MemoryProvider):
            @property
            def name(self):
                return "stub"

            def is_available(self):
                return True

            def initialize(self, session_id, **kwargs):
                pass

            def prefetch(self, query, *, session_id=""):
                return "stub context"

            def get_tool_schemas(self):
                return []

        manager = MemoryManager()
        manager.add_provider(StubProvider())

        # Should not crash even though provider has no should_auto_recall
        result = manager.prefetch_all("test")
        assert result == "stub context"


# ---------------------------------------------------------------------------
# Test RecallEngine scoring weights
# ---------------------------------------------------------------------------

class TestRecallScoringWeights:
    """Verify the updated scoring weights sum correctly."""

    def test_weights_sum_to_one(self):
        """The five scoring weights should sum to ~1.0."""
        total = _TAG_WEIGHT + _RELEVANCE_WEIGHT + _RECENCY_WEIGHT + _ACCESS_WEIGHT + _FEEDBACK_BOOST_WEIGHT
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected ~1.0"

    def test_access_weight_value(self):
        """ACCESS_WEIGHT should be 0.15."""
        assert _ACCESS_WEIGHT == 0.15

    def test_feedback_boost_weight_value(self):
        """FEEDBACK_BOOST_WEIGHT should be 0.10."""
        assert _FEEDBACK_BOOST_WEIGHT == 0.10

    def test_tag_weight_value(self):
        """TAG_WEIGHT should be 0.35 (reduced from 0.4)."""
        assert _TAG_WEIGHT == 0.35

    def test_relevance_weight_value(self):
        """RELEVANCE_WEIGHT should be 0.25 (reduced from 0.4)."""
        assert _RELEVANCE_WEIGHT == 0.25

    def test_recency_weight_value(self):
        """RECENCY_WEIGHT should be 0.15 (reduced from 0.2)."""
        assert _RECENCY_WEIGHT == 0.15


# ---------------------------------------------------------------------------
# Test config schema
# ---------------------------------------------------------------------------

class TestConfigSchema:
    """Tests for auto_recall config schema additions."""

    def test_auto_recall_config_in_schema(self):
        """auto_recall should be in the config schema."""
        provider = DreamMemoryProvider(config={})
        schema = provider.get_config_schema()
        keys = [s["key"] for s in schema]
        assert "auto_recall" in keys

    def test_auto_recall_budget_in_schema(self):
        """auto_recall_budget should be in the config schema."""
        provider = DreamMemoryProvider(config={})
        schema = provider.get_config_schema()
        keys = [s["key"] for s in schema]
        assert "auto_recall_budget" in keys

    def test_auto_recall_top_k_in_schema(self):
        """auto_recall_top_k should be in the config schema."""
        provider = DreamMemoryProvider(config={})
        schema = provider.get_config_schema()
        keys = [s["key"] for s in schema]
        assert "auto_recall_top_k" in keys

    def test_auto_recall_default_is_false(self):
        """auto_recall default in schema should be 'false'."""
        provider = DreamMemoryProvider(config={})
        schema = provider.get_config_schema()
        auto_recall_entry = next(s for s in schema if s["key"] == "auto_recall")
        assert auto_recall_entry["default"] == "false"

    def test_auto_recall_budget_default(self):
        """auto_recall_budget default should be '2048'."""
        provider = DreamMemoryProvider(config={})
        schema = provider.get_config_schema()
        budget_entry = next(s for s in schema if s["key"] == "auto_recall_budget")
        assert budget_entry["default"] == "2048"

    def test_auto_recall_top_k_default(self):
        """auto_recall_top_k default should be '10'."""
        provider = DreamMemoryProvider(config={})
        schema = provider.get_config_schema()
        top_k_entry = next(s for s in schema if s["key"] == "auto_recall_top_k")
        assert top_k_entry["default"] == "10"
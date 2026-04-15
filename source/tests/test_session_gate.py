"""Tests for Heartbeat 2: Session Count Gate for Dream Memory consolidation.

Tests the dual-gate consolidation logic (time + session count),
session counter persistence, and the Anthropic-compatible bypass rules.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from plugins.memory.dream.store import DreamStore
from plugins.memory.dream.taxonomy import MEMORY_TYPES, render_frontmatter
from plugins.memory.dream.consolidation import (
    orient,
    run_consolidation,
    MIN_SESSIONS_FOR_CONSOLIDATION,
    CONSOLIDATION_INTERVAL_HOURS,
)


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


def _write_consolidation_log(store: DreamStore, hours_ago: float = 25.0) -> None:
    """Write a consolidation_log.json entry to simulate a past consolidation."""
    log_path = store.dream_root / "consolidation_log.json"
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    log_entry = [{"timestamp": ts, "actions": [], "summary": {}}]
    log_path.write_text(json.dumps(log_entry), encoding="utf-8")


def _write_oversized_memory(store: DreamStore, lines: int = 300) -> Path:
    """Write a memory that exceeds the default max_lines to trigger urgent bypass."""
    content = "\n".join([f"Line {i}" for i in range(lines)])
    return store.add_memory(
        memory_type="user",
        content=content,
        tags=["big"],
        source="test",
        relevance=0.5,
    )


# ---------------------------------------------------------------------------
# Session counter unit tests
# ---------------------------------------------------------------------------


class TestSessionCounter:
    """Unit tests for DreamStore session counter methods."""

    def test_session_counter_starts_at_zero(self, vault: DreamStore):
        """New vault has 0 sessions."""
        assert vault.get_sessions_since_consolidation() == 0

    def test_increment_session_counter(self, vault: DreamStore):
        """Incrementing increases count."""
        new_count = vault.increment_session_counter()
        assert new_count == 1
        assert vault.get_sessions_since_consolidation() == 1

    def test_increment_multiple_times(self, vault: DreamStore):
        """Multiple increments accumulate."""
        for _ in range(5):
            vault.increment_session_counter()
        assert vault.get_sessions_since_consolidation() == 5

    def test_reset_session_counter(self, vault: DreamStore):
        """Reset sets count back to 0."""
        vault.increment_session_counter()
        vault.increment_session_counter()
        assert vault.get_sessions_since_consolidation() == 2
        vault.reset_session_counter()
        assert vault.get_sessions_since_consolidation() == 0

    def test_session_counter_persists(self, tmp_path: Path):
        """Counter persisted across DreamStore instances."""
        vault_path = tmp_path / "dream_vault"
        store1 = DreamStore(vault_path)
        store1.initialize()
        store1.increment_session_counter()
        store1.increment_session_counter()

        # Create a new store instance pointing at the same vault
        store2 = DreamStore(vault_path)
        store2.initialize()
        assert store2.get_sessions_since_consolidation() == 2

    def test_session_counter_file_format(self, vault: DreamStore):
        """Counter file has correct JSON format with last_increment timestamp."""
        vault.increment_session_counter()
        counter_path = vault.dream_root / ".session_counter.json"
        data = json.loads(counter_path.read_text())
        assert "count" in data
        assert data["count"] == 1
        assert "last_increment" in data
        # Verify timestamp is valid ISO format
        dt = datetime.fromisoformat(data["last_increment"])
        assert dt.tzinfo is not None

    def test_corrupted_counter_returns_zero(self, vault: DreamStore):
        """Corrupted .session_counter.json returns 0 (backward compatible)."""
        counter_path = vault.dream_root / ".session_counter.json"
        counter_path.write_text("not valid json{{{")
        assert vault.get_sessions_since_consolidation() == 0


# ---------------------------------------------------------------------------
# orient() dual-gate tests
# ---------------------------------------------------------------------------


class TestOrientSessionGate:
    """Tests for orient() session count gate logic."""

    def test_orient_rejects_insufficient_sessions(self, vault: DreamStore):
        """orient returns needs_consolidation=False when sessions < min_sessions
        and time gate has passed."""
        # Add a memory so vault isn't empty
        _add_memory(vault)

        # Write consolidation log 25h ago (time gate passed)
        _write_consolidation_log(vault, hours_ago=25.0)

        # Only 2 sessions (less than default 5)
        vault.increment_session_counter()
        vault.increment_session_counter()

        result = orient(vault)
        assert result.needs_consolidation is False
        assert "session" in result.reason.lower() or "2/5" in result.reason

    def test_orient_accepts_sufficient_sessions_and_time(self, vault: DreamStore):
        """orient returns True when both time gate and session gate pass."""
        _add_memory(vault)
        _write_consolidation_log(vault, hours_ago=25.0)

        # 5 sessions (meets default min_sessions)
        for _ in range(5):
            vault.increment_session_counter()

        result = orient(vault)
        assert result.needs_consolidation is True
        assert "session" in result.reason.lower() or "5" in result.reason

    def test_orient_first_consolidation_with_data(self, vault: DreamStore):
        """First-ever consolidation runs when there's data (no session gate)."""
        _add_memory(vault)

        # No consolidation log = never consolidated
        # Even with 0 sessions, first consolidation should run
        result = orient(vault)
        assert result.needs_consolidation is True
        assert "no previous consolidation" in result.reason

    def test_orient_first_consolidation_no_data(self, vault: DreamStore):
        """First-ever consolidation runs even on empty vault (harmless no-op)."""
        # No memories, no consolidation log
        result = orient(vault)
        # First consolidation always runs; gather will find no entries and skip
        assert result.needs_consolidation is True
        assert "no previous consolidation" in result.reason

    def test_orient_bypasses_on_urgent(self, vault: DreamStore):
        """Oversized files bypass session gate — urgent always runs."""
        _add_memory(vault)
        _write_consolidation_log(vault, hours_ago=1.0)  # recent, time gate NOT passed

        # Only 1 session
        vault.increment_session_counter()

        # Add oversized memory to trigger urgent bypass
        _write_oversized_memory(vault)

        result = orient(vault)
        assert result.needs_consolidation is True
        assert "urgent" in result.reason.lower()

    def test_orient_time_not_passed_no_sessions(self, vault: DreamStore):
        """Time gate not passed — consolidation should not fire regardless of sessions."""
        _add_memory(vault)
        _write_consolidation_log(vault, hours_ago=1.0)  # only 1h ago

        # 10 sessions (plenty)
        for _ in range(10):
            vault.increment_session_counter()

        result = orient(vault)
        assert result.needs_consolidation is False

    def test_orient_session_stats_in_result(self, vault: DreamStore):
        """orient result includes session info in stats."""
        _add_memory(vault)
        for _ in range(3):
            vault.increment_session_counter()

        result = orient(vault)
        assert "sessions_since_consolidation" in result.stats
        assert result.stats["sessions_since_consolidation"] == 3
        assert "min_sessions" in result.stats
        assert result.stats["min_sessions"] == MIN_SESSIONS_FOR_CONSOLIDATION


class TestOrientSessionGateConfigurable:
    """Tests for configurable min_sessions_for_consolidation."""

    def test_session_counter_configurable(self, tmp_path: Path):
        """min_sessions can be set via store config."""
        config = {"min_sessions_for_consolidation": 2}
        store = DreamStore(tmp_path / "dream_vault", config=config)
        store.initialize()
        _add_memory(store)
        _write_consolidation_log(store, hours_ago=25.0)

        # 2 sessions meets the configured minimum
        store.increment_session_counter()
        store.increment_session_counter()

        result = orient(store)
        assert result.needs_consolidation is True
        assert result.stats["min_sessions"] == 2

    def test_session_counter_config_not_met(self, tmp_path: Path):
        """Configured min_sessions not met blocks consolidation."""
        config = {"min_sessions_for_consolidation": 10}
        store = DreamStore(tmp_path / "dream_vault", config=config)
        store.initialize()
        _add_memory(store)
        _write_consolidation_log(store, hours_ago=25.0)

        # Only 5 sessions — less than configured 10
        for _ in range(5):
            store.increment_session_counter()

        result = orient(store)
        assert result.needs_consolidation is False
        assert result.stats["min_sessions"] == 10

    def test_session_counter_config_string_value(self, tmp_path: Path):
        """min_sessions as string (from YAML config) is handled correctly."""
        config = {"min_sessions_for_consolidation": "3"}
        store = DreamStore(tmp_path / "dream_vault", config=config)
        store.initialize()
        _add_memory(store)
        _write_consolidation_log(store, hours_ago=25.0)

        for _ in range(3):
            store.increment_session_counter()

        result = orient(store)
        assert result.needs_consolidation is True
        assert result.stats["min_sessions"] == 3


# ---------------------------------------------------------------------------
# Session counter reset after consolidation
# ---------------------------------------------------------------------------


class TestSessionCounterReset:
    """Tests for session counter being reset after successful consolidation."""

    def test_session_counter_reset_after_consolidation(self, vault: DreamStore):
        """Successful consolidation resets the session counter."""
        _add_memory(vault)
        _add_memory(vault, content="another memory", tags=["test"])

        # Simulate 5+ sessions and 25h since last consolidation
        _write_consolidation_log(vault, hours_ago=25.0)
        for _ in range(5):
            vault.increment_session_counter()

        assert vault.get_sessions_since_consolidation() >= 5

        # Run consolidation (not dry run)
        result = run_consolidation(vault, config={"max_lines": 200}, dry_run=False)

        # Manifest should have been updated (prune always reloads manifest in non-dry-run)
        assert result.prune.manifest_updated is True

        # Session counter should be reset to 0
        # Re-read from disk (avoid any caching issues)
        counter_path = vault.dream_root / ".session_counter.json"
        data = json.loads(counter_path.read_text())
        assert data["count"] == 0

    def test_session_counter_not_reset_on_dry_run(self, vault: DreamStore):
        """Dry-run consolidation does NOT reset the session counter."""
        _add_memory(vault)
        _write_consolidation_log(vault, hours_ago=25.0)
        for _ in range(5):
            vault.increment_session_counter()

        count_before = vault.get_sessions_since_consolidation()
        run_consolidation(vault, config={"max_lines": 200}, dry_run=True)

        # Counter should NOT be reset
        assert vault.get_sessions_since_consolidation() == count_before
"""Tests for Dream Memory Cron Integration.

Tests cover:
  - run_cron_consolidation() in consolidation.py
  - DreamMemoryProvider.setup_cron() / remove_cron() / cron_status()
  - Standalone helpers: setup_consolidation_cron / remove_consolidation_cron / get_consolidation_cron_status
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from plugins.memory.dream.consolidation import run_cron_consolidation
from plugins.memory.dream.store import DreamStore


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
def hermes_home_with_config(tmp_path: Path) -> Path:
    """Create a minimal hermes_home with config.yaml and dream config."""
    config = tmp_path / "config.yaml"
    config.write_text(
        "plugins:\n"
        "  dream:\n"
        "    vault_path: {vault}\n"
        "    max_lines: 100\n"
        "    max_bytes: 50000\n"
        "    consolidate_cron: '0 3 * * *'\n".format(
            vault=str(tmp_path / "dream_vault")
        ),
        encoding="utf-8",
    )
    return tmp_path


# ---------------------------------------------------------------------------
# run_cron_consolidation tests
# ---------------------------------------------------------------------------


class TestRunCronConsolidation:
    """Test the standalone cron entry point in consolidation.py."""

    def test_smoke_run_with_empty_vault(self, hermes_home_with_config: Path):
        """run_cron_consolidation should succeed even with an empty vault."""
        # Create the vault directory (empty vault)
        vault_path = hermes_home_with_config / "dream_vault"
        vault_path.mkdir(exist_ok=True)

        result = run_cron_consolidation(hermes_home=str(hermes_home_with_config))
        assert isinstance(result, str)
        assert "[dream cron]" in result
        # Empty vault → "not needed" or "no entries"
        assert "SKIP" in result or "completed" in result.lower() or "entries" in result.lower()

    def test_run_with_existing_memories(self, hermes_home_with_config: Path):
        """run_cron_consolidation should consolidate and return a summary."""
        # Create the vault and add memories
        vault_path = hermes_home_with_config / "dream_vault"
        store = DreamStore(vault_path)
        store.initialize()

        # Add some similar memories to trigger consolidation
        store.add_memory("user", "I prefer dark mode for editing", tags=["editor"], relevance=0.8)
        store.add_memory("user", "I prefer dark mode for all tasks", tags=["editor"], relevance=0.7)

        result = run_cron_consolidation(hermes_home=str(hermes_home_with_config))
        assert isinstance(result, str)
        assert "[dream cron]" in result

    def test_run_without_hermes_home_uses_default(self, tmp_path: Path):
        """When hermes_home is None, it should attempt to fall back."""
        # Patch hermes_constants.get_hermes_home to return tmp_path
        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            # Create a vault so it doesn't skip
            vault_path = tmp_path / "dream_vault"
            vault_path.mkdir(exist_ok=True)
            result = run_cron_consolidation(hermes_home=None)
            assert isinstance(result, str)

    def test_missing_vault_path_returns_skip(self, tmp_path: Path):
        """If the vault path doesn't exist, should return a SKIP message."""
        config = tmp_path / "config.yaml"
        config.write_text(
            "plugins:\n  dream:\n    vault_path: /nonexistent/path\n"
        )
        result = run_cron_consolidation(hermes_home=str(tmp_path))
        assert "SKIP" in result

    def test_no_config_file_uses_defaults(self, tmp_path: Path):
        """Without a config.yaml, should use defaults and try the default vault."""
        # Create the default vault location
        default_vault = tmp_path / "dream_vault"
        default_vault.mkdir(exist_ok=True)
        DreamStore(default_vault).initialize()

        result = run_cron_consolidation(hermes_home=str(tmp_path))
        assert isinstance(result, str)
        assert "[dream cron]" in result

    def test_consolidation_failure_returns_error(self, hermes_home_with_config: Path):
        """If consolidation fails, should return an ERROR message."""
        vault_path = hermes_home_with_config / "dream_vault"
        vault_path.mkdir(exist_ok=True)

        with patch(
            "plugins.memory.dream.consolidation.run_consolidation",
            side_effect=RuntimeError("test failure"),
        ):
            result = run_cron_consolidation(hermes_home=str(hermes_home_with_config))
            assert "ERROR" in result
            assert "test failure" in result


# ---------------------------------------------------------------------------
# DreamMemoryProvider cron methods
# ---------------------------------------------------------------------------


class TestDreamCronMethods:
    """Test setup_cron, remove_cron, cron_status on DreamMemoryProvider."""

    def test_setup_cron_creates_job(self):
        """setup_cron should call cron.jobs.create_job."""
        provider = MagicMock(spec=["_config"])
        provider._config = {"consolidate_cron": "0 3 * * *"}

        mock_job = {
            "id": "abc123",
            "name": "dream-consolidation",
            "schedule_display": "0 3 * * *",
            "enabled": True,
        }

        with patch("plugins.memory.dream.remove_consolidation_cron", return_value=False):
            with patch("cron.jobs.create_job", return_value=mock_job) as mock_create:
                # Import the real class to test setup_cron
                from plugins.memory.dream import DreamMemoryProvider

                real_provider = DreamMemoryProvider(
                    config={"consolidate_cron": "0 3 * * *"}
                )
                with patch.object(real_provider, "_config", {"consolidate_cron": "0 3 * * *"}):
                    result = real_provider.setup_cron()

                assert result.get("name") == "dream-consolidation"
                mock_create.assert_called_once()

    def test_setup_cron_uses_config_schedule(self):
        """setup_cron should use the consolidate_cron config value."""
        from plugins.memory.dream import DreamMemoryProvider

        provider = DreamMemoryProvider(config={"consolidate_cron": "0 5 * * *"})

        with patch("plugins.memory.dream.remove_consolidation_cron", return_value=False):
            with patch("cron.jobs.create_job", return_value={"id": "x", "name": "dream-consolidation"}) as mock_create:
                provider.setup_cron()
                call_kwargs = mock_create.call_args
                assert call_kwargs[1]["schedule"] == "0 5 * * *" or call_kwargs[0][1] == "0 5 * * *" or "0 5 * * *" in str(call_kwargs)

    def test_setup_cron_default_schedule(self):
        """setup_cron should default to '0 3 * * *' when no config."""
        from plugins.memory.dream import DreamMemoryProvider

        provider = DreamMemoryProvider(config={})

        with patch("plugins.memory.dream.remove_consolidation_cron", return_value=False):
            with patch("cron.jobs.create_job", return_value={"id": "x", "name": "dream-consolidation"}) as mock_create:
                provider.setup_cron()
                # Check either positional or keyword arg for schedule
                call_args = mock_create.call_args
                schedule = call_args.kwargs.get("schedule") or call_args[1].get("schedule") if len(call_args) > 1 and isinstance(call_args[1], dict) else None
                if schedule is None:
                    # It's a positional arg in the (prompt, schedule, name, ...) order
                    # create_job(prompt, schedule, name=...)
                    schedule = call_args[0][1] if len(call_args[0]) > 1 else call_args.kwargs.get("schedule")
                assert schedule == "0 3 * * *"

    def test_remove_cron_success(self):
        """remove_cron should return removed=True when a job is found."""
        from plugins.memory.dream import DreamMemoryProvider

        provider = DreamMemoryProvider(config={})

        with patch("plugins.memory.dream.remove_consolidation_cron", return_value=True):
            result = provider.remove_cron()
            assert result["removed"] is True

    def test_remove_cron_no_job(self):
        """remove_cron should return removed=False when no job exists."""
        from plugins.memory.dream import DreamMemoryProvider

        provider = DreamMemoryProvider(config={})

        with patch("plugins.memory.dream.remove_consolidation_cron", return_value=False):
            result = provider.remove_cron()
            assert result["removed"] is False

    def test_cron_status_existing_job(self):
        """cron_status should return job details when a job exists."""
        from plugins.memory.dream import DreamMemoryProvider

        provider = DreamMemoryProvider(config={})

        mock_job = {
            "id": "abc123",
            "name": "dream-consolidation",
            "enabled": True,
            "schedule_display": "0 3 * * *",
            "next_run_at": "2025-01-01T03:00:00+00:00",
            "last_run_at": None,
            "last_status": None,
        }

        with patch("cron.jobs.list_jobs", return_value=[mock_job]):
            result = provider.cron_status()
            assert result["exists"] is True
            assert result["enabled"] is True
            assert result["schedule"] == "0 3 * * *"
            assert result["job_id"] == "abc123"

    def test_cron_status_no_job(self):
        """cron_status should return exists=False when no job exists."""
        from plugins.memory.dream import DreamMemoryProvider

        provider = DreamMemoryProvider(config={})

        with patch("cron.jobs.list_jobs", return_value=[]):
            result = provider.cron_status()
            assert result["exists"] is False

    def test_cron_status_error(self):
        """cron_status should handle errors gracefully."""
        from plugins.memory.dream import DreamMemoryProvider

        provider = DreamMemoryProvider(config={})

        with patch("cron.jobs.list_jobs", side_effect=Exception("db error")):
            result = provider.cron_status()
            assert result["exists"] is False
            assert "error" in result


# ---------------------------------------------------------------------------
# Standalone helper tests
# ---------------------------------------------------------------------------


class TestStandaloneCronHelpers:
    """Test the module-level setup_consolidation_cron, remove_consolidation_cron, get_consolidation_cron_status."""

    def test_setup_consolidation_cron(self):
        """setup_consolidation_cron should delegate to provider.setup_cron()."""
        from plugins.memory.dream import setup_consolidation_cron

        mock_job = {"id": "test-id", "name": "dream-consolidation"}
        with patch("plugins.memory.dream._load_plugin_config", return_value={}):
            with patch("hermes_constants.get_hermes_home", return_value=Path("/tmp")):
                with patch("plugins.memory.dream.remove_consolidation_cron", return_value=False):
                    with patch("cron.jobs.create_job", return_value=mock_job):
                        result = setup_consolidation_cron(hermes_home="/tmp")
                        assert result.get("name") == "dream-consolidation"

    def test_remove_consolidation_cron(self):
        """remove_consolidation_cron should find and remove matching jobs."""
        from plugins.memory.dream import remove_consolidation_cron

        mock_jobs = [
            {"id": "abc", "name": "dream-consolidation"},
            {"id": "def", "name": "other-job"},
        ]
        with patch("cron.jobs.list_jobs", return_value=mock_jobs):
            with patch("cron.jobs.remove_job", return_value=True) as mock_remove:
                result = remove_consolidation_cron(hermes_home="/tmp")
                assert result is True
                mock_remove.assert_called_once_with("abc")

    def test_remove_consolidation_cron_no_match(self):
        """remove_consolidation_cron returns False when no matching job."""
        from plugins.memory.dream import remove_consolidation_cron

        with patch("cron.jobs.list_jobs", return_value=[{"id": "xyz", "name": "other-job"}]):
            result = remove_consolidation_cron(hermes_home="/tmp")
            assert result is False

    def test_get_consolidation_cron_status(self):
        """get_consolidation_cron_status should return job info."""
        from plugins.memory.dream import get_consolidation_cron_status

        mock_job = {
            "id": "abc123",
            "name": "dream-consolidation",
            "enabled": True,
            "schedule_display": "0 3 * * *",
            "next_run_at": "2025-01-01T03:00:00+00:00",
            "last_run_at": None,
            "last_status": None,
        }

        with patch("plugins.memory.dream._load_plugin_config", return_value={}):
            with patch("cron.jobs.list_jobs", return_value=[mock_job]):
                result = get_consolidation_cron_status(hermes_home="/tmp")
                assert result["exists"] is True
                assert result["job_id"] == "abc123"


# ---------------------------------------------------------------------------
# Integration: run_cron_consolidation with real vault
# ---------------------------------------------------------------------------


class TestCronConsolidationIntegration:
    """Integration test running run_cron_consolidation with a real vault."""

    def test_full_cycle_with_duplicates(self, tmp_path: Path):
        """Run cron consolidation on a vault with duplicate memories."""
        vault_path = tmp_path / "dream_vault"
        store = DreamStore(vault_path)
        store.initialize()

        # Add duplicate memories
        store.add_memory("user", "The user prefers dark mode for coding", tags=["preference"], relevance=0.8)
        store.add_memory("user", "The user prefers dark mode for coding", tags=["preference"], relevance=0.7)

        # Write config
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            "plugins:\n  dream:\n    vault_path: {v}\n    max_lines: 100\n    max_bytes: 50000\n".format(
                v=str(vault_path)
            ),
            encoding="utf-8",
        )

        result = run_cron_consolidation(hermes_home=str(tmp_path))
        assert isinstance(result, str)
        # Should have processed the vault (either SKIP or completed)
        assert "[dream cron]" in result
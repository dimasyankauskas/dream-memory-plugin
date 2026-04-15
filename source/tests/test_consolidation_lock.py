"""Tests for ConsolidationLock — PID-based file lock for dream consolidation."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from plugins.memory.dream.consolidation import (
    ConsolidationLock,
    ConsolidationLockError,
    ConsolidationResult,
    run_consolidation,
)
from plugins.memory.dream.store import DreamStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vault_path(tmp_path: Path) -> Path:
    """Create a temp vault directory."""
    vp = tmp_path / "dream_vault"
    vp.mkdir(parents=True, exist_ok=True)
    return vp


@pytest.fixture
def lock(vault_path: Path) -> ConsolidationLock:
    """Create a ConsolidationLock for the vault."""
    return ConsolidationLock(vault_path)


@pytest.fixture
def store(tmp_path: Path) -> DreamStore:
    """Create a DreamStore rooted in a temp directory, initialised."""
    s = DreamStore(tmp_path / "dream_vault")
    s.initialize()
    return s


# ---------------------------------------------------------------------------
# ConsolidationLock unit tests
# ---------------------------------------------------------------------------


class TestConsolidationLock:
    """Unit tests for the ConsolidationLock class."""

    def test_acquire_creates_lock_file(self, lock: ConsolidationLock, vault_path: Path):
        """Acquiring creates .consolidation.lock with PID."""
        assert not lock.lock_path.exists()
        result = lock.acquire()
        assert result is True
        assert lock.lock_path.exists()
        pid_str = lock.lock_path.read_text(encoding="utf-8").strip()
        assert int(pid_str) == os.getpid()
        # Clean up
        lock.release()

    def test_release_deletes_lock_file(self, lock: ConsolidationLock, vault_path: Path):
        """Release removes the lock file."""
        lock.acquire()
        assert lock.lock_path.exists()
        lock.release()
        assert not lock.lock_path.exists()

    def test_concurrent_acquire_fails(self, lock: ConsolidationLock, vault_path: Path):
        """Second acquire on same path returns False."""
        lock.acquire()
        # Create a second lock object for the same path
        lock2 = ConsolidationLock(vault_path)
        result = lock2.acquire()
        assert result is False
        # First lock still holds it
        assert lock.lock_path.exists()
        pid_str = lock.lock_path.read_text(encoding="utf-8").strip()
        assert int(pid_str) == os.getpid()
        lock.release()

    def test_stale_lock_is_broken(self, vault_path: Path):
        """Lock file older than 1hr is auto-broken."""
        lock = ConsolidationLock(vault_path)
        # Create a lock file with current PID but old mtime
        lock.lock_path.write_text(str(os.getpid()), encoding="utf-8")
        # Set mtime to 2 hours ago (older than STALE_THRESHOLD_SECONDS=3600)
        old_time = time.time() - 7200
        os.utime(lock.lock_path, (old_time, old_time))

        # A new lock should be able to break the stale one
        lock2 = ConsolidationLock(vault_path)
        result = lock2.acquire()
        assert result is True
        # Lock is now held by lock2 (which is also our PID - it's fine since same process)
        lock2.release()

    def test_dead_pid_lock_is_broken(self, vault_path: Path):
        """Lock with dead PID is auto-broken."""
        # Create a lock file with a PID that definitely doesn't exist
        # Use a very high PID number which is extremely unlikely to be alive
        dead_pid = 999999999
        lock_file = vault_path / ".consolidation.lock"
        lock_file.write_text(str(dead_pid), encoding="utf-8")

        lock = ConsolidationLock(vault_path)
        result = lock.acquire()
        assert result is True
        # Verify the lock now has our PID
        pid_str = lock.lock_path.read_text(encoding="utf-8").strip()
        assert int(pid_str) == os.getpid()
        lock.release()

    def test_context_manager(self, lock: ConsolidationLock, vault_path: Path):
        """Works as context manager, auto-releases on exit."""
        with lock:
            assert lock.lock_path.exists()
        # After exiting, lock should be released
        assert not lock.lock_path.exists()

    def test_context_manager_exception(self, lock: ConsolidationLock, vault_path: Path):
        """Lock released even if exception occurs inside context manager."""
        try:
            with lock:
                assert lock.lock_path.exists()
                raise ValueError("test error")
        except ValueError:
            pass
        # Lock should still be released despite the exception
        assert not lock.lock_path.exists()

    def test_consolidation_lock_error(self, vault_path: Path):
        """Context manager raises ConsolidationLockError when locked."""
        lock1 = ConsolidationLock(vault_path)
        lock1.acquire()

        lock2 = ConsolidationLock(vault_path)
        with pytest.raises(ConsolidationLockError, match="Consolidation already in progress"):
            with lock2:
                pass

        lock1.release()

    def test_is_locked_property(self, lock: ConsolidationLock, vault_path: Path):
        """is_locked returns True when locked, False when released."""
        assert not lock.is_locked
        lock.acquire()
        assert lock.is_locked
        lock.release()
        assert not lock.is_locked

    def test_acquire_idempotent(self, lock: ConsolidationLock, vault_path: Path):
        """Acquiring an already-held lock returns True (same lock object)."""
        lock.acquire()
        # Acquiring again on the same object should still return True
        # since we already own it (the PID matches)
        result = lock.acquire()
        assert result is True
        lock.release()


# ---------------------------------------------------------------------------
# Integration tests with run_consolidation
# ---------------------------------------------------------------------------


class TestRunConsolidationLock:
    """Integration tests for ConsolidationLock wired into run_consolidation."""

    def test_run_consolidation_with_lock(self, store: DreamStore):
        """run_consolidation acquires and releases lock."""
        lock_path = store.dream_root / ".consolidation.lock"
        assert not lock_path.exists()

        result = run_consolidation(store, dry_run=False)
        # After completion, lock should be released
        # (It may or may not have done consolidation depending on vault state,
        # but the lock should definitely be released)
        assert not lock_path.exists()
        assert isinstance(result, ConsolidationResult)

    def test_run_consolidation_concurrent_blocked(self, store: DreamStore):
        """Second consolidation while first runs is blocked."""
        lock_path = store.dream_root / ".consolidation.lock"

        # Simulate an active consolidation by manually acquiring the lock
        lock = ConsolidationLock(store.dream_root)
        lock.acquire()
        assert lock_path.exists()

        # Try to run consolidation — should be blocked
        result = run_consolidation(store, dry_run=False)
        assert result.orient.reason == "consolidation already in progress"

        # Clean up
        lock.release()

    def test_lock_file_location(self, store: DreamStore):
        """Lock file is at dream_root/.consolidation.lock."""
        lock = ConsolidationLock(store.dream_root)
        assert lock.lock_path == store.dream_root / ".consolidation.lock"
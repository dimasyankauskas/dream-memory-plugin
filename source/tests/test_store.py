"""Tests for dream memory store — CRUD, directory creation, manifest updates."""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from plugins.memory.dream.store import DreamStore, slugify, _MANIFEST_FILE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def vault(tmp_path: Path) -> DreamStore:
    """Create a DreamStore rooted in a temp directory, initialised."""
    store = DreamStore(tmp_path / "dream_vault")
    store.initialize()
    return store


# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------

class TestSlugify:
    def test_basic(self):
        assert slugify("Hello World") == "hello-world"

    def test_special_chars(self):
        assert slugify("Config: ~/.bashrc & aliases") == "config-bashrc-aliases"

    def test_long_text(self):
        result = slugify("one two three four five six seven eight")
        # limited to 6 words
        assert result == "one-two-three-four-five-six"

    def test_empty(self):
        assert slugify("") == "memory"


# ---------------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------------

class TestInitialize:
    def test_creates_type_subdirs(self, vault: DreamStore):
        for type_name in ("user", "feedback", "project", "reference"):
            assert (vault.vault_path / type_name).is_dir()

    def test_creates_manifest(self, vault: DreamStore):
        manifest = vault.vault_path / _MANIFEST_FILE
        assert manifest.exists()
        data = json.loads(manifest.read_text())
        assert data == []


# ---------------------------------------------------------------------------
# add_memory
# ---------------------------------------------------------------------------

class TestAddMemory:
    def test_creates_file(self, vault: DreamStore):
        path = vault.add_memory("user", "Prefers vim editor", tags=["editor"], source="sess-1")
        assert path.exists()
        assert path.suffix == ".md"

    def test_file_has_frontmatter(self, vault: DreamStore):
        path = vault.add_memory("project", "Using Python 3.12", tags=["python"])
        content = path.read_text()
        assert content.startswith("---")
        assert "type: project" in content

    def test_auto_generates_filename(self, vault: DreamStore):
        path = vault.add_memory("feedback", "Wrong indentation style")
        # Filename should contain a slug
        assert "wrong-indentation-style" in path.name or "wrong" in path.name

    def test_invalid_type_raises(self, vault: DreamStore):
        with pytest.raises(ValueError, match="Invalid memory type"):
            vault.add_memory("bogus", "test content")

    def test_updates_manifest(self, vault: DreamStore):
        vault.add_memory("user", "test content")
        manifest_path = vault.vault_path / _MANIFEST_FILE
        data = json.loads(manifest_path.read_text())
        assert len(data) == 1
        assert data[0]["type"] == "user"


# ---------------------------------------------------------------------------
# read_memory
# ---------------------------------------------------------------------------

class TestReadMemory:
    def test_read_roundtrip(self, vault: DreamStore):
        path = vault.add_memory("reference", "FastAPI docs", tags=["api"], source="sess-2", relevance=0.9)
        filename = path.name
        result = vault.read_memory("reference", filename)
        assert result["meta"]["type"] == "reference"
        assert "FastAPI docs" in result["body"]

    def test_read_nonexistent_raises(self, vault: DreamStore):
        with pytest.raises(FileNotFoundError):
            vault.read_memory("user", "nonexistent.md")


# ---------------------------------------------------------------------------
# update_memory
# ---------------------------------------------------------------------------

class TestUpdateMemory:
    def test_update_content(self, vault: DreamStore):
        path = vault.add_memory("user", "Original content")
        filename = path.name

        vault.update_memory("user", filename, content="Updated content")
        result = vault.read_memory("user", filename)
        assert "Updated content" in result["body"]

    def test_update_tags(self, vault: DreamStore):
        path = vault.add_memory("project", "My project", tags=["old-tag"])
        filename = path.name

        vault.update_memory("project", filename, tags=["new-tag", "important"])
        result = vault.read_memory("project", filename)
        assert "new-tag" in result["meta"]["tags"]

    def test_update_relevance(self, vault: DreamStore):
        path = vault.add_memory("feedback", "something", relevance=0.3)
        filename = path.name

        vault.update_memory("feedback", filename, relevance=0.8)
        result = vault.read_memory("feedback", filename)
        assert abs(result["meta"]["relevance"] - 0.8) < 0.01

    def test_update_nonexistent_raises(self, vault: DreamStore):
        with pytest.raises(FileNotFoundError):
            vault.update_memory("user", "nope.md", content="nope")


# ---------------------------------------------------------------------------
# delete_memory
# ---------------------------------------------------------------------------

class TestDeleteMemory:
    def test_delete_removes_file(self, vault: DreamStore):
        path = vault.add_memory("user", "To be deleted")
        filename = path.name
        assert path.exists()

        result = vault.delete_memory("user", filename)
        assert result is True
        assert not path.exists()

    def test_delete_nonexistent_returns_false(self, vault: DreamStore):
        result = vault.delete_memory("user", "nonexistent.md")
        assert result is False

    def test_delete_updates_manifest(self, vault: DreamStore):
        path = vault.add_memory("user", "temp")
        filename = path.name
        vault.delete_memory("user", filename)

        manifest_path = vault.vault_path / _MANIFEST_FILE
        data = json.loads(manifest_path.read_text())
        assert len(data) == 0


# ---------------------------------------------------------------------------
# list_memories
# ---------------------------------------------------------------------------

class TestListMemories:
    def test_list_empty(self, vault: DreamStore):
        result = vault.list_memories()
        assert result == []

    def test_list_all(self, vault: DreamStore):
        vault.add_memory("user", "User memory")
        vault.add_memory("project", "Project memory")

        result = vault.list_memories()
        assert len(result) == 2

    def test_list_filtered_by_type(self, vault: DreamStore):
        vault.add_memory("user", "User memory")
        vault.add_memory("project", "Project memory")
        vault.add_memory("feedback", "Feedback memory")

        result = vault.list_memories(memory_type="user")
        assert len(result) == 1
        assert result[0]["type"] == "user"

    def test_list_invalid_type_returns_empty(self, vault: DreamStore):
        result = vault.list_memories(memory_type="bogus")
        assert result == []


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_empty_stats(self, vault: DreamStore):
        stats = vault.stats()
        assert stats["total"] == 0
        for t in ("user", "feedback", "project", "reference"):
            assert stats["counts"][t] == 0

    def test_stats_with_memories(self, vault: DreamStore):
        vault.add_memory("user", "mem1")
        vault.add_memory("user", "mem2")
        vault.add_memory("project", "mem3")

        stats = vault.stats()
        assert stats["total"] == 3
        assert stats["counts"]["user"] == 2
        assert stats["counts"]["project"] == 1
        assert stats["counts"]["feedback"] == 0


# ---------------------------------------------------------------------------
# get_memory_path
# ---------------------------------------------------------------------------

class TestGetMemoryPath:
    def test_resolves_path(self, vault: DreamStore):
        path = vault.get_memory_path("user", "test.md")
        assert path == vault.vault_path / "user" / "test.md"

    def test_adds_md_extension(self, vault: DreamStore):
        path = vault.get_memory_path("project", "test")
        assert path.suffix == ".md"

    def test_invalid_type_raises(self, vault: DreamStore):
        with pytest.raises(ValueError, match="Invalid memory type"):
            vault.get_memory_path("bogus", "test.md")
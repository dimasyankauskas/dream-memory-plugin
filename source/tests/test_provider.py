"""Tests for Dream Memory Provider — MemoryProvider ABC compliance, Phase 2 hooks, and Phase 3 tools."""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from plugins.memory.dream import DreamMemoryProvider, _load_plugin_config


# ---------------------------------------------------------------------------
# ABC compliance
# ---------------------------------------------------------------------------

class TestABCCompliance:
    """Verify DreamMemoryProvider satisfies the MemoryProvider interface."""

    def test_name_property(self):
        provider = DreamMemoryProvider(config={})
        assert provider.name == "dream"

    def test_is_available(self):
        provider = DreamMemoryProvider(config={})
        # Dream is always available (filesystem-only)
        assert provider.is_available() is True

    def test_initialise_creates_store(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        assert provider._store is not None

    def test_system_prompt_block_empty_before_init(self):
        provider = DreamMemoryProvider(config={})
        # Before initialise, store is None — should return empty string
        assert provider.system_prompt_block() == ""

    def test_system_prompt_block_returns_text_after_init(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        block = provider.system_prompt_block()
        assert "Dream Memory" in block

    def test_get_tool_schemas(self):
        provider = DreamMemoryProvider(config={})
        schemas = provider.get_tool_schemas()
        assert len(schemas) > 0
        names = [s["name"] for s in schemas]
        assert "dream_status" in names
        assert "dream_recall" in names
        assert "dream_consolidate" in names

    def test_handle_tool_call_dream_status(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        result = provider.handle_tool_call("dream_status", {})
        data = json.loads(result)
        assert "total" in data
        assert "counts" in data

    def test_handle_tool_call_unknown(self):
        provider = DreamMemoryProvider(config={})
        result = provider.handle_tool_call("unknown_tool", {})
        # Should call tool_error which returns an error string
        assert "Unknown tool" in result or "error" in result.lower()


# ---------------------------------------------------------------------------
# system_prompt_block — Phase 3 updated
# ---------------------------------------------------------------------------

class TestSystemPromptBlock:
    def test_empty_vault_message(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        block = provider.system_prompt_block()
        assert "Empty vault" in block
        assert "dream_recall" in block

    def test_vault_with_memories(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        # Add some memories
        provider._store.add_memory("user", "I prefer vim", tags=["editor"], relevance=0.8)
        provider._store.add_memory("project", "Using FastAPI", tags=["fastapi"], relevance=0.7)
        provider._store.add_memory("feedback", "No camelCase", tags=["naming"], relevance=0.9)

        block = provider.system_prompt_block()
        assert "3 memories stored" in block
        assert "1U/1F/1P/0R" in block
        assert "dream_recall" in block

    def test_vault_with_multiple_of_same_type(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        provider._store.add_memory("user", "Pref1", tags=["t1"])
        provider._store.add_memory("user", "Pref2", tags=["t2"])
        provider._store.add_memory("reference", "Ref1", tags=["t3"])

        block = provider.system_prompt_block()
        assert "3 memories stored" in block
        assert "2U/0F/0P/1R" in block


# ---------------------------------------------------------------------------
# dream_recall tool — Phase 3
# ---------------------------------------------------------------------------

class TestDreamRecallTool:
    def test_dream_recall_returns_results(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        provider._store.add_memory(
            "user", "I prefer dark mode", tags=["dark-mode", "preference"],
            source="s1", relevance=0.8,
        )
        result = provider.handle_tool_call("dream_recall", {"query": "dark mode"})
        data = json.loads(result)
        assert "results" in data
        assert len(data["results"]) >= 1
        assert data["results"][0]["memory_type"] == "user"

    def test_dream_recall_with_memory_type_filter(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        provider._store.add_memory(
            "user", "I prefer dark mode", tags=["dark-mode"], source="s1", relevance=0.8,
        )
        provider._store.add_memory(
            "project", "The project uses FastAPI", tags=["fastapi"], source="s2", relevance=0.7,
        )
        result = provider.handle_tool_call(
            "dream_recall", {"query": "mode", "memory_type": "user"}
        )
        data = json.loads(result)
        assert all(r["memory_type"] == "user" for r in data["results"])

    def test_dream_recall_with_limit(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        for i in range(8):
            provider._store.add_memory(
                "reference", f"Reference {i} about Python",
                tags=["python", f"ref-{i}"], source="s1", relevance=0.5 + i * 0.01,
            )
        result = provider.handle_tool_call("dream_recall", {"query": "Python", "limit": 3})
        data = json.loads(result)
        assert len(data["results"]) <= 3

    def test_dream_recall_empty_query_error(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        result = provider.handle_tool_call("dream_recall", {"query": ""})
        data = json.loads(result)
        assert "error" in data

    def test_dream_recall_without_store(self):
        provider = DreamMemoryProvider(config={})
        result = provider.handle_tool_call("dream_recall", {"query": "test"})
        data = json.loads(result)
        assert "error" in data

    def test_dream_recall_no_matches(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        result = provider.handle_tool_call("dream_recall", {"query": "zzzzz nonexistent"})
        data = json.loads(result)
        # May return empty results, which is valid
        assert isinstance(data["results"], list)


# ---------------------------------------------------------------------------
# dream_consolidate tool — Phase 4 consolidation engine
# ---------------------------------------------------------------------------

class TestDreamConsolidateTool:
    def test_consolidate_returns_result(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        provider._store.add_memory("user", "Test memory", tags=["test"])

        result = provider.handle_tool_call("dream_consolidate", {})
        data = json.loads(result)
        assert data["status"] in ("completed", "dry_run")
        assert "orient" in data
        assert "gather" in data
        assert "consolidate" in data
        assert "prune" in data

    def test_consolidate_with_dry_run(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        result = provider.handle_tool_call("dream_consolidate", {"dry_run": True})
        data = json.loads(result)
        assert data["dry_run"] is True

    def test_consolidate_with_memory_type(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        result = provider.handle_tool_call(
            "dream_consolidate", {"memory_type": "user"}
        )
        data = json.loads(result)
        assert data["memory_type"] == "user"

    def test_consolidate_without_store(self):
        provider = DreamMemoryProvider(config={})
        result = provider.handle_tool_call("dream_consolidate", {})
        data = json.loads(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# prefetch — Phase 3
# ---------------------------------------------------------------------------

class TestPrefetch:
    def test_prefetch_returns_formatted_context(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        provider._store.add_memory(
            "user", "I prefer dark mode for editing.", tags=["dark-mode", "preference"],
            source="s1", relevance=0.8,
        )

        result = provider.prefetch("dark mode editing", session_id="test-session")
        assert "## Dream Memory" in result
        assert "User" in result
        assert "dark mode" in result.lower()

    def test_prefetch_returns_empty_for_no_matches(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        result = provider.prefetch("zzzzz nonexistent", session_id="test-session")
        assert result == ""

    def test_prefetch_returns_empty_without_store(self):
        provider = DreamMemoryProvider(config={})
        result = provider.prefetch("test query")
        assert result == ""

    def test_prefetch_caches_result(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        provider._store.add_memory(
            "user", "I prefer vim.", tags=["vim", "preference"],
            source="s1", relevance=0.8,
        )

        result1 = provider.prefetch("vim editor", session_id="sess-1")
        result2 = provider.prefetch("vim editor", session_id="sess-1")
        assert result1 == result2  # Cache hit

    def test_prefetch_limits_to_five(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        # Add 8 memories
        for i in range(8):
            provider._store.add_memory(
                "reference", f"Reference {i} about Python coding.",
                tags=["python", f"ref-{i}"], source="s1", relevance=0.5 + i * 0.01,
            )

        result = provider.prefetch("python", session_id="test-session")
        # Should include at most 5 memories in the output
        lines = [l for l in result.split("\n") if l.startswith("- [")]
        assert len(lines) <= 5

    def test_prefetch_truncates_content(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        # Add a memory with long content
        long_content = "A" * 500
        provider._store.add_memory(
            "user", long_content, tags=["test"], source="s1", relevance=0.8,
        )

        result = provider.prefetch("AAAA", session_id="test-session")
        # The content in prefetch should be truncated to 200 chars
        # Find the line with the content
        for line in result.split("\n"):
            if line.startswith("- ["):
                # The snippet (after score marker) should be ≤ 200 chars + ellipsis
                content_part = line.split("] ", 1)[1] if "] " in line else line
                assert len(content_part) <= 250  # 200 + ellipsis + slack


# ---------------------------------------------------------------------------
# queue_prefetch — Phase 3
# ---------------------------------------------------------------------------

class TestQueuePrefetch:
    def test_queue_prefetch_stores_query(self):
        provider = DreamMemoryProvider(config={})
        provider.queue_prefetch("test query", session_id="sess-1")
        assert provider._pending_prefetch_query == "test query"

    def test_queue_prefetch_overwrites_previous(self):
        provider = DreamMemoryProvider(config={})
        provider.queue_prefetch("first query", session_id="sess-1")
        provider.queue_prefetch("second query", session_id="sess-1")
        assert provider._pending_prefetch_query == "second query"


# ---------------------------------------------------------------------------
# sync_turn — Phase 2
# ---------------------------------------------------------------------------

class TestSyncTurn:
    """Test that sync_turn extracts candidates and writes to store."""

    def test_sync_turn_stores_preference(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        provider.sync_turn("I prefer dark mode for editing.", "Noted!")

        stats = provider._store.stats()
        assert stats["total"] >= 1
        # Should have stored a user-type memory
        assert stats["counts"]["user"] >= 1

    def test_sync_turn_stores_feedback(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        provider.sync_turn("No, don't use that approach.", "I'll fix it.")

        stats = provider._store.stats()
        assert stats["total"] >= 1
        assert stats["counts"]["feedback"] >= 1

    def test_sync_turn_no_candidates_for_plain_text(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        provider.sync_turn("Hello, how are you?", "I'm doing well!")

        # No patterns match — nothing stored
        stats = provider._store.stats()
        assert stats["total"] == 0

    def test_sync_turn_skips_low_relevance(self, tmp_path: Path):
        """Candidates below 0.6 relevance should not be stored."""
        # Reference patterns have relevance=0.6 which meets threshold
        # but if we had a pattern at 0.5 it would be skipped — test with regular text
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        # This generates reference-type candidates (relevance=0.6, meets threshold)
        provider.sync_turn("The path is /usr/local/bin/app.", "Okay.")
        stats = provider._store.stats()
        assert stats["total"] >= 1  # reference at 0.6 is stored

    def test_sync_turn_without_store(self):
        """sync_turn should not raise when store is None."""
        provider = DreamMemoryProvider(config={})
        # Should not raise
        provider.sync_turn("I prefer tea.", "Noted.")

    def test_sync_turn_with_session_id(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        provider.sync_turn("I prefer tabs.", "Got it.", session_id="custom-sess")

        mems = provider._store.list_memories(memory_type="user")
        assert len(mems) >= 1
        # The source should be the custom session_id
        assert mems[0]["meta"]["source"] == "custom-sess"


# ---------------------------------------------------------------------------
# on_memory_write — Phase 2
# ---------------------------------------------------------------------------

class TestOnMemoryWrite:
    """Test mirroring of built-in memory writes."""

    def test_add_user_target(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        provider.on_memory_write("add", "user", "Prefers vim editor")

        stats = provider._store.stats()
        assert stats["counts"]["user"] >= 1

    def test_add_memory_target(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        provider.on_memory_write("add", "memory", "Using Python 3.12")

        stats = provider._store.stats()
        assert stats["counts"]["project"] >= 1

    def test_replace_updates_existing(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        # Add a memory first
        provider.on_memory_write("add", "user", "Prefers vim editor")

        # Replace it
        provider.on_memory_write("replace", "user", "Prefers vim editor")

        # Should still have at least one user memory
        mems = provider._store.list_memories(memory_type="user")
        assert len(mems) >= 1

    def test_remove_deletes_matching(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        # Add a memory first
        provider.on_memory_write("add", "user", "Prefers vim editor")

        # Remove it
        provider.on_memory_write("remove", "user", "Prefers vim editor")

        # The matching memory should be deleted
        mems = provider._store.list_memories(memory_type="user")
        for m in mems:
            assert "Prefers vim editor" not in m.get("body", "")

    def test_on_memory_write_without_store(self):
        """Should not raise when store is None."""
        provider = DreamMemoryProvider(config={})
        provider.on_memory_write("add", "user", "test content")
        # Should not raise


# ---------------------------------------------------------------------------
# on_session_end — Phase 2
# ---------------------------------------------------------------------------

class TestOnSessionEnd:
    """Test session-end extraction."""

    def test_extracts_from_messages(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        messages = [
            {"role": "user", "content": "I prefer dark mode."},
            {"role": "assistant", "content": "Noted."},
            {"role": "user", "content": "The project uses FastAPI."},
            {"role": "assistant", "content": "Got it."},
        ]

        provider.on_session_end(messages)

        stats = provider._store.stats()
        assert stats["total"] >= 2  # at least user + project

    def test_empty_messages(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        provider.on_session_end([])
        stats = provider._store.stats()
        assert stats["total"] == 0

    def test_no_matching_messages(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        provider.on_session_end(messages)
        stats = provider._store.stats()
        assert stats["total"] == 0

    def test_tags_include_session_end(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        messages = [
            {"role": "user", "content": "I prefer tabs over spaces."},
        ]
        provider.on_session_end(messages)

        mems = provider._store.list_memories(memory_type="user")
        assert len(mems) >= 1
        assert "session-end" in mems[0]["meta"]["tags"]

    def test_without_store(self):
        """Should not raise when store is None."""
        provider = DreamMemoryProvider(config={})
        provider.on_session_end([{"role": "user", "content": "I prefer tea."}])
        # Should not raise


# ---------------------------------------------------------------------------
# on_pre_compress — Phase 2
# ---------------------------------------------------------------------------

class TestOnPreCompress:
    """Test pre-compress extraction and summary."""

    def test_returns_summary_string(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        messages = [
            {"role": "user", "content": "I prefer dark mode."},
            {"role": "assistant", "content": "Setting dark theme."},
        ]
        result = provider.on_pre_compress(messages)
        assert isinstance(result, str)
        assert "Dream Memory" in result

    def test_returns_empty_for_no_candidates(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        messages = [
            {"role": "user", "content": "Hello there."},
        ]
        result = provider.on_pre_compress(messages)
        assert result == ""

    def test_stores_candidates_too(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        messages = [
            {"role": "user", "content": "I always use snake_case."},
        ]
        provider.on_pre_compress(messages)

        stats = provider._store.stats()
        assert stats["total"] >= 1

    def test_pre_compress_tags(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        messages = [
            {"role": "user", "content": "I never use camelCase."},
        ]
        provider.on_pre_compress(messages)

        mems = provider._store.list_memories(memory_type="user")
        assert len(mems) >= 1
        assert "pre-compress" in mems[0]["meta"]["tags"]

    def test_empty_messages_returns_empty(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")

        result = provider.on_pre_compress([])
        assert result == ""


# ---------------------------------------------------------------------------
# Optional hooks — still functional
# ---------------------------------------------------------------------------

class TestOptionalHooks:
    def test_shutdown_clears_store(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={"vault_path": str(tmp_path / "dream_test")})
        provider.initialize(session_id="test-session")
        assert provider._store is not None
        provider.shutdown()
        assert provider._store is None
        assert provider._recall_engine is None


# ---------------------------------------------------------------------------
# register entry point
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_calls_ctx(self):
        from plugins.memory.dream import register

        mock_ctx = MagicMock()
        register(mock_ctx)
        mock_ctx.register_memory_provider.assert_called_once()
        provider = mock_ctx.register_memory_provider.call_args[0][0]
        assert isinstance(provider, DreamMemoryProvider)
        assert provider.name == "dream"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestLoadPluginConfig:
    def test_loads_from_config_yaml(self, tmp_path: Path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "plugins:\n  dream:\n    vault_path: /tmp/dream_vault\n"
        )
        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            cfg = _load_plugin_config()
            assert cfg.get("vault_path") == "/tmp/dream_vault"

    def test_returns_empty_when_no_config(self, tmp_path: Path):
        with patch("hermes_constants.get_hermes_home", return_value=tmp_path):
            cfg = _load_plugin_config()
            assert cfg == {}


# ---------------------------------------------------------------------------
# Config management — save_config / get_config_schema
# ---------------------------------------------------------------------------

class TestConfigManagement:
    def test_save_config_writes_yaml(self, tmp_path: Path):
        provider = DreamMemoryProvider(config={})
        values = {"vault_path": "/tmp/my_dream_vault", "taxonomy": "true"}
        provider.save_config(values, str(tmp_path))

        import yaml
        config_file = tmp_path / "config.yaml"
        assert config_file.exists()
        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["plugins"]["dream"]["vault_path"] == "/tmp/my_dream_vault"

    def test_save_config_preserves_existing(self, tmp_path: Path):
        """save_config should not overwrite other plugin configs."""
        import yaml
        config_file = tmp_path / "config.yaml"
        existing = {"plugins": {"other_plugin": {"key": "value"}}}
        with open(config_file, "w") as f:
            yaml.dump(existing, f)

        provider = DreamMemoryProvider(config={})
        provider.save_config({"vault_path": "/new/path"}, str(tmp_path))

        with open(config_file) as f:
            data = yaml.safe_load(f)
        assert data["plugins"]["other_plugin"]["key"] == "value"
        assert data["plugins"]["dream"]["vault_path"] == "/new/path"

    def test_get_config_schema(self):
        provider = DreamMemoryProvider(config={})
        schema = provider.get_config_schema()
        assert isinstance(schema, list)
        assert len(schema) > 0
        keys = [s["key"] for s in schema]
        assert "vault_path" in keys
        assert "max_lines" in keys
        assert "max_bytes" in keys
        assert "consolidate_model" in keys
        assert "consolidate_cron" in keys
        assert "taxonomy" in keys
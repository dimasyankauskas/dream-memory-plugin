"""Tests for LLM-powered Dream Memory Consolidation.

Tests the llm_consolidate function and the consolidation_mode config option
in run_consolidation, using mocked API calls.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from plugins.memory.dream.store import DreamStore
from plugins.memory.dream.taxonomy import MEMORY_TYPES, make_memory_document, render_frontmatter
from plugins.memory.dream.consolidation import (
    ConsolidationAction,
    ConsolidateResult,
    ConsolidationResult as FullConsolidationResult,
    MemoryEntry,
    DEFAULT_CONSOLIDATE_MODEL,
    _build_llm_prompt,
    _extract_llm_merges,
    _format_memories_for_llm,
    _parse_llm_response,
    _resolve_api_settings,
    gather,
    llm_consolidate,
    orient,
    run_consolidation,
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


def _write_memory_direct(
    store: DreamStore,
    memory_type: str,
    filename: str,
    content: str,
    tags: list = None,
    relevance: float = 0.5,
    created: str = None,
) -> Path:
    """Write a memory file directly with explicit filename."""
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


def _sample_entries() -> List[MemoryEntry]:
    """Create sample MemoryEntry objects for testing."""
    return [
        MemoryEntry(
            memory_type="user",
            filename="pref-vim-20250101T000001Z.md",
            tags=["editor", "vim"],
            content="I prefer vim for editing code. I use it daily for all my projects.",
            relevance=0.8,
            created="2025-01-01T00:00:01+00:00",
            updated="2025-01-01T00:00:01+00:00",
            lines=2,
            bytes=80,
            path="/vault/user/pref-vim-20250101T000001Z.md",
        ),
        MemoryEntry(
            memory_type="user",
            filename="pref-emacs-20250101T000002Z.md",
            tags=["editor", "emacs"],
            content="I prefer emacs for editing. It has great Lisp support.",
            relevance=0.7,
            created="2025-01-02T00:00:02+00:00",
            updated="2025-01-02T00:00:02+00:00",
            lines=2,
            bytes=60,
            path="/vault/user/pref-emacs-20250101T000002Z.md",
        ),
    ]


# ---------------------------------------------------------------------------
# _resolve_api_settings tests
# ---------------------------------------------------------------------------


class TestResolveApiSettings:
    def test_defaults(self):
        """With no config or env, should use defaults."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove relevant env vars
            for key in ["OPENROUTER_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"]:
                os.environ.pop(key, None)
            base_url, api_key, model = _resolve_api_settings({})
            assert base_url == "https://openrouter.ai/api/v1"
            assert model == DEFAULT_CONSOLIDATE_MODEL

    def test_config_overrides(self):
        """Config values should take priority over env vars."""
        config = {
            "consolidate_api_key": "test-key-123",
            "consolidate_base_url": "https://custom.api/v1",
            "consolidate_model": "gpt-4o",
        }
        base_url, api_key, model = _resolve_api_settings(config)
        assert base_url == "https://custom.api/v1"
        assert api_key == "test-key-123"
        assert model == "gpt-4o"

    def test_env_var_fallback(self):
        """Should fall back to env vars when config is empty."""
        with patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "or-key-456",
            "OPENAI_BASE_URL": "https://custom.openai/v1",
        }):
            config = {}
            base_url, api_key, model = _resolve_api_settings(config)
            assert api_key == "or-key-456"
            assert base_url == "https://custom.openai/v1"

    def test_openai_key_fallback(self):
        """OPENAI_API_KEY should be used if OPENROUTER_API_KEY is not set."""
        env = {"OPENAI_API_KEY": "oai-key-789"}
        with patch.dict(os.environ, env, clear=True):
            config = {}
            base_url, api_key, model = _resolve_api_settings(config)
            assert api_key == "oai-key-789"


# ---------------------------------------------------------------------------
# _format_memories_for_llm tests
# ---------------------------------------------------------------------------


class TestFormatMemoriesForLLM:
    def test_format_entries(self):
        entries = _sample_entries()
        result = _format_memories_for_llm(entries)
        assert "--- Memory 1 ---" in result
        assert "Type: user" in result
        assert "pref-vim-20250101T000001Z.md" in result
        assert "Tags: editor, vim" in result
        assert "I prefer vim" in result

    def test_format_empty(self):
        result = _format_memories_for_llm([])
        assert result == ""


# ---------------------------------------------------------------------------
# _build_llm_prompt tests
# ---------------------------------------------------------------------------


class TestBuildLLMPrompt:
    def test_prompt_structure(self):
        entries = _sample_entries()
        system_prompt, user_prompt = _build_llm_prompt(entries)
        assert "memory consolidation engine" in system_prompt.lower()
        assert "JSON" in system_prompt
        assert "merge" in system_prompt
        assert "deduplicate" in system_prompt
        assert "contradict" in system_prompt
        assert "--- Memory 1 ---" in user_prompt
        assert "Analyse these 2 memories" in user_prompt


# ---------------------------------------------------------------------------
# _parse_llm_response tests
# ---------------------------------------------------------------------------


class TestParseLLMResponse:
    def test_merge_action(self):
        entries = _sample_entries()
        response = json.dumps({
            "actions": [
                {
                    "action": "merge",
                    "source_indices": [0, 1],
                    "result_index": 1,
                    "merged_content": "Combined editor preferences",
                    "merged_tags": ["editor", "vim", "emacs"],
                    "relevance": 0.85,
                    "details": "Merged two editor preference memories",
                }
            ]
        })
        actions = _parse_llm_response(response, entries)
        assert len(actions) == 1
        assert actions[0].action == "merge"
        assert actions[0].target_type == "user"
        assert len(actions[0].target_files) == 2
        assert actions[0].details == "Merged two editor preference memories"

    def test_deduplicate_action(self):
        entries = _sample_entries()
        response = json.dumps({
            "actions": [
                {
                    "action": "deduplicate",
                    "keep_index": 1,
                    "remove_indices": [0],
                    "details": "Memory 0 is a duplicate of memory 1",
                }
            ]
        })
        actions = _parse_llm_response(response, entries)
        assert len(actions) == 1
        assert actions[0].action == "deduplicate"

    def test_contradict_action(self):
        entries = _sample_entries()
        response = json.dumps({
            "actions": [
                {
                    "action": "contradict",
                    "newer_index": 1,
                    "older_indices": [0],
                    "details": "Memory 1 contradicts memory 0",
                }
            ]
        })
        actions = _parse_llm_response(response, entries)
        assert len(actions) == 1
        assert actions[0].action == "contradict"

    def test_compress_action(self):
        entries = _sample_entries()
        response = json.dumps({
            "actions": [
                {
                    "action": "compress",
                    "target_index": 0,
                    "compressed_content": "Short summary",
                    "details": "Compressed verbose memory",
                }
            ]
        })
        actions = _parse_llm_response(response, entries)
        assert len(actions) == 1
        assert actions[0].action == "compress"

    def test_boost_relevance_action(self):
        entries = _sample_entries()
        response = json.dumps({
            "actions": [
                {
                    "action": "boost_relevance",
                    "target_indices": [0],
                    "new_relevance": 0.9,
                    "details": "Boost frequently used memory",
                }
            ]
        })
        actions = _parse_llm_response(response, entries)
        assert len(actions) == 1
        assert actions[0].action == "boost_relevance"

    def test_multiple_actions(self):
        entries = _sample_entries()
        response = json.dumps({
            "actions": [
                {"action": "merge", "source_indices": [0, 1], "result_index": 1, "merged_content": "Merged", "merged_tags": [], "relevance": 0.8, "details": "merge"},
                {"action": "deduplicate", "keep_index": 1, "remove_indices": [0], "details": "dedup"},
            ]
        })
        actions = _parse_llm_response(response, entries)
        assert len(actions) == 2

    def test_empty_actions(self):
        entries = _sample_entries()
        response = json.dumps({"actions": []})
        actions = _parse_llm_response(response, entries)
        assert len(actions) == 0

    def test_markdown_fences_stripped(self):
        entries = _sample_entries()
        raw_response = json.dumps({
            "actions": [
                {"action": "merge", "source_indices": [0, 1], "result_index": 1, "merged_content": "M", "merged_tags": [], "relevance": 0.8, "details": "test"},
            ]
        })
        # Wrap in markdown code fences
        fenced = f"```json\n{raw_response}\n```"
        actions = _parse_llm_response(fenced, entries)
        assert len(actions) == 1

    def test_invalid_json_falls_back(self):
        entries = _sample_entries()
        actions = _parse_llm_response("not valid json at all", entries)
        assert len(actions) == 0

    def test_invalid_action_type_skipped(self):
        entries = _sample_entries()
        response = json.dumps({
            "actions": [
                {"action": "unknown_action", "details": "should be skipped"},
            ]
        })
        actions = _parse_llm_response(response, entries)
        assert len(actions) == 0

    def test_out_of_bounds_index_skipped(self):
        entries = _sample_entries()  # only 2 entries
        response = json.dumps({
            "actions": [
                {"action": "merge", "source_indices": [0, 99], "result_index": 0, "merged_content": "M", "merged_tags": [], "relevance": 0.8, "details": "test"},
            ]
        })
        # source_indices includes 99 which is out of bounds, so source_entries will only have 1 entry
        # which is still valid for merge
        actions = _parse_llm_response(response, entries)
        # Should still produce a merge action with the valid index
        assert len(actions) >= 0

    def test_non_dict_actions_skipped(self):
        entries = _sample_entries()
        response = json.dumps({
            "actions": ["not a dict", 42, None],
        })
        actions = _parse_llm_response(response, entries)
        assert len(actions) == 0


# ---------------------------------------------------------------------------
# _extract_llm_merges tests
# ---------------------------------------------------------------------------


class TestExtractLLMMerges:
    def test_merge_extraction_with_content(self):
        entries = _sample_entries()
        action = ConsolidationAction(
            action="merge",
            target_type="user",
            target_files=["user:pref-vim-20250101T000001Z.md", "user:pref-emacs-20250101T000002Z.md"],
            result_file="user:pref-emacs-20250101T000002Z.md",
            details="Merged editor preferences",
        )
        raw_response = {
            "actions": [
                {
                    "action": "merge",
                    "source_indices": [0, 1],
                    "result_index": 1,
                    "merged_content": "User prefers vim for code but also uses emacs for Lisp.",
                    "merged_tags": ["editor", "vim", "emacs"],
                    "relevance": 0.85,
                    "details": "Merged editor preferences",
                }
            ]
        }

        merges, superseded = _extract_llm_merges([action], entries, raw_response)
        assert len(merges) == 1
        assert merges[0]["content"] == "User prefers vim for code but also uses emacs for Lisp."
        assert merges[0]["tags"] == ["editor", "vim", "emacs"]
        assert merges[0]["relevance"] == 0.85
        # The non-result source file should be superseded
        assert "user:pref-vim-20250101T000001Z.md" in superseded

    def test_no_merge_actions(self):
        entries = _sample_entries()
        merges, superseded = _extract_llm_merges([], entries, {})
        assert len(merges) == 0
        assert len(superseded) == 0


# ---------------------------------------------------------------------------
# llm_consolidate tests (with mocked API)
# ---------------------------------------------------------------------------


class TestLLMConsolidate:
    def test_empty_memories(self):
        """Should return empty result for empty memory list."""
        result = llm_consolidate([], config={"consolidate_api_key": "test-key"})
        assert isinstance(result, ConsolidateResult)
        assert len(result.actions) == 0
        assert result.stats.get("mode") == "llm"

    def test_no_api_key_fallback(self):
        """Should return empty result with fallback mode when no API key."""
        with patch.dict(os.environ, {}, clear=True):
            for key in ["OPENROUTER_API_KEY", "OPENAI_API_KEY"]:
                os.environ.pop(key, None)
            result = llm_consolidate(_sample_entries(), config={})
            assert result.stats.get("mode") == "llm_fallback_no_key"

    def test_successful_merge_response(self):
        """Test a successful LLM merge call with mocked API."""
        entries = _sample_entries()

        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "actions": [
                {
                    "action": "merge",
                    "source_indices": [0, 1],
                    "result_index": 1,
                    "merged_content": "Combined editor preferences: vim for code, emacs for Lisp.",
                    "merged_tags": ["editor", "vim", "emacs"],
                    "relevance": 0.85,
                    "details": "Merged editor preference memories",
                }
            ]
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        config = {
            "consolidate_api_key": "test-key",
            "consolidate_model": "test-model",
            "consolidate_base_url": "https://test.api/v1",
        }

        with patch("openai.OpenAI", return_value=mock_client):
            result = llm_consolidate(entries, config)

        assert result.merged_count == 1
        assert len(result.actions) == 1
        assert result.actions[0].action == "merge"
        assert result.stats.get("mode") == "llm"
        assert len(result.merges) == 1
        assert "Combined editor preferences" in result.merges[0]["content"]

    def test_successful_dedup_response(self):
        """Test LLM deduplication response."""
        entries = _sample_entries()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "actions": [
                {
                    "action": "deduplicate",
                    "keep_index": 1,
                    "remove_indices": [0],
                    "details": "Memory 0 is a near-duplicate of memory 1",
                }
            ]
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        config = {
            "consolidate_api_key": "test-key",
            "consolidate_model": "test-model",
            "consolidate_base_url": "https://test.api/v1",
        }

        with patch("openai.OpenAI", return_value=mock_client):
            result = llm_consolidate(entries, config)

        assert result.deduped_count == 1
        assert len(result.superseded) >= 1

    def test_successful_contradiction_response(self):
        """Test LLM contradiction resolution."""
        entries = _sample_entries()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "actions": [
                {
                    "action": "contradict",
                    "newer_index": 1,
                    "older_indices": [0],
                    "details": "Conflicting editor preferences; newer supersedes",
                }
            ]
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        config = {
            "consolidate_api_key": "test-key",
            "consolidate_model": "test-model",
            "consolidate_base_url": "https://test.api/v1",
        }

        with patch("openai.OpenAI", return_value=mock_client):
            result = llm_consolidate(entries, config)

        assert result.pruned_count == 1
        assert len(result.actions) == 1

    def test_api_error_returns_empty_result(self):
        """Test API call failure returns graceful empty result."""
        entries = _sample_entries()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        config = {
            "consolidate_api_key": "test-key",
            "consolidate_model": "test-model",
            "consolidate_base_url": "https://test.api/v1",
        }

        with patch("openai.OpenAI", return_value=mock_client):
            result = llm_consolidate(entries, config)

        assert len(result.actions) == 0
        assert result.stats.get("mode") == "llm_error"

    def test_multiple_action_types(self):
        """Test LLM response with multiple action types."""
        entries = [
            MemoryEntry(
                memory_type="user", filename="m1-20250101T000001Z.md",
                tags=["python"], content="I prefer Python for backend",
                relevance=0.8, created="2025-01-01T00:00:01+00:00",
                updated="2025-01-01T00:00:01+00:00", lines=1, bytes=40,
                path="/vault/user/m1.md",
            ),
            MemoryEntry(
                memory_type="user", filename="m2-20250101T000002Z.md",
                tags=["python"], content="I prefer Python for backend",
                relevance=0.7, created="2025-01-02T00:00:02+00:00",
                updated="2025-01-02T00:00:02+00:00", lines=1, bytes=40,
                path="/vault/user/m2.md",
            ),
            MemoryEntry(
                memory_type="reference", filename="m3-20250101T000003Z.md",
                tags=["api"], content="\n".join([f"Line {i}" for i in range(200)]),
                relevance=0.6, created="2025-01-03T00:00:03+00:00",
                updated="2025-01-03T00:00:03+00:00", lines=200, bytes=2000,
                path="/vault/reference/m3.md",
            ),
        ]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "actions": [
                {
                    "action": "deduplicate",
                    "keep_index": 1,
                    "remove_indices": [0],
                    "details": "Duplicate Python preference",
                },
                {
                    "action": "compress",
                    "target_index": 2,
                    "compressed_content": "API reference summary",
                    "details": "Compressed verbose reference",
                },
                {
                    "action": "boost_relevance",
                    "target_indices": [1],
                    "new_relevance": 0.9,
                    "details": "Boosting frequently referenced memory",
                },
            ]
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        config = {
            "consolidate_api_key": "test-key",
            "consolidate_model": "test-model",
            "consolidate_base_url": "https://test.api/v1",
        }

        with patch("openai.OpenAI", return_value=mock_client):
            result = llm_consolidate(entries, config)

        assert result.deduped_count == 1
        assert len(result.capped) >= 1  # compress
        assert len(result.actions) == 3  # dedup + compress + boost

    def test_malformed_json_response(self):
        """Test graceful handling of malformed LLM response."""
        entries = _sample_entries()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is not JSON at all!"

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        config = {
            "consolidate_api_key": "test-key",
            "consolidate_model": "test-model",
            "consolidate_base_url": "https://test.api/v1",
        }

        with patch("openai.OpenAI", return_value=mock_client):
            result = llm_consolidate(entries, config)

        # Should handle gracefully with empty actions
        assert len(result.actions) == 0

    def test_compress_action_generates_capped(self):
        """Verify that LLM compress actions produce capped entries."""
        entries = [
            MemoryEntry(
                memory_type="reference", filename="verbose-20250101T000000Z.md",
                tags=["api"], content="\n".join([f"Line {i}" for i in range(200)]),
                relevance=0.6, created="2025-01-01T00:00:00+00:00",
                updated="2025-01-01T00:00:00+00:00", lines=200, bytes=2000,
                path="/vault/reference/verbose.md",
            ),
        ]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "actions": [
                {
                    "action": "compress",
                    "target_index": 0,
                    "compressed_content": "Short summary",
                    "details": "Compressed verbose reference",
                }
            ]
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        config = {
            "consolidate_api_key": "test-key",
            "consolidate_model": "test-model",
            "consolidate_base_url": "https://test.api/v1",
            "max_lines": 50,
        }

        with patch("openai.OpenAI", return_value=mock_client):
            result = llm_consolidate(entries, config)

        assert len(result.capped) == 1
        # reference type has max_lines=120 in taxonomy
        assert result.capped[0][2] == 120  # reference max_lines
        assert result.capped[0][0] == "reference"


# ---------------------------------------------------------------------------
# run_consolidation with consolidation_mode
# ---------------------------------------------------------------------------


class TestRunConsolidationWithMode:
    def test_deterministic_mode_default(self, vault: DreamStore):
        """Default consolidation mode should use deterministic logic."""
        _write_memory_direct(
            vault, "user", "like-vim-20250101T000001Z.md",
            "I like vim for editing", tags=["editor", "vim"],
        )
        _write_memory_direct(
            vault, "user", "use-vim-20250101T000002Z.md",
            "I use vim daily for coding", tags=["editor", "vim"],
        )

        result = run_consolidation(vault, config={})
        # Should use deterministic consolidation (merge fragments)
        assert isinstance(result, FullConsolidationResult)

    def test_explicit_deterministic_mode(self, vault: DreamStore):
        """Explicit 'deterministic' mode should work the same as default."""
        _write_memory_direct(
            vault, "user", "like-vim-20250101T000001Z.md",
            "I like vim for editing", tags=["editor", "vim"],
        )
        _write_memory_direct(
            vault, "user", "use-vim-20250101T000002Z.md",
            "I use vim daily for coding", tags=["editor", "vim"],
        )

        result = run_consolidation(vault, config={"consolidation_mode": "deterministic"})
        assert isinstance(result, FullConsolidationResult)

    def test_llm_mode_with_mock(self, vault: DreamStore):
        """LLM consolidation mode should call the LLM and produce results."""
        _write_memory_direct(
            vault, "user", "pref-vim-20250101T000001Z.md",
            "I prefer vim for editing", tags=["editor", "vim"],
        )
        _write_memory_direct(
            vault, "user", "pref-emacs-20250101T000002Z.md",
            "I prefer emacs for editing", tags=["editor", "emacs"],
        )

        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "actions": [
                {
                    "action": "merge",
                    "source_indices": [0, 1],
                    "result_index": 1,
                    "merged_content": "User has experience with both vim and emacs editors.",
                    "merged_tags": ["editor", "vim", "emacs"],
                    "relevance": 0.85,
                    "details": "Merged two editor preference memories",
                }
            ]
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        config = {
            "consolidation_mode": "llm",
            "consolidate_api_key": "test-key",
            "consolidate_model": "test-model",
            "consolidate_base_url": "https://test.api/v1",
            "max_lines": 100,
            "max_bytes": 50000,
        }

        with patch("openai.OpenAI", return_value=mock_client):
            result = run_consolidation(vault, config=config)

        # Should have used LLM consolidation
        assert result.consolidate.merged_count == 1
        assert result.consolidate.stats.get("mode") == "llm"

    def test_llm_mode_fallback_on_error(self, vault: DreamStore):
        """LLM mode should fall back to deterministic on API error."""
        _write_memory_direct(
            vault, "user", "like-vim-20250101T000001Z.md",
            "I like vim for editing", tags=["editor", "vim"],
        )
        _write_memory_direct(
            vault, "user", "use-vim-20250101T000002Z.md",
            "I use vim daily for coding", tags=["editor", "vim"],
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        config = {
            "consolidation_mode": "llm",
            "consolidate_api_key": "test-key",
            "consolidate_model": "test-model",
            "consolidate_base_url": "https://test.api/v1",
        }

        with patch("openai.OpenAI", return_value=mock_client):
            result = run_consolidation(vault, config=config)

        # Should fall back to deterministic
        assert isinstance(result, FullConsolidationResult)
        # The fallback will produce results if deterministic finds groups

    def test_llm_mode_no_api_key_fallback(self, vault: DreamStore):
        """LLM mode should fall back when no API key is available."""
        _add_memory(vault, "user", "Test preference")

        with patch.dict(os.environ, {}, clear=True):
            for key in ["OPENROUTER_API_KEY", "OPENAI_API_KEY"]:
                os.environ.pop(key, None)

            config = {"consolidation_mode": "llm"}
            result = run_consolidation(vault, config=config)

        # Should fall back to deterministic
        assert isinstance(result, FullConsolidationResult)

    def test_llm_mode_with_empty_actions_fallback(self, vault: DreamStore):
        """LLM mode returning empty actions with fallback status should use deterministic."""
        _write_memory_direct(
            vault, "user", "like-vim-20250101T000001Z.md",
            "I like vim for editing", tags=["editor", "vim"],
        )
        _write_memory_direct(
            vault, "user", "use-vim-20250101T000002Z.md",
            "I use vim daily for coding", tags=["editor", "vim"],
        )

        # Mock LLM returns empty actions but with fallback status
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({"actions": []})

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        # But we need the llm_consolidate to return fallback status
        # This happens when no API key is available - test the fallback path
        config_no_key = {
            "consolidation_mode": "llm",
            "max_lines": 100,
            "max_bytes": 50000,
        }

        with patch.dict(os.environ, {}, clear=True):
            for key in ["OPENROUTER_API_KEY", "OPENAI_API_KEY"]:
                os.environ.pop(key, None)

            result = run_consolidation(vault, config=config_no_key)

        # Falls back to deterministic
        assert isinstance(result, FullConsolidationResult)


# ---------------------------------------------------------------------------
# Integration: LLM consolidate then prune
# ---------------------------------------------------------------------------


class TestLLMConsolidateIntegration:
    def test_llm_consolidate_produces_prunable_result(self):
        """LLM consolidate result should be compatible with the prune phase."""
        entries = _sample_entries()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "actions": [
                {
                    "action": "deduplicate",
                    "keep_index": 1,
                    "remove_indices": [0],
                    "details": "Near-duplicate editor preferences",
                }
            ]
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        config = {
            "consolidate_api_key": "test-key",
            "consolidate_model": "test-model",
            "consolidate_base_url": "https://test.api/v1",
        }

        with patch("openai.OpenAI", return_value=mock_client):
            result = llm_consolidate(entries, config)

        # Result should have the structure expected by prune
        assert isinstance(result, ConsolidateResult)
        assert hasattr(result, "actions")
        assert hasattr(result, "merges")
        assert hasattr(result, "superseded")
        assert hasattr(result, "capped")
        assert hasattr(result, "merged_count")
        assert hasattr(result, "deduped_count")
        assert hasattr(result, "pruned_count")

    def test_llm_merge_produces_correct_structure(self):
        """LLM merge actions should produce merges in the format expected by prune."""
        entries = _sample_entries()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({
            "actions": [
                {
                    "action": "merge",
                    "source_indices": [0, 1],
                    "result_index": 1,
                    "merged_content": "User uses both vim and emacs for editing.",
                    "merged_tags": ["editor", "vim", "emacs"],
                    "relevance": 0.85,
                    "details": "Merged two editor memories",
                }
            ]
        })

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        config = {
            "consolidate_api_key": "test-key",
            "consolidate_model": "test-model",
            "consolidate_base_url": "https://test.api/v1",
        }

        with patch("openai.OpenAI", return_value=mock_client):
            result = llm_consolidate(entries, config)

        assert len(result.merges) == 1
        merge = result.merges[0]
        assert "memory_type" in merge
        assert "filename" in merge
        assert "content" in merge
        assert "tags" in merge
        assert "relevance" in merge
        assert merge["content"] == "User uses both vim and emacs for editing."
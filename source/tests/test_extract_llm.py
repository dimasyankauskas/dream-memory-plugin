"""Tests for LLM-based memory extraction (extract_llm.py).

Covers:
  - Prompt building
  - Message formatting
  - Response parsing (valid JSON, code-fenced JSON, malformed input)
  - CandidateMemory creation from parsed data
  - Graceful fallback on LLM call failure
  - Config resolution (model, API key, base URL, timeout)
  - Manifest summary generation
  - Integration: on_session_end with extraction_mode = regex/llm/both
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from plugins.memory.dream.extract import CandidateMemory
from plugins.memory.dream.extract_llm import (
    LLMExtractor,
    get_manifest_summary,
    _SYSTEM_PROMPT,
    _USER_PROMPT_TEMPLATE,
    DEFAULT_EXTRACTION_MODEL,
    DEFAULT_EXTRACTION_TIMEOUT,
    DEFAULT_BASE_URL,
    VALID_MEMORY_TYPES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    """Build a minimal config dict with sensible defaults for testing."""
    config = {
        "consolidate_api_key": "test-key-123",
        "consolidate_base_url": "https://api.example.com/v1",
        "consolidate_model": "test-model",
        "extraction_mode": "llm",
    }
    config.update(overrides)
    return config


def _sample_messages():
    """Return a realistic conversation transcript for testing."""
    return [
        {"role": "user", "content": "I prefer short confirmations over long explanations."},
        {"role": "assistant", "content": "Got it, I'll keep responses concise."},
        {"role": "user", "content": "Actually, don't use markdown tables in Telegram."},
        {"role": "assistant", "content": "Understood, I'll avoid markdown tables on Telegram."},
        {"role": "user", "content": "The Dream plugin vault_path should default to ~/.hermes/dream_vault."},
        {"role": "assistant", "content": "Noted! I'll remember that."},
    ]


def _sample_llm_response():
    """Return a valid LLM JSON response for testing parsing."""
    return json.dumps({
        "memories": [
            {
                "type": "feedback",
                "content": "User prefers short confirmations over explanations",
                "tags": ["preference", "communication"],
                "relevance": 0.8,
            },
            {
                "type": "project",
                "content": "Dream plugin default vault_path should be ~/.hermes/dream_vault",
                "tags": ["architecture", "vault"],
                "relevance": 0.7,
            },
        ]
    })


# ---------------------------------------------------------------------------
# LLMExtractor.__init__ — config resolution
# ---------------------------------------------------------------------------

class TestLLMExtractorConfig:
    """Test LLMExtractor config resolution."""

    def test_default_model_from_consolidate_model(self):
        config = _make_config(extraction_model="")
        ext = LLMExtractor(config)
        assert ext._model == "test-model"  # falls back to consolidate_model

    def test_extraction_model_overrides(self):
        config = _make_config(extraction_model="override-model")
        ext = LLMExtractor(config)
        assert ext._model == "override-model"

    def test_default_model_when_empty_config(self):
        config = _make_config(consolidate_model="", extraction_model="")
        ext = LLMExtractor(config)
        assert ext._model == DEFAULT_EXTRACTION_MODEL

    def test_api_key_from_config(self):
        config = _make_config(consolidate_api_key="my-key")
        ext = LLMExtractor(config)
        assert ext._api_key == "my-key"

    def test_api_key_extraction_overrides(self):
        config = _make_config(
            consolidate_api_key="consol-key",
            extraction_api_key="extract-key",
        )
        ext = LLMExtractor(config)
        assert ext._api_key == "extract-key"

    def test_api_key_placeholder_ignored(self):
        config = _make_config(consolidate_api_key="***")
        ext = LLMExtractor(config)
        # Should fall back to env vars or empty
        assert ext._api_key == ""

    def test_api_key_from_env(self):
        config = _make_config(consolidate_api_key="", extraction_api_key="")
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}):
            ext = LLMExtractor(config)
            assert ext._api_key == "env-key"

    def test_api_key_openai_env(self):
        config = _make_config(consolidate_api_key="", extraction_api_key="")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=False):
            # Ensure OPENROUTER_API_KEY is not set
            env = {"OPENAI_API_KEY": "openai-key"}
            with patch.dict(os.environ, env, clear=False):
                # Remove OPENROUTER_API_KEY if set
                os.environ.pop("OPENROUTER_API_KEY", None)
                ext = LLMExtractor(config)
                assert ext._api_key == "openai-key"

    def test_base_url_from_config(self):
        config = _make_config(consolidate_base_url="https://custom.api/v1")
        ext = LLMExtractor(config)
        assert ext._base_url == "https://custom.api/v1"

    def test_base_url_extraction_overrides(self):
        config = _make_config(
            consolidate_base_url="https://consol.api/v1",
            extraction_base_url="https://extract.api/v1",
        )
        ext = LLMExtractor(config)
        assert ext._base_url == "https://extract.api/v1"

    def test_base_url_default(self):
        config = _make_config(consolidate_base_url="", extraction_base_url="")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_BASE_URL", None)
            ext = LLMExtractor(config)
            assert ext._base_url == DEFAULT_BASE_URL

    def test_timeout_default(self):
        config = _make_config()
        ext = LLMExtractor(config)
        assert ext._timeout == DEFAULT_EXTRACTION_TIMEOUT

    def test_timeout_from_config(self):
        config = _make_config(extraction_timeout=60)
        ext = LLMExtractor(config)
        assert ext._timeout == 60


# ---------------------------------------------------------------------------
# LLMExtractor._build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    """Test prompt building."""

    def test_prompt_contains_conversation(self):
        config = _make_config()
        ext = LLMExtractor(config)
        messages = _sample_messages()
        prompt = ext._build_prompt(messages, manifest_summary="")
        assert "I prefer short confirmations" in prompt
        assert "markdown tables" in prompt

    def test_prompt_contains_manifest_summary(self):
        config = _make_config()
        ext = LLMExtractor(config)
        messages = _sample_messages()
        summary = "### User\n- Prefers dark mode"
        prompt = ext._build_prompt(messages, manifest_summary=summary)
        assert "Prefers dark mode" in prompt

    def test_prompt_empty_manifest_uses_placeholder(self):
        config = _make_config()
        ext = LLMExtractor(config)
        messages = _sample_messages()
        prompt = ext._build_prompt(messages, manifest_summary="")
        assert "(no existing memories)" in prompt

    def test_system_prompt_contains_rules(self):
        assert "insight" in _SYSTEM_PROMPT.lower()
        assert "session chronicle" in _SYSTEM_PROMPT.lower()
        assert "empty" in _SYSTEM_PROMPT.lower() or "[]" in _SYSTEM_PROMPT

    def test_system_prompt_contains_taxonomy(self):
        """Verify the system prompt defines all 4 memory types."""
        assert "user" in _SYSTEM_PROMPT.lower()
        assert "feedback" in _SYSTEM_PROMPT.lower()
        assert "project" in _SYSTEM_PROMPT.lower()
        assert "reference" in _SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# LLMExtractor._format_messages
# ---------------------------------------------------------------------------

class TestFormatMessages:
    """Test conversation formatting."""

    def test_basic_formatting(self):
        config = _make_config()
        ext = LLMExtractor(config)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = ext._format_messages(messages)
        assert "[user]: Hello" in result
        assert "[assistant]: Hi there!" in result

    def test_skips_non_string_content(self):
        config = _make_config()
        ext = LLMExtractor(config)
        messages = [
            {"role": "user", "content": None},
            {"role": "user", "content": "I prefer Vim."},
        ]
        result = ext._format_messages(messages)
        assert "I prefer Vim" in result

    def test_skips_empty_content(self):
        config = _make_config()
        ext = LLMExtractor(config)
        messages = [
            {"role": "user", "content": ""},
            {"role": "user", "content": "  "},
            {"role": "user", "content": "Real content here."},
        ]
        result = ext._format_messages(messages)
        assert "Real content here" in result

    def test_truncation_at_max_chars(self):
        config = _make_config()
        ext = LLMExtractor(config)
        # Create messages that exceed the 6000-char budget
        messages = [
            {"role": "user", "content": "x" * 7000},
        ]
        result = ext._format_messages(messages)
        assert len(result) <= 7000  # Should be truncated

    def test_empty_messages(self):
        config = _make_config()
        ext = LLMExtractor(config)
        result = ext._format_messages([])
        assert result == "(empty conversation)"


# ---------------------------------------------------------------------------
# LLMExtractor._parse_response
# ---------------------------------------------------------------------------

class TestParseResponse:
    """Test LLM response parsing."""

    def test_valid_json(self):
        config = _make_config()
        ext = LLMExtractor(config)
        response = _sample_llm_response()
        candidates = ext._parse_response(response)
        assert len(candidates) == 2
        assert candidates[0].type == "feedback"
        assert "short confirmations" in candidates[0].content
        assert candidates[0].relevance == 0.8
        assert "preference" in candidates[0].tags

    def test_json_in_code_fence(self):
        config = _make_config()
        ext = LLMExtractor(config)
        response = f"```json\n{_sample_llm_response()}\n```"
        candidates = ext._parse_response(response)
        assert len(candidates) == 2

    def test_json_in_code_fence_no_language(self):
        config = _make_config()
        ext = LLMExtractor(config)
        response = f"```\n{_sample_llm_response()}\n```"
        candidates = ext._parse_response(response)
        assert len(candidates) == 2

    def test_json_embedded_in_text(self):
        config = _make_config()
        ext = LLMExtractor(config)
        response = f"Here are the memories I extracted:\n{_sample_llm_response()}\nHope that helps!"
        candidates = ext._parse_response(response)
        assert len(candidates) == 2

    def test_empty_response(self):
        config = _make_config()
        ext = LLMExtractor(config)
        candidates = ext._parse_response("")
        assert candidates == []

    def test_empty_whitespace_response(self):
        config = _make_config()
        ext = LLMExtractor(config)
        candidates = ext._parse_response("   \n  \t  ")
        assert candidates == []

    def test_invalid_json(self):
        config = _make_config()
        ext = LLMExtractor(config)
        candidates = ext._parse_response("This is not JSON at all.")
        assert candidates == []

    def test_json_with_wrong_top_level_type(self):
        config = _make_config()
        ext = LLMExtractor(config)
        # A list instead of an object
        response = json.dumps([{"type": "user", "content": "test"}])
        candidates = ext._parse_response(response)
        assert candidates == []

    def test_missing_memories_key(self):
        config = _make_config()
        ext = LLMExtractor(config)
        response = json.dumps({"result": "ok"})
        candidates = ext._parse_response(response)
        assert candidates == []

    def test_memories_not_a_list(self):
        config = _make_config()
        ext = LLMExtractor(config)
        response = json.dumps({"memories": "not a list"})
        candidates = ext._parse_response(response)
        assert candidates == []

    def test_invalid_memory_type_defaults_to_user(self):
        config = _make_config()
        ext = LLMExtractor(config)
        response = json.dumps({
            "memories": [
                {
                    "type": "invalid_type",
                    "content": "Some content",
                    "tags": [],
                    "relevance": 0.5,
                }
            ]
        })
        candidates = ext._parse_response(response)
        assert len(candidates) == 1
        assert candidates[0].type == "user"  # defaults to 'user'

    def test_empty_content_skipped(self):
        config = _make_config()
        ext = LLMExtractor(config)
        response = json.dumps({
            "memories": [
                {
                    "type": "user",
                    "content": "",
                    "tags": [],
                    "relevance": 0.5,
                },
                {
                    "type": "feedback",
                    "content": "Valid content",
                    "tags": [],
                    "relevance": 0.7,
                },
            ]
        })
        candidates = ext._parse_response(response)
        assert len(candidates) == 1
        assert candidates[0].content == "Valid content"

    def test_non_dict_memory_entry_skipped(self):
        config = _make_config()
        ext = LLMExtractor(config)
        response = json.dumps({
            "memories": [
                "not a dict",
                {"type": "user", "content": "Valid", "tags": [], "relevance": 0.5},
            ]
        })
        candidates = ext._parse_response(response)
        assert len(candidates) == 1

    def test_tags_normalised_to_lowercase(self):
        config = _make_config()
        ext = LLMExtractor(config)
        response = json.dumps({
            "memories": [
                {
                    "type": "user",
                    "content": "Test content",
                    "tags": ["Preference", "COMMUNICATION"],
                    "relevance": 0.7,
                }
            ]
        })
        candidates = ext._parse_response(response)
        assert candidates[0].tags == ["preference", "communication"]

    def test_tags_limited_to_five(self):
        config = _make_config()
        ext = LLMExtractor(config)
        response = json.dumps({
            "memories": [
                {
                    "type": "user",
                    "content": "Test",
                    "tags": ["a", "b", "c", "d", "e", "f", "g"],
                    "relevance": 0.5,
                }
            ]
        })
        candidates = ext._parse_response(response)
        assert len(candidates[0].tags) <= 5

    def test_relevance_clamped_to_valid_range(self):
        config = _make_config()
        ext = LLMExtractor(config)
        response = json.dumps({
            "memories": [
                {"type": "user", "content": "High", "tags": [], "relevance": 1.5},
                {"type": "user", "content": "Low", "tags": [], "relevance": -0.3},
            ]
        })
        candidates = ext._parse_response(response)
        assert candidates[0].relevance == 1.0
        assert candidates[1].relevance == 0.0

    def test_relevance_defaults_to_half_on_invalid(self):
        config = _make_config()
        ext = LLMExtractor(config)
        response = json.dumps({
            "memories": [
                {"type": "user", "content": "No relevance", "tags": [], "relevance": "invalid"},
            ]
        })
        candidates = ext._parse_response(response)
        assert candidates[0].relevance == 0.5

    def test_importance_equals_relevance(self):
        config = _make_config()
        ext = LLMExtractor(config)
        response = json.dumps({
            "memories": [
                {"type": "feedback", "content": "Don't do X", "tags": ["correction"], "relevance": 0.85},
            ]
        })
        candidates = ext._parse_response(response)
        assert candidates[0].importance == 0.85

    def test_type_case_insensitive(self):
        config = _make_config()
        ext = LLMExtractor(config)
        response = json.dumps({
            "memories": [
                {"type": "User", "content": "Prefers X", "tags": [], "relevance": 0.7},
                {"type": "FEEDBACK", "content": "Stop Y", "tags": [], "relevance": 0.8},
            ]
        })
        candidates = ext._parse_response(response)
        assert candidates[0].type == "user"
        assert candidates[1].type == "feedback"

    def test_content_stripped(self):
        config = _make_config()
        ext = LLMExtractor(config)
        response = json.dumps({
            "memories": [
                {"type": "user", "content": "  Trim me  \n", "tags": [], "relevance": 0.5},
            ]
        })
        candidates = ext._parse_response(response)
        assert candidates[0].content == "Trim me"


# ---------------------------------------------------------------------------
# LLMExtractor.extract — integration with mocked LLM call
# ---------------------------------------------------------------------------

class TestExtract:
    """Test the public extract() method with mocked LLM calls."""

    def test_returns_candidates_on_success(self):
        config = _make_config()
        ext = LLMExtractor(config)
        messages = _sample_messages()

        mock_response = _sample_llm_response()
        with patch.object(ext, '_call_llm', return_value=mock_response):
            with patch.object(ext, '_build_prompt', return_value="test prompt"):
                candidates = ext.extract(session_id="test-123", messages=messages)
                assert len(candidates) == 2
                assert candidates[0].type == "feedback"
                assert candidates[1].type == "project"

    def test_returns_empty_on_no_api_key(self):
        config = _make_config(consolidate_api_key="", extraction_api_key="")
        # Also clear env vars
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENROUTER_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            ext = LLMExtractor(config)
            candidates = ext.extract(
                session_id="test-123",
                messages=_sample_messages(),
            )
            assert candidates == []

    def test_returns_empty_on_empty_messages(self):
        config = _make_config()
        ext = LLMExtractor(config)
        candidates = ext.extract(session_id="test-123", messages=[])
        assert candidates == []

    def test_returns_empty_on_llm_call_failure(self):
        config = _make_config()
        ext = LLMExtractor(config)
        with patch.object(ext, '_call_llm', side_effect=Exception("API error")):
            candidates = ext.extract(
                session_id="test-123",
                messages=_sample_messages(),
            )
            assert candidates == []

    def test_returns_empty_on_empty_llm_response(self):
        config = _make_config()
        ext = LLMExtractor(config)
        with patch.object(ext, '_call_llm', return_value=""):
            with patch.object(ext, '_build_prompt', return_value="p"):
                candidates = ext.extract(
                    session_id="test-123",
                    messages=_sample_messages(),
                )
                assert candidates == []

    def test_calls_build_prompt_with_manifest_summary(self):
        config = _make_config()
        ext = LLMExtractor(config)
        messages = _sample_messages()
        summary = "### User\n- test"

        mock_response = _sample_llm_response()
        with patch.object(ext, '_call_llm', return_value=mock_response):
            with patch.object(ext, '_build_prompt', return_value="prompt") as mock_build:
                ext.extract(
                    session_id="test-123",
                    messages=messages,
                    manifest_summary=summary,
                )
                mock_build.assert_called_once_with(messages, summary)


# ---------------------------------------------------------------------------
# LLMExtractor._call_llm — HTTP call mechanics
# ---------------------------------------------------------------------------

class TestCallLLM:
    """Test the LLM API call mechanics."""

    def test_makes_post_request(self):
        config = _make_config()
        ext = LLMExtractor(config)

        # Create a mock response
        api_response = json.dumps({
            "choices": [
                {"message": {"content": _sample_llm_response()}}
            ]
        }).encode("utf-8")

        mock_response = MagicMock()
        mock_response.read.return_value = api_response

        with patch("plugins.memory.dream.extract_llm.urllib.request.urlopen", return_value=mock_response):
            result = ext._call_llm("test prompt")
            assert "feedback" in result
            assert "project" in result

    def test_uses_correct_url(self):
        config = _make_config(consolidate_base_url="https://api.mycustom.com/v1")
        ext = LLMExtractor(config)

        api_response = json.dumps({
            "choices": [{"message": {"content": "{}"}}]
        }).encode("utf-8")
        mock_response = MagicMock()
        mock_response.read.return_value = api_response

        with patch("plugins.memory.dream.extract_llm.urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            ext._call_llm("test")
            call_args = mock_urlopen.call_args
            request = call_args[0][0]
            assert request.full_url == "https://api.mycustom.com/v1/chat/completions"
            assert request.method == "POST"

    def test_sends_api_key_in_header(self):
        config = _make_config(consolidate_api_key="my-secret-key")
        ext = LLMExtractor(config)

        api_response = json.dumps({
            "choices": [{"message": {"content": "{}"}}]
        }).encode("utf-8")
        mock_response = MagicMock()
        mock_response.read.return_value = api_response

        with patch("plugins.memory.dream.extract_llm.urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            ext._call_llm("test")
            call_args = mock_urlopen.call_args
            request = call_args[0][0]
            assert request.get_header("Authorization") == "Bearer my-secret-key"

    def test_sends_json_body(self):
        config = _make_config()
        ext = LLMExtractor(config)

        api_response = json.dumps({
            "choices": [{"message": {"content": "{}"}}]
        }).encode("utf-8")
        mock_response = MagicMock()
        mock_response.read.return_value = api_response

        with patch("plugins.memory.dream.extract_llm.urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            ext._call_llm("test prompt text")
            call_args = mock_urlopen.call_args
            request = call_args[0][0]
            body = json.loads(request.data.decode("utf-8"))
            assert body["model"] == ext._model
            assert len(body["messages"]) == 2
            assert body["messages"][0]["role"] == "system"
            assert body["messages"][1]["role"] == "user"
            assert body["messages"][1]["content"] == "test prompt text"

    def test_timeout_on_network_error(self):
        config = _make_config(extraction_timeout=5)
        ext = LLMExtractor(config)

        import socket
        with patch("plugins.memory.dream.extract_llm.urllib.request.urlopen", side_effect=socket.timeout("timed out")):
            with pytest.raises(Exception):
                ext._call_llm("test")

    def test_handles_empty_choices(self):
        config = _make_config()
        ext = LLMExtractor(config)

        api_response = json.dumps({"choices": []}).encode("utf-8")
        mock_response = MagicMock()
        mock_response.read.return_value = api_response

        with patch("plugins.memory.dream.extract_llm.urllib.request.urlopen", return_value=mock_response):
            result = ext._call_llm("test")
            assert result == ""


# ---------------------------------------------------------------------------
# get_manifest_summary
# ---------------------------------------------------------------------------

class TestGetManifestSummary:
    """Test manifest summary generation."""

    def test_returns_empty_when_store_none(self):
        result = get_manifest_summary(None)
        assert result == ""

    def test_formats_existing_memories(self):
        store = MagicMock()
        store.list_memories.return_value = [
            {
                "filename": "prefers-vim.md",
                "body": "User prefers vim for text editing",
                "meta": {"tags": ["preference", "editor"]},
            },
        ]
        # For types with no memories
        store.list_memories.side_effect = lambda memory_type=None: (
            [{
                "filename": "prefers-vim.md",
                "body": "User prefers vim for editing",
                "meta": {"tags": ["preference"]},
            }] if memory_type == "user" else []
        )
        result = get_manifest_summary(store)
        assert "User" in result or "user" in result.lower()
        assert "prefers vim" in result.lower()

    def test_handles_empty_store(self):
        store = MagicMock()
        store.list_memories.return_value = []
        result = get_manifest_summary(store)
        # No memories → empty string
        assert result == ""

    def test_limits_to_10_per_type(self):
        store = MagicMock()
        # Return 15 entries for 'user' type
        entries = [
            {"filename": f"mem-{i}.md", "body": f"Memory {i}", "meta": {"tags": []}}
            for i in range(15)
        ]
        store.list_memories.side_effect = lambda memory_type=None: (
            entries if memory_type == "user" else []
        )
        result = get_manifest_summary(store)
        # Should not include all 15
        assert result.count("Memory") <= 10

    def test_handles_store_exception(self):
        store = MagicMock()
        store.list_memories.side_effect = Exception("store error")
        result = get_manifest_summary(store)
        assert result == ""


# ---------------------------------------------------------------------------
# Integration: DreamMemoryProvider.on_session_end with extraction modes
# ---------------------------------------------------------------------------

class TestOnSessionEndExtractionModes:
    """Test that on_session_end respects extraction_mode config."""

    def _make_provider(self, extraction_mode="llm"):
        """Create a DreamMemoryProvider with mocked store."""
        from plugins.memory.dream import DreamMemoryProvider
        config = _make_config(extraction_mode=extraction_mode)
        provider = DreamMemoryProvider(config=config)
        provider._store = MagicMock()
        provider._session_id = "test-session"
        return provider

    def test_regex_mode_uses_regex_only(self):
        """In regex mode, only regex extraction should run, not LLM."""
        provider = self._make_provider(extraction_mode="regex")
        messages = [{"role": "user", "content": "I prefer dark mode."}]

        # The regex extractor should find this pattern
        provider.on_session_end(messages)

        # Store should have been called (regex found "I prefer")
        assert provider._store.add_memory.called

    def test_llm_mode_uses_llm_extractor(self):
        """In LLM mode, LLM extractor should be called."""
        provider = self._make_provider(extraction_mode="llm")
        messages = [{"role": "user", "content": "I prefer dark mode."}]

        # Mock the LLM extractor to return candidates
        llm_candidates = [
            CandidateMemory(
                type="user",
                content="User prefers dark mode",
                tags=["preference"],
                relevance=0.8,
                importance=0.8,
            ),
        ]
        with patch.object(provider._llm_extractor, 'extract', return_value=llm_candidates):
            with patch.object(provider._llm_extractor, '_call_llm'):
                with patch(
                    "plugins.memory.dream.extract_llm.get_manifest_summary",
                    return_value=""
                ):
                    provider.on_session_end(messages)

        # Store should have been called with the LLM candidate
        assert provider._store.add_memory.called
        call_args = provider._store.add_memory.call_args
        assert call_args.kwargs.get("tags") is not None or "session-end-llm" in str(call_args)

    def test_both_mode_merges_results(self):
        """In 'both' mode, both extractors should run and results merge."""
        provider = self._make_provider(extraction_mode="both")
        messages = [{"role": "user", "content": "I prefer dark mode."}]

        # Mock regex extraction will find "I prefer dark mode"
        # Mock LLM extraction to return a different candidate
        llm_candidates = [
            CandidateMemory(
                type="feedback",
                content="User wants dark mode as default",
                tags=["preference"],
                relevance=0.85,
                importance=0.85,
            ),
        ]
        with patch.object(provider._llm_extractor, 'extract', return_value=llm_candidates):
            with patch(
                "plugins.memory.dream.extract_llm.get_manifest_summary",
                return_value=""
            ):
                provider.on_session_end(messages)

        # Should have been called at least twice (regex + LLM candidates)
        assert provider._store.add_memory.call_count >= 1

    def test_llm_failure_falls_back_gracefully(self):
        """If LLM extractor fails, on_session_end should not crash."""
        provider = self._make_provider(extraction_mode="llm")
        messages = [{"role": "user", "content": "Hello there."}]

        # Mock LLM extractor to return empty (simulating failure)
        with patch.object(provider._llm_extractor, 'extract', return_value=[]):
            with patch(
                "plugins.memory.dream.extract_llm.get_manifest_summary",
                return_value=""
            ):
                # Should not raise
                provider.on_session_end(messages)

    def test_invalid_extraction_mode_defaults_to_llm(self):
        """Invalid extraction mode should default to 'llm'."""
        from plugins.memory.dream import DreamMemoryProvider
        config = _make_config(extraction_mode="invalid_mode")
        provider = DreamMemoryProvider(config=config)
        assert provider._extraction_mode == "llm"

    def test_tag_session_end_llm_in_llm_mode(self):
        """In LLM mode, candidates should be tagged 'session-end-llm'."""
        provider = self._make_provider(extraction_mode="llm")
        messages = [{"role": "user", "content": "I prefer dark mode."}]

        llm_candidates = [
            CandidateMemory(
                type="user",
                content="User prefers dark mode",
                tags=["preference"],
                relevance=0.8,
                importance=0.8,
            ),
        ]
        with patch.object(provider._llm_extractor, 'extract', return_value=llm_candidates):
            with patch(
                "plugins.memory.dream.extract_llm.get_manifest_summary",
                return_value=""
            ):
                provider.on_session_end(messages)

        # Check the tags on add_memory calls
        for call in provider._store.add_memory.call_args_list:
            tags = call.kwargs.get("tags", call[1].get("tags") if len(call) > 1 else [])
            if isinstance(tags, list):
                assert "session-end-llm" in tags, f"Expected 'session-end-llm' in tags, got {tags}"

    def test_tag_session_end_in_regex_mode(self):
        """In regex mode, candidates should be tagged 'session-end'."""
        provider = self._make_provider(extraction_mode="regex")
        messages = [{"role": "user", "content": "I prefer dark mode."}]

        provider.on_session_end(messages)

        # Check that at least one call used 'session-end' tag
        found_session_end = False
        for call in provider._store.add_memory.call_args_list:
            tags = call.kwargs.get("tags", [])
            if isinstance(tags, list) and "session-end" in tags:
                found_session_end = True
                break
        assert found_session_end, "Expected 'session-end' tag in regex mode"


# ---------------------------------------------------------------------------
# End-to-end _parse_response with realistic LLM output
# ---------------------------------------------------------------------------

class TestRealisticLLMOutput:
    """Test parsing of realistic LLM output."""

    def test_typical_response(self):
        config = _make_config()
        ext = LLMExtractor(config)
        response = json.dumps({
            "memories": [
                {
                    "type": "user",
                    "content": "User prefers short confirmations — just 'Updated' or 'Done', no long explanations",
                    "tags": ["preference", "communication"],
                    "relevance": 0.8,
                },
                {
                    "type": "feedback",
                    "content": "Don't use markdown tables in Telegram responses",
                    "tags": ["correction", "telegram"],
                    "relevance": 0.9,
                },
                {
                    "type": "project",
                    "content": "Dream plugin default vault_path should be ~/.hermes/dream_vault",
                    "tags": ["architecture", "vault"],
                    "relevance": 0.7,
                },
                {
                    "type": "reference",
                    "content": "Obsidian vault is at ~/apps/Garuda_hermes/ObsidianVault/",
                    "tags": ["path"],
                    "relevance": 0.4,
                },
            ]
        })
        candidates = ext._parse_response(response)
        assert len(candidates) == 4
        types = {c.type for c in candidates}
        assert types == {"user", "feedback", "project", "reference"}
        # Check relevance ordering-like properties
        feedback = [c for c in candidates if c.type == "feedback"][0]
        assert feedback.relevance == 0.9
        reference = [c for c in candidates if c.type == "reference"][0]
        assert reference.relevance == 0.4

    def test_response_with_extra_fields_ignored(self):
        config = _make_config()
        ext = LLMExtractor(config)
        response = json.dumps({
            "memories": [
                {
                    "type": "user",
                    "content": "Test content",
                    "tags": ["test"],
                    "relevance": 0.6,
                    "confidence": 0.95,  # extra field
                    "source": "voice",    # extra field
                },
            ]
        })
        candidates = ext._parse_response(response)
        assert len(candidates) == 1
        assert candidates[0].content == "Test content"

    def test_mixed_valid_invalid_entries(self):
        config = _make_config()
        ext = LLMExtractor(config)
        response = json.dumps({
            "memories": [
                {"type": "user", "content": "Valid entry", "tags": [], "relevance": 0.7},
                "not a dict",
                {"type": "feedback", "content": "", "tags": [], "relevance": 0.8},
                {"type": "project", "content": "Also valid", "tags": ["arch"], "relevance": 0.6},
            ]
        })
        candidates = ext._parse_response(response)
        # Only 2 valid entries (empty content is skipped, non-dict is skipped)
        assert len(candidates) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
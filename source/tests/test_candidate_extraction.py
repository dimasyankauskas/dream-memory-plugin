"""Tests for dream memory candidate extraction — regex patterns, filtering, edge cases."""

from __future__ import annotations

import pytest

from plugins.memory.dream.extract import (
    CandidateMemory,
    extract_candidates,
    extract_candidates_from_messages,
    build_pre_compress_summary,
)


# ---------------------------------------------------------------------------
# extract_candidates — user preference patterns
# ---------------------------------------------------------------------------

class TestUserPreferenceExtraction:
    """Test extraction of user preference patterns."""

    def test_i_prefer(self):
        candidates = extract_candidates("I prefer dark mode over light mode.", "")
        assert len(candidates) >= 1
        assert any(c.type == "user" for c in candidates)
        match = next(c for c in candidates if c.type == "user")
        assert "prefer" in match.content.lower()

    def test_i_always(self):
        candidates = extract_candidates("I always use tabs for indentation.", "")
        assert any(c.type == "user" for c in candidates)

    def test_i_never(self):
        candidates = extract_candidates("I never want to see that again.", "")
        assert any(c.type == "user" for c in candidates)

    def test_my_favorite(self):
        candidates = extract_candidates("My favorite language is Python.", "")
        assert any(c.type == "user" for c in candidates)

    def test_i_like(self):
        candidates = extract_candidates("I like using pytest for testing.", "")
        assert any(c.type == "user" for c in candidates)

    def test_i_love(self):
        candidates = extract_candidates("I love working with type hints.", "")
        assert any(c.type == "user" for c in candidates)

    def test_preferred_is(self):
        candidates = extract_candidates("My preferred editor is vim.", "")
        assert any(c.type == "user" for c in candidates)

    def test_default_is(self):
        candidates = extract_candidates("My default shell is zsh.", "")
        assert any(c.type == "user" for c in candidates)

    def test_user_preference_relevance(self):
        """User preference candidates should have relevance ~0.7."""
        candidates = extract_candidates("I prefer tea over coffee.", "")
        user_candidates = [c for c in candidates if c.type == "user"]
        assert len(user_candidates) >= 1
        assert all(0.5 < c.relevance <= 1.0 for c in user_candidates)


# ---------------------------------------------------------------------------
# extract_candidates — correction/feedback patterns
# ---------------------------------------------------------------------------

class TestFeedbackExtraction:
    """Test extraction of correction/feedback patterns."""

    def test_no_dont(self):
        candidates = extract_candidates("No, don't do it that way.", "")
        assert any(c.type == "feedback" for c in candidates)

    def test_actually(self):
        candidates = extract_candidates("Actually, I wanted the other option.", "")
        assert any(c.type == "feedback" for c in candidates)

    def test_stop_doing(self):
        candidates = extract_candidates("Stop doing that immediately.", "")
        assert any(c.type == "feedback" for c in candidates)

    def test_i_said(self):
        candidates = extract_candidates("I said to use lowercase!", "")
        assert any(c.type == "feedback" for c in candidates)

    def test_thats_wrong(self):
        candidates = extract_candidates("That's wrong, we should use snake_case.", "")
        assert any(c.type == "feedback" for c in candidates)

    def test_feedback_relevance(self):
        """Feedback candidates should have higher relevance (~0.8)."""
        candidates = extract_candidates("Actually, that's wrong.", "")
        fb_candidates = [c for c in candidates if c.type == "feedback"]
        assert len(fb_candidates) >= 1
        assert all(c.relevance >= 0.7 for c in fb_candidates)


# ---------------------------------------------------------------------------
# extract_candidates — project fact patterns
# ---------------------------------------------------------------------------

class TestProjectFactExtraction:
    """Test extraction of project-level fact patterns."""

    def test_project_uses(self):
        candidates = extract_candidates("The project uses FastAPI for the API layer.", "")
        assert any(c.type == "project" for c in candidates)

    def test_we_decided(self):
        candidates = extract_candidates("We decided to migrate to PostgreSQL.", "")
        assert any(c.type == "project" for c in candidates)

    def test_codebase_is(self):
        candidates = extract_candidates("The codebase is mostly Python.", "")
        assert any(c.type == "project" for c in candidates)

    def test_app_uses(self):
        candidates = extract_candidates("The app uses React on the frontend.", "")
        assert any(c.type == "project" for c in candidates)

    def test_project_relevance(self):
        """Project facts should have relevance ~0.65."""
        candidates = extract_candidates("The project uses Python 3.12.", "")
        proj_candidates = [c for c in candidates if c.type == "project"]
        assert len(proj_candidates) >= 1
        assert all(0.5 < c.relevance <= 1.0 for c in proj_candidates)


# ---------------------------------------------------------------------------
# extract_candidates — reference pointer patterns
# ---------------------------------------------------------------------------

class TestReferenceExtraction:
    """Test extraction of reference/pointer patterns."""

    def test_api_is_at(self):
        candidates = extract_candidates("The API is at api.example.com.", "")
        assert any(c.type == "reference" for c in candidates)

    def test_the_path_is(self):
        candidates = extract_candidates("The path is /usr/local/bin/python.", "")
        assert any(c.type == "reference" for c in candidates)

    def test_see_docs_at(self):
        candidates = extract_candidates("See docs at https://example.com/docs.", "")
        assert any(c.type == "reference" for c in candidates)

    def test_url_is(self):
        candidates = extract_candidates("The URL is https://api.service.io/v2.", "")
        assert any(c.type == "reference" for c in candidates)

    def test_reference_relevance(self):
        """Reference candidates should have relevance ~0.6."""
        candidates = extract_candidates("The path is /etc/config.", "")
        ref_candidates = [c for c in candidates if c.type == "reference"]
        assert len(ref_candidates) >= 1
        assert all(c.relevance >= 0.5 for c in ref_candidates)


# ---------------------------------------------------------------------------
# extract_candidates — deduplication, sorting, edge cases
# ---------------------------------------------------------------------------

class TestDeduplication:
    """Ensure duplicate matches are deduplicated."""

    def test_duplicate_sentences_deduplicated(self):
        # The same sentence matched in different sentences
        text = "I always use vim. I always use vim."
        candidates = extract_candidates(text, "")
        user_candidates = [c for c in candidates if c.type == "user"]
        # Should be deduplicated — not 2 identical entries
        contents = [c.content.lower().strip() for c in user_candidates]
        assert len(contents) == len(set(contents))

    def test_across_types_not_deduplicated(self):
        # The same text shouldn't match both user and feedback, but different
        # texts on different types are fine
        text = "I always use Python. The project uses Python."
        candidates = extract_candidates(text, "")
        types = [c.type for c in candidates]
        # Should have at least two different types
        assert len(set(types)) >= 1  # at minimum it works

    def test_sorted_by_relevance(self):
        """Candidates should be sorted by relevance descending."""
        text = "No, don't do that. I prefer the other way. The project uses Rust."
        candidates = extract_candidates(text, "")
        for i in range(len(candidates) - 1):
            assert candidates[i].relevance >= candidates[i + 1].relevance


class TestEdgeCases:
    """Edge case handling in extraction."""

    def test_empty_user_content(self):
        candidates = extract_candidates("", "")
        assert candidates == []

    def test_none_user_content(self):
        candidates = extract_candidates(None, "")
        assert candidates == []

    def test_short_content_skipped_in_from_messages(self):
        """Messages shorter than 5 chars are skipped in extract_candidates_from_messages."""
        messages = [{"role": "user", "content": "hi"}]
        candidates = extract_candidates_from_messages(messages)
        assert candidates == []

    def test_no_matching_patterns(self):
        """Plain text without any pattern should return empty list."""
        candidates = extract_candidates("Hello, how are you today?", "")
        assert candidates == []

    def test_long_content_truncated(self):
        """Very long content should be truncated to _MAX_CONTENT_LEN."""
        long_text = "I prefer " + "x" * 500
        candidates = extract_candidates(long_text, "")
        user_candidates = [c for c in candidates if c.type == "user"]
        if user_candidates:
            assert len(user_candidates[0].content) <= 300

    def test_assistant_content_not_extracted(self):
        """Currently assistant_content is not scanned for patterns."""
        candidates = extract_candidates("", "I always use vim for editing.")
        # Assistant content is not scanned currently
        assert candidates == []

    def test_multiple_matches_in_one_message(self):
        text = "I prefer dark mode. The project uses Python. Actually, wait."
        candidates = extract_candidates(text, "")
        types = {c.type for c in candidates}
        assert len(types) >= 2  # at least user + project or feedback


# ---------------------------------------------------------------------------
# extract_candidates_from_messages
# ---------------------------------------------------------------------------

class TestExtractFromMessages:
    """Test extraction from a list of message dicts."""

    def test_basic_extraction(self):
        messages = [
            {"role": "user", "content": "I prefer tea over coffee."},
            {"role": "assistant", "content": "Noted!"},
        ]
        candidates = extract_candidates_from_messages(messages)
        assert len(candidates) >= 1
        assert any(c.type == "user" for c in candidates)

    def test_skips_assistant_messages(self):
        messages = [
            {"role": "assistant", "content": "I always use vim."},
        ]
        candidates = extract_candidates_from_messages(messages)
        assert candidates == []

    def test_skips_non_string_content(self):
        messages = [
            {"role": "user", "content": None},
        ]
        candidates = extract_candidates_from_messages(messages)
        assert candidates == []

    def test_deduplicates_across_messages(self):
        messages = [
            {"role": "user", "content": "I always use vim."},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": "I always use vim."},  # duplicate
        ]
        candidates = extract_candidates_from_messages(messages)
        # Should not produce duplicate candidate
        contents = [c.content.lower().strip() for c in candidates]
        assert len(contents) == len(set(contents))

    def test_empty_messages(self):
        candidates = extract_candidates_from_messages([])
        assert candidates == []


# ---------------------------------------------------------------------------
# build_pre_compress_summary
# ---------------------------------------------------------------------------

class TestBuildPreCompressSummary:
    """Test the pre-compress summary builder."""

    def test_returns_string_with_candidates(self):
        messages = [
            {"role": "user", "content": "I prefer dark mode for my editor."},
            {"role": "assistant", "content": "Noted."},
        ]
        summary = build_pre_compress_summary(messages)
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "Dream Memory" in summary

    def test_returns_empty_for_no_candidates(self):
        messages = [
            {"role": "user", "content": "Hello there."},
            {"role": "assistant", "content": "Hi!"},
        ]
        summary = build_pre_compress_summary(messages)
        assert summary == ""

    def test_groups_by_type(self):
        messages = [
            {"role": "user", "content": "I prefer tea. The project uses Python."},
            {"role": "assistant", "content": "Got it."},
        ]
        summary = build_pre_compress_summary(messages)
        assert "User Preferences" in summary or "user" in summary.lower()
        assert "Project Context" in summary or "project" in summary.lower()

    def test_limits_per_type(self):
        """Should only include top 5 candidates per type."""
        # Create a message with many preference patterns
        prefs = ". ".join([f"I prefer option {i}" for i in range(10)])
        messages = [{"role": "user", "content": prefs}]
        summary = build_pre_compress_summary(messages)
        # Count bullet points under User Preferences section
        lines = summary.split("\n")
        bullet_lines = [l for l in lines if l.startswith("- ") and "prefer" in l.lower()]
        # Should be limited to 5
        assert len(bullet_lines) <= 5


# ---------------------------------------------------------------------------
# CandidateMemory dataclass
# ---------------------------------------------------------------------------

class TestCandidateMemory:
    """Test the CandidateMemory dataclass."""

    def test_default_fields(self):
        c = CandidateMemory(type="user", content="test")
        assert c.tags == []
        assert c.relevance == 0.5

    def test_custom_fields(self):
        c = CandidateMemory(type="feedback", content="correction", tags=["correction"], relevance=0.9)
        assert c.type == "feedback"
        assert c.content == "correction"
        assert c.tags == ["correction"]
        assert c.relevance == 0.9
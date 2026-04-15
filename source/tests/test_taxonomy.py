"""Tests for dream memory taxonomy — frontmatter parsing, rendering, type validation."""

from __future__ import annotations

import pytest

from plugins.memory.dream.taxonomy import (
    MEMORY_TYPES,
    MemoryTypeSpec,
    make_memory_document,
    parse_frontmatter,
    render_frontmatter,
    validate_memory_type,
)


# ---------------------------------------------------------------------------
# validate_memory_type
# ---------------------------------------------------------------------------

class TestValidateMemoryType:
    def test_valid_types(self):
        for t in ("user", "feedback", "project", "reference"):
            assert validate_memory_type(t) is True

    def test_invalid_type(self):
        assert validate_memory_type("bogus") is False

    def test_case_sensitive(self):
        assert validate_memory_type("User") is False
        assert validate_memory_type("USER") is False


# ---------------------------------------------------------------------------
# MEMORY_TYPES constant
# ---------------------------------------------------------------------------

class TestMemoryTypes:
    def test_all_four_types_present(self):
        expected = {"user", "feedback", "project", "reference"}
        assert set(MEMORY_TYPES.keys()) == expected

    def test_each_spec_has_required_fields(self):
        for name, spec in MEMORY_TYPES.items():
            assert spec.name == name
            assert isinstance(spec.description, str) and len(spec.description) > 0
            assert isinstance(spec.max_lines, int) and spec.max_lines > 0
            assert isinstance(spec.filename_pattern, str) and "{slug}" in spec.filename_pattern


# ---------------------------------------------------------------------------
# Frontmatter round-trip
# ---------------------------------------------------------------------------

class TestFrontmatterRoundTrip:
    def test_roundtrip_simple(self):
        meta = {
            "type": "user",
            "created": "2025-04-13T12:00:00+00:00",
            "updated": "2025-04-13T12:00:00+00:00",
            "relevance": 0.7,
            "tags": ["preference", "editor"],
            "source": "session-abc123",
        }
        rendered = render_frontmatter(meta)
        parsed = parse_frontmatter(rendered)
        assert parsed["type"] == "user"
        assert parsed["source"] == "session-abc123"
        assert abs(parsed["relevance"] - 0.7) < 0.01

    def test_roundtrip_with_body(self):
        meta = {
            "type": "project",
            "created": "2025-01-01T00:00:00+00:00",
            "updated": "2025-01-01T00:00:00+00:00",
            "relevance": 0.9,
            "tags": ["python", "ml"],
            "source": "sess-1",
        }
        body = "This is the memory body.\nSecond line."
        doc = render_frontmatter(meta) + "\n" + body
        parsed_meta = parse_frontmatter(doc)
        assert parsed_meta["type"] == "project"
        # Body should still be present in the document
        assert "This is the memory body." in doc

    def test_empty_frontmatter(self):
        assert parse_frontmatter("no frontmatter here") == {}

    def test_frontmatter_no_closing_delim(self):
        assert parse_frontmatter("---\ntype: user\n") == {}

    def test_relevance_clamped(self):
        doc = make_memory_document(
            content="test",
            memory_type="user",
            relevance=1.5,
        )
        parsed = parse_frontmatter(doc)
        assert parsed["relevance"] <= 1.0

    def test_relevance_negative_clamped(self):
        doc = make_memory_document(
            content="test",
            memory_type="feedback",
            relevance=-0.5,
        )
        parsed = parse_frontmatter(doc)
        assert parsed["relevance"] >= 0.0


# ---------------------------------------------------------------------------
# make_memory_document
# ---------------------------------------------------------------------------

class TestMakeMemoryDocument:
    def test_creates_valid_document(self):
        doc = make_memory_document(
            content="User prefers dark theme",
            memory_type="user",
            tags=["preference"],
            source="sess-42",
            relevance=0.8,
        )
        # Starts with frontmatter
        assert doc.startswith("---")
        # Contains the content
        assert "User prefers dark theme" in doc
        # Can be parsed back
        meta = parse_frontmatter(doc)
        assert meta["type"] == "user"
        assert meta["source"] == "sess-42"

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid memory type"):
            make_memory_document(
                content="bad",
                memory_type="invalid_type",
            )

    def test_default_tags_and_source(self):
        doc = make_memory_document(
            content="hello",
            memory_type="reference",
        )
        meta = parse_frontmatter(doc)
        assert meta["tags"] == []
        assert meta["source"] == ""
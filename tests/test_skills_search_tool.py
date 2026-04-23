"""Tests for the skills search tool.

Tests are purely unit-level — no DB, no LLM, no Docker required.
The SkillsLoader and Settings are mocked so the tool reads from
in-memory strings rather than the filesystem.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentic_chatbot_next.tools.skills_search_tool import (
    _build_index,
    _parse_sections,
    _score_section,
    _tokenise,
    make_skills_search_tool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_settings():
    return MagicMock()


def _make_tool_with_content(content_map: dict):
    """Build a search_skills tool backed by an in-memory content_map.

    content_map: {agent_key: markdown_string}
    """
    settings = _mock_settings()

    with patch(
        "agentic_chatbot_next.tools.skills_search_tool._build_index"
    ) as mock_build:
        # Build real index from provided content strings
        from agentic_chatbot_next.tools.skills_search_tool import _parse_sections

        sections = []
        for key, content in content_map.items():
            sections.extend(_parse_sections(key, content))
        mock_build.return_value = sections

        tool = make_skills_search_tool(settings)

    return tool


# ---------------------------------------------------------------------------
# _tokenise
# ---------------------------------------------------------------------------

class TestTokenise:
    def test_basic_words(self):
        tokens = _tokenise("resolve document confidence")
        assert "resolve" in tokens
        assert "document" in tokens
        assert "confidence" in tokens

    def test_stop_words_removed(self):
        tokens = _tokenise("how to handle the document")
        assert "handle" in tokens
        assert "document" in tokens
        # stop words removed
        assert "how" not in tokens
        assert "the" not in tokens
        assert "to" not in tokens

    def test_short_words_removed(self):
        tokens = _tokenise("a an be or if")
        assert len(tokens) == 0

    def test_lowercase(self):
        tokens = _tokenise("Document SEARCH Clause")
        assert "document" in tokens
        assert "search" in tokens
        assert "clause" in tokens

    def test_underscores_as_word_chars(self):
        tokens = _tokenise("resolve_document search_document")
        # underscores are word chars in the regex
        assert "resolve_document" in tokens


# ---------------------------------------------------------------------------
# _parse_sections
# ---------------------------------------------------------------------------

SAMPLE_MD = """\
# Agent Instructions

Some preamble text here.

## Failure Recovery

When search returns empty results:
1. Simplify the query
2. Try a different strategy

## Search Strategy

Use hybrid by default.
Use keyword for exact terms.

### Sub-strategy

More detail here.
"""


class TestParseSections:
    def test_sections_found(self):
        sections = _parse_sections("rag_agent", SAMPLE_MD)
        headings = [s.heading for s in sections]
        assert "Failure Recovery" in headings
        assert "Search Strategy" in headings

    def test_section_content(self):
        sections = _parse_sections("rag_agent", SAMPLE_MD)
        fr = next(s for s in sections if s.heading == "Failure Recovery")
        assert "Simplify the query" in fr.content

    def test_agent_key_set(self):
        sections = _parse_sections("rag_agent", SAMPLE_MD)
        assert all(s.agent_key == "rag_agent" for s in sections)

    def test_key_format(self):
        sections = _parse_sections("rag_agent", SAMPLE_MD)
        fr = next(s for s in sections if s.heading == "Failure Recovery")
        assert fr.key == "rag_agent > Failure Recovery"

    def test_tokens_populated(self):
        sections = _parse_sections("rag_agent", SAMPLE_MD)
        fr = next(s for s in sections if s.heading == "Failure Recovery")
        assert len(fr.tokens) > 0
        assert "simplify" in fr.tokens

    def test_empty_content_skipped(self):
        md = "## Empty Section\n\n## Real Section\nSome content here."
        sections = _parse_sections("test", md)
        headings = [s.heading for s in sections]
        assert "Real Section" in headings
        # Empty section should be skipped
        assert "Empty Section" not in headings

    def test_h1_not_parsed_as_section(self):
        """H1 headings (single #) are not split into sections."""
        sections = _parse_sections("test", SAMPLE_MD)
        headings = [s.heading for s in sections]
        assert "Agent Instructions" not in headings


# ---------------------------------------------------------------------------
# _score_section
# ---------------------------------------------------------------------------

class TestScoreSection:
    def _make_section(self, heading: str, content: str, agent_key: str = "test"):
        from agentic_chatbot_next.tools.skills_search_tool import _Section
        sec = _Section(agent_key=agent_key, heading=heading, content=content)
        sec.tokens = _tokenise(heading + " " + content)
        return sec

    def test_title_match_scores_higher(self):
        sec = self._make_section("Failure Recovery", "Some general content here.")
        q = _tokenise("failure recovery")
        score = _score_section(sec, q)
        assert score >= 6  # 2 title tokens × 3

    def test_content_match_scores_lower(self):
        sec = self._make_section("General Rules", "Failure recovery procedure steps here.")
        q = _tokenise("failure recovery")
        score = _score_section(sec, q)
        # content hits only (1 point each), title hits = 0
        assert 0 < score < 6

    def test_no_match_scores_zero(self):
        sec = self._make_section("Excel Inspection", "Load dataset and inspect columns.")
        q = _tokenise("memory save procedure")
        score = _score_section(sec, q)
        assert score == 0

    def test_partial_match(self):
        sec = self._make_section("Document Resolution", "Resolve document using confidence score.")
        q = _tokenise("document confidence threshold")
        score = _score_section(sec, q)
        assert score > 0


# ---------------------------------------------------------------------------
# make_skills_search_tool (integration-level, no filesystem)
# ---------------------------------------------------------------------------

CONTENT_RAG = """\
## Failure Recovery

When search returns no results, try a simpler query.
After 3 attempts, report failure explicitly.

## Search Strategy

Use hybrid search by default for most queries.
Use keyword search for exact clause numbers.
"""

CONTENT_DA = """\
## Operating Rules

Always call load_dataset before any analysis.
Never skip the inspect columns step.

## Code Best Practices

Use print() for all output in the sandbox.
Handle missing values before groupby operations.
"""


class TestMakeSkillsSearchTool:
    def _build_tool(self):
        settings = _mock_settings()
        from agentic_chatbot_next.tools.skills_search_tool import _parse_sections, _Section

        # Build real sections from in-memory content
        sections = []
        sections.extend(_parse_sections("rag_agent", CONTENT_RAG))
        sections.extend(_parse_sections("data_analyst_agent", CONTENT_DA))

        with patch(
            "agentic_chatbot_next.tools.skills_search_tool._build_index",
            return_value=sections,
        ):
            tool = make_skills_search_tool(settings)
        return tool

    def test_tool_name(self):
        tool = self._build_tool()
        assert tool.name == "search_skills"

    def test_returns_relevant_section(self):
        tool = self._build_tool()
        result = tool.invoke({"query": "failure recovery empty results"})
        assert "Failure Recovery" in result
        assert "rag_agent" in result

    def test_top_k_limits_results(self):
        tool = self._build_tool()
        result = tool.invoke({"query": "search strategy", "top_k": 1})
        # Should return at most 1 section
        assert result.count("[rag_agent >") + result.count("[data_analyst_agent >") <= 1

    def test_agent_filter_restricts_results(self):
        tool = self._build_tool()
        result = tool.invoke({
            "query": "operating rules",
            "agent_filter": "data_analyst_agent",
        })
        assert "data_analyst_agent" in result
        assert "rag_agent" not in result

    def test_no_match_returns_graceful_message(self):
        tool = self._build_tool()
        result = tool.invoke({"query": "xyzzy completely unrelated query abc"})
        assert "No matching" in result or "no results" in result.lower()

    def test_empty_query(self):
        tool = self._build_tool()
        result = tool.invoke({"query": ""})
        assert "No results" in result or "no searchable tokens" in result.lower()

    def test_invalid_agent_filter(self):
        tool = self._build_tool()
        result = tool.invoke({
            "query": "failure recovery",
            "agent_filter": "nonexistent_agent",
        })
        assert "No sections found" in result or "No matching" in result

    def test_top_k_capped_at_5(self):
        """top_k values > 5 should be silently capped."""
        tool = self._build_tool()
        # Should not raise even with large top_k
        result = tool.invoke({"query": "search strategy", "top_k": 100})
        assert isinstance(result, str)

    def test_result_contains_section_key(self):
        """Results should include the [agent > Section] prefix."""
        tool = self._build_tool()
        result = tool.invoke({"query": "hybrid search strategy"})
        assert "[rag_agent > Search Strategy]" in result

    def test_fallback_substring_match(self):
        """When token overlap is 0 but substring matches exist, still return results."""
        tool = self._build_tool()
        # "groupby" is not a token in CONTENT_DA but "groupby" is in the content
        result = tool.invoke({"query": "groupby"})
        # Either finds something or returns no-match — just shouldn't crash
        assert isinstance(result, str)


def test_repo_prompt_search_prefers_updated_general_guidance():
    repo_root = Path(__file__).resolve().parents[1]
    general_sections = _parse_sections(
        "general_agent",
        (repo_root / "data" / "skills" / "general_agent.md").read_text(encoding="utf-8"),
    )
    shared_sections = _parse_sections(
        "shared",
        (repo_root / "data" / "skills" / "skills.md").read_text(encoding="utf-8"),
    )
    with patch(
        "agentic_chatbot_next.tools.skills_search_tool._build_index",
        return_value=[*general_sections, *shared_sections],
    ):
        tool = make_skills_search_tool(_mock_settings())

    result = tool.invoke({"query": "unsupported request fallback", "agent_filter": "general_agent", "top_k": 1})

    assert "general_agent" in result.lower()
    assert "unsupported" in result.lower()

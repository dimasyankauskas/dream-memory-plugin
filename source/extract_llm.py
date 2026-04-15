"""Dream Memory LLM-based Extraction — high-quality memory extraction via LLM.

Uses an OpenAI-compatible API to extract structured memories from conversation
transcripts.  This is the QUALITY path (vs. the regex-based fast path in
extract.py) and is called at session end.

Architecture:
  - extract.py (regex): per-turn, fast, zero-cost → catches obvious patterns
  - extract_llm.py (LLM): session-end, slower, higher quality → extracts insights

Config keys (reused from consolidation, with optional overrides):
  - extraction_mode: 'regex' | 'llm' | 'both' (default: 'llm')
  - extraction_model: model identifier (default: consolidate_model)
  - extraction_api_key: API key (default: consolidate_api_key → env vars)
  - extraction_base_url: base URL (default: consolidate_base_url → env vars)
  - extraction_timeout: seconds to wait for LLM response (default: 30)
  - consolidate_model, consolidate_api_key, consolidate_base_url: fallbacks
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from .extract import CandidateMemory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_EXTRACTION_MODEL: str = "glm-5.1:cloud"
DEFAULT_EXTRACTION_TIMEOUT: int = 60  # seconds (Ollama/cloud models need more time)
DEFAULT_BASE_URL: str = "https://openrouter.ai/api/v1"

# Valid extraction modes
_VALID_MODES = {"regex", "llm", "both"}

# Valid memory types (must match taxonomy)
VALID_MEMORY_TYPES = {"user", "feedback", "project", "reference"}

# ---------------------------------------------------------------------------
# Extraction prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a memory extraction agent. Analyze the conversation below and extract memories worth persisting.

## Memory Types

Write each memory with the most appropriate type:
- **user**: Preferences, communication style, goals, personal context (e.g., "User prefers short confirmations over explanations")
- **feedback**: Corrections, "don't do this again" directives, explicit instructions (e.g., "Don't use markdown tables in Telegram")
- **project**: Active project facts, technical decisions, architecture choices (e.g., "Dream plugin vault_path defaults to ~/.hermes/dream_vault")
- **reference**: Stable facts — paths, URLs, API details, tool quirks (e.g., "Obsidian vault is at ~/apps/Garuda_hermes/ObsidianVault/")

## What NOT to Save
- Greetings, acknowledgments, small talk
- Already-known information (check existing memories below)
- Transient/ephemeral details (current time, session IDs)
- Verbose conversation — extract the INSIGHT, not the exchange
- Raw voice transcription artifacts (incomplete sentences, filler words)
- Subjective opinions not expressed as preferences

## Rules
- Extract INSIGHTS, not raw sentences. "I love it" → "User approves of this approach" or better yet, what specifically they approved
- Each memory should be self-contained and understandable without the conversation context
- Use concise, factual language
- Assign relevance: 0.7-0.9 for preferences/corrections, 0.5-0.7 for project facts, 0.3-0.5 for references
- Assign tags: 1-3 short lowercase tags per memory
- Return valid JSON only — no markdown code fences, no commentary

## Output Format

Return a JSON object with a single "memories" array. Each entry has:
- type: one of "user", "feedback", "project", "reference"
- content: concise, self-contained fact string
- tags: array of 1-3 short lowercase strings
- relevance: float between 0.0 and 1.0

Example:
```json
{
  "memories": [
    {
      "type": "feedback",
      "content": "User prefers short confirmations — just 'Updated' or 'Done', no long explanations",
      "tags": ["preference", "communication"],
      "relevance": 0.8
    }
  ]
}
```
"""

_USER_PROMPT_TEMPLATE = """## Existing Memories
{manifest_summary}

## Conversation
{conversation}"""

# ---------------------------------------------------------------------------
# LLMExtractor
# ---------------------------------------------------------------------------


class LLMExtractor:
    """Extract candidate memories from conversations using an LLM call.

    Uses an OpenAI-compatible API (via urllib, no external deps) to send
    the conversation transcript to an LLM and parse the structured JSON
    response into CandidateMemory objects.
    """

    def __init__(self, config: dict):
        """Initialise the extractor from plugin config.

        Resolution order for each setting:
          extraction_X → consolidate_X → env var → default
        """
        self._config = config

        # Resolve API key
        self._api_key = (
            config.get("extraction_api_key", "")
            or config.get("consolidate_api_key", "")
            or os.getenv("OPENROUTER_API_KEY", "")
            or os.getenv("OPENAI_API_KEY", "")
        )
        # Treat placeholder values as absent
        if self._api_key in ("", "***"):
            self._api_key = ""

        # Resolve base URL
        self._base_url = (
            config.get("extraction_base_url", "")
            or config.get("consolidate_base_url", "")
            or os.getenv("OPENAI_BASE_URL", "")
            or DEFAULT_BASE_URL
        ).rstrip("/")

        # Resolve model
        self._model = (
            config.get("extraction_model", "")
            or config.get("consolidate_model", "")
            or DEFAULT_EXTRACTION_MODEL
        )

        # Timeout
        self._timeout = int(config.get("extraction_timeout", DEFAULT_EXTRACTION_TIMEOUT))

    # -- Public API ----------------------------------------------------------

    def extract(
        self,
        session_id: str,
        messages: List[dict],
        manifest_summary: str = "",
    ) -> List[CandidateMemory]:
        """Extract memories from conversation messages using LLM.

        Falls back gracefully: if the LLM call fails or times out,
        returns an empty list (logs a warning).

        Parameters
        ----------
        session_id:
            Session identifier (used for logging).
        messages:
            List of message dicts, each with ``role`` and ``content``.
        manifest_summary:
            Summary of existing memories for dedup awareness (optional).

        Returns
        -------
        List[CandidateMemory]
            Extracted candidate memories. Empty on failure.
        """
        # Ollama / local servers don't need API keys — skip the check
        # when the base URL points at localhost or a private network.
        is_local = bool(
            self._base_url
            and (
                "localhost" in self._base_url
                or "127.0.0.1" in self._base_url
                or "0.0.0.0" in self._base_url
                or "::1" in self._base_url
                or self._base_url.startswith("http://192.168")
                or self._base_url.startswith("http://10.")
                or self._base_url.startswith("http://172.")
            )
        )

        if not self._api_key and not is_local:
            logger.warning(
                "LLM extraction skipped: no API key configured "
                "(checked extraction_api_key, consolidate_api_key, "
                "OPENROUTER_API_KEY, OPENAI_API_KEY)"
            )
            return []

        if not messages:
            return []

        prompt = self._build_prompt(messages, manifest_summary)
        try:
            raw_response = self._call_llm(prompt)
        except Exception as exc:
            logger.warning("LLM extraction call failed for session %s: %s", session_id, exc)
            return []

        if not raw_response:
            logger.warning("LLM extraction returned empty response for session %s", session_id)
            return []

        candidates = self._parse_response(raw_response)
        logger.info(
            "LLM extraction: session %s produced %d candidates",
            session_id, len(candidates),
        )
        return candidates

    # -- Prompt building -----------------------------------------------------

    def _build_prompt(self, messages: List[dict], manifest_summary: str = "") -> str:
        """Build the user-facing extraction prompt."""
        conversation = self._format_messages(messages)
        summary = manifest_summary.strip() if manifest_summary else "(no existing memories)"
        return _USER_PROMPT_TEMPLATE.format(
            manifest_summary=summary,
            conversation=conversation,
        )

    def _format_messages(self, messages: List[dict]) -> str:
        """Format conversation messages into a readable transcript.

        Each message is rendered as::

            [role]: content

        Non-string content is skipped. Truncated to ~6000 chars to stay
        within reasonable context limits.
        """
        lines: List[str] = []
        total_chars = 0
        max_chars = 6000

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            line = f"[{role}]: {content.strip()}"
            if total_chars + len(line) > max_chars:
                # Truncate the last line to fit
                remaining = max_chars - total_chars
                if remaining > 50:
                    lines.append(line[:remaining] + "…")
                break
            lines.append(line)
            total_chars += len(line) + 1  # +1 for newline

        return "\n".join(lines) if lines else "(empty conversation)"

    # -- LLM call ------------------------------------------------------------

    def _call_llm(self, user_prompt: str) -> str:
        """Make an OpenAI-compatible API call via urllib.

        Returns the raw response text (content of the first choice).

        Raises
        ------
        urllib.error.URLError
            On network failure.
        Exception
            On API error, timeout, etc.
        """
        url = f"{self._base_url}/chat/completions"
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 4096,
        }

        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )

        response = urllib.request.urlopen(req, timeout=self._timeout)
        raw = response.read().decode("utf-8")
        parsed = json.loads(raw)

        # Extract the content from the first choice
        choices = parsed.get("choices", [])
        if not choices:
            logger.warning("LLM extraction: no choices in response")
            return ""

        content = choices[0].get("message", {}).get("content", "")
        return content or ""

    # -- Response parsing ----------------------------------------------------

    def _parse_response(self, response_text: str) -> List[CandidateMemory]:
        """Parse the LLM response into CandidateMemory objects.

        Tries to extract JSON from the response text. Handles:
        - Raw JSON response
        - JSON wrapped in markdown code fences (```json ... ```)
        - JSON embedded in text (searches for first { to last })

        Returns an empty list on any parse failure (no crash).
        """
        if not response_text or not response_text.strip():
            return []

        # Strip markdown code fences if present
        stripped = response_text.strip()
        if stripped.startswith("```"):
            first_nl = stripped.find("\n")
            if first_nl != -1:
                stripped = stripped[first_nl + 1:]
            if stripped.rstrip().endswith("```"):
                stripped = stripped.rstrip()[:-3].rstrip()

        # Try direct JSON parse
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            json_match = re.search(r'\{[\s\S]*\}', stripped)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    logger.warning("LLM extraction: could not parse JSON from response")
                    return []
            else:
                logger.warning("LLM extraction: no JSON object found in response")
                return []

        if not isinstance(data, dict):
            logger.warning("LLM extraction: response is not a JSON object")
            return []

        memories_raw = data.get("memories", [])
        if not isinstance(memories_raw, list):
            logger.warning("LLM extraction: 'memories' is not an array")
            return []

        candidates: List[CandidateMemory] = []
        for i, mem in enumerate(memories_raw):
            if not isinstance(mem, dict):
                logger.debug("LLM extraction: skipping non-dict memory at index %d", i)
                continue

            mem_type = mem.get("type", "")
            content = mem.get("content", "")
            tags = mem.get("tags", [])
            relevance = mem.get("relevance", 0.5)

            # Validate and normalise type
            if mem_type not in VALID_MEMORY_TYPES:
                # Try to map common variations
                type_lower = str(mem_type).lower().strip()
                if type_lower in VALID_MEMORY_TYPES:
                    mem_type = type_lower
                else:
                    logger.debug(
                        "LLM extraction: invalid type %r at index %d, defaulting to 'user'",
                        mem_type, i,
                    )
                    mem_type = "user"

            # Validate content
            if not isinstance(content, str) or not content.strip():
                logger.debug("LLM extraction: skipping empty content at index %d", i)
                continue

            # Validate and normalise tags
            if not isinstance(tags, list):
                tags = []
            else:
                tags = [
                    str(t).lower().strip()
                    for t in tags
                    if isinstance(t, str) and t.strip()
                ]
            # Limit to 5 tags
            tags = tags[:5]

            # Validate and clamp relevance
            try:
                relevance = float(relevance)
            except (TypeError, ValueError):
                relevance = 0.5
            relevance = max(0.0, min(1.0, relevance))

            # Use relevance as initial importance
            candidates.append(CandidateMemory(
                type=mem_type,
                content=content.strip(),
                tags=tags,
                relevance=relevance,
                importance=relevance,
            ))

        return candidates


# ---------------------------------------------------------------------------
# Helper: get manifest summary from store
# ---------------------------------------------------------------------------

def get_manifest_summary(store) -> str:
    """Build a human-readable summary of existing memories for dedup awareness.

    Parameters
    ----------
    store : DreamStore
        The dream store instance (must have list_memories).

    Returns
    -------
    str
        A formatted summary string, or '' if no memories or store is None.
    """
    if store is None:
        return ""

    lines: List[str] = []
    try:
        for mem_type in ("user", "feedback", "project", "reference"):
            entries = store.list_memories(memory_type=mem_type)
            if not entries:
                continue
            lines.append(f"\n### {mem_type.capitalize()}")
            for entry in entries[:10]:  # Cap at 10 per type for context budget
                body = entry.get("body", "") or entry.get("content", "")
                # First 100 chars of content
                snippet = body[:100].replace("\n", " ").strip()
                tags = entry.get("meta", {}).get("tags", [])
                tag_str = ", ".join(tags[:3]) if tags else ""
                line = f"- {snippet}"
                if tag_str:
                    line += f"  [{tag_str}]"
                lines.append(line)
    except Exception as exc:
        logger.warning("Failed to build manifest summary for extraction: %s", exc)
        return ""

    return "\n".join(lines) if lines else ""
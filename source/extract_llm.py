"""Dream v2 LLM Extraction — Distillation prompt for significant memories.

This is the core fix from v1: instead of "what happened" (chronicles),
this asks "what matters" (distilled insights).

Model selection: glm-5.1:agentic
- Long-horizon optimization (sustains hundreds of rounds)
- Best SWE-Pro (58.4%) — good for analytical distillation
- From MODEL_PORTFOLIO_INDEX.md
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default Ollama endpoint
DEFAULT_BASE_URL = "http://localhost:11434/v1"


@dataclass
class ExtractedMemory:
    content: str
    type: str
    tags: List[str]
    importance: float
    slug: str


class LLMExtractor:
    """Extract significant memories using LLM distillation."""

    def __init__(
        self,
        model: str = "glm-5.1:agentic",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url or DEFAULT_BASE_URL
        self.timeout = timeout
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Lazy-init openai-compatible client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key or "ollama",
                    timeout=self.timeout,
                )
            except ImportError:
                try:
                    import httpx
                    self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout)
                except ImportError:
                    return None
        return self._client

    def extract_from_messages(
        self,
        messages: List[Dict[str, Any]],
        max_memories: int = 3,
        significance_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Extract significant memories from conversation messages.

        Args:
            messages: List of {role, content} dicts
            max_memories: Maximum memories to extract (per-session cap)
            significance_threshold: Minimum importance score to store

        Returns:
            List of {content, type, tags, importance} dicts
        """
        prompt = build_distillation_prompt(messages, max_memories)

        response = self._call_llm(prompt)
        if not response:
            return []

        candidates = self._parse_response(response)
        return [c for c in candidates if c.get("importance", 0) >= significance_threshold][:max_memories]

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM with the distillation prompt."""
        client = self._get_client()
        if client is None:
            logger.error("[LLMExtractor] No client available")
            return None

        try:
            # OpenAI-compatible API
            import openai
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error("[LLMExtractor] LLM call failed: %s", e)
            return None

    def _parse_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into memory dicts.

        Expected format — markdown code block with JSON:
        ```json
        [
          {
            "type": "consciousness/self",
            "content": "The memory content...",
            "tags": ["tag1", "tag2"],
            "importance": 0.8,
            "slug": "short-descriptive-slug"
          }
        ]
        ```
        """
        if not response:
            return []

        # Extract JSON from code blocks or raw
        json_str = response
        if "```json" in response:
            parts = response.split("```json")
            if len(parts) > 1:
                json_str = parts[1].split("```")[0]
        elif "```" in response:
            parts = response.split("```")
            if len(parts) > 1:
                json_str = parts[1].strip()

        json_str = json_str.strip()

        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "memories" in data:
                return data["memories"]
            return []
        except json.JSONDecodeError as e:
            logger.warning("[LLMExtractor] Failed to parse JSON: %s\nResponse: %s", e, response[:500])
            return []


def build_distillation_prompt(messages: List[Dict[str, Any]], max_memories: int = 3) -> str:
    """Build the distillation prompt for memory extraction.

    This is the KEY change from v1:
    - v1 asked "what happened" → session chronicles
    - v2 asks "what matters" → distilled insights

    The prompt enforces:
    1. Significance gate — non-obvious only
    2. Per-session cap — max memories
    3. Why it matters — not just what
    4. Going-forward actionability — what to do differently
    """

    # Format conversation for the LLM
    conversation = _format_conversation(messages)

    prompt = f"""You are a memory distillation agent. Your job is to extract 1-{max_memories} memories that are **non-obvious** — things that would genuinely be lost without explicit recording.

## Memory Types

Choose the best type for each memory:
- **consciousness/self**: What the agent learned about its own capabilities, limitations, strengths, or patterns
- **consciousness/relationship**: What the agent learned about the user's preferences, communication style, or values
- **consciousness/work**: Project or technical learnings — what worked, what failed, process insights
- **decisions**: Explicit agreements or decisions made (and why)
- **feedback**: Corrections or directives — what the user told the agent to do differently
- **reference**: Stable facts worth remembering (APIs, tools, paths)

## Significance Gate

Extract ONLY if the memory is non-obvious:
- Extract: decisions with reasoning, real preferences (not stated), lessons learned, capability insights
- DO NOT extract: obvious facts, tool names, file paths, raw conversation summaries, ephemeral debugging

## Output Format

Return a JSON array with up to {max_memories} memories. Each memory has:
- type: the memory type
- content: 3-10 sentences that capture what happened, why it matters, and what to do differently
- tags: 2-4 relevant tags
- importance: 0.0-1.0 (0.9 = critical, 0.7 = significant, 0.5 = worth noting)
- slug: a 3-5 word lowercase hyphenated identifier

## Conversation to Analyze

{conversation}

## Output

```json
[
  {{
    "type": "consciousness/self",
    "content": "I struggle with long-range planning but self-correct well in code review. When given a large project, I should explicitly ask Kedar to help break it into milestones rather than attempting to architect the whole thing alone.",
    "tags": ["self-knowledge", "planning", "capability"],
    "importance": 0.8,
    "slug": "self-planning-limitation"
  }}
]
```
"""
    return prompt


def _format_conversation(messages: List[Dict[str, Any]]) -> str:
    """Format messages for the prompt. Focus on actual content."""
    lines = []
    for msg in messages[-20:]:  # Last 20 messages to avoid context overflow
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if not content or isinstance(content, list):
            continue

        # Truncate very long messages
        if len(content) > 1000:
            content = content[:1000] + "..."

        lines.append(f"**{role.upper()}**: {content}")

    if not lines:
        return "(No conversation content)"

    return "\n".join(lines)
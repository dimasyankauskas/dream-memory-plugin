"""Dream v2 Shared — Config loading utilities."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def load_dream_v2_config() -> dict:
    """Read the ``plugins.dream_v2`` section from config.yaml."""
    from hermes_constants import get_hermes_home

    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}

    try:
        import yaml
        with open(config_path) as f:
            all_config = yaml.safe_load(f) or {}
        dream_config = all_config.get("plugins", {}).get("dream_v2", {}) or {}
    except Exception:
        return {}

    # Default extraction model to glm-5.1:agentic
    dream_config.setdefault("extraction_model", "glm-5.1:agentic")
    dream_config.setdefault("consolidation_model", "glm-5.1:agentic")
    dream_config.setdefault("extraction_mode", "llm")
    dream_config.setdefault("hybrid_mode", True)
    dream_config.setdefault("max_memories_per_session", 3)
    dream_config.setdefault("significance_threshold", 0.7)
    dream_config.setdefault("max_lines_per_file", 200)
    dream_config.setdefault("auto_recall", False)
    dream_config.setdefault("consolidation_mode", "manual")
    dream_config.setdefault("vault_path", "")  # resolved at runtime via resolve_vault_path()

    return dream_config


def resolve_vault_path(config: dict) -> Path:
    """Resolve vault path from config."""
    explicit = config.get("vault_path", "")
    if explicit:
        return Path(explicit).resolve()

    from hermes_constants import get_hermes_home
    return (get_hermes_home() / "dream_vault").resolve()
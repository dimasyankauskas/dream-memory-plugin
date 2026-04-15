"""Dream Memory Plugin — shared configuration and path utilities.

Centralises config loading and vault path resolution so that __init__.py
and cli.py don't duplicate the same logic.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def load_dream_config() -> dict:
    """Read the ``plugins.dream`` section from config.yaml.

    For ``consolidate_api_key``, prefers environment variables
    ``CONSOLIDATE_API_KEY`` or ``OPENROUTER_API_KEY`` over the config value.
    The config.yaml value is treated as a fallback; if it's a placeholder
    like ``"***"``, it is ignored.
    """
    from hermes_constants import get_hermes_home

    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path) as f:
            all_config = yaml.safe_load(f) or {}
        dream_config = all_config.get("plugins", {}).get("dream", {}) or {}
    except Exception:
        return {}

    # Prefer env vars over config.yaml for the API key
    env_key = os.getenv("CONSOLIDATE_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
    config_key = dream_config.get("consolidate_api_key", "")
    # Treat placeholder values as absent
    if config_key in ("", "***"):
        config_key = ""
    dream_config["consolidate_api_key"] = env_key or config_key

    # Also resolve extraction_api_key with same env var fallback
    extraction_env_key = os.getenv("EXTRACTION_API_KEY", "")
    extraction_config_key = dream_config.get("extraction_api_key", "")
    if extraction_config_key in ("", "***"):
        extraction_config_key = ""
    dream_config["extraction_api_key"] = extraction_env_key or extraction_config_key

    # Ensure vault_subdir has a default value
    dream_config.setdefault("vault_subdir", "")

    # Default extraction_mode to 'llm' if not set
    dream_config.setdefault("extraction_mode", "llm")

    return dream_config


def resolve_vault_path(dream_config: dict) -> Path:
    """Resolve the vault path from config, defaulting to $HERMES_HOME/dream_vault.

    Validates that the resolved path is under hermes_home ONLY when
    the vault_path is derived from the default (not explicitly configured).
    An explicit vault_path in config is trusted as-is.
    """
    from hermes_constants import get_hermes_home

    hermes_home = get_hermes_home()
    explicit_path = dream_config.get("vault_path")

    if explicit_path:
        # Expand $HERMES_HOME variable if present
        vault_path_str = str(explicit_path)
        vault_path_str = vault_path_str.replace("$HERMES_HOME", str(hermes_home))
        vault_path_str = vault_path_str.replace("${HERMES_HOME}", str(hermes_home))
        resolved = Path(vault_path_str).resolve()
        # Explicitly configured paths are trusted — no validation
        return resolved

    # Default path — validate it stays under hermes_home
    resolved = (hermes_home / "dream_vault").resolve()
    try:
        resolved.relative_to(hermes_home.resolve())
    except ValueError:
        logger.error(
            "Dream vault path %s escapes hermes_home %s; falling back to default",
            resolved, hermes_home,
        )
        resolved = hermes_home / "dream_vault"

    return resolved


def save_dream_config(dream_config: dict) -> None:
    """Write the ``plugins.dream`` section back to config.yaml.

    Masks the consolidate_api_key with '***' so the real key
    is never persisted in the yaml file.  The key should be
    provided via environment variable instead.
    """
    from hermes_constants import get_hermes_home

    config_path = get_hermes_home() / "config.yaml"
    try:
        import yaml
        existing: dict = {}
        if config_path.exists():
            with open(config_path) as f:
                existing = yaml.safe_load(f) or {}
        existing.setdefault("plugins", {})
        # Mask API key before writing to disk
        safe_config = dict(dream_config)  # shallow copy
        if safe_config.get("consolidate_api_key") and safe_config["consolidate_api_key"] != "***":
            safe_config["consolidate_api_key"] = "***"
        existing["plugins"]["dream"] = safe_config
        with open(config_path, "w") as f:
            yaml.dump(existing, f, default_flow_style=False)
    except Exception as exc:
        logger.warning("Dream save_dream_config failed: %s", exc)
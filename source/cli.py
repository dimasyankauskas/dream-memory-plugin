"""CLI commands for Dream Memory Plugin management.

Handles: hermes dream status | consolidate | recall | list | setup | enable | disable
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home
from hermes_cli.colors import Colors, color

from .shared import load_dream_config as _load_plugin_config, resolve_vault_path as _get_vault_path, save_dream_config as _save_plugin_config


# ---------------------------------------------------------------------------
# Prompt helper
# ---------------------------------------------------------------------------

def _prompt(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    sys.stdout.write(f"  {label}{suffix}: ")
    sys.stdout.flush()
    val = sys.stdin.readline().strip()
    return val or (default or "")


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def cmd_status(args) -> None:
    """Show vault statistics — count by type, total size, last consolidation, config."""
    from .store import DreamStore
    from .consolidation import _read_last_consolidation_ts

    dream_config = _load_plugin_config()
    vault_path = _get_vault_path(dream_config)

    store = DreamStore(vault_path, config=dream_config)
    if not store.dream_root.exists():
        print(f"\n  Dream vault not found at {store.dream_root}")
        print("  Run 'hermes dream setup' to configure, or start a session to auto-create.\n")
        return

    store.initialize()
    stats = store.stats()

    print(f"\nDream Memory status\n" + "─" * 40)
    print(f"  Vault path:     {stats['vault_path']}")

    counts = stats.get("counts", {})
    total = stats.get("total", 0)

    type_labels = {
        "user": "User",
        "feedback": "Feedback",
        "project": "Project",
        "reference": "Reference",
    }

    print(f"  Total memories:  {total}")
    print()
    for mt in ("user", "feedback", "project", "reference"):
        label = type_labels.get(mt, mt.capitalize())
        count = counts.get(mt, 0)
        bar = "█" * min(count, 40)
        print(f"    {label:<12} {count:>4}  {bar}")
    print()

    # Total vault size on disk
    total_bytes = 0
    dream_root = store.dream_root
    for mt in ("user", "feedback", "project", "reference"):
        type_dir = dream_root / mt
        if type_dir.exists():
            for f in type_dir.glob("*.md"):
                try:
                    total_bytes += f.stat().st_size
                except OSError:
                    pass

    if total_bytes < 1024:
        size_str = f"{total_bytes} B"
    elif total_bytes < 1024 * 1024:
        size_str = f"{total_bytes / 1024:.1f} KB"
    else:
        size_str = f"{total_bytes / (1024 * 1024):.1f} MB"
    print(f"  Vault size:      {size_str}")

    # Last consolidation
    last_ts = _read_last_consolidation_ts(store.dream_root)
    if last_ts:
        print(f"  Last consolidate: {last_ts}")
    else:
        print(f"  Last consolidate: never")

    # Active config
    print()
    print(f"  Configuration:")
    print(f"    max_lines:       {dream_config.get('max_lines', '(default: 100)')}")
    print(f"    max_bytes:       {dream_config.get('max_bytes', '(default: 50000)')}")
    print(f"    consolidate_model: {dream_config.get('consolidate_model', '(not set)')}")
    print(f"    consolidation_mode: {dream_config.get('consolidation_mode', '(default: auto)')}")
    print()


def cmd_consolidate(args) -> None:
    """Run consolidation immediately (Orient → Gather → Consolidate → Prune)."""
    from .store import DreamStore
    from .consolidation import run_consolidation

    dream_config = _load_plugin_config()
    vault_path = _get_vault_path(dream_config)

    if not vault_path.exists():
        print(f"\n  Dream vault not found at {vault_path}")
        print("  Start a session first to create the vault.\n")
        return

    dry_run = getattr(args, "dry_run", False)
    memory_type = getattr(args, "type", None)

    if memory_type and memory_type not in ("user", "feedback", "project", "reference"):
        print(f"\n  Invalid memory type: {memory_type!r}")
        print("  Valid types: user, feedback, project, reference\n")
        return

    store = DreamStore(vault_path, config=dream_config)
    store.initialize()

    consolidation_config = {
        "max_lines": dream_config.get("max_lines", 100),
        "max_bytes": dream_config.get("max_bytes", 50000),
    }

    print(f"\nDream consolidation {'(dry-run)' if dry_run else ''}\n" + "─" * 40)

    try:
        result = run_consolidation(
            store=store,
            config=consolidation_config,
            dry_run=dry_run,
            memory_type=memory_type,
        )
    except Exception as exc:
        print(f"  Consolidation failed: {exc}\n")
        return

    # Orient results
    orient = result.orient
    print(f"  Orient:")
    print(f"    Needs consolidation: {orient.needs_consolidation}")
    print(f"    Reason:             {orient.reason}")
    if orient.stale_files:
        print(f"    Stale files:        {len(orient.stale_files)}")
    if orient.oversized_files:
        print(f"    Oversized files:    {len(orient.oversized_files)}")

    # Gather results
    gather = result.gather
    print(f"\n  Gather:")
    print(f"    Entries loaded:     {gather.stats.get('entries_loaded', 0)}")
    print(f"    Groups found:       {gather.stats.get('groups_found', 0)}")
    print(f"    Duplicates found:   {gather.stats.get('duplicates_found', 0)}")
    print(f"    Contradictions:     {gather.stats.get('contradictions_found', 0)}")

    # Consolidate results
    consolidate = result.consolidate
    print(f"\n  Consolidate:")
    print(f"    Merged:             {consolidate.merged_count}")
    print(f"    Deduped:            {consolidate.deduped_count}")
    print(f"    Contradictions:     {consolidate.pruned_count}")
    print(f"    Total actions:      {len(consolidate.actions)}")

    # Show action details
    if consolidate.actions:
        print()
        for action in consolidate.actions:
            desc = f"    {action.action}: {action.target_type}"
            if action.target_files:
                desc += f" {', '.join(action.target_files[:3])}"
                if len(action.target_files) > 3:
                    desc += f" (+{len(action.target_files) - 3} more)"
            if action.result_file:
                desc += f" → {action.result_file}"
            print(desc)

    # Prune results
    prune = result.prune
    if prune.deleted_files or prune.capped_files:
        print(f"\n  Prune:")
        if prune.deleted_files:
            print(f"    Deleted: {len(prune.deleted_files)} files")
        if prune.capped_files:
            print(f"    Capped:  {len(prune.capped_files)} files")
        print(f"    Manifest updated: {prune.manifest_updated}")

    status_label = "dry-run" if dry_run else "completed"
    print(f"\n  Status: {status_label}")
    print()


def cmd_recall(args) -> None:
    """Search memories by query using manifest-based recall."""
    from .store import DreamStore
    from .recall import RecallEngine

    dream_config = _load_plugin_config()
    vault_path = _get_vault_path(dream_config)

    if not vault_path.exists():
        print(f"\n  Dream vault not found at {vault_path}")
        print("  Start a session first to create the vault.\n")
        return

    query = getattr(args, "query", "")
    if not query or not query.strip():
        print("\n  Usage: hermes dream recall <query>\n")
        return

    memory_type = getattr(args, "type", None)
    limit = getattr(args, "limit", 5)

    if memory_type and memory_type not in ("user", "feedback", "project", "reference"):
        print(f"\n  Invalid memory type: {memory_type!r}")
        print("  Valid types: user, feedback, project, reference\n")
        return

    store = DreamStore(vault_path, config=dream_config)
    store.initialize()
    engine = RecallEngine(store)

    results = engine.recall(query=query, memory_type=memory_type, limit=limit)

    print(f"\nDream recall: \"{query}\"\n" + "─" * 40)

    if not results:
        print("  No matching memories found.\n")
        return

    type_labels = {
        "user": "User",
        "feedback": "Feedback",
        "project": "Project",
        "reference": "Reference",
    }

    for i, r in enumerate(results, 1):
        label = type_labels.get(r.memory_type, r.memory_type.capitalize())
        print(f"  {i}. [{label}] {r.filename}")
        print(f"     Score: {r.score:.4f}")
        # Show first 150 chars of content
        content_preview = r.content[:150].replace("\n", " ").strip()
        if len(r.content) > 150:
            content_preview += "…"
        print(f"     {content_preview}")
        tags = r.frontmatter.get("tags", [])
        if tags:
            print(f"     Tags: {', '.join(tags[:5])}")
        print()

    print(f"  {len(results)} result(s) for \"{query}\"")
    print()


def cmd_list(args) -> None:
    """List all memories, optionally filtered by type."""
    from .store import DreamStore

    dream_config = _load_plugin_config()
    vault_path = _get_vault_path(dream_config)

    if not vault_path.exists():
        print(f"\n  Dream vault not found at {vault_path}")
        print("  Start a session first to create the vault.\n")
        return

    memory_type = getattr(args, "type", None)

    if memory_type and memory_type not in ("user", "feedback", "project", "reference"):
        print(f"\n  Invalid memory type: {memory_type!r}")
        print("  Valid types: user, feedback, project, reference\n")
        return

    store = DreamStore(vault_path, config=dream_config)
    store.initialize()
    memories = store.list_memories(memory_type=memory_type)

    type_labels = {
        "user": "User",
        "feedback": "Feedback",
        "project": "Project",
        "reference": "Reference",
    }

    filter_label = f" (type: {memory_type})" if memory_type else ""
    print(f"\nDream memories{filter_label}\n" + "─" * 40)

    if not memories:
        print("  No memories found.\n")
        return

    for entry in memories:
        mt = entry.get("type", "?")
        label = type_labels.get(mt, mt.capitalize())
        filename = entry.get("filename", "?")
        body = entry.get("body", "").replace("\n", " ").strip()[:80]
        meta = entry.get("meta", {})
        relevance = meta.get("relevance", "?")
        tags = meta.get("tags", [])
        tag_str = f" [{', '.join(tags[:3])}]" if tags else ""

        print(f"  {label:<10} {filename}  (r={relevance}){tag_str}")
        print(f"             {body}")

    print(f"\n  {len(memories)} memories listed.\n")


def cmd_setup(args) -> None:
    """Interactive setup wizard for Dream Memory configuration."""
    dream_config = _load_plugin_config()
    hermes_home = get_hermes_home()

    print("\nDream Memory setup\n" + "─" * 40)
    print("  Dream gives Hermes structured file-based memory with")
    print("  taxonomy, manifest-based recall, and scheduled consolidation.")
    print()

    # --- 1. Vault path ---
    current_vault = dream_config.get("vault_path", str(hermes_home / "dream_vault"))
    new_vault = _prompt("Vault path", default=current_vault)
    if new_vault:
        dream_config["vault_path"] = new_vault

    # --- 2. max_lines ---
    current_lines = str(dream_config.get("max_lines", 100))
    new_lines = _prompt("Max lines per memory", default=current_lines)
    try:
        dream_config["max_lines"] = int(new_lines)
    except (ValueError, TypeError):
        pass  # keep default

    # --- 3. max_bytes ---
    current_bytes = str(dream_config.get("max_bytes", 50000))
    new_bytes = _prompt("Max bytes per memory", default=current_bytes)
    try:
        dream_config["max_bytes"] = int(new_bytes)
    except (ValueError, TypeError):
        pass

    # --- 4. consolidate_model ---
    current_model = dream_config.get("consolidate_model", "")
    print("\n  Consolidation model (LLM model for Phase 4 consolidation).")
    print("  Leave blank to use deterministic-only (no LLM) consolidation.")
    new_model = _prompt("Consolidation model", default=current_model or "(none)")
    if new_model and new_model != "(none)":
        dream_config["consolidate_model"] = new_model
    else:
        dream_config.pop("consolidate_model", None)

    # --- 5. consolidation_mode ---
    current_mode = dream_config.get("consolidation_mode", "auto")
    print("\n  Consolidation mode:")
    print("    auto    — consolidate automatically when conditions are met (default)")
    print("    manual  — only consolidate when explicitly triggered")
    print("    off     — disable consolidation entirely")
    new_mode = _prompt("Consolidation mode", default=current_mode)
    if new_mode in ("auto", "manual", "off"):
        dream_config["consolidation_mode"] = new_mode
    else:
        dream_config["consolidation_mode"] = "auto"

    # --- 6. consolidate_api_key (secret) ---
    current_key = dream_config.get("consolidate_api_key", "")
    masked_key = "***" if current_key and current_key != "***" else current_key
    print("\n  API key for LLM consolidation.")
    print("  Recommended: set CONSOLIDATE_API_KEY or OPENROUTER_API_KEY env var instead.")
    print(f"  Current: {masked_key if masked_key else '(not set)'}")
    new_key = _prompt("Consolidate API key (leave blank to keep or use env var)", default="")
    if new_key:
        dream_config["consolidate_api_key"] = new_key
    elif not new_key and current_key and current_key != "***":
        # Keep existing key if user didn't change it
        pass
    else:
        dream_config.pop("consolidate_api_key", None)

    # --- Save ---
    _save_plugin_config(dream_config)
    print(f"\n  Configuration saved.")

    # Show what was saved
    vault_path = _get_vault_path(dream_config)
    print(f"  Vault path:           {vault_path}")
    print(f"  Max lines:            {dream_config.get('max_lines', 100)}")
    print(f"  Max bytes:            {dream_config.get('max_bytes', 50000)}")
    print(f"  Consolidation model:  {dream_config.get('consolidate_model', '(none)')}")
    print(f"  Consolidation mode:   {dream_config.get('consolidation_mode', 'auto')}")
    display_key = "***" if dream_config.get("consolidate_api_key") else "(not set)"
    print(f"  API key:              {display_key}")
    print(f"\n  Config saved to {hermes_home / 'config.yaml'}")
    print()

    # Create vault directory if it doesn't exist
    if not vault_path.exists():
        from .store import DreamStore
        print("  Creating vault directory...")
        store = DreamStore(vault_path, config=dream_config)
        store.initialize()
        print(f"  Vault created at {vault_path}\n")
    else:
        print(f"  Vault already exists at {vault_path}\n")


def cmd_enable(args) -> None:
    """Enable dream as memory provider (set memory.provider=dream in config.yaml)."""
    from hermes_cli.config import load_config, save_config

    config = load_config()
    current_provider = config.get("memory", {}).get("provider")

    if current_provider == "dream":
        print("\n  Dream is already the active memory provider.\n")
        return

    if current_provider and current_provider != "none":
        print(f"\n  Current memory provider is '{current_provider}'.")
        answer = _prompt(f"Switch to dream?", default="y")
        if answer.lower() not in ("y", "yes"):
            print("  Cancelled.\n")
            return

    config.setdefault("memory", {})["provider"] = "dream"
    save_config(config)

    # Ensure vault exists
    dream_config = _load_plugin_config()
    vault_path = _get_vault_path(dream_config)
    if not vault_path.exists():
        from .store import DreamStore
        store = DreamStore(vault_path, config=dream_config)
        store.initialize()

    print(f"\n  Dream memory provider enabled.")
    print(f"  Vault: {vault_path}")
    print(f"  Set memory.provider=dream in config.yaml\n")


def cmd_disable(args) -> None:
    """Disable dream provider (set memory.provider back to none)."""
    from hermes_cli.config import load_config, save_config

    config = load_config()
    current_provider = config.get("memory", {}).get("provider")

    if not current_provider or current_provider == "none":
        print("\n  No memory provider is currently active.\n")
        return

    if current_provider != "dream":
        print(f"\n  Current memory provider is '{current_provider}', not 'dream'.")
        print("  Run 'hermes {current_provider} disable' instead.\n")
        return

    answer = _prompt("Disable dream memory provider?", default="n")
    if answer.lower() not in ("y", "yes"):
        print("  Cancelled.\n")
        return

    config["memory"]["provider"] = "none"
    save_config(config)

    print(f"\n  Dream memory provider disabled.")
    print(f"  Set memory.provider=none in config.yaml")
    print(f"  Vault data preserved on disk.\n")


# ---------------------------------------------------------------------------
# Command router
# ---------------------------------------------------------------------------

def dream_command(args) -> None:
    """Route dream subcommands."""
    sub = getattr(args, "dream_command", None)

    if sub is None or sub == "status":
        cmd_status(args)
    elif sub == "consolidate":
        cmd_consolidate(args)
    elif sub == "recall":
        cmd_recall(args)
    elif sub == "list":
        cmd_list(args)
    elif sub == "setup":
        cmd_setup(args)
    elif sub == "enable":
        cmd_enable(args)
    elif sub == "disable":
        cmd_disable(args)
    else:
        print(f"  Unknown dream command: {sub}")
        print("  Available: status, consolidate, recall, list, setup, enable, disable\n")


def register_cli(subparser) -> None:
    """Build the ``hermes dream`` argparse subcommand tree.

    Called by the plugin CLI registration system during argparse setup.
    The *subparser* is the parser for ``hermes dream``.
    """

    subparser.add_argument(
        "--target-profile", metavar="NAME", dest="target_profile",
        help="Target a specific profile's Dream config without switching",
    )
    subs = subparser.add_subparsers(dest="dream_command")

    subs.add_parser(
        "status",
        help="Show vault statistics (count by type, total size, last consolidation)",
    )

    consolidate_parser = subs.add_parser(
        "consolidate",
        help="Run consolidation immediately (Orient → Gather → Consolidate → Prune)",
    )
    consolidate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to disk",
    )
    consolidate_parser.add_argument(
        "--type",
        choices=("user", "feedback", "project", "reference"),
        help="Consolidate only this memory type",
    )

    recall_parser = subs.add_parser(
        "recall",
        help="Search memories by query using manifest-based recall",
    )
    recall_parser.add_argument(
        "query",
        help="Search query for memory recall",
    )
    recall_parser.add_argument(
        "--type",
        choices=("user", "feedback", "project", "reference"),
        help="Filter by memory type",
    )
    recall_parser.add_argument(
        "--limit", type=int, default=5,
        help="Maximum number of results (default: 5)",
    )

    list_parser = subs.add_parser(
        "list",
        help="List all memories, optionally filtered by type",
    )
    list_parser.add_argument(
        "--type",
        choices=("user", "feedback", "project", "reference"),
        help="Filter by memory type",
    )

    subs.add_parser(
        "setup",
        help="Interactive setup wizard for Dream Memory configuration",
    )

    subs.add_parser(
        "enable",
        help="Enable dream as memory provider",
    )

    subs.add_parser(
        "disable",
        help="Disable dream provider (set memory.provider back to none)",
    )

    subparser.set_defaults(func=dream_command)
"""Dream v2 Staging Manager — Pre-compress rescue workaround.

Bug #6: upstream silently discards on_pre_compress return value.
Workaround: write candidates to a staging file. Next session_start merges to vault.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

STAGING_DIR = "staging"
STAGING_FILE = "pending_memories.jsonl"


class StagingManager:
    """Manages staging directory for pre-compress rescue."""

    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.staging_dir = self.vault_path / STAGING_DIR
        self.staging_dir.mkdir(exist_ok=True)
        self.staging_file = self.staging_dir / STAGING_FILE

    def write_candidates(self, candidates: List[Dict[str, Any]], session_id: str) -> None:
        """Write extracted candidates to staging file."""
        if not candidates:
            return

        try:
            # Append to staging file as JSONL
            with open(self.staging_file, "a") as f:
                for candidate in candidates:
                    record = {
                        "session_id": session_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "candidate": candidate,
                    }
                    f.write(json.dumps(record) + "\n")

            logger.info("[Staging] Wrote %d candidates for session %s", len(candidates), session_id)
        except Exception as e:
            logger.error("[Staging] Failed to write staging: %s", e)

    def merge_to_vault(self) -> int:
        """Merge all staging records into the vault. Returns count merged."""
        if not self.staging_file.exists():
            return 0

        merged = 0
        try:
            # Read all records
            records = []
            with open(self.staging_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

            if not records:
                return 0

            # Import here to avoid circular
            from .store import DreamStore
            store = DreamStore(str(self.vault_path))

            for record in records:
                candidate = record.get("candidate", {})
                path = store.add_memory(
                    content=candidate.get("content", ""),
                    memory_type=candidate.get("type", "reference"),
                    tags=candidate.get("tags", []),
                    source=f"staging:{record.get('session_id', 'unknown')}",
                    importance=candidate.get("importance", 0.3),
                )
                if path:
                    merged += 1

            # Clear staging file
            self.staging_file.unlink()
            store.rebuild_index()

            logger.info("[Staging] Merged %d memories to vault", merged)
        except Exception as e:
            logger.error("[Staging] Merge failed: %s", e)

        return merged

    def pending_count(self) -> int:
        """Return number of pending staged memories."""
        if not self.staging_file.exists():
            return 0
        try:
            with open(self.staging_file) as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0
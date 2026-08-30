"""Shared manifest/provenance helpers for the data-processing scripts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_entry(
    name: str,
    description: str,
    path: Path,
    repo_root: Path,
    source: str,
    source_release: str,
) -> dict:
    """Build a processed-artifact manifest entry with checksum and size."""
    return {
        "name": name,
        "description": description,
        "source": source,
        "source_release": source_release,
        "path": str(path.relative_to(repo_root)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def register_processed(manifest_path: Path, entries: list[dict]) -> None:
    """Idempotently upsert ``processed/`` entries by name, keeping others."""
    manifest = json.loads(manifest_path.read_text())
    by_name = {d["name"]: d for d in manifest["datasets"]
               if d["name"].startswith("processed/")}
    for entry in entries:
        by_name[entry["name"]] = entry
    kept = [d for d in manifest["datasets"]
            if not d["name"].startswith("processed/")]
    manifest["datasets"] = kept + list(by_name.values())
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

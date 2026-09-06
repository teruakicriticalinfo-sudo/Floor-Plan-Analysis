"""Persistent cache for expensive vision-stage floor-plan results."""

import hashlib
import json
from pathlib import Path
from typing import Any

from .analyzer import PIPELINE_CACHE_VERSION


def structure_cache_key(
    image_bytes: bytes,
    provider: str,
    model: str,
    num_ctx: int,
    max_images: int,
) -> str:
    digest = hashlib.sha256(image_bytes)
    settings = f"{PIPELINE_CACHE_VERSION}|{provider}|{model}|{num_ctx}|{max_images}"
    digest.update(settings.encode("utf-8"))
    return digest.hexdigest()


def load_structure_cache(path: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("pipeline_version") != PIPELINE_CACHE_VERSION:
            return None
        return payload["draft_structure"], payload["structure"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_structure_cache(
    path: Path,
    draft: dict[str, Any],
    structure: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    payload = {
        "pipeline_version": PIPELINE_CACHE_VERSION,
        "metadata": metadata,
        "draft_structure": draft,
        "structure": structure,
    }
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

"""Experimental, ground-truth-independent sliding wall-patch vision pass."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .targeted_vision import _ask, _valid_box


PATCH_VERSION = "window-wall-patches-v1"
DETECTION_SCHEMA = {
    "type": "object", "required": ["detections"],
    "properties": {"detections": {"type": "array", "items": {
        "type": "object", "required": ["kind", "bbox", "confidence", "evidence"],
        "properties": {
            "kind": {"type": "string", "enum": ["window", "balcony_door", "garage_window_door"]},
            "bbox": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "evidence": {"type": "string"},
        },
    }}},
}


def _starts(low: int, high: int, size: int, stride: int) -> list[int]:
    if high - low <= size:
        return [low]
    starts = list(range(low, high - size + 1, stride))
    last = high - size
    if starts[-1] != last:
        starts.append(last)
    return starts


def build_patches(image_size: tuple[int, int], plan_regions: list[dict[str, Any]],
                  size_px: int = 144, stride_px: int = 96) -> list[dict[str, Any]]:
    """Tile each floor without consulting labeled window or door positions."""
    if size_px < 64 or stride_px < 32 or stride_px > size_px:
        raise ValueError("パッチ寸法または移動幅が不正です")
    width, height = image_size
    regions = plan_regions or [{"floor_id": "unknown", "bbox": [0, 0, 1, 1]}]
    patches = []
    seen = set()
    for region_index, region in enumerate(regions, 1):
        box = region.get("bbox")
        if not _valid_box(box):
            raise ValueError("図面領域のbboxが不正です")
        left, top = round(box[0] * width), round(box[1] * height)
        right, bottom = round(box[2] * width), round(box[3] * height)
        patch_width = min(size_px, right - left)
        patch_height = min(size_px, bottom - top)
        for y in _starts(top, bottom, patch_height, stride_px):
            for x in _starts(left, right, patch_width, stride_px):
                pixel_box = [x, y, x + patch_width, y + patch_height]
                if tuple(pixel_box) in seen:
                    continue
                seen.add(tuple(pixel_box))
                patches.append({"id": f"P{region_index}_{len(patches)+1}",
                                "floor_id": region.get("floor_id"), "pixel_bbox": pixel_box})
    return patches


def _prompt() -> str:
    return (
        "これは日本の間取り図の小さな切出しです。この切出しで完全に見える外壁の窓記号と、"
        "外部へ出るガラス戸だけを探してください。外壁の太い黒線を横切る細い白線・二重線は窓の手掛かりです。"
        "室内扉、収納扉、家具、文字、単なる壁の切れ目は窓ではありません。"
        "円弧で開くベランダへの扉はbalcony_door、車庫へ出るガラス戸はgarage_window_doorとし、windowと区別します。"
        "見えた記号の外接bboxを切出しの左上[0,0]、右下[1,1]で返してください。"
        "各候補はkind,bbox,confidence,evidenceを持ちます。高・中確信度以外は後で除外します。"
        "見えなければdetections空配列です。位置を推測しないでください。最大4件のJSONだけを返してください。"
    )


def _global_box(local_box: list[float], patch: dict[str, Any], image_size: tuple[int, int]) -> list[float]:
    x1, y1, x2, y2 = patch["pixel_bbox"]
    width, height = image_size
    return [round((x1 + local_box[0] * (x2 - x1)) / width, 5),
            round((y1 + local_box[1] * (y2 - y1)) / height, 5),
            round((x1 + local_box[2] * (x2 - x1)) / width, 5),
            round((y1 + local_box[3] * (y2 - y1)) / height, 5)]


def _upscale(image: Image.Image) -> Image.Image:
    scale = min(4, 768 / max(image.size))
    if scale <= 1:
        return image
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)


def scan_patches(image: Image.Image, patches: list[dict[str, Any]], client: Any, model: str,
                 *, checkpoint_path: Path | None = None,
                 progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Checkpoint each patch so interrupted local-model runs can resume."""
    completed: dict[str, Any] = {}
    if checkpoint_path and checkpoint_path.is_file():
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if saved.get("version") == PATCH_VERSION and saved.get("model") == model:
            completed = saved.get("patch_results", {})
    for index, patch in enumerate(patches, 1):
        patch_id = patch["id"]
        if patch_id in completed:
            if progress:
                progress(f"PATCH HIT {index}/{len(patches)} {patch_id}")
            continue
        if progress:
            progress(f"PATCH SCAN {index}/{len(patches)} {patch_id}")
        crop = _upscale(image.crop(tuple(patch["pixel_bbox"])))
        try:
            answer = _ask(client, model, _prompt(), crop, DETECTION_SCHEMA)
            if not isinstance(answer.get("detections"), list):
                raise ValueError("detections配列がありません")
            observations = []
            for item in answer["detections"][:4]:
                if (not isinstance(item, dict) or item.get("kind") not in
                    {"window", "balcony_door", "garage_window_door"}
                    or not _valid_box(item.get("bbox"))):
                    continue
                observations.append({"kind": item["kind"], "bbox": _global_box(item["bbox"], patch, image.size),
                                     "confidence": item.get("confidence"),
                                     "evidence": str(item.get("evidence", ""))[:100], "patch_id": patch_id})
            completed[patch_id] = {"status": "complete", "detections": observations}
        except (ValueError, RuntimeError, KeyError) as exc:
            completed[patch_id] = {"status": "error", "error": str(exc), "detections": []}
        if checkpoint_path:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"version": PATCH_VERSION, "model": model, "patches": patches,
                       "patch_results": completed}
            temporary = checkpoint_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(checkpoint_path)
    raw = [item for result in completed.values() for item in result.get("detections", [])]
    accepted = []
    rejected = []
    patch_boxes = {patch["id"]: patch["pixel_bbox"] for patch in patches}
    for item in raw:
        reason = _rejection_reason(item, image.size, patch_boxes.get(item["patch_id"]))
        if reason:
            rejected.append({**item, "rejection_reason": reason})
        else:
            accepted.append(item)
    return {"version": PATCH_VERSION, "model": model, "patches": patches,
            "patch_results": completed, "raw_detections": raw,
            "rejected_detections": rejected,
            "accepted_detections": deduplicate(accepted, image.size)}


def _rejection_reason(item: dict[str, Any], image_size: tuple[int, int],
                      patch_box: list[int] | None = None) -> str | None:
    if item.get("confidence") not in {"high", "medium"}:
        return "低確信度"
    box = item["bbox"]
    width = (box[2] - box[0]) * image_size[0]
    height = (box[3] - box[1]) * image_size[1]
    if patch_box and width >= 0.85 * (patch_box[2] - patch_box[0]) and height >= 0.85 * (patch_box[3] - patch_box[1]):
        return "切出し全体を囲むため開口位置が特定できない"
    if item.get("kind") == "window":
        if min(width, height) > 12 or max(width, height) / max(min(width, height), 0.1) < 1.8:
            return "細長い窓記号の形状ではない"
    return None


def deduplicate(detections: list[dict[str, Any]], image_size: tuple[int, int],
                radius_px: float = 12) -> list[dict[str, Any]]:
    """Collapse only nearby detections of the same kind from overlapping patches."""
    result = []
    width, height = image_size
    for item in sorted(detections, key=lambda value: value.get("confidence") != "high"):
        box = item["bbox"]
        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        if any(item["kind"] == prior["kind"] and
               (((center[0] - (prior["bbox"][0] + prior["bbox"][2]) / 2) * width) ** 2
                + ((center[1] - (prior["bbox"][1] + prior["bbox"][3]) / 2) * height) ** 2) ** 0.5
               <= radius_px for prior in result):
            continue
        result.append(item)
    return result


def scan_cache_key(image_bytes: bytes, model: str, patches: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256(image_bytes)
    digest.update(PATCH_VERSION.encode())
    digest.update(model.encode())
    digest.update(json.dumps(patches, sort_keys=True).encode())
    return digest.hexdigest()

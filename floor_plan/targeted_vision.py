"""Small-area vision probes. Results are review candidates, never scoring evidence."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from typing import Any

from .analyzer import _generate_with_retry, _inventory_anchor, _point_to_bbox_edge_distance


PROBE_VERSION = "targeted-vision-v2"
ROOM_SCHEMA = {
    "type": "object", "required": ["found", "bbox", "confidence", "evidence"],
    "properties": {
        "found": {"type": "boolean"},
        "bbox": {"type": ["array", "null"], "items": {"type": "number"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "evidence": {"type": "string"},
    },
}
WINDOW_PROBE_SCHEMA = {
    "type": "object", "required": ["windows"],
    "properties": {"windows": {"type": "array", "items": {
        "type": "object", "required": ["position", "confidence", "evidence"],
        "properties": {
            "position": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "evidence": {"type": "string"},
        },
    }}},
}


def _valid_box(box: Any) -> bool:
    return (isinstance(box, list) and len(box) == 4
            and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and 0 <= value <= 1 for value in box)
            and box[0] < box[2] and box[1] < box[3])


def _valid_point(point: Any) -> bool:
    return (isinstance(point, list) and len(point) == 2
            and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and 0 <= value <= 1 for value in point))


def padded_box(box: list[float], x_pad: float, y_pad: float) -> list[float]:
    return [max(0, box[0] - x_pad), max(0, box[1] - y_pad),
            min(1, box[2] + x_pad), min(1, box[3] + y_pad)]


def map_point(point: list[float], box: list[float]) -> list[float]:
    return [round(box[0] + point[0] * (box[2] - box[0]), 5),
            round(box[1] + point[1] * (box[3] - box[1]), 5)]


def map_box(inner: list[float], outer: list[float]) -> list[float]:
    return map_point(inner[:2], outer) + map_point(inner[2:], outer)


def _crop(image: Any, box: list[float]) -> Any:
    width, height = image.size
    cropped = image.crop((round(box[0] * width), round(box[1] * height),
                          round(box[2] * width), round(box[3] * height)))
    if hasattr(cropped, "resize") and max(cropped.size) < 900:
        from PIL import Image

        scale = min(4, 1200 / max(cropped.size))
        cropped = cropped.resize((round(cropped.size[0] * scale), round(cropped.size[1] * scale)),
                                 Image.Resampling.LANCZOS)
    return cropped


def _ask(client: Any, model: str, prompt: str, image: Any, schema: dict[str, Any]) -> dict[str, Any]:
    response = _generate_with_retry(client,
        model=model, contents=[prompt, image],
        config={"response_mime_type": "application/json", "response_json_schema": schema,
                "temperature": 0, "max_output_tokens": 512},
    )
    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("小領域の読取結果がJSONオブジェクトではありません")
    return value


def _floor_box(space: dict[str, Any], regions: list[dict[str, Any]]) -> list[float] | None:
    for region in regions:
        if region.get("floor_id") == space.get("floor_id") and _valid_box(region.get("bbox")):
            return region["bbox"]
    return None


def select_probes(structure: dict[str, Any], max_window_rooms: int = 4,
                  window_wall_strips: bool = False) -> list[dict[str, Any]]:
    """Use existing geometry only to choose crops, not as proof of a feature."""
    regions = structure.get("plan_regions") or []
    spaces = [space for space in structure.get("spaces", []) if _valid_box(space.get("bbox"))]
    probes: list[dict[str, Any]] = []
    for wash in spaces:
        if _inventory_anchor(wash) != "washroom":
            continue
        floor = _floor_box(wash, regions)
        neighbors = [space for space in spaces if space.get("floor_id") == wash.get("floor_id")
                     and _inventory_anchor(space) == "bath"]
        boxes = [wash["bbox"], *[space["bbox"] for space in neighbors]]
        union = [min(box[0] for box in boxes), min(box[1] for box in boxes),
                 max(box[2] for box in boxes), max(box[3] for box in boxes)]
        crop = padded_box(union, 0.035, 0.06)
        if floor:
            crop = [max(crop[0], floor[0]), max(crop[1], floor[1]),
                    min(crop[2], floor[2]), min(crop[3], floor[3])]
        if _valid_box(crop):
            probes.append({"kind": "washroom", "space_id": wash["id"],
                           "label": wash.get("label", "洗面所"), "crop_bbox": crop})

    rooms = [space for space in spaces if space.get("space_type") == "room"]
    rooms.sort(key=lambda space: (space["bbox"][2] - space["bbox"][0])
               * (space["bbox"][3] - space["bbox"][1]), reverse=True)
    for space in rooms[:max_window_rooms]:
        floor = _floor_box(space, regions)
        box = space["bbox"]
        if window_wall_strips and floor:
            distances = {"top": box[1] - floor[1], "bottom": floor[3] - box[3],
                         "left": box[0] - floor[0], "right": floor[2] - box[2]}
            edges = sorted(distances, key=distances.get)[:2]
            crops = {
                "top": [box[0] - 0.035, box[1] - 0.07, box[2] + 0.035, box[1] + 0.04],
                "bottom": [box[0] - 0.035, box[3] - 0.04, box[2] + 0.035, box[3] + 0.07],
                "left": [box[0] - 0.07, box[1] - 0.035, box[0] + 0.04, box[3] + 0.035],
                "right": [box[2] - 0.04, box[1] - 0.035, box[2] + 0.07, box[3] + 0.035],
            }
            candidates = [(edge, crops[edge]) for edge in edges]
        else:
            candidates = [(None, padded_box(box, 0.035, 0.045))]
        for edge, crop in candidates:
            if floor:
                crop = [max(crop[0], floor[0]), max(crop[1], floor[1]),
                        min(crop[2], floor[2]), min(crop[3], floor[3])]
            crop = [max(0, crop[0]), max(0, crop[1]), min(1, crop[2]), min(1, crop[3])]
            if _valid_box(crop):
                probes.append({"kind": "window", "space_id": space["id"],
                               "label": space.get("label", "部屋"), "edge": edge, "crop_bbox": crop})
    return probes


def run_probes(image: Any, structure: dict[str, Any], client: Any, model: str,
               *, max_window_rooms: int = 4, window_wall_strips: bool = False,
               progress: Any = None) -> dict[str, Any]:
    spaces = {space["id"]: space for space in structure.get("spaces", [])}
    results = []
    for probe in select_probes(structure, max_window_rooms, window_wall_strips):
        if progress:
            progress(f"小領域読取: {probe['kind']} {probe['space_id']}")
        crop = _crop(image, probe["crop_bbox"])
        result = {**probe, "candidates": [], "status": "completed"}
        try:
            if probe["kind"] == "washroom":
                prompt = ("この切出し図面内で、洗面所・脱衣所の空間を特定してください。浴槽のある浴室、便器のあるトイレとは別です。"
                          "洗面所・脱衣所が実際に読める場合だけfound=trueにし、その空間全体のbboxを切出し画像の左上[0,0]、右下[1,1]で返してください。"
                          "洗面台のアイコンだけで空間境界が分からなければfound=false。推測禁止。"
                          "found,bbox,confidence,evidenceだけのJSONを返してください。")
                answer = _ask(client, model, prompt, crop, ROOM_SCHEMA)
                candidate = {"found": answer.get("found") is True,
                             "confidence": answer.get("confidence"), "evidence": answer.get("evidence", "")}
                if candidate["found"] and _valid_box(answer.get("bbox")):
                    candidate["bbox"] = map_box(answer["bbox"], probe["crop_bbox"])
                    candidate["review_status"] = ("candidate" if candidate["confidence"] in {"high", "medium"}
                                                  else "low_confidence")
                else:
                    candidate["review_status"] = "not_confirmed"
                result["candidates"].append(candidate)
            else:
                prompt = (f"この切出し図面は「{probe['label']}」の外壁付近です。この部屋の外壁に実際に描かれた窓記号だけを探してください。"
                          "室内扉、収納扉、壁線、バルコニー開口の推測は窓に含めません。"
                          "各窓のpositionは切出し画像の左上[0,0]、右下[1,1]に正規化した窓記号の中心です。"
                          "根拠が曖昧なら登録せず、windows空配列にしてください。"
                          "windows配列の各要素はposition,confidence,evidenceだけです。")
                answer = _ask(client, model, prompt, crop, WINDOW_PROBE_SCHEMA)
                if not isinstance(answer.get("windows"), list):
                    raise ValueError("窓の配列がありません")
                for item in answer["windows"]:
                    if not isinstance(item, dict) or not _valid_point(item.get("position")):
                        continue
                    position = map_point(item["position"], probe["crop_bbox"])
                    edge_distance = _point_to_bbox_edge_distance(position, spaces[probe["space_id"]]["bbox"])
                    confidence = item.get("confidence")
                    result["candidates"].append({
                        "position": position, "confidence": confidence,
                        "evidence": item.get("evidence", ""),
                        "edge_distance": round(edge_distance, 5),
                        "review_status": ("candidate" if confidence in {"high", "medium"} and edge_distance <= 0.035
                                          else "rejected_geometry_or_confidence"),
                    })
        except (ValueError, RuntimeError, KeyError) as exc:
            result["status"] = "error"
            result["error"] = str(exc)
        results.append(result)
    return {"version": PROBE_VERSION, "model": model, "probes": results,
            "note": "候補は人手確認用。承認までは採点・正解データへ自動反映しない。"}


def cache_key(image_path: Path, structure_path: Path, model: str,
              max_window_rooms: int = 4, window_wall_strips: bool = False) -> str:
    digest = hashlib.sha256()
    for value in (PROBE_VERSION.encode(), model.encode(), str(max_window_rooms).encode(),
                  str(window_wall_strips).encode(), image_path.read_bytes(), structure_path.read_bytes()):
        digest.update(value)
    return digest.hexdigest()


def render_review_html(image_path: Path, result: dict[str, Any], output_path: Path,
                       reference_boxes: dict[str, Any] | None = None) -> None:
    image_uri = image_path.resolve().as_uri()
    marks = []
    rows = []
    for space_id, reference in (reference_boxes or {}).items():
        box = reference.get("bbox") if isinstance(reference, dict) else None
        if not _valid_box(box):
            continue
        x1, y1, x2, y2 = box
        marks.append(f'<rect x="{x1*100:.3f}" y="{y1*100:.3f}" width="{(x2-x1)*100:.3f}" height="{(y2-y1)*100:.3f}" fill="none" stroke="#d09b00" stroke-width="0.55" stroke-dasharray="1.2 0.7"/>')
        marks.append(f'<text x="{x1*100:.3f}" y="{max(2,y1*100-0.4):.3f}" fill="#a97900" font-size="2">{html.escape(space_id)}</text>')
    for index, probe in enumerate(result["probes"], 1):
        x1, y1, x2, y2 = probe["crop_bbox"]
        color = "#8e44ad" if probe["kind"] == "washroom" else "#1177cc"
        marks.append(f'<rect x="{x1*100:.3f}" y="{y1*100:.3f}" width="{(x2-x1)*100:.3f}" height="{(y2-y1)*100:.3f}" fill="none" stroke="{color}" stroke-width="0.45"/>')
        marks.append(f'<text x="{x1*100:.3f}" y="{max(2,y1*100-0.4):.3f}" fill="{color}" font-size="2">{index}</text>')
        for candidate in probe["candidates"]:
            status = candidate.get("review_status", "")
            mark_color = "#10a050" if status == "candidate" else "#d24536"
            if _valid_box(candidate.get("bbox")):
                bx = candidate["bbox"]
                marks.append(f'<rect x="{bx[0]*100:.3f}" y="{bx[1]*100:.3f}" width="{(bx[2]-bx[0])*100:.3f}" height="{(bx[3]-bx[1])*100:.3f}" fill="none" stroke="{mark_color}" stroke-width="0.55"/>')
            if _valid_point(candidate.get("position")):
                px, py = candidate["position"]
                marks.append(f'<circle cx="{px*100:.3f}" cy="{py*100:.3f}" r="0.9" fill="{mark_color}"/>')
        summary = ", ".join(f"{item.get('review_status')}: {item.get('evidence', '')}" for item in probe["candidates"]) or "候補なし"
        if probe.get("error"):
            summary = f"エラー: {probe['error']}"
        rows.append(f"<tr><td>{index}</td><td>{html.escape(probe['kind'])}</td><td>{html.escape(probe['space_id'])}</td><td>{html.escape(summary)}</td></tr>")
    page = ("<!doctype html><html lang=ja><meta charset=utf-8><title>小領域読取の確認</title>"
            "<style>body{font-family:sans-serif;max-width:1100px;margin:24px auto;padding:0 16px}"
            ".plan{position:relative}.plan img{width:100%;display:block}.plan svg{position:absolute;inset:0;width:100%;height:100%}"
            "table{border-collapse:collapse;width:100%;margin-top:16px}td,th{border:1px solid #bbb;padding:6px;text-align:left}</style>"
            "<h1>小領域読取の確認</h1><p>紫・青枠は再読取範囲、緑は候補、赤は除外候補、金色の破線は人手確認済み参照範囲です。"
            "候補は採点に自動反映されません。</p>"
            f'<div class="plan"><img src="{html.escape(image_uri, quote=True)}" alt="間取り図">'
            f'<svg viewBox="0 0 100 100" preserveAspectRatio="none">{"".join(marks)}</svg></div>'
            f'<table><tr><th>#</th><th>種別</th><th>空間ID</th><th>結果</th></tr>{"".join(rows)}</table></html>')
    output_path.write_text(page, encoding="utf-8")

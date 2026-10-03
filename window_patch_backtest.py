"""Experimental small-patch window scan and separate labeled-position evaluation."""

import argparse
import html
import json
import math
import os
import re
import sys
from pathlib import Path

from PIL import Image, ImageOps

from floor_plan.cache import structure_cache_key
from floor_plan.providers import OllamaClient
from floor_plan.wall_window import build_patches, scan_cache_key, scan_patches


ROOT = Path(__file__).resolve().parent


def _box_distance_px(point: list[float], box: list[float], size: tuple[int, int]) -> float:
    width, height = size
    return math.hypot(max((box[0] - point[0]) * width, 0, (point[0] - box[2]) * width),
                      max((box[1] - point[1]) * height, 0, (point[1] - box[3]) * height))


def evaluate_detections(result: dict, annotation: dict, size: tuple[int, int], tolerance_px: float = 12) -> dict:
    """Match after inference; annotations are never used to choose patches or prompt the model."""
    references = annotation.get("reference_rectangles", [])
    windows = [item for item in references if item.get("kind") == "window"]
    doors = [item for item in references if item.get("kind") != "window"]
    predicted_windows = [item for item in result["accepted_detections"] if item["kind"] == "window"]
    edges = []
    for predicted_index, prediction in enumerate(predicted_windows):
        box = prediction["bbox"]
        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        for truth_index, reference in enumerate(windows):
            distance = _box_distance_px(center, reference["bbox"], size)
            if distance <= tolerance_px:
                edges.append((distance, predicted_index, truth_index))
    used_predictions = set()
    used_truth = set()
    matches = []
    for distance, predicted_index, truth_index in sorted(edges):
        if predicted_index in used_predictions or truth_index in used_truth:
            continue
        used_predictions.add(predicted_index)
        used_truth.add(truth_index)
        matches.append({"reference_id": windows[truth_index]["id"],
                        "patch_id": predicted_windows[predicted_index]["patch_id"],
                        "distance_px": round(distance, 1)})
    door_false_positives = []
    for prediction in predicted_windows:
        box = prediction["bbox"]
        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        for door in doors:
            if _box_distance_px(center, door["bbox"], size) <= tolerance_px:
                door_false_positives.append({"patch_id": prediction["patch_id"], "door_id": door["id"]})
                break
    return {
        "reference_window_rectangles": len(windows),
        "reference_exterior_door_rectangles": len(doors),
        "raw_window_candidates": sum(item["kind"] == "window" for item in result.get("raw_detections", [])),
        "rejected_window_candidates": sum(item["kind"] == "window" for item in result.get("rejected_detections", [])),
        "accepted_window_candidates": len(predicted_windows),
        "raw_exterior_door_candidates": sum(item["kind"] != "window" for item in result.get("raw_detections", [])),
        "rejected_exterior_door_candidates": sum(item["kind"] != "window" for item in result.get("rejected_detections", [])),
        "accepted_exterior_door_candidates": sum(item["kind"] != "window" for item in result["accepted_detections"]),
        "matched_window_rectangles": len(matches),
        "recall_on_marked_rectangles": round(len(matches) / len(windows), 3) if windows else None,
        "missed_reference_ids": [item["id"] for index, item in enumerate(windows) if index not in used_truth],
        "door_misclassified_as_window": door_false_positives,
        "matches": matches,
        "precision": None,
        "note": "赤枠は物理的な窓の枚数ではなく、網羅性も未確認。再現率は赤枠に対する暫定値で、precisionや正式な全窓精度は出しません。",
    }


def render_review(image_path: Path, result: dict, evaluation: dict, annotation: dict, output: Path) -> None:
    with Image.open(image_path) as image:
        width, height = image.size
    marks = []
    for patch in result["patches"]:
        x1, y1, x2, y2 = patch["pixel_bbox"]
        marks.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="#94a3b8" stroke-width="0.4"/>')
    for reference in annotation.get("reference_rectangles", []):
        box = reference["bbox"]
        x1, y1, x2, y2 = box[0]*width, box[1]*height, box[2]*width, box[3]*height
        color = "#dc2626" if reference["kind"] == "window" else "#7c3aed"
        marks.append(f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{x2-x1:.1f}" height="{y2-y1:.1f}" fill="none" stroke="{color}" stroke-width="1.7"/>')
    for prediction in result.get("rejected_detections", []):
        box = prediction["bbox"]
        x1, y1, x2, y2 = box[0]*width, box[1]*height, box[2]*width, box[3]*height
        marks.append(f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{x2-x1:.1f}" height="{y2-y1:.1f}" fill="none" stroke="#475569" stroke-width="1.5" stroke-dasharray="3 2"/>')
    for prediction in result["accepted_detections"]:
        box = prediction["bbox"]
        x1, y1, x2, y2 = box[0]*width, box[1]*height, box[2]*width, box[3]*height
        color = "#16a34a" if prediction["kind"] == "window" else "#d97706"
        marks.append(f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{x2-x1:.1f}" height="{y2-y1:.1f}" fill="none" stroke="{color}" stroke-width="2"/>')
    summary = (f"窓の赤枠一致 {evaluation['matched_window_rectangles']}/{evaluation['reference_window_rectangles']}、"
               f"採用候補の窓 {evaluation['accepted_window_candidates']}、"
               f"扉の窓誤認 {len(evaluation['door_misclassified_as_window'])}")
    page = ("<!doctype html><html lang=ja><meta charset=utf-8><title>窓パッチ読取の照合</title>"
            "<style>body{font:16px sans-serif;max-width:950px;margin:24px auto;padding:0 16px}.plan{position:relative}"
            ".plan img{width:100%;display:block}.plan svg{position:absolute;inset:0;width:100%;height:100%}</style>"
            "<h1>窓パッチ読取の照合</h1><p>灰色はモデルへ渡したパッチ、赤は利用者の窓赤枠、紫は外部扉赤枠、"
            "緑は採用したモデル窓候補、橙はモデルの外部扉候補、濃灰破線は形状・確信度で除外した候補です。"
            "赤枠は推論前には使用していません。</p>"
            f"<p>{html.escape(summary)}</p>"
            f'<div class="plan"><img src="{html.escape(image_path.resolve().as_uri(), quote=True)}" alt="間取り図">'
            f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none">{"".join(marks)}</svg></div>'
            "<p>この結果は実験用で、採点や正式な全窓ベンチマークには自動反映しません。</p></html>")
    output.write_text(page, encoding="utf-8")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="外壁候補パッチから窓・外部扉を個別に再読取")
    parser.add_argument("--image", type=Path, default=Path("floor_sample/sample1.webp"))
    parser.add_argument("--structure", type=Path, help="構造JSON。省略時は現在の解析キャッシュ")
    parser.add_argument("--annotations", type=Path, default=Path("manual_corrections/sample1_windows_draft.json"))
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen3-vl:8b-instruct-q4_K_M"))
    parser.add_argument("--size-px", type=int, default=144)
    parser.add_argument("--stride-px", type=int, default=96)
    parser.add_argument("--output-dir", type=Path, default=Path("targeted_vision_results"))
    args = parser.parse_args()
    image_path = (ROOT / args.image).resolve()
    if not image_path.is_file():
        parser.error(f"画像がありません: {image_path}")
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
    if args.structure:
        structure_path = (ROOT / args.structure).resolve()
    else:
        key = structure_cache_key(image_path.read_bytes(), "ollama", args.model, num_ctx, 1)
        structure_path = ROOT / ".analysis_cache" / f"{key}.json"
    if not structure_path.is_file():
        parser.error(f"現在の構造キャッシュがありません: {structure_path}")
    payload = json.loads(structure_path.read_text(encoding="utf-8"))
    structure = payload.get("structure", payload)
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    patches = build_patches(image.size, structure.get("plan_regions") or [], args.size_px, args.stride_px)
    name = re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    key = scan_cache_key(image_path.read_bytes(), args.model, patches)
    checkpoint = output_dir / f"{image_path.stem}__{name}.{key[:12]}.wall_patches.json"
    client = OllamaClient(host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
                          timeout=int(os.getenv("OLLAMA_TIMEOUT", "1200")), num_ctx=num_ctx)
    result = scan_patches(image, patches, client, args.model, checkpoint_path=checkpoint, progress=print)
    annotation = json.loads((ROOT / args.annotations).read_text(encoding="utf-8"))
    if annotation.get("image") != image_path.name:
        parser.error("評価用注釈と入力画像の名前が一致しません")
    evaluation = evaluate_detections(result, annotation, image.size)
    run_name = f"{image_path.stem}__{name}__s{args.size_px}_st{args.stride_px}"
    evaluation["scan_settings"] = {"size_px": args.size_px, "stride_px": args.stride_px,
                                   "patch_count": len(patches), "checkpoint": checkpoint.name}
    evaluation["patch_errors"] = sum(item["status"] != "complete" for item in result["patch_results"].values())
    report_path = output_dir / f"{run_name}.wall_patch_eval.json"
    page_path = output_dir / f"{run_name}.wall_patch_review.html"
    report_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    render_review(image_path, result, evaluation, annotation, page_path)
    print(f"照合: {evaluation['matched_window_rectangles']}/{evaluation['reference_window_rectangles']}、"
          f"扉の窓誤認: {len(evaluation['door_misclassified_as_window'])}")
    print(f"レポート: {report_path}\n確認ページ: {page_path}")
    return 0 if all(item["status"] == "complete" for item in result["patch_results"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())

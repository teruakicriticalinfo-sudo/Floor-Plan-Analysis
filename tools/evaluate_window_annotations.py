"""Measure recall on user-marked window points without claiming full-plan accuracy."""

import argparse
import json
import math
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from floor_plan.analyzer import sanitize_visual_features


def evaluate(annotation: dict, structure: dict, image_size: tuple[int, int], tolerance_px: float = 12) -> dict:
    if annotation.get("classification_review_status") != "user_confirmed":
        raise ValueError("利用者が分類を確認した注釈だけを評価します")
    if tolerance_px <= 0:
        raise ValueError("座標許容距離は正の値にしてください")
    marks = [*annotation.get("windows", []),
             *(item for item in annotation.get("user_markers", []) if item.get("kind") == "window")]
    ids = [item.get("id") for item in marks]
    if len(ids) != len(set(ids)) or any(not ident for ident in ids):
        raise ValueError("窓マーカーIDが重複または欠落しています")
    accepted = sanitize_visual_features(structure).get("windows", [])
    width, height = image_size
    choices = []
    for predicted_index, window in enumerate(accepted):
        point = window.get("position")
        if not isinstance(point, list) or len(point) != 2:
            continue
        for marked_index, mark in enumerate(marks):
            target = mark.get("position")
            if not isinstance(target, list) or len(target) != 2:
                raise ValueError(f"注釈座標が不正です: {mark.get('id')}")
            distance = math.hypot((point[0] - target[0]) * width,
                                  (point[1] - target[1]) * height)
            if distance <= tolerance_px:
                choices.append((distance, predicted_index, marked_index))
    used_predictions = set()
    used_marks = set()
    matches = []
    for distance, predicted_index, marked_index in sorted(choices):
        if predicted_index in used_predictions or marked_index in used_marks:
            continue
        used_predictions.add(predicted_index)
        used_marks.add(marked_index)
        matches.append({"marker_id": marks[marked_index]["id"],
                        "model_window_id": accepted[predicted_index].get("id"),
                        "distance_px": round(distance, 1)})
    return {
        "classification": "user_confirmed",
        "coverage": annotation.get("coverage_review_status", "unknown"),
        "geometry": annotation.get("geometry_review_status", "unknown"),
        "annotated_window_markers": len(marks),
        "annotated_exterior_doors_excluded": sum(
            item.get("kind") in {"balcony_door", "garage_window_door"}
            for item in annotation.get("user_markers", [])
        ),
        "raw_model_windows": len(structure.get("windows", [])),
        "accepted_model_windows": len(accepted),
        "matched_markers": len(matches),
        "recall_on_marked_positions": round(len(matches) / len(marks), 3) if marks else None,
        "unmatched_marker_ids": [item["id"] for index, item in enumerate(marks) if index not in used_marks],
        "matches": matches,
        "precision": None,
        "note": "これは利用者が示した窓位置に対する暫定的な再現率です。窓の網羅性・重複が未確認のため、正式な全窓ベンチマークや採点へは転用しません。外部扉は窓の分母に入れません。",
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="利用者が示した窓マーカーに対する暫定再現率を測定")
    parser.add_argument("--annotations", type=Path, default=Path("manual_corrections/sample1_windows_draft.json"))
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=Path("floor_sample/sample1.webp"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance-px", type=float, default=12)
    args = parser.parse_args()
    annotation = json.loads((ROOT / args.annotations).read_text(encoding="utf-8"))
    payload = json.loads((ROOT / args.structure).read_text(encoding="utf-8"))
    structure = payload.get("structure", payload)
    with Image.open(ROOT / args.image) as image:
        result = evaluate(annotation, structure, image.size, args.tolerance_px)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        destination = ROOT / args.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

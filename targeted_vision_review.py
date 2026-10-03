"""Review small-area washroom and window readings without altering scores."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from PIL import Image, ImageOps

from floor_plan.cache import structure_cache_key
from floor_plan.providers import OllamaClient
from floor_plan.targeted_vision import cache_key, render_review_html, run_probes


ROOT = Path(__file__).resolve().parent


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="洗面所と窓を小領域から再読取し、候補を確認ページに表示します。")
    parser.add_argument("--image", default="sample1.webp", help="floor_sample内の画像名")
    parser.add_argument("--input-dir", type=Path, default=Path("floor_sample"))
    parser.add_argument("--structure", type=Path, help="使用する構造化JSON。省略時は現在の画像認識キャッシュ")
    parser.add_argument("--corrections-dir", type=Path, default=Path("manual_corrections"),
                        help="承認済み参照範囲を確認ページに描画する補正JSONのフォルダ")
    parser.add_argument("--output-dir", type=Path, default=Path("targeted_vision_results"))
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen3-vl:8b-instruct-q4_K_M"))
    parser.add_argument("--max-window-rooms", type=int, default=4)
    parser.add_argument("--whole-room-windows", action="store_true",
                        help="外壁帯ではなく部屋全体を切り出して窓を探す")
    parser.add_argument("--refresh", action="store_true", help="保存済み小領域結果を使わず再読取")
    args = parser.parse_args()
    if args.max_window_rooms < 0 or args.max_window_rooms > 12:
        parser.error("--max-window-rooms は0〜12にしてください")
    image_path = (ROOT / args.input_dir / args.image).resolve()
    if not image_path.is_file() or image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        parser.error(f"画像が見つかりません: {image_path}")
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
    current_key = structure_cache_key(image_path.read_bytes(), "ollama", args.model, num_ctx, 1)
    current_cache = ROOT / ".analysis_cache" / f"{current_key}.json"
    structure_path = (ROOT / args.structure).resolve() if args.structure else current_cache
    if not structure_path.is_file():
        parser.error(f"現在の構造キャッシュがありません。先にlive_backtest.pyを実行するか --structure を指定してください: {structure_path}")
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = output_dir / f"{image_path.stem}__{safe_model}.targeted.json"
    output_html = output_dir / f"{image_path.stem}__{safe_model}.targeted.html"
    wall_strips = not args.whole_room_windows
    key = cache_key(image_path, structure_path, args.model, args.max_window_rooms, wall_strips)
    result = None
    if output_json.is_file() and not args.refresh:
        saved = json.loads(output_json.read_text(encoding="utf-8"))
        if saved.get("cache_key") == key:
            result = saved
            print("CACHE HIT: 小領域の画像認識結果を再利用")
    if result is None:
        print("CACHE MISS: 小領域の画像認識を実行")
        payload = json.loads(structure_path.read_text(encoding="utf-8"))
        structure = payload.get("structure", payload)
        with Image.open(image_path) as loaded:
            image = ImageOps.exif_transpose(loaded).convert("RGB")
        client = OllamaClient(
            host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "1200")),
            num_ctx=num_ctx,
        )
        result = run_probes(image, structure, client, args.model,
                            max_window_rooms=args.max_window_rooms,
                            window_wall_strips=wall_strips, progress=print)
        result["cache_key"] = key
        result["image"] = image_path.name
        output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    reference_boxes = None
    corrections_path = (ROOT / args.corrections_dir / f"{image_path.stem}.json").resolve()
    if corrections_path.is_file():
        corrections = json.loads(corrections_path.read_text(encoding="utf-8"))
        if corrections.get("bbox_review_status") == "approved":
            reference_boxes = corrections.get("spaces")
    render_review_html(image_path, result, output_html, reference_boxes)
    probes = result["probes"]
    print(f"読取範囲 {len(probes)}、確認候補 {sum(item.get('review_status') == 'candidate' for p in probes for item in p['candidates'])}、エラー {sum(p['status'] == 'error' for p in probes)}")
    print(f"確認ページ: {output_html}")
    print(f"詳細JSON: {output_json}")
    return 0 if not any(p["status"] == "error" for p in probes) else 2


if __name__ == "__main__":
    raise SystemExit(main())

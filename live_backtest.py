"""Run the analyzer against every supported image in a chosen folder."""

import argparse
import os
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageOps

from floor_plan import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    SCORE_CRITERIA,
    analyze_cached_structure,
    analyze_floor_plan,
    create_analysis_client,
    load_knowledge,
    load_structure_cache,
    save_structure_cache,
    structure_cache_key,
)


APP_DIR = Path(__file__).resolve().parent
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def safe_model_name(model: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", model).strip("_") or "model"


def check_response(response: str) -> list[str]:
    """Return basic structural problems found in a model response."""
    problems = []
    total_match = re.search(r"総合評価\s*[:：]\s*(\d+)\s*/\s*100", response)
    if total_match is None:
        problems.append("総合点を抽出できない")
    elif not 0 <= int(total_match.group(1)) <= 100:
        problems.append("総合点が0〜100の範囲外")

    for name, _, _ in SCORE_CRITERIA:
        if name not in response:
            problems.append(f"採点項目が不足: {name}")

    for heading in ("読み取り条件", "採点表", "良い点", "気になる点", "改善案", "専門家に確認すべき事項"):
        if heading not in response:
            problems.append(f"出力セクションが不足: {heading}")
    return problems


def resolve_app_path(value: Path) -> Path:
    """Resolve relative CLI paths from the project directory."""
    return value if value.is_absolute() else APP_DIR / value


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="間取り画像を構造化して採点します。")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("floor_photo"),
        help="分析する画像フォルダ（既定: floor_photo）",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("backtest_results"),
        help="結果Markdown/JSONの保存先（既定: backtest_results）",
    )
    parser.add_argument("--no-cache", action="store_true", help="キャッシュを読み書きしない")
    parser.add_argument("--refresh-cache", action="store_true", help="画像認識をやり直してキャッシュを更新する")
    args = parser.parse_args()

    load_dotenv(APP_DIR / ".env")
    image_dir = resolve_app_path(args.input_dir)
    if not image_dir.is_dir():
        print(f"ERROR: 画像フォルダがありません: {image_dir}", file=sys.stderr)
        return 1
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not image_paths:
        print("ERROR: floor_photoに対応画像がありません。", file=sys.stderr)
        return 1

    knowledge = load_knowledge(APP_DIR / "knowledge.md")
    provider = os.getenv("ANALYSIS_PROVIDER", "ollama").strip().lower()
    default_model = DEFAULT_MODEL if provider == "ollama" else "gemini-2.5-flash"
    model = os.getenv("OLLAMA_MODEL" if provider == "ollama" else "GEMINI_MODEL", default_model).strip() or default_model
    fallback = os.getenv("ENABLE_GEMINI_FALLBACK", "false").strip().lower() in {"1", "true", "yes"}
    ollama_num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
    ollama_max_images = int(os.getenv("OLLAMA_MAX_IMAGES", "1"))
    ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT", "1200"))
    try:
        client = create_analysis_client(
            provider,
            ollama_host=os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
            ollama_timeout=ollama_timeout,
            ollama_num_ctx=ollama_num_ctx,
            ollama_max_images=ollama_max_images,
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            enable_gemini_fallback=fallback,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    result_dir = resolve_app_path(args.results_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = APP_DIR / ".analysis_cache"
    failed = False

    for image_path in image_paths:
        print(f"===== {image_path.name} =====")
        try:
            started = time.perf_counter()
            key = structure_cache_key(image_path.read_bytes(), provider, model, ollama_num_ctx, ollama_max_images)
            cache_path = cache_dir / f"{key}.json"
            cached = None if args.no_cache or args.refresh_cache else load_structure_cache(cache_path)
            if cached:
                print(f"CACHE HIT: 画像認識を省略 ({cache_path.name})")
                result = analyze_cached_structure(cached[0], cached[1], client, knowledge, model)
                cache_status = "hit"
            else:
                print("CACHE MISS: 画像認識を実行")
                with Image.open(image_path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                result = analyze_floor_plan(image, client, knowledge, model)
                cache_status = "disabled" if args.no_cache else "miss"
                if not args.no_cache:
                    save_structure_cache(
                        cache_path,
                        result.draft_structure,
                        result.structure,
                        {"image": image_path.name, "provider": provider, "model": model, "num_ctx": ollama_num_ctx, "max_images": ollama_max_images},
                    )
            elapsed = time.perf_counter() - started
            problems = check_response(result.report)
            print("----- DRAFT STRUCTURE -----")
            print(json.dumps(result.draft_structure, ensure_ascii=False, indent=2))
            print("\n----- VERIFIED STRUCTURE -----")
            print(json.dumps(result.structure, ensure_ascii=False, indent=2))
            print("\n----- ANALYSIS -----")
            print(result.report)
            print("\n----- FORMAT CHECK -----")
            if problems:
                failed = True
                for problem in problems:
                    print(f"NG: {problem}")
                check_text = "\n".join(f"- NG: {problem}" for problem in problems)
            else:
                print("OK: 必須の採点項目とセクションを確認")
                check_text = "- OK: 必須の採点項目とセクションを確認"

            report = (
                f"# 実画像バックテスト: {image_path.name}\n\n"
                f"- 使用モデル: `{model}`\n"
                f"- プロバイダー: `{client.provider_name}`\n"
                f"- 知識ファイル: `knowledge.md` 全文\n\n"
                f"- 構造キャッシュ: `{cache_status}`\n"
                f"- 処理時間: `{elapsed:.1f}秒`\n\n"
                f"## 第1段階：最初の読取結果\n\n"
                f"```json\n{json.dumps(result.draft_structure, ensure_ascii=False, indent=2)}\n```\n\n"
                f"## 第1段階：再照合後の読取結果\n\n"
                f"```json\n{json.dumps(result.structure, ensure_ascii=False, indent=2)}\n```\n\n"
                f"## 第2段階：採点結果\n\n"
                f"### 検証済み採点JSON\n\n"
                f"```json\n{json.dumps(result.scoring, ensure_ascii=False, indent=2)}\n```\n\n"
                f"### 確定レポート\n\n"
                f"{result.report}\n\n"
                f"## 自動形式チェック\n\n{check_text}\n"
            )
            result_stem = f"{image_path.stem}__{safe_model_name(model)}"
            result_path = result_dir / f"{result_stem}.md"
            result_path.write_text(report, encoding="utf-8")
            structure_path = result_dir / f"{result_stem}.structure.json"
            structure_path.write_text(
                json.dumps(result.structure, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"保存先: {result_path}")
            print(f"処理時間: {elapsed:.1f}秒 / cache={cache_status}")
        except Exception as exc:
            failed = True
            print(f"ERROR: {type(exc).__name__}: {exc}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

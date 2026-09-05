"""Run the analyzer against every supported image in floor_photo."""

import os
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageOps

from floor_plan import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_HOST,
    SCORE_CRITERIA,
    analyze_floor_plan,
    create_analysis_client,
    load_knowledge,
)


APP_DIR = Path(__file__).resolve().parent
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


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


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    load_dotenv(APP_DIR / ".env")
    image_dir = APP_DIR / "floor_photo"
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
    try:
        client = create_analysis_client(
            provider,
            ollama_host=os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
            ollama_num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "16384")),
            ollama_max_images=int(os.getenv("OLLAMA_MAX_IMAGES", "1")),
            gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            enable_gemini_fallback=fallback,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    result_dir = APP_DIR / "backtest_results"
    result_dir.mkdir(exist_ok=True)
    failed = False

    for image_path in image_paths:
        print(f"===== {image_path.name} =====")
        try:
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            result = analyze_floor_plan(image, client, knowledge, model)
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
            result_path = result_dir / f"{image_path.stem}.md"
            result_path.write_text(report, encoding="utf-8")
            print(f"保存先: {result_path}")
        except Exception as exc:
            failed = True
            print(f"ERROR: {type(exc).__name__}: {exc}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

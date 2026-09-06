"""Create or update human-review templates for a local floor-plan benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def reviewed_connection_labels(structure: dict) -> list[list[str]]:
    """Return the current accepted edges as a *draft*, never as approved truth."""
    spaces = {item.get("id"): item for item in structure.get("spaces", [])}
    labels: set[tuple[str, str]] = set()
    for connection in structure.get("connections", []):
        if connection.get("traversable") is not True:
            continue
        if connection.get("validation_status") != "accepted":
            continue
        first = str(spaces.get(connection.get("space_a"), {}).get("label", "")).strip()
        second = str(spaces.get(connection.get("space_b"), {}).get("label", "")).strip()
        if first and second and first != second:
            labels.add(tuple(sorted((first, second))))
    return [list(pair) for pair in sorted(labels)]


def result_for_image(structure_dir: Path, image_path: Path) -> Path | None:
    matches = sorted(structure_dir.glob(f"{image_path.stem}__*.structure.json"))
    if len(matches) > 1:
        raise RuntimeError(
            f"{image_path.name} に対応する構造JSONが複数あります。モデル別の結果フォルダを指定してください: {matches}"
        )
    return matches[0] if matches else None


def make_template(image_path: Path, structure_path: Path | None) -> dict:
    template = {
        "image": image_path.name,
        "review_status": "pending",
        "notes": "画像を見て、収納を除く直接通行可能な接続だけを connections に記入してください。",
        "ignore_space_types": ["storage"],
        "connections": [],
        "label_aliases": {},
    }
    if structure_path is not None:
        structure = json.loads(structure_path.read_text(encoding="utf-8"))
        template["review_status"] = "draft"
        template["notes"] = (
            "AIの構造認識結果を下書きとして入れています。画像を見て接続、表記、"
            "label_aliases を必ず確認し、完了後に review_status を approved に変更してください。"
        )
        template["connections"] = reviewed_connection_labels(structure)
        template["draft_source"] = structure_path.name
    return template


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="ローカル間取り画像の正解データ用テンプレートを作成します。")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument(
        "--structure-dir",
        type=Path,
        help="分析済み .structure.json のフォルダ。指定すると接続候補を下書きに使います。",
    )
    parser.add_argument("--overwrite", action="store_true", help="既存テンプレートを上書きする")
    args = parser.parse_args()

    if not args.images_dir.is_dir():
        parser.error(f"画像フォルダがありません: {args.images_dir}")
    if args.structure_dir is not None and not args.structure_dir.is_dir():
        parser.error(f"構造JSONフォルダがありません: {args.structure_dir}")

    images = sorted(path for path in args.images_dir.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES)
    if not images:
        parser.error("対応する画像がありません")
    args.ground_truth_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    preserved = 0
    manifest_images = []
    for image_path in images:
        truth_path = args.ground_truth_dir / f"{image_path.stem}.json"
        manifest_images.append({"image": image_path.name, "ground_truth": truth_path.name})
        if truth_path.exists() and not args.overwrite:
            preserved += 1
            continue
        structure_path = result_for_image(args.structure_dir, image_path) if args.structure_dir else None
        truth_path.write_text(
            json.dumps(make_template(image_path, structure_path), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        created += 1

    manifest = {
        "name": args.ground_truth_dir.name,
        "image_count": len(images),
        "review_instruction": "各JSONで connections を人手確認し、完了したものだけ review_status を approved に変更します。",
        "images": manifest_images,
    }
    (args.ground_truth_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"テンプレート作成: {created}件 / 既存保持: {preserved}件 / 合計: {len(images)}件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate every approved ground-truth file against matching structure JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluate_ground_truth import evaluate


def structure_for_truth(structure_dir: Path, truth: dict, model: str | None) -> Path | None:
    image = Path(str(truth["image"]))
    pattern = f"{image.stem}__{model}.structure.json" if model else f"{image.stem}__*.structure.json"
    matches = sorted(structure_dir.glob(pattern))
    if len(matches) > 1:
        raise RuntimeError(
            f"{image.name} に対応する構造JSONが複数あります。--model で対象モデルを指定してください: {matches}"
        )
    return matches[0] if matches else None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="確認済み正解データを一括評価します。")
    parser.add_argument("--structure-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--model", help="結果ファイル名に含まれる安全化済みモデル名")
    parser.add_argument("--output", type=Path, help="集計JSONの保存先")
    args = parser.parse_args()
    if not args.structure_dir.is_dir() or not args.ground_truth_dir.is_dir():
        parser.error("structure-dir と ground-truth-dir は既存フォルダを指定してください")

    rows = []
    skipped = []
    for truth_path in sorted(args.ground_truth_dir.glob("*.json")):
        if truth_path.name == "benchmark_manifest.json":
            continue
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        if truth.get("review_status") != "approved":
            skipped.append({"ground_truth": truth_path.name, "reason": "未確認"})
            continue
        structure_path = structure_for_truth(args.structure_dir, truth, args.model)
        if structure_path is None:
            skipped.append({"ground_truth": truth_path.name, "reason": "構造JSONなし"})
            continue
        result = evaluate(json.loads(structure_path.read_text(encoding="utf-8")), truth)
        rows.append({"image": truth["image"], "ground_truth": truth_path.name, "structure": structure_path.name, **result})

    true_positive = sum(len(row["true_positive"]) for row in rows)
    false_positive = sum(len(row["false_positive"]) for row in rows)
    false_negative = sum(len(row["false_negative"]) for row in rows)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    report = {
        "approved_count": len(rows),
        "skipped_count": len(skipped),
        "micro_average": {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        },
        "per_image": rows,
        "skipped": skipped,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Apply a reviewed floor-plan correction file to a saved structure JSON."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from floor_plan.corrections import apply_reviewed_corrections


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="画像認識後の部屋位置・接続を人手確認データで補正します。")
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-draft", action="store_true", help="未承認の下書きを試算する。正式な評価には使わない")
    args = parser.parse_args()
    if args.output.resolve() in {args.structure.resolve(), args.corrections.resolve()}:
        parser.error("出力先は入力ファイルとは別にしてください")
    try:
        structure = json.loads(args.structure.read_text(encoding="utf-8"))
        corrections = json.loads(args.corrections.read_text(encoding="utf-8"))
        corrected = apply_reviewed_corrections(structure, corrections, allow_draft=args.preview_draft)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(corrected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"保存先: {args.output}")
    print(f"修正状態: {corrected['manual_correction_status']}")
    print(f"採用した通行可能接続: {len(corrected['verified_topology']['direct_connections'])}件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

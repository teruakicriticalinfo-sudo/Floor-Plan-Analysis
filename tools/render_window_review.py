"""Render a numbered, unapproved window-candidate overlay for human review."""

import argparse
import html
import json
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def render(annotation: dict, image_path: Path) -> str:
    if annotation.get("review_status") != "draft":
        raise ValueError("この確認ページは未承認の候補JSONだけを表示します")
    if Path(str(annotation.get("image", ""))).name != image_path.name:
        raise ValueError("候補JSONと画像のファイル名が一致しません")
    with Image.open(image_path) as image:
        width, height = image.size
    marks = []
    rows = []
    seen = set()
    for item in annotation.get("windows", []):
        ident = item.get("id")
        point = item.get("position")
        if not ident or ident in seen or not isinstance(point, list) or len(point) != 2 or not all(
            isinstance(value, (float, int)) and not isinstance(value, bool) and 0 <= value <= 1 for value in point
        ):
            raise ValueError(f"窓候補IDまたは座標が不正です: {ident}")
        seen.add(ident)
        x, y = point[0] * width, point[1] * height
        marks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#f59e0b" fill-opacity=".45" stroke="#a33b00" stroke-width="2"/>')
        marks.append(f'<text x="{x+9:.1f}" y="{y-9:.1f}" fill="#9a3100" font-size="15" font-weight="bold">{html.escape(str(ident))}</text>')
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            ident, item.get("floor_id", ""), item.get("room", ""), item.get("description", ""),
        )) + "</tr>")
    return ("<!doctype html><html lang=ja><meta charset=utf-8><title>窓位置の確認候補</title>"
            "<style>body{font:16px sans-serif;max-width:950px;margin:24px auto;padding:0 16px;color:#222}"
            ".plan{position:relative;max-width:100%}.plan img{width:100%;display:block}.plan svg{position:absolute;inset:0;width:100%;height:100%}"
            "table{border-collapse:collapse;width:100%;margin-top:20px}th,td{border:1px solid #bbb;padding:7px;text-align:left}</style>"
            "<h1>sample1 窓位置の確認候補</h1>"
            "<p>橙丸は画像を目視して選んだ未承認の候補です。Qwenの検出結果でも正解データでもありません。"
            "各番号が窓を指しているか確認してください。候補以外の窓があれば、その位置も教えてください。</p>"
            f'<div class="plan"><img src="{html.escape(image_path.resolve().as_uri(), quote=True)}" alt="間取り図">'
            f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none">{"".join(marks)}</svg></div>'
            f'<table><tr><th>番号</th><th>階</th><th>部屋</th><th>候補の説明</th></tr>{"".join(rows)}</table>'
            "<p>このページを見て、例えば「W1・W3・W5は窓、W2・W4は違う。ほかに…」のようにお知らせください。"
            "承認があるまで分析結果やベンチマークの正解に反映しません。</p></html>")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="未承認の窓候補を元画像に重ねた確認ページを作成")
    parser.add_argument("--annotations", type=Path, default=Path("manual_corrections/sample1_windows_draft.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("floor_sample"))
    parser.add_argument("--output", type=Path, default=Path("manual_corrections/sample1_window_review.html"))
    args = parser.parse_args()
    annotation_path = (ROOT / args.annotations).resolve()
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    image_path = (ROOT / args.images_dir / annotation["image"]).resolve()
    output_path = (ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render(annotation, image_path), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

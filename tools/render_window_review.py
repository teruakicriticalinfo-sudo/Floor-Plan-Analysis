"""Render a numbered, unapproved window-candidate overlay for human review."""

import argparse
import html
import json
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MARKER_KINDS = {
    "window": "窓",
    "balcony_door": "ベランダに出る扉",
    "garage_window_door": "車庫に出られる窓扉",
}


def _check_point(item: dict, seen: set[str]) -> tuple[str, list[float]]:
    ident = item.get("id")
    point = item.get("position")
    if not ident or ident in seen or not isinstance(point, list) or len(point) != 2 or not all(
        isinstance(value, (float, int)) and not isinstance(value, bool) and 0 <= value <= 1 for value in point
    ):
        raise ValueError(f"印のIDまたは座標が不正です: {ident}")
    seen.add(ident)
    return str(ident), point


def render(annotation: dict, image_path: Path) -> str:
    if annotation.get("review_status") != "draft":
        raise ValueError("この確認ページは未承認の候補JSONだけを表示します")
    if Path(str(annotation.get("image", ""))).name != image_path.name:
        raise ValueError("候補JSONと画像のファイル名が一致しません")
    with Image.open(image_path) as image:
        width, height = image.size
    marks = []
    candidate_rows = []
    marker_rows = []
    seen = set()
    for item in annotation.get("windows", []):
        ident, point = _check_point(item, seen)
        x, y = point[0] * width, point[1] * height
        marks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="#f59e0b" fill-opacity=".45" stroke="#a33b00" stroke-width="2"/>')
        marks.append(f'<text x="{x+9:.1f}" y="{y-9:.1f}" fill="#9a3100" font-size="15" font-weight="bold">{html.escape(str(ident))}</text>')
        candidate_rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            ident, item.get("floor_id", ""), item.get("room", ""), item.get("description", ""),
        )) + "</tr>")
    for item in annotation.get("user_markers", []):
        ident, point = _check_point(item, seen)
        kind = item.get("kind")
        if kind not in MARKER_KINDS:
            raise ValueError(f"利用者の印の種別が不正です: {ident}")
        x, y = point[0] * width, point[1] * height
        color = "#1769dd" if kind == "window" else "#7b2cbf"
        marks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}" stroke="white" stroke-width="1.5"/>')
        marks.append(f'<text x="{x+8:.1f}" y="{y-7:.1f}" fill="{color}" font-size="12" font-weight="bold">{html.escape(ident)}</text>')
        marker_rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            ident, item.get("comment", ""), MARKER_KINDS[kind],
        )) + "</tr>")
    return ("<!doctype html><html lang=ja><meta charset=utf-8><title>窓位置の確認候補</title>"
            "<style>body{font:16px sans-serif;max-width:950px;margin:24px auto;padding:0 16px;color:#222}"
            ".plan{position:relative;max-width:100%}.plan img{width:100%;display:block}.plan svg{position:absolute;inset:0;width:100%;height:100%}"
            "table{border-collapse:collapse;width:100%;margin-top:20px}th,td{border:1px solid #bbb;padding:7px;text-align:left}</style>"
            "<h1>sample1 窓・外部扉の確認</h1>"
            "<p>橙のW1〜W5は作業者の目視による未承認候補です。青のM1〜M12は利用者が窓とコメントした位置、"
            "紫のM13〜M15は利用者が外部へ出る扉とコメントした位置です。利用者コメントの座標は画面上の印から概算変換したため、"
            "数ピクセルずれる可能性があります。いずれもQwenの検出結果ではありません。</p>"
            f'<div class="plan"><img src="{html.escape(image_path.resolve().as_uri(), quote=True)}" alt="間取り図">'
            f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none">{"".join(marks)}</svg></div>'
            f'<h2>作業者の未承認候補</h2><table><tr><th>番号</th><th>階</th><th>部屋</th><th>候補の説明</th></tr>{"".join(candidate_rows)}</table>'
            f'<h2>利用者が付けた印</h2><table><tr><th>番号</th><th>コメント番号</th><th>利用者の分類</th></tr>{"".join(marker_rows)}</table>'
            "<p>W1〜W5にも窓かどうかの確認が必要です。M1〜M15の位置にずれがあればお知らせください。"
            "確認が終わるまで採点やベンチマークの正解に反映しません。</p></html>")


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

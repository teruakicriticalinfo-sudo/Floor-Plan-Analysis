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
        confirmed = annotation.get("classification_review_status") == "user_confirmed"
        color = "#16803c" if confirmed else "#f59e0b"
        text_color = "#086328" if confirmed else "#9a3100"
        marks.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{color}" fill-opacity=".45" stroke="{text_color}" stroke-width="2"/>')
        marks.append(f'<text x="{x+9:.1f}" y="{y-9:.1f}" fill="{text_color}" font-size="15" font-weight="bold">{html.escape(str(ident))}</text>')
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
            "<p>緑のW1〜W5は利用者が窓と確認した元候補です。青のM1〜M12も利用者が窓とコメントした位置、"
            "紫のM13〜M15は利用者が外部へ出る扉とコメントした位置です。M系列の位置は画面上の印から"
            "元画像へ変換し、近くの壁・開口線へ補間しました。分類は確認済みですが、全窓の網羅性・重複は未確認です。"
            "いずれもQwenの検出結果ではありません。</p>"
            f'<div class="plan"><img src="{html.escape(image_path.resolve().as_uri(), quote=True)}" alt="間取り図">'
            f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none">{"".join(marks)}</svg></div>'
            f'<h2>元候補（利用者が窓と確認）</h2><table><tr><th>番号</th><th>階</th><th>部屋</th><th>位置の説明</th></tr>{"".join(candidate_rows)}</table>'
            f'<h2>利用者が付けた印</h2><table><tr><th>番号</th><th>コメント番号</th><th>利用者の分類</th></tr>{"".join(marker_rows)}</table>'
            "<p>このページは分類を記録した作業下書きです。網羅性と重複を確認するまで、"
            "採点や正式なベンチマークの正解には反映しません。</p></html>")


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

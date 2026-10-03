"""Generate a label-free wall-gap candidate scan and a local human-review page."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import mimetypes
import sys
from pathlib import Path

from PIL import Image

from floor_plan.window_cv import DETECTOR_VERSION, detect_wall_gap_candidates


ROOT = Path(__file__).resolve().parent


def _distance_to_box(point: tuple[float, float], box: list[float], size: tuple[int, int]) -> float:
    width, height = size
    dx = max((box[0] - point[0]) * width, 0, (point[0] - box[2]) * width)
    dy = max((box[1] - point[1]) * height, 0, (point[1] - box[3]) * height)
    return math.hypot(dx, dy)


def evaluate_candidate_coverage(candidates: list[dict], annotation: dict,
                                image_size: tuple[int, int], tolerance_px: float = 12) -> dict:
    """Measure marked-position coverage only after candidate extraction is done."""
    if annotation.get("classification_review_status") != "user_confirmed":
        raise ValueError("利用者が分類を確認した赤枠だけを照合します")
    if tolerance_px <= 0:
        raise ValueError("照合許容距離が不正です")
    references = annotation.get("reference_rectangles") or []
    windows = [item for item in references if item.get("kind") == "window"]
    doors = [item for item in references if item.get("kind") in {"balcony_door", "garage_window_door"}]
    edges = []
    for candidate_index, candidate in enumerate(candidates):
        box = candidate["bbox"]
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        for reference_index, reference in enumerate(windows):
            distance = _distance_to_box(center, reference["bbox"], image_size)
            if distance <= tolerance_px:
                edges.append((distance, candidate_index, reference_index))
    used_candidates = set()
    used_windows = set()
    matches = []
    for distance, candidate_index, reference_index in sorted(edges):
        if candidate_index in used_candidates or reference_index in used_windows:
            continue
        used_candidates.add(candidate_index)
        used_windows.add(reference_index)
        matches.append({"reference_id": windows[reference_index]["id"],
                        "candidate_id": candidates[candidate_index]["id"],
                        "distance_px": round(distance, 1)})
    covered_doors = []
    for door in doors:
        if any(_distance_to_box(((candidate["bbox"][0] + candidate["bbox"][2]) / 2,
                                 (candidate["bbox"][1] + candidate["bbox"][3]) / 2),
                                door["bbox"], image_size) <= tolerance_px for candidate in candidates):
            covered_doors.append(door["id"])
    return {"detector_version": DETECTOR_VERSION,
            "candidate_count": len(candidates),
            "marked_window_rectangles": len(windows),
            "covered_marked_windows": len(matches),
            "marked_window_coverage": round(len(matches) / len(windows), 3) if windows else None,
            "missed_reference_ids": [item["id"] for index, item in enumerate(windows)
                                     if index not in used_windows],
            "marked_exterior_door_rectangles": len(doors),
            "covered_exterior_door_ids": covered_doors,
            "matches": matches,
            "precision": None,
            "note": "候補は窓と断定していません。sample1は検出器の開発に使った画像で、位置一致は独立評価ではありません。赤枠の網羅性も未確認のためprecisionは計算しません。"}


def render_review_page(image_bytes: bytes, image_name: str, image_size: tuple[int, int],
                       candidates: list[dict], evaluation: dict,
                       annotation: dict, output: Path) -> None:
    width, height = image_size
    fingerprint = hashlib.sha256(image_bytes).hexdigest()
    image_mime = mimetypes.guess_type(image_name)[0] or "application/octet-stream"
    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    candidate_shapes = []
    rows = []
    for candidate in candidates:
        x1, y1, x2, y2 = candidate["pixel_bbox"]
        ident = candidate["id"]
        candidate_shapes.append(
            f'<g class="candidate" data-id="{ident}">'
            f'<rect class="outline" x="{x1-2}" y="{y1-2}" width="{x2-x1+4}" height="{y2-y1+4}"/>'
            f'<rect class="hitbox" x="{x1-6}" y="{y1-6}" width="{x2-x1+12}" height="{y2-y1+12}"/>'
            f'<text x="{x1+3}" y="{max(9,y1-4)}">{ident[1:]}</text></g>')
        direction = "縦" if candidate["orientation"] == "vertical" else "横"
        rows.append(
            f'<label class="review-row" id="row-{ident}" data-id="{ident}">'
            f'<span><strong>{ident}</strong> {direction} {x1},{y1}–{x2},{y2}</span>'
            f'<select aria-label="{ident} の判定" data-id="{ident}">'
            '<option value="unreviewed">未判定</option>'
            '<option value="window">窓</option>'
            '<option value="exterior_door">外部扉</option>'
            '<option value="not_opening">窓・扉ではない</option>'
            '<option value="uncertain">判別不能</option>'
            '</select></label>')
    reference_shapes = []
    for reference in annotation.get("reference_rectangles", []):
        box = reference["bbox"]
        x1, y1, x2, y2 = box[0] * width, box[1] * height, box[2] * width, box[3] * height
        color = "#dc2626" if reference["kind"] == "window" else "#7c3aed"
        reference_shapes.append(
            f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{x2-x1:.1f}" height="{y2-y1:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="1.5"/>')
    data = {"image": image_name, "image_sha256": fingerprint,
            "detector_version": DETECTOR_VERSION,
            "candidates": [{"id": candidate["id"], "bbox": candidate["bbox"],
                            "orientation": candidate["orientation"]} for candidate in candidates],
            "candidate_ids": [candidate["id"] for candidate in candidates]}
    script_data = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    coverage_text = (f"{evaluation['covered_marked_windows']}/{evaluation['marked_window_rectangles']}"
                     if evaluation["marked_window_rectangles"] is not None else "赤枠照合なし")
    page = f"""<!doctype html>
<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>窓候補の確認 — {html.escape(image_name)}</title>
<style>
body{{font:16px/1.55 system-ui,sans-serif;margin:0;color:#17212f;background:#f7f9fc}}
header{{padding:16px 24px;background:white;border-bottom:1px solid #d8dfe8}}
h1{{font-size:1.5rem;margin:0 0 4px}}p{{margin:6px 0}}
main{{display:grid;grid-template-columns:minmax(0,2fr) minmax(300px,1fr);gap:18px;padding:18px}}
.panel{{background:white;border:1px solid #d8dfe8;border-radius:10px;padding:12px}}
.plan{{position:relative;width:100%}}.plan img{{width:100%;display:block;image-rendering:auto}}
.plan svg{{position:absolute;inset:0;width:100%;height:100%}}
.candidate .outline{{fill:none;stroke:#0876bf;stroke-width:1.6}}
.candidate .hitbox{{fill:transparent;cursor:pointer}}
.candidate text{{font-size:7px;font-weight:700;fill:#004a80;pointer-events:none;paint-order:stroke;stroke:white;stroke-width:2px}}
.candidate[data-status="window"] .outline{{stroke:#159447;stroke-width:2.3}}
.candidate[data-status="exterior_door"] .outline{{stroke:#bd6b00;stroke-width:2.3}}
.candidate[data-status="not_opening"] .outline{{stroke:#9ca3af}}
.candidate[data-status="uncertain"] .outline{{stroke:#8b5cf6;stroke-width:2.3}}
.candidate.selected .outline{{stroke-width:3.5}}
.review-list{{max-height:70vh;overflow:auto;display:grid;gap:4px}}
.review-row{{display:flex;align-items:center;justify-content:space-between;gap:6px;padding:5px 7px;border:1px solid #e3e8ef;border-radius:5px;font-size:13px}}
.review-row.selected{{background:#eff7ff;border-color:#0876bf}}
select,button{{font:inherit;padding:4px 6px}}button{{cursor:pointer}}
.controls{{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:8px 0}}
.small{{font-size:13px;color:#475569}}
@media(max-width:900px){{main{{grid-template-columns:1fr}}.review-list{{max-height:none}}}}
</style>
<header><h1>窓候補の確認 — {html.escape(image_name)}</h1>
<p>青は画像処理が見つけた「壁線の切れ目」です。窓とは限りません。候補をクリックして分類してください。</p>
<p class="small">このページは採点を変更しません。判定はブラウザ内に一時保存され、「確認結果JSONを保存」で書き出せます。</p></header>
<main><section class="panel"><div class="controls">
<label><input id="show-reference" type="checkbox"> 元の赤枠を表示（赤＝窓、紫＝外部扉）</label>
<span>候補 {len(candidates)}件 / 開発用赤枠の位置一致 {coverage_text}</span>
</div><div class="plan"><img src="data:{image_mime};base64,{encoded_image}" alt="間取り図">
<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none">
<g id="reference-layer" style="display:none">{''.join(reference_shapes)}</g>
{''.join(candidate_shapes)}</svg></div>
<p class="small">位置一致はsample1での開発中の値です。窓の自動認定や独立した精度評価ではありません。</p></section>
<aside class="panel"><div class="controls"><strong id="progress"></strong>
<button id="export" type="button">確認結果JSONを保存</button>
<label>JSONを読み込む<input id="import" type="file" accept=".json,application/json"></label></div>
<div class="review-list">{''.join(rows)}</div></aside></main>
<script>
const meta = {script_data};
const valid = new Set(['unreviewed','window','exterior_door','not_opening','uncertain']);
const storageKey = `window-cv-review:${{meta.image_sha256}}:${{meta.detector_version}}`;
let decisions = {{}};
try {{ decisions = JSON.parse(localStorage.getItem(storageKey) || '{{}}'); }} catch (_) {{ decisions = {{}}; }}
function render() {{
  let reviewed = 0;
  for (const id of meta.candidate_ids) {{
    const choice = valid.has(decisions[id]) ? decisions[id] : 'unreviewed';
    if (choice !== 'unreviewed') reviewed++;
    document.querySelector(`select[data-id="${{id}}"]`).value = choice;
    document.querySelector(`g.candidate[data-id="${{id}}"]`).dataset.status = choice;
  }}
  document.getElementById('progress').textContent = `判定済み ${{reviewed}}/${{meta.candidate_ids.length}}`;
}}
for (const select of document.querySelectorAll('select[data-id]')) {{
  select.addEventListener('change', () => {{
    decisions[select.dataset.id] = select.value;
    try {{ localStorage.setItem(storageKey, JSON.stringify(decisions)); }} catch (_) {{}}
    render();
  }});
}}
for (const shape of document.querySelectorAll('g.candidate')) {{
  shape.addEventListener('click', () => {{
    document.querySelectorAll('.selected').forEach(node => node.classList.remove('selected'));
    const row = document.getElementById(`row-${{shape.dataset.id}}`);
    row.classList.add('selected'); shape.classList.add('selected');
    row.scrollIntoView({{behavior:'smooth',block:'center'}});
    row.querySelector('select').focus();
  }});
}}
document.getElementById('show-reference').addEventListener('change', event => {{
  document.getElementById('reference-layer').style.display = event.target.checked ? '' : 'none';
}});
document.getElementById('export').addEventListener('click', () => {{
  const entries = meta.candidates.map(item => ({{candidate_id:item.id,bbox:item.bbox,
    orientation:item.orientation,classification:decisions[item.id] || 'unreviewed'}}));
  const payload = {{schema_version:1,image:meta.image,image_sha256:meta.image_sha256,
    detector_version:meta.detector_version,review_status:entries.every(x => x.classification !== 'unreviewed')
    ? 'user_reviewed' : 'draft',decisions:entries}};
  const blob = new Blob([JSON.stringify(payload,null,2)],{{type:'application/json'}});
  const link = document.createElement('a');link.href=URL.createObjectURL(blob);
  link.download=meta.image.replace(/\\.[^.]+$/,'')+'_window_cv_review.json';link.click();
  setTimeout(() => URL.revokeObjectURL(link.href),1000);
}});
document.getElementById('import').addEventListener('change', async event => {{
  const file=event.target.files[0];if(!file)return;
  try {{
    const data=JSON.parse(await file.text());
    if(data.image_sha256!==meta.image_sha256 || data.detector_version!==meta.detector_version)
      throw new Error('画像または検出器の版が異なります');
    const allowed=new Set(meta.candidate_ids);decisions={{}};
    for(const item of data.decisions || [])
      if(allowed.has(item.candidate_id) && valid.has(item.classification))
        decisions[item.candidate_id]=item.classification;
    try {{ localStorage.setItem(storageKey,JSON.stringify(decisions)); }} catch (_) {{}}
    render();
  }} catch(error) {{ alert('JSONを読み込めません: '+error.message); }}
}});
render();
</script></html>"""
    output.write_text(page, encoding="utf-8")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="壁線の切れ目から窓候補を作り、人手確認ページを生成")
    parser.add_argument("--image", type=Path, default=Path("floor_sample/sample1.webp"))
    parser.add_argument("--annotations", type=Path,
                        help="照合する赤枠JSON。sample1.webpでは省略時に既存の赤枠を使用")
    parser.add_argument("--output-dir", type=Path, default=Path("targeted_vision_results"))
    args = parser.parse_args()
    image_path = (ROOT / args.image).resolve()
    annotation_path = ((ROOT / args.annotations).resolve() if args.annotations else
                       ROOT / "manual_corrections/sample1_windows_draft.json"
                       if image_path.name == "sample1.webp" else None)
    if not image_path.is_file():
        parser.error(f"画像がありません: {image_path}")
    image_bytes = image_path.read_bytes()
    with Image.open(image_path) as opened:
        candidates = detect_wall_gap_candidates(opened)
        image_size = opened.size
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{image_path.stem}.{DETECTOR_VERSION}"
    candidates_path = output_dir / f"{stem}.candidates.json"
    candidates_path.write_text(json.dumps({"image": image_path.name,
                                           "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                                           "detector_version": DETECTOR_VERSION,
                                           "candidates": candidates}, ensure_ascii=False, indent=2), encoding="utf-8")
    if annotation_path:
        if not annotation_path.is_file():
            parser.error(f"照合用赤枠がありません: {annotation_path}")
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        if annotation.get("image") != image_path.name:
            parser.error("赤枠の画像名が入力画像と一致しません")
        evaluation = evaluate_candidate_coverage(candidates, annotation, image_size)
    else:
        annotation = {"reference_rectangles": []}
        evaluation = {"detector_version": DETECTOR_VERSION, "candidate_count": len(candidates),
                      "marked_window_rectangles": None, "covered_marked_windows": None,
                      "marked_window_coverage": None, "precision": None,
                      "note": "赤枠がないため候補位置の精度は未測定です。候補は窓と断定していません。"}
    evaluation_path = output_dir / f"{stem}.coverage.json"
    evaluation_path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    review_path = output_dir / f"{stem}.review.html"
    render_review_page(image_bytes, image_path.name, image_size,
                       candidates, evaluation, annotation, review_path)
    if evaluation["marked_window_rectangles"] is None:
        print(f"候補 {len(candidates)}件、赤枠照合なし")
    else:
        print(f"候補 {len(candidates)}件、既知の窓赤枠への位置一致 "
              f"{evaluation['covered_marked_windows']}/{evaluation['marked_window_rectangles']}")
    print(f"候補JSON: {candidates_path}\n照合JSON: {evaluation_path}\n確認ページ: {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

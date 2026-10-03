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
REVIEW_SCHEMA_VERSION = 2


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
    door_edges = []
    for candidate_index, candidate in enumerate(candidates):
        box = candidate["bbox"]
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        for door_index, door in enumerate(doors):
            distance = _distance_to_box(center, door["bbox"], image_size)
            if distance <= tolerance_px:
                door_edges.append((distance, candidate_index, door_index))
    used_door_candidates = set()
    used_doors = set()
    door_matches = []
    for distance, candidate_index, door_index in sorted(door_edges):
        if candidate_index in used_door_candidates or door_index in used_doors:
            continue
        used_door_candidates.add(candidate_index)
        used_doors.add(door_index)
        door_matches.append({"reference_id": doors[door_index]["id"],
                             "candidate_id": candidates[candidate_index]["id"],
                             "distance_px": round(distance, 1)})
    return {"detector_version": DETECTOR_VERSION,
            "candidate_count": len(candidates),
            "marked_window_rectangles": len(windows),
            "covered_marked_windows": len(matches),
            "marked_window_coverage": round(len(matches) / len(windows), 3) if windows else None,
            "missed_reference_ids": [item["id"] for index, item in enumerate(windows)
                                     if index not in used_windows],
            "marked_exterior_door_rectangles": len(doors),
            "covered_exterior_door_ids": [item["reference_id"] for item in door_matches],
            "missed_exterior_door_ids": [item["id"] for index, item in enumerate(doors)
                                         if index not in used_doors],
            "door_matches": door_matches,
            "matches": matches,
            "precision": None,
            "note": "候補は窓と断定していません。sample1は検出器の開発に使った画像で、位置一致は独立評価ではありません。赤枠の網羅性も未確認のためprecisionは計算しません。"}


def migrate_review(review: dict, candidates: list[dict], image_name: str,
                   image_bytes: bytes) -> dict:
    """Validate a downloaded review and make legacy generic doors unresolved."""
    version = review.get("schema_version")
    if version not in {1, REVIEW_SCHEMA_VERSION}:
        raise ValueError("確認JSONの形式が不正です")
    if (review.get("image") != image_name
            or review.get("image_sha256") != hashlib.sha256(image_bytes).hexdigest()
            or review.get("detector_version") != DETECTOR_VERSION):
        raise ValueError("確認JSONの画像または検出器の版が一致しません")
    source_decisions = review.get("decisions")
    if not isinstance(source_decisions, list) or len(source_decisions) != len(candidates):
        raise ValueError("確認JSONの候補件数が一致しません")
    by_id = {item["id"]: item for item in candidates}
    normalized = []
    seen = set()
    allowed = {"unreviewed", "window", "interior_door", "exterior_door",
               "door_unclassified", "not_opening", "uncertain"}
    for decision in source_decisions:
        ident = decision.get("candidate_id")
        if ident not in by_id or ident in seen:
            raise ValueError("確認JSONの候補IDが不正または重複しています")
        candidate = by_id[ident]
        if (decision.get("bbox") != candidate["bbox"]
                or decision.get("orientation") != candidate["orientation"]):
            raise ValueError(f"確認JSONの候補位置が一致しません: {ident}")
        classification = decision.get("classification")
        if classification not in allowed:
            raise ValueError(f"確認JSONの分類が不正です: {ident}")
        if version == 1 and classification == "exterior_door":
            classification = "door_unclassified"
        normalized.append({"candidate_id": ident, "bbox": candidate["bbox"],
                           "orientation": candidate["orientation"],
                           "classification": classification})
        seen.add(ident)
    normalized.sort(key=lambda item: item["candidate_id"])
    unresolved = {"unreviewed", "door_unclassified"}
    return {"schema_version": REVIEW_SCHEMA_VERSION, "image": image_name,
            "image_sha256": review["image_sha256"], "detector_version": DETECTOR_VERSION,
            "review_status": ("draft" if any(item["classification"] in unresolved for item in normalized)
                              else "user_reviewed"),
            "source_schema_version": version,
            "notes": "旧形式の外部扉は扉全般として選ばれたため、内外未分類に変換。採点には自動反映しない。"
            if version == 1 else "採点には自動反映しない。",
            "decisions": normalized}


def render_review_page(image_bytes: bytes, image_name: str, image_size: tuple[int, int],
                       candidates: list[dict], evaluation: dict,
                       annotation: dict, output: Path,
                       initial_decisions: dict[str, str] | None = None) -> None:
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
            '<option value="interior_door">室内扉</option>'
            '<option value="exterior_door">外部扉</option>'
            '<option value="door_unclassified">扉（内外未分類）</option>'
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
            "initial_decisions": initial_decisions or {},
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
.candidate[data-status="interior_door"] .outline{{stroke:#0f766e;stroke-width:2.3}}
.candidate[data-status="door_unclassified"] .outline{{stroke:#eab308;stroke-width:2.3}}
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
<p>青は画像処理が見つけた「壁線の切れ目」です。窓とは限りません。候補をクリックし、室内扉と外部扉も分けて分類してください。</p>
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
<p id="import-note" class="small">旧版のJSONを読み込むと、「外部扉」は内外未分類に戻ります。扉だけ再確認してください。</p>
<div class="review-list">{''.join(rows)}</div></aside></main>
<script>
const meta = {script_data};
const valid = new Set(['unreviewed','window','interior_door','exterior_door','door_unclassified','not_opening','uncertain']);
const unresolved = new Set(['unreviewed','door_unclassified']);
const storageKey = `window-cv-review-v2:${{meta.image_sha256}}:${{meta.detector_version}}`;
let decisions = {{...meta.initial_decisions}};
try {{
  const saved = JSON.parse(localStorage.getItem(storageKey) || '{{}}');
  if(Object.keys(saved).length) decisions = saved;
}} catch (_) {{}}
function render() {{
  let reviewed = 0, unclassifiedDoors = 0;
  for (const id of meta.candidate_ids) {{
    const choice = valid.has(decisions[id]) ? decisions[id] : 'unreviewed';
    if (!unresolved.has(choice)) reviewed++;
    if (choice === 'door_unclassified') unclassifiedDoors++;
    document.querySelector(`select[data-id="${{id}}"]`).value = choice;
    document.querySelector(`g.candidate[data-id="${{id}}"]`).dataset.status = choice;
  }}
  document.getElementById('progress').textContent =
    `分類確定 ${{reviewed}}/${{meta.candidate_ids.length}}・内外未分類の扉 ${{unclassifiedDoors}}`;
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
  const payload = {{schema_version:{REVIEW_SCHEMA_VERSION},image:meta.image,image_sha256:meta.image_sha256,
    detector_version:meta.detector_version,review_status:entries.every(x => !unresolved.has(x.classification))
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
    if(data.schema_version!==1 && data.schema_version!=={REVIEW_SCHEMA_VERSION})
      throw new Error('確認JSONの形式が異なります');
    const allowed=new Set(meta.candidate_ids);decisions={{}};
    let legacyDoors=0;
    for(const item of data.decisions || []) {{
      if(!allowed.has(item.candidate_id) || !valid.has(item.classification)) continue;
      if(data.schema_version===1 && item.classification==='exterior_door') {{
        decisions[item.candidate_id]='door_unclassified';legacyDoors++;
      }} else decisions[item.candidate_id]=item.classification;
    }}
    try {{ localStorage.setItem(storageKey,JSON.stringify(decisions)); }} catch (_) {{}}
    render();
    document.getElementById('import-note').textContent = legacyDoors
      ? `旧版の扉 ${{legacyDoors}} 件を内外未分類に戻しました。室内扉・外部扉を選び直してください。`
      : 'JSONを読み込みました。';
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
    parser.add_argument("--review-file", type=Path,
                        help="以前に保存した確認JSON。旧形式の扉は内外未分類に変換して再確認ページへ反映")
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
    initial_decisions = None
    if args.review_file:
        source_path = (ROOT / args.review_file).resolve()
        if not source_path.is_file():
            parser.error(f"確認JSONがありません: {source_path}")
        source_review = json.loads(source_path.read_text(encoding="utf-8"))
        migrated = migrate_review(source_review, candidates, image_path.name, image_bytes)
        draft_path = output_dir / f"{stem}.review_draft.json"
        draft_path.write_text(json.dumps(migrated, ensure_ascii=False, indent=2), encoding="utf-8")
        initial_decisions = {item["candidate_id"]: item["classification"]
                             for item in migrated["decisions"]}
        print(f"既存の確認結果を読込: {len(initial_decisions)}件、内外未分類の扉 "
              f"{sum(value == 'door_unclassified' for value in initial_decisions.values())}件")
        print(f"再確認用の下書きJSON: {draft_path}")
    review_path = output_dir / f"{stem}.{'recheck' if args.review_file else 'review'}.html"
    render_review_page(image_bytes, image_path.name, image_size,
                       candidates, evaluation, annotation, review_path, initial_decisions)
    if evaluation["marked_window_rectangles"] is None:
        print(f"候補 {len(candidates)}件、赤枠照合なし")
    else:
        print(f"候補 {len(candidates)}件、既知の窓赤枠への位置一致 "
              f"{evaluation['covered_marked_windows']}/{evaluation['marked_window_rectangles']}")
    print(f"候補JSON: {candidates_path}\n照合JSON: {evaluation_path}\n確認ページ: {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

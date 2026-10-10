"""Create a candidate-blind page for drawing window reference rectangles."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import mimetypes
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
REFERENCE_SCHEMA_VERSION = 1
KINDS = {"window", "window_door", "exterior_door", "uncertain"}


def validate_reference(data: dict, image_name: str, image_bytes: bytes,
                       image_size: tuple[int, int]) -> None:
    """Reject a mismatched or malformed downloaded reference JSON."""
    if (data.get("schema_version") != REFERENCE_SCHEMA_VERSION
            or data.get("image") != image_name
            or data.get("image_sha256") != hashlib.sha256(image_bytes).hexdigest()
            or data.get("image_size") != list(image_size)):
        raise ValueError("参照JSONの画像・版・寸法が一致しません")
    rectangles = data.get("reference_rectangles")
    if not isinstance(rectangles, list):
        raise ValueError("参照JSONの矩形一覧が不正です")
    seen = set()
    for item in rectangles:
        if not isinstance(item, dict):
            raise ValueError("参照JSONの矩形が不正です")
        ident, kind, box = item.get("id"), item.get("kind"), item.get("bbox")
        if (not isinstance(ident, str) or not ident.startswith("R") or ident in seen
                or kind not in KINDS or not isinstance(box, list) or len(box) != 4
                or any(not isinstance(value, (float, int)) or isinstance(value, bool)
                       or not 0 <= value <= 1 for value in box)
                or box[0] >= box[2] or box[1] >= box[3]):
            raise ValueError(f"参照JSONの矩形ID・種類・座標が不正です: {ident}")
        seen.add(ident)
    status = data.get("review_status")
    if status not in {"draft", "approved"}:
        raise ValueError("参照JSONの確認状態が不正です")
    if status == "approved" and (not rectangles
                                 or any(item["kind"] == "uncertain" for item in rectangles)
                                 or data.get("classification_review_status") != "user_confirmed"
                                 or data.get("geometry_review_status") != "user_confirmed"
                                 or data.get("coverage_review_status")
                                 != "user_confirmed_no_missing_windows"):
        raise ValueError("全窓確認を終えていない参照JSONは承認済みにできません")


def render_label_page(image_bytes: bytes, image_name: str,
                      image_size: tuple[int, int]) -> str:
    width, height = image_size
    mime = mimetypes.guess_type(image_name)[0] or "application/octet-stream"
    image_data = f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    meta = {"schema_version": REFERENCE_SCHEMA_VERSION, "image": image_name,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "image_size": [width, height]}
    meta_json = json.dumps(meta, ensure_ascii=False).replace("<", "\\u003c")
    page = r'''<!doctype html>
<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>窓の正解枠を作る — __IMAGE_NAME__</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;background:#f7f9fc;color:#17212f;margin:0}
header{background:white;border-bottom:1px solid #d8dfe8;padding:14px 20px}
h1{font-size:1.35rem;margin:0 0 5px}p{margin:5px 0}
main{display:grid;grid-template-columns:minmax(0,2fr) minmax(290px,1fr);gap:16px;padding:16px}
.panel{background:white;border:1px solid #d8dfe8;border-radius:9px;padding:12px}
.plan{position:relative;width:100%;max-width:1000px;margin:auto;user-select:none}
.plan img{display:block;width:100%}.plan svg{position:absolute;inset:0;width:100%;height:100%;touch-action:none;cursor:crosshair}
.mark{fill:#dc262622;stroke:#dc2626;stroke-width:1.6;pointer-events:none}
.mark.window_door{fill:#d9770622;stroke:#d97706}.mark.exterior_door{fill:#7c3aed22;stroke:#7c3aed}
.mark.uncertain{fill:#6b728022;stroke:#6b7280}.label{font-size:9px;font-weight:bold;fill:#004a80;pointer-events:none;paint-order:stroke;stroke:white;stroke-width:2px}
#preview{fill:#0876bf22;stroke:#0876bf;stroke-width:1.4;pointer-events:none}
.row{display:flex;align-items:center;justify-content:space-between;gap:7px;border:1px solid #e5e7eb;border-radius:5px;padding:5px;margin:4px 0;font-size:13px}
button,select{font:inherit;padding:4px 7px}.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:8px 0}
.small{color:#475569;font-size:13px}.warn{color:#9a3412}
@media(max-width:900px){main{grid-template-columns:1fr}}
</style>
<header><h1>窓の正解枠を作る — __IMAGE_NAME__</h1>
<p>元画像だけを表示しています。検出器の候補は表示しません。窓記号を一つずつドラッグで囲み、種類を選んでください。</p>
<p class="small">赤＝窓、橙＝通行できる窓扉、紫＝その他の外部扉。矩形は窓の物理的な枚数ではなく、図面上の印の位置です。</p></header>
<main><section class="panel">
<div class="controls"><label>新しい枠の種類 <select id="new-kind">
<option value="window">窓</option><option value="window_door">通行できる窓扉</option>
<option value="exterior_door">その他の外部扉</option><option value="uncertain">判別不能</option>
</select></label><span id="count"></span></div>
<div class="plan"><img src="__IMAGE_DATA__" alt="間取り図">
<svg id="overlay" viewBox="0 0 __WIDTH__ __HEIGHT__" preserveAspectRatio="none">
<g id="marks"></g><rect id="preview" style="display:none"></rect></svg></div>
<p class="small">間違えた枠は右側の一覧から削除できます。判定はブラウザ内にも一時保存されます。</p>
</section><aside class="panel"><strong>記録した枠</strong><div id="rows"></div>
<label><input id="complete" type="checkbox"> すべての窓・窓扉を確認し、付け漏れはない</label>
<p class="small warn">チェックして保存した場合だけ承認済みになります。候補検出・採点には自動反映しません。</p>
<div class="controls"><button id="export" type="button">確認JSONを保存</button>
<label>JSONを読み込む <input id="import" type="file" accept=".json,application/json"></label></div>
<p id="message" class="small"></p></aside></main>
<script>
const meta=__META__;
const kinds=['window','window_door','exterior_door','uncertain'];
const names={window:'窓',window_door:'通行できる窓扉',exterior_door:'その他の外部扉',uncertain:'判別不能'};
const key=`manual-window-label-v1:${meta.image_sha256}`;
let rectangles=[];let complete=false;let start=null;
const overlay=document.getElementById('overlay'),preview=document.getElementById('preview');
function saveLocal(){try{localStorage.setItem(key,JSON.stringify({rectangles,complete}));}catch(_){}}
function point(event){const r=overlay.getBoundingClientRect();return [Math.max(0,Math.min(meta.image_size[0],(event.clientX-r.left)*meta.image_size[0]/r.width)),Math.max(0,Math.min(meta.image_size[1],(event.clientY-r.top)*meta.image_size[1]/r.height))];}
function round(value){return Math.round(value*100000)/100000;}
function validBox(box){return Array.isArray(box)&&box.length===4&&box.every(v=>Number.isFinite(v)&&v>=0&&v<=1)&&box[0]<box[2]&&box[1]<box[3];}
function draw(){
 const marks=document.getElementById('marks');marks.replaceChildren();
 const rows=document.getElementById('rows');rows.replaceChildren();
 rectangles.forEach((item,index)=>{
  const id=`R${String(index+1).padStart(2,'0')}`;
  const [x1,y1,x2,y2]=item.bbox;
  const rect=document.createElementNS('http://www.w3.org/2000/svg','rect');
  rect.setAttribute('x',x1*meta.image_size[0]);rect.setAttribute('y',y1*meta.image_size[1]);
  rect.setAttribute('width',(x2-x1)*meta.image_size[0]);rect.setAttribute('height',(y2-y1)*meta.image_size[1]);
  rect.setAttribute('class',`mark ${item.kind}`);marks.append(rect);
  const label=document.createElementNS('http://www.w3.org/2000/svg','text');
  label.setAttribute('x',x1*meta.image_size[0]+2);label.setAttribute('y',Math.max(9,y1*meta.image_size[1]-2));
  label.setAttribute('class','label');label.textContent=id;marks.append(label);
  const row=document.createElement('div');row.className='row';
  const title=document.createElement('strong');title.textContent=id;row.append(title);
  const select=document.createElement('select');
  for(const kind of kinds){const option=document.createElement('option');option.value=kind;option.textContent=names[kind];select.append(option);}
  select.value=item.kind;select.addEventListener('change',()=>{rectangles[index].kind=select.value;saveLocal();draw();});row.append(select);
  const remove=document.createElement('button');remove.type='button';remove.textContent='削除';
  remove.addEventListener('click',()=>{rectangles.splice(index,1);complete=false;saveLocal();draw();});row.append(remove);rows.append(row);
 });
 document.getElementById('count').textContent=`${rectangles.length}枠`;
 document.getElementById('complete').checked=complete;
}
overlay.addEventListener('pointerdown',event=>{start=point(event);overlay.setPointerCapture(event.pointerId);preview.style.display='';});
overlay.addEventListener('pointermove',event=>{
 if(!start)return;const end=point(event);const x=Math.min(start[0],end[0]),y=Math.min(start[1],end[1]);
 preview.setAttribute('x',x);preview.setAttribute('y',y);
 preview.setAttribute('width',Math.abs(end[0]-start[0]));preview.setAttribute('height',Math.abs(end[1]-start[1]));
});
overlay.addEventListener('pointerup',event=>{
 if(!start)return;const end=point(event);const x1=Math.min(start[0],end[0]),y1=Math.min(start[1],end[1]);
 const x2=Math.max(start[0],end[0]),y2=Math.max(start[1],end[1]);start=null;preview.style.display='none';
 if(x2-x1<3||y2-y1<3)return;
 rectangles.push({kind:document.getElementById('new-kind').value,
  bbox:[round(x1/meta.image_size[0]),round(y1/meta.image_size[1]),round(x2/meta.image_size[0]),round(y2/meta.image_size[1])]});
 complete=false;saveLocal();draw();
});
overlay.addEventListener('pointercancel',()=>{start=null;preview.style.display='none';});
document.getElementById('complete').addEventListener('change',event=>{complete=event.target.checked;saveLocal();});
document.getElementById('export').addEventListener('click',()=>{
 if(complete&&!rectangles.length){alert('少なくとも1つの枠を記録してください');return;}
 if(complete&&rectangles.some(item=>item.kind==='uncertain')){alert('判別不能の枠がある間は承認済みにできません');return;}
 const references=rectangles.map((item,index)=>({id:`R${String(index+1).padStart(2,'0')}`,kind:item.kind,bbox:item.bbox}));
 const payload={...meta,review_status:complete?'approved':'draft',source:'candidate_blind_manual_browser',
  classification_review_status:complete?'user_confirmed':'draft',
  geometry_review_status:complete?'user_confirmed':'draft',
  coverage_review_status:complete?'user_confirmed_no_missing_windows':'not_confirmed',
  reference_rectangles:references};
 const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});
 const link=document.createElement('a');link.href=URL.createObjectURL(blob);
 link.download=meta.image.replace(/\.[^.]+$/,'')+'_window_reference.json';link.click();
 setTimeout(()=>URL.revokeObjectURL(link.href),1000);
});
document.getElementById('import').addEventListener('change',async event=>{
 const file=event.target.files[0];if(!file)return;
 try{
  const data=JSON.parse(await file.text());
  if(data.schema_version!==meta.schema_version||data.image!==meta.image||data.image_sha256!==meta.image_sha256||JSON.stringify(data.image_size)!==JSON.stringify(meta.image_size))throw Error('画像・版・寸法が一致しません');
  if(!Array.isArray(data.reference_rectangles)||!data.reference_rectangles.every(item=>kinds.includes(item.kind)&&validBox(item.bbox)))throw Error('矩形が不正です');
  rectangles=data.reference_rectangles.map(item=>({kind:item.kind,bbox:item.bbox}));
  complete=data.review_status==='approved'&&data.coverage_review_status==='user_confirmed_no_missing_windows';
  saveLocal();draw();document.getElementById('message').textContent=`${rectangles.length}枠を読み込みました。`;
 }catch(error){alert('JSONを読み込めません: '+error.message);}
});
try{const saved=JSON.parse(localStorage.getItem(key)||'{}');
 if(Array.isArray(saved.rectangles)&&saved.rectangles.every(item=>kinds.includes(item.kind)&&validBox(item.bbox)))rectangles=saved.rectangles;
 complete=Boolean(saved.complete);}catch(_){}
draw();
</script></html>'''
    return (page.replace("__IMAGE_NAME__", html.escape(image_name))
            .replace("__IMAGE_DATA__", image_data)
            .replace("__WIDTH__", str(width)).replace("__HEIGHT__", str(height))
            .replace("__META__", meta_json))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="検出候補を表示せず、窓の正解枠を手描きするページを生成")
    parser.add_argument("--image", type=Path, default=Path("floor_sample/sample3.webp"))
    parser.add_argument("--output-dir", type=Path, default=Path("manual_corrections"))
    args = parser.parse_args()
    image_path = (ROOT / args.image).resolve()
    if not image_path.is_file():
        parser.error(f"画像がありません: {image_path}")
    image_bytes = image_path.read_bytes()
    with Image.open(image_path) as image:
        size = image.size
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{image_path.stem}_window_label.html"
    output.write_text(render_label_page(image_bytes, image_path.name, size), encoding="utf-8")
    print(f"候補を表示しない窓ラベルページ: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

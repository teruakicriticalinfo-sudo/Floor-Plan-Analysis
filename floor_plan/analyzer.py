"""Two-stage floor-plan extraction and evaluation with a vision model."""

import json
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "qwen3-vl:4b-instruct"

SCORE_CRITERIA = (
    ("生活動線", 15, "帰宅、家事、来客、各室間の移動に無駄や交錯がないか"),
    ("ゾーニング・プライバシー", 15, "公私の分離、視線、トイレや寝室の配置が適切か"),
    ("採光・通風", 10, "方位など確認できる条件の範囲で、窓と室配置が適切か"),
    ("家具配置・居住性", 15, "家具を置いた後も有効幅、開閉、滞在空間を確保できるか"),
    ("収納", 10, "量だけでなく、使用場所と収納場所が対応しているか"),
    ("家事効率", 15, "調理、配膳、洗濯、片付け、買い物後の動線が効率的か"),
    ("安全性・バリアフリー", 10, "階段、扉の干渉、段差、避難・搬入経路に問題がないか"),
    ("将来対応・可変性", 10, "家族構成、加齢、用途変更などの変化に対応できるか"),
)

STRUCTURE_JSON_SCHEMA = {
    "type": "object",
    "required": ["image_quality", "orientation", "spaces", "connections", "windows", "fixtures", "negative_observations", "unreadable_items"],
    "properties": {
        "image_quality": {"type": "object"},
        "orientation": {"type": "object"},
        "spaces": {"type": "array", "items": {"type": "object", "required": ["id", "label", "space_type", "confidence"], "properties": {"id": {"type": "string"}, "label": {"type": "string"}, "space_type": {"type": "string"}, "area_text": {"type": ["string", "null"]}, "bbox": {"type": "array"}, "confidence": {"type": "string"}, "evidence": {"type": "string"}}}},
        "connections": {"type": "array", "items": {"type": "object", "required": ["id", "space_a", "space_b", "boundary_relation", "traversable", "confidence"], "properties": {"id": {"type": "string"}, "space_a": {"type": "string"}, "space_b": {"type": "string"}, "boundary_relation": {"type": "string"}, "traversable": {"type": "boolean"}, "position": {"type": "array"}, "confidence": {"type": "string"}, "evidence": {"type": "string"}}}},
        "windows": {"type": "array", "items": {"type": "object"}},
        "fixtures": {"type": "array", "items": {"type": "object"}},
        "negative_observations": {"type": "array", "items": {"type": "object"}},
        "unreadable_items": {"type": "array", "items": {"type": "string"}},
    },
}


@dataclass(frozen=True)
class FloorPlanAnalysis:
    draft_structure: dict[str, Any]
    structure: dict[str, Any]
    report: str


def load_knowledge(path: str | Path) -> str:
    knowledge_path = Path(path)
    if not knowledge_path.is_file():
        raise FileNotFoundError(f"知識ファイルが見つかりません: {knowledge_path}")

    knowledge = knowledge_path.read_text(encoding="utf-8").strip()
    if not knowledge:
        raise ValueError("knowledge.md が空です。")
    return knowledge


def build_extraction_prompt() -> str:
    """Prompt used only to transcribe spatial facts from the image."""
    return """
間取り画像を観察し、空間の接続関係をJSONだけで記録してください。この段階では評価・改善提案・一般常識による補完をしません。
簡潔に記録し、同じ空間・境界・窓を重複登録しないでください。evidenceは20文字以内とし、画像に存在する数を超えて項目を繰り返さないでください。

重要な区別:
- 壁を共有するだけの空間と、扉や開口を通って直接移動できる空間を区別する。
- バルコニーに面して見えることと、バルコニーへ直接出入りできることを区別する。
- 窓と扉を区別する。判別できなければ unknown とする。
- 線が不鮮明な場合は推測せず confidence を low、値を null または unknown にする。
- 画像に描かれていない収納、通路、扉、窓を追加しない。
- すべての空間に一意のIDを付け、connections/windowsからはそのIDだけを参照する。
- 屋外、バルコニー、玄関ホール、廊下、収納、水回りも独立したspaceとして登録する。
- 扉は、扉記号が描かれた壁の両側にある2空間だけを接続する。単に近い空間とは接続しない。
- 開口の座標が両空間の境界上にあるかを確認してから traversable=true にする。
- 玄関の外部ドアも屋外とのconnectionとして記録する。
- 掃き出し窓・ガラス戸は、通行可能ならconnection、採光可能ならwindowにも記録する。
- 検出されなかったことは不存在を意味しない。明確に「ない」と確認できたものだけnegative_observationsへ記録する。

次の形のJSONオブジェクトだけを返す:
{
  "image_quality": {"level": "high|medium|low", "notes": ["..."]},
  "orientation": {"value": null, "confidence": "high|medium|low", "evidence": "..."},
  "spaces": [
    {"id": "S1", "label": "LDK", "space_type": "room|hall|storage|sanitary|exterior|other", "area_text": "11.1帖|null", "bbox": [0.0, 0.0, 1.0, 1.0], "confidence": "high|medium|low", "evidence": "画像上の文字や位置"}
  ],
  "connections": [
    {"id": "C1", "space_a": "S1", "space_b": "S2", "boundary_relation": "door|glazed_door|open_passage|shared_wall_only|unknown", "traversable": true, "position": [0.0, 0.0], "confidence": "high|medium|low", "evidence": "扉記号や開口の位置"}
  ],
  "windows": [
    {"id": "W1", "space_id": "S1", "faces_space_id": "S9|null", "faces_exterior": true, "position": [0.0, 0.0], "confidence": "high|medium|low", "evidence": "窓記号の位置"}
  ],
  "fixtures": [
    {"space_id": "S1", "fixture": "設備名", "confidence": "high|medium|low", "evidence": "..."}
  ],
  "negative_observations": [
    {"space_id": "S1", "feature": "収納", "confidence": "high|medium|low", "evidence": "空間全体が明瞭で収納記号なし"}
  ],
  "unreadable_items": ["確認できなかった事項"]
}

connectionsには、画像から確認できる空間境界を記録する。直接通行できると確認できた場合だけ traversable をtrueにする。
JSON以外の説明文やMarkdownコードフェンスは出力しない。
""".strip()


def parse_structure_response(text: str) -> dict[str, Any]:
    """Parse and minimally validate the extraction model's JSON."""
    if not text or not text.strip():
        raise RuntimeError("モデルから構造化結果が返されませんでした。")

    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
    else:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end >= start:
            candidate = candidate[start : end + 1]

    try:
        structure = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"構造化結果をJSONとして解釈できません: {exc}") from exc

    if not isinstance(structure, dict):
        raise RuntimeError("構造化結果がJSONオブジェクトではありません。")

    required = {"image_quality", "orientation", "spaces", "connections", "windows", "fixtures", "negative_observations", "unreadable_items"}
    missing = required.difference(structure)
    if missing:
        raise RuntimeError(f"構造化結果の必須項目が不足しています: {', '.join(sorted(missing))}")
    if not isinstance(structure["spaces"], list) or not structure["spaces"]:
        raise RuntimeError("構造化結果に空間がありません。")

    space_ids = [space.get("id") for space in structure["spaces"] if isinstance(space, dict)]
    if len(space_ids) != len(structure["spaces"]) or any(not value for value in space_ids):
        raise RuntimeError("すべての空間にIDが必要です。")
    if len(space_ids) != len(set(space_ids)):
        raise RuntimeError("空間IDが重複しています。")

    known_ids = set(space_ids)
    for connection in structure["connections"]:
        if not isinstance(connection, dict):
            raise RuntimeError("connectionsの要素が不正です。")
        endpoints = {connection.get("space_a"), connection.get("space_b")}
        unknown_ids = endpoints.difference(known_ids)
        if unknown_ids:
            raise RuntimeError(f"接続関係が未定義の空間を参照しています: {', '.join(sorted(map(str, unknown_ids)))}")
    return structure


def add_verified_topology(structure: dict[str, Any]) -> dict[str, Any]:
    """Create a deterministic adjacency map from traversable model observations."""
    enriched = deepcopy(structure)
    neighbors = {space["id"]: [] for space in enriched["spaces"]}
    direct_connections = []
    for connection in enriched["connections"]:
        if connection.get("traversable") is True:
            space_a, space_b = connection["space_a"], connection["space_b"]
            neighbors[space_a].append(space_b)
            neighbors[space_b].append(space_a)
            direct_connections.append(
                {"connection_id": connection.get("id"), "spaces": [space_a, space_b]}
            )

    space_types = {space["id"]: space.get("space_type") for space in enriched["spaces"]}
    enriched["verified_topology"] = {
        "traversable_neighbors": {key: sorted(value) for key, value in neighbors.items()},
        "direct_connections": direct_connections,
        "non_transit_spaces": sorted(
            space_id for space_id, space_type in space_types.items() if space_type == "storage"
        ),
        "confirmed_absences": [
            observation
            for observation in enriched["negative_observations"]
            if observation.get("confidence") == "high"
        ],
    }
    return enriched


def find_topology_anomalies(structure: dict[str, Any]) -> list[str]:
    """Find suspicious graph patterns that should trigger visual reinspection."""
    enriched = add_verified_topology(structure)
    neighbors = enriched["verified_topology"]["traversable_neighbors"]
    spaces = {space["id"]: space for space in enriched["spaces"]}
    anomalies = []
    for space_id, space in spaces.items():
        connected = neighbors[space_id]
        if space.get("space_type") == "storage" and not connected:
            anomalies.append(f"{space_id}({space.get('label')})に収納扉が記録されていない")
        if space.get("space_type") != "room":
            continue
        if not connected:
            anomalies.append(f"{space_id}({space.get('label')})に通行可能な入口がない")
            continue
        neighbor_types = {spaces[item].get("space_type") for item in connected}
        if neighbor_types.issubset({"exterior", "storage"}):
            anomalies.append(
                f"{space_id}({space.get('label')})への入口が屋外または収納経由しかない"
            )

    toilet_spaces = {
        fixture.get("space_id")
        for fixture in enriched["fixtures"]
        if "トイレ" in str(fixture.get("fixture", ""))
    }
    for space_id in toilet_spaces:
        connected = neighbors.get(space_id, [])
        if connected and all(spaces[item].get("space_type") == "room" for item in connected):
            anomalies.append(
                f"{space_id}(トイレ)の入口が居室だけに接続している。近接する廊下・玄関ホール側の扉を再確認"
            )

    hall_ids = [key for key, value in spaces.items() if value.get("space_type") == "hall"]
    living_ids = [
        key for key, value in spaces.items()
        if value.get("space_type") == "room" and "LDK" in str(value.get("label", "")).upper()
    ]
    blocked_types = {"sanitary", "storage", "exterior"}
    for hall_id in hall_ids:
        reachable = {hall_id}
        queue = [hall_id]
        while queue:
            current = queue.pop(0)
            for neighbor in neighbors[current]:
                if neighbor in reachable:
                    continue
                if spaces[neighbor].get("space_type") in blocked_types:
                    continue
                reachable.add(neighbor)
                queue.append(neighbor)
        if living_ids and not any(item in reachable for item in living_ids):
            anomalies.append(
                f"{hall_id}(玄関・廊下)からLDKへ行くのに水回り・収納・屋外を通る。ホール形状とLDK扉を再確認"
            )
    return anomalies


def build_visual_contents(prompt: str, image: Any) -> list[Any]:
    """Provide a whole-plan view plus overlapping enlarged crops for small plans."""
    if not all(hasattr(image, attr) for attr in ("size", "resize", "crop")):
        return [prompt, image]

    width, height = image.size
    if not width or not height:
        return [prompt, image]

    try:
        from PIL import Image as PilImage

        resampling = PilImage.Resampling.LANCZOS
    except (ImportError, AttributeError):
        return [prompt, image]

    scale = max(1.0, min(4.0, 1800 / width))
    full = image.resize((round(width * scale), round(height * scale)), resampling)
    contents: list[Any] = [
        prompt,
        "同じ間取りの全体拡大図です。空間全体の位置関係に使用してください。",
        full,
    ]

    overlap_x, overlap_y = round(width * 0.08), round(height * 0.08)
    regions = (
        ("左上", (0, 0, width // 2 + overlap_x, height // 2 + overlap_y)),
        ("右上", (width // 2 - overlap_x, 0, width, height // 2 + overlap_y)),
        ("左下", (0, height // 2 - overlap_y, width // 2 + overlap_x, height)),
        ("右下", (width // 2 - overlap_x, height // 2 - overlap_y, width, height)),
    )
    for label, box in regions:
        crop = image.crop(box)
        crop_scale = max(1.0, min(4.0, 1200 / crop.size[0]))
        crop = crop.resize(
            (round(crop.size[0] * crop_scale), round(crop.size[1] * crop_scale)),
            resampling,
        )
        contents.extend((f"同じ間取りの{label}拡大図です。扉・壁・窓記号の確認に使用してください。", crop))
    return contents


def build_verification_prompt(draft: dict[str, Any]) -> str:
    """Ask a fresh visual pass to correct the draft topology before scoring."""
    anomalies = find_topology_anomalies(draft)
    anomaly_text = "\n".join(f"- {item}" for item in anomalies) or "- 自動検出なし"
    return f"""
間取り画像と、別の読取処理が作った【暫定JSON】を照合し、誤った空間接続を修正してください。評価や改善提案は行わず、修正後の完全なJSONだけを返してください。

重点確認:
- 暫定JSONを正しいと仮定せず、すべての traversable=true を元画像の扉・開口記号と再照合する。
- 扉記号がある壁の両側の空間だけを接続する。近くにある別室と接続しない。
- 玄関ホールや廊下はL字など不整形になり得る。単純なbboxだけで接続先を決めない。
- 各居室に通常の屋内入口があるか確認する。バルコニーや収納を通らないと居室へ入れない結果は、画像に明白な根拠がない限り誤読として再確認する。
- トイレ、洗面所、浴室の扉が、玄関ホール・廊下・LDKのどれに実際に開くか、扉の円弧と壁の切れ目から確認する。
- shared_wall_only、door、glazed_door、windowを混同しない。
- 掃き出し窓は、通行可能ならglazed_doorとしてconnectionsに、採光可能ならwindowsにも記録する。
- 明確な不存在だけnegative_observationsへ残す。単に検出できなかった項目は削除し、unreadable_itemsへ移す。
- confidenceは再照合後の確信度に修正する。不鮮明ならlowにする。
- 元のJSONと同じ必須キーをすべて含め、verified_topologyは含めない。

【コードが検出した要再確認事項】
{anomaly_text}

【暫定JSON】
{json.dumps(draft, ensure_ascii=False, indent=2)}
""".strip()


def _response_text(response: Any, stage: str) -> str:
    text = getattr(response, "text", None)
    if not text or not text.strip():
        raise RuntimeError(f"モデルから{stage}の本文が返されませんでした。")
    return text.strip()


def _generate_with_retry(client: Any, **kwargs: Any) -> Any:
    """Retry transient local-server and cloud capacity failures."""
    for attempt in range(3):
        try:
            if hasattr(client, "generate_content"):
                return client.generate_content(**kwargs)
            return client.models.generate_content(**kwargs)
        except Exception as exc:
            message = str(exc).upper()
            transient = any(marker in message for marker in ("429", "503", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
            if not transient or attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("モデルAPIの再試行に失敗しました。")


def _parse_or_repair_structure(
    text: str,
    client: Any,
    model: str,
    stage: str,
) -> dict[str, Any]:
    """Repair syntax-only JSON mistakes without asking the model to inspect the image again."""
    try:
        return parse_structure_response(text)
    except RuntimeError as original_error:
        repair_prompt = f"""
次の間取り読取JSONは構文が壊れています。JSON構文だけを修復し、JSONオブジェクトだけを返してください。
内容の追加、削除、推測、評価、要約、ID変更は禁止です。必須キーと配列の全要素を維持してください。

【壊れたJSON】
{text}
""".strip()
        response = _generate_with_retry(
            client,
            model=model,
            contents=[repair_prompt],
            config={"response_mime_type": "application/json", "response_json_schema": STRUCTURE_JSON_SCHEMA, "temperature": 0},
        )
        repaired_text = _response_text(response, f"{stage}のJSON修復結果")
        try:
            return parse_structure_response(repaired_text)
        except RuntimeError as repair_error:
            raise RuntimeError(
                f"{stage}のJSONが不正で、自動修復にも失敗しました。"
                f" 元のエラー: {original_error}; 修復後: {repair_error}"
            ) from repair_error


def extract_floor_plan_structure(image: Any, client: Any, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Stage 1: extract observable topology without judging it."""
    response = _generate_with_retry(
        client,
        model=model,
        contents=build_visual_contents(build_extraction_prompt(), image),
        config={"response_mime_type": "application/json", "response_json_schema": STRUCTURE_JSON_SCHEMA, "temperature": 0},
    )
    return _parse_or_repair_structure(
        _response_text(response, "構造化結果"), client, model, "構造化結果"
    )


def verify_floor_plan_structure(
    image: Any,
    draft: dict[str, Any],
    client: Any,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Stage 1 verification: re-check suspicious and all traversable edges."""
    response = _generate_with_retry(
        client,
        model=model,
        contents=build_visual_contents(build_verification_prompt(draft), image),
        config={"response_mime_type": "application/json", "response_json_schema": STRUCTURE_JSON_SCHEMA, "temperature": 0},
    )
    verified = _parse_or_repair_structure(
        _response_text(response, "再照合結果"), client, model, "再照合結果"
    )
    return add_verified_topology(verified)


def build_analysis_prompt(knowledge: str, structure: dict[str, Any]) -> str:
    """Stage 2 prompt: evaluate only the extracted facts."""
    if not knowledge.strip():
        raise ValueError("参考知識が空です。")

    rubric_lines = "\n".join(
        f"- {name}: {points}点 — {description}"
        for name, points, description in SCORE_CRITERIA
    )
    structure_json = json.dumps(structure, ensure_ascii=False, indent=2)

    return f"""
あなたは住宅の間取りをレビューする分析者です。
【構造化された画像読取結果】だけを間取りの事実として扱い、【参考知識】を評価観点として使用してください。
元画像を想像して再解釈したり、JSONにない扉・窓・収納・バルコニー接続を補ったりしてはいけません。

【接続関係の絶対ルール】
- 移動経路に使えるのは connections で traversable=true の接続だけ。
- shared_wall_only は隣接しているだけで、移動・採光・バルコニーアクセスの根拠にしない。
- バルコニーへ出られると述べるには、対象空間とバルコニー間に traversable=true の接続が必要。
- 窓があるだけでは、その先へ出入りできると判断しない。
- confidence=low または unknown/null の情報は断定や強い減点の根拠にしない。
- 「図面で検出されなかった」と「存在しない」を区別する。項目が配列にないという理由だけで、不足・不存在・問題点として絶対に加点減点しない。
- 不足や不存在を問題にできるのは negative_observations に該当項目が明記され、そのconfidenceがhighの場合だけ。
- 採光の根拠にできるのはwindowsの記録、またはglazed_doorの記録だけ。逆に、それらの記録がないだけでは暗いと判断しない。
- 方位、隣家、居住人数、構造、法令適合性など、JSONにない情報は「確認不能」とする。
- 参考知識は、対象設備・条件が構造化結果で確認できる場合だけ適用する。
- 既に独立した個室が複数あることを「将来2部屋に分割できない」の減点理由にしない。将来分割を想定した大部屋が確認できる場合だけ適用する。
- 根拠には必ずspace ID、connection IDまたはwindow IDを併記する。
- 経路は verified_topology.traversable_neighbors の組だけで構成し、各移動をIDで示す。non_transit_spacesは経路の途中に使わない。
- 不足を指摘できる対象は verified_topology.confirmed_absences に列挙されたものだけ。この配列にない収納・窓などを「ない」「不足」と書かない。

【固定採点表・合計100点】
{rubric_lines}

【出力形式】
# 総合評価: XX / 100点

## 読み取り条件
- 確認できた情報
- 確認不能な情報
- 画像の判読性: 高・中・低

## 採点表
| 評価項目 | 得点 | 配点 | 構造化結果上の根拠 | 参考にした知識 |
|---|---:|---:|---|---|
8項目すべてを記載し、得点合計と総合評価を一致させる。

## 良い点
## 気になる点
## 改善案
優先度を「高・中・低」で示す。
## 専門家に確認すべき事項

最後に、建築士による法的・構造的確認の代替ではないことを短く明記する。

【構造化された画像読取結果ここから】
{structure_json}
【構造化された画像読取結果ここまで】

【参考知識ここから】
{knowledge}
【参考知識ここまで】
""".strip()


def analyze_from_structure(
    structure: dict[str, Any],
    client: Any,
    knowledge: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """Stage 2: score the already-extracted structure without the source image."""
    response = _generate_with_retry(
        client,
        model=model,
        contents=[build_analysis_prompt(knowledge, structure)],
        config={"temperature": 0},
    )
    return _response_text(response, "分析結果")


def analyze_floor_plan(
    image: Any,
    client: Any,
    knowledge: str,
    model: str = DEFAULT_MODEL,
) -> FloorPlanAnalysis:
    """Run topology extraction first, then evaluate the extracted JSON."""
    draft = extract_floor_plan_structure(image, client, model)
    structure = verify_floor_plan_structure(image, draft, client, model)
    report = analyze_from_structure(structure, client, knowledge, model)
    return FloorPlanAnalysis(draft_structure=draft, structure=structure, report=report)

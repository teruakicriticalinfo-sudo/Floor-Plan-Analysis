"""Two-stage floor-plan extraction and evaluation with a vision model."""

import json
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


DEFAULT_MODEL = "qwen3-vl:4b-instruct"
PIPELINE_CACHE_VERSION = "floor-regions-refined-v6"

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
    "required": ["image_quality", "orientation", "spaces", "openings", "connections", "windows", "fixtures", "negative_observations", "unreadable_items"],
    "properties": {
        "image_quality": {"type": "object"},
        "orientation": {"type": "object"},
        "plan_regions": {"type": "array", "items": {"type": "object", "required": ["floor_id", "bbox"], "properties": {"floor_id": {"type": "string"}, "bbox": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}}}}},
        "spaces": {"type": "array", "items": {"type": "object", "required": ["id", "label", "space_type", "bbox", "confidence"], "properties": {"id": {"type": "string"}, "label": {"type": "string"}, "space_type": {"type": "string"}, "floor_id": {"type": ["string", "null"]}, "area_text": {"type": ["string", "null"]}, "bbox": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}}, "confidence": {"type": "string"}, "evidence": {"type": "string"}}}},
        "openings": {"type": "array", "items": {"type": "object", "required": ["id", "opening_type", "position", "confidence"], "properties": {"id": {"type": "string"}, "opening_type": {"type": "string"}, "position": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}, "confidence": {"type": "string"}, "evidence": {"type": "string"}}}},
        "connections": {"type": "array", "items": {"type": "object", "required": ["id", "opening_id", "space_a", "space_b", "boundary_relation", "traversable", "position", "confidence"], "properties": {"id": {"type": "string"}, "opening_id": {"type": "string"}, "space_a": {"type": "string"}, "space_b": {"type": "string"}, "boundary_relation": {"type": "string"}, "traversable": {"type": "boolean"}, "position": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}, "confidence": {"type": "string"}, "evidence": {"type": "string"}}}},
        "windows": {"type": "array", "items": {"type": "object", "required": ["id", "space_id", "position", "confidence"], "properties": {"id": {"type": "string"}, "space_id": {"type": "string"}, "faces_exterior": {"type": "boolean"}, "position": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}, "confidence": {"type": "string"}, "evidence": {"type": "string"}}}},
        "fixtures": {"type": "array", "items": {"type": "object", "required": ["space_id", "fixture", "confidence"], "properties": {"space_id": {"type": "string"}, "fixture": {"type": "string"}, "confidence": {"type": "string"}, "evidence": {"type": "string"}}}},
        "negative_observations": {"type": "array", "items": {"type": "object"}},
        "unreadable_items": {"type": "array", "items": {"type": "string"}},
    },
}

SPACE_INVENTORY_SCHEMA = {
    "type": "object",
    "required": ["image_quality", "orientation", "spaces", "plan_regions"],
    "properties": {
        "image_quality": STRUCTURE_JSON_SCHEMA["properties"]["image_quality"],
        "orientation": STRUCTURE_JSON_SCHEMA["properties"]["orientation"],
        "plan_regions": STRUCTURE_JSON_SCHEMA["properties"]["plan_regions"],
        "spaces": STRUCTURE_JSON_SCHEMA["properties"]["spaces"],
    },
}

CROPPED_INVENTORY_SCHEMA = {
    "type": "object", "required": ["spaces"],
    "properties": {"spaces": STRUCTURE_JSON_SCHEMA["properties"]["spaces"]},
}

HALL_PROBE_SCHEMA = {
    "type": "object", "required": ["separate_hall", "bbox", "confidence", "evidence"],
    "properties": {
        "separate_hall": {"type": "boolean"},
        "bbox": {"type": ["array", "null"], "items": {"type": "number"}},
        "confidence": {"type": "string"},
        "evidence": {"type": "string"},
    },
}

TOPOLOGY_OBSERVATION_SCHEMA = {
    "type": "object",
    "required": ["openings", "connections", "windows", "fixtures", "negative_observations", "unreadable_items"],
    "properties": {
        key: STRUCTURE_JSON_SCHEMA["properties"][key]
        for key in ("openings", "connections", "windows", "fixtures", "negative_observations", "unreadable_items")
    },
}

FEATURE_SCHEMA = {
    "type": "object", "required": ["windows", "fixtures", "unreadable_items"],
    "properties": {
        "windows": STRUCTURE_JSON_SCHEMA["properties"]["windows"],
        "fixtures": STRUCTURE_JSON_SCHEMA["properties"]["fixtures"],
        "unreadable_items": STRUCTURE_JSON_SCHEMA["properties"]["unreadable_items"],
    },
}

SCORE_JSON_SCHEMA = {
    "type": "object",
    "required": ["criteria", "good_points", "concerns", "improvements", "expert_checks"],
    "properties": {
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "proposed_score", "status", "evidence_ids", "reason", "knowledge_basis"],
                "properties": {
                    "name": {"type": "string"},
                    "proposed_score": {"type": "integer"},
                    "status": {"type": "string", "enum": ["confirmed", "partial", "unverifiable"]},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                    "knowledge_basis": {"type": "string"},
                },
            },
        },
        "good_points": {"type": "array", "items": {"type": "string"}},
        "concerns": {"type": "array", "items": {"type": "string"}},
        "improvements": {"type": "array", "items": {"type": "string"}},
        "expert_checks": {"type": "array", "items": {"type": "string"}},
    },
}

VERIFICATION_JSON_SCHEMA = {
    "type": "object",
    "required": ["decisions"],
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["connection_id", "verdict", "confidence", "reason"],
                "properties": {
                    "connection_id": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["accept", "reject", "uncertain"]},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "reason": {"type": "string"},
                },
            },
        }
    },
}


@dataclass(frozen=True)
class FloorPlanAnalysis:
    draft_structure: dict[str, Any]
    structure: dict[str, Any]
    scoring: dict[str, Any]
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
- 最初にspacesを矩形bboxで登録し、次に画像上で実際に見える扉・開口をopeningsへ登録する。その後だけconnectionsを作る。
- bboxとpositionは画像左上を[0,0]、右下を[1,1]とした正規化座標にする。
- connectionは必ず実在するopening_idを1つ参照し、そのopeningのpositionをconnection.positionにも複写する。
- 壁を共有するだけの空間と、扉や開口を通って直接移動できる空間を区別する。
- バルコニーに面して見えることと、バルコニーへ直接出入りできることを区別する。
- 窓と扉を区別する。判別できなければ unknown とする。
- 線が不鮮明な場合は推測せず confidence を low、値を null または unknown にする。
- 画像に描かれていない収納、通路、扉、窓を追加しない。
- すべての空間に一意のIDを付け、connections/windowsからはそのIDだけを参照する。
- 屋外、バルコニー、玄関ホール、廊下、収納、水回りも独立したspaceとして登録する。階段はspace_type=vertical_circulationとする。
- 「1階平面図」「2階平面図」など複数階が同じ画像にある場合、各spaceにfloor_id（例: "1F", "2F"）を必ず付ける。別階のspaceを直接connectionにしてはいけない。階段を通る接続だけは例外。
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
    {"id": "S1", "label": "LDK", "space_type": "room|hall|storage|sanitary|exterior|vertical_circulation|other", "floor_id": "1F|null", "area_text": "11.1帖|null", "bbox": [0.0, 0.0, 1.0, 1.0], "confidence": "high|medium|low", "evidence": "画像上の文字や位置"}
  ],
  "openings": [
    {"id": "O1", "opening_type": "door|glazed_door|open_passage|unknown", "position": [0.5, 0.5], "confidence": "high|medium|low", "evidence": "扉円弧・壁の切れ目"}
  ],
  "connections": [
    {"id": "C1", "opening_id": "O1", "space_a": "S1", "space_b": "S2", "boundary_relation": "door|glazed_door|open_passage|shared_wall_only|unknown", "traversable": true, "position": [0.5, 0.5], "confidence": "high|medium|low", "evidence": "扉記号や開口の位置"}
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
openingsにない扉をconnectionsのために追加してはいけない。1つのopeningを複数のconnectionへ使わない。
JSON以外の説明文やMarkdownコードフェンスは出力しない。
""".strip()


def build_space_inventory_prompt() -> str:
    """Read and locate spaces without attempting any door association."""
    return """
間取り画像から空間だけを列挙してください。扉・接続・評価は扱いません。
画像左上を[0,0]、右下を[1,1]として、各空間のbboxを必ず記録してください。

重要:
- 玄関ラベル周辺だけでなく、玄関からLDKまで続く廊下・ホール全体を1つのhallとしてbboxに含める。
- 浴室、洗面所・脱衣所、トイレを別々のsanitary空間として確認する。
- キッチンがLDK内の設備なら独立roomにしない。
- 上下など離れたバルコニーは別々のexterior空間にする。
- CL、押入などの収納も個別のstorage空間にする。
- 複数階が同一画像にある場合は、1階・2階などの図面ごとにfloor_idを分ける。階段はvertical_circulationとして登録する。
- 最初に独立した平面図ごとのplan_regionsを作る。1階平面図・2階平面図の印字からfloor_idを読み、図面全体をbboxで囲む。単一図面ならplan_regionsは1件、階が読めなければfloor_idは"unknown"。小さな屋根裏収納などの付属図は独立した階と断定しない。
- 各spaceのfloor_idは所属するplan_regionsのfloor_idと一致させる。図面の位置だけから階数を推測しない。
- 読めない空間を推測で作らない。bboxは文字だけでなく壁で囲まれた領域全体を示す。

JSON形式:
{"image_quality":{"level":"high|medium|low","notes":[]},"orientation":{"value":null,"confidence":"high|medium|low","evidence":"..."},"plan_regions":[{"floor_id":"1F","bbox":[0,0,0.5,1]},{"floor_id":"2F","bbox":[0.5,0,1,1]}],"spaces":[{"id":"S1","label":"LDK","space_type":"room|hall|storage|sanitary|exterior|vertical_circulation|other","floor_id":"1F|null","area_text":"14.5帖|null","bbox":[0,0,1,1],"confidence":"high|medium|low","evidence":"..."}]}
JSON以外は返さない。
""".strip()


def build_cropped_inventory_prompt(floor_id: str) -> str:
    return f"""
これは{floor_id}の平面図だけを切り出した画像です。この図面の壁で区切られた空間を、画像を直接見て列挙してください。
特に玄関と居室を分ける廊下・ホール、浴室・洗面所・トイレ、階段、収納を正しい位置と大きさで登録してください。
外部に離して描かれた収納小図を、LDKに接する部屋として広げてはいけません。
各bboxはこの切出し画像の左上[0,0]、右下[1,1]に正規化します。文字の位置ではなく、壁に囲まれた空間の外形です。
読めない空間は推測せず省略します。扉や接続、評価はまだ扱いません。
返答はspaces配列を持つJSONオブジェクトだけにしてください。各要素はid、label、space_type、bbox、confidence、evidenceを持ちます。
space_typeはroom、hall、sanitary、storage、exterior、vertical_circulation、otherのいずれか1語だけにします。候補を縦線で連結しないでください。
図面全体を1部屋のbboxとして返さず、壁で区切られた部屋を個別に列挙してください。
""".strip()


def build_hall_probe_prompt(floor_id: str, spaces: list[dict[str, Any]]) -> str:
    labels = [f"{space.get('id')}: {space.get('label')}" for space in spaces]
    return f"""
これは{floor_id}だけの間取り図です。既存の部屋一覧に、通行用の廊下・ホールが抜けていないか画像を確認してください。
既存の部屋: {', '.join(labels)}
玄関の土間・靴脱ぎ部分と、そこから居室・水回り・階段へ分岐する細長い共用動線を分けて見てください。通路に文字ラベルや玄関との扉がなくても、壁の配置、細長い形、床色の違いで動線部分を特定できればseparate_hallをtrueにします。階段と一体のオープンな通路でも構いません。
玄関の中だけ、階段の踏み面だけ、またはLDK内の通路ならfalseにしてください。一般的な住宅なら廊下があるはず、という推測はしないでください。
trueの場合のbboxは、この切出し画像の左上[0,0]、右下[1,1]に正規化した通路全体の外接矩形です。confidenceはhigh、medium、lowのいずれかです。
見分けられなければfalse、bboxはnull、confidenceはlowにしてください。evidenceは図面上で見えた根拠を短く記録してください。
separate_hall、bbox、confidence、evidenceの4項目だけをJSONで返してください。
""".strip()


def _valid_hall_probe(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("separate_hall") is not True:
        return False
    bbox = value.get("bbox")
    if not _valid_normalized_coordinates(bbox, 4) or value.get("confidence") != "high":
        return False
    x1, y1, x2, y2 = bbox
    return x1 < x2 and y1 < y2 and (x2 - x1) * (y2 - y1) < 0.5 and bool(str(value.get("evidence") or "").strip())


def _valid_cropped_inventory(value: Any, expected_count: int) -> bool:
    spaces = value.get("spaces") if isinstance(value, dict) else None
    if not isinstance(spaces, list) or not spaces:
        return False
    if expected_count >= 3 and len(spaces) < 2:
        return False
    allowed_types = {"room", "hall", "sanitary", "storage", "exterior", "vertical_circulation", "other"}
    ids = [item.get("id") for item in spaces if isinstance(item, dict)]
    return (
        len(ids) == len(spaces) and len(ids) == len(set(ids))
        and all(item.get("id") and item.get("space_type") in allowed_types
                and _valid_normalized_coordinates(item.get("bbox"), 4)
                and item["bbox"][0] < item["bbox"][2] and item["bbox"][1] < item["bbox"][3]
                for item in spaces)
    )


def build_topology_prompt(inventory: dict[str, Any]) -> str:
    """Detect openings first, then associate only the fixed inventory IDs."""
    return f"""
間取り画像と確定済みの空間一覧を使い、扉・開口を先に検出してから接続先を割り当ててください。
空間の追加・削除・ID変更は禁止です。

手順:
1. 扉の円弧、引戸線、壁の切れ目、掃き出し窓をopeningsへ重複なく登録する。
2. 各opening.positionの両側にあるspace IDを、bboxだけでなく壁の形状も見て決める。
3. 対応できるopeningだけconnectionsへ登録する。不明なら接続を推測せずunreadable_itemsへ記録する。
4. 窓と設備は後の専用工程で読むため、windowsとfixturesは空配列にする。明確な不存在だけ記録する。

出力を短く保つ:
- 指定されたキー以外は出力しない。evidenceは12文字以内、unreadable_itemsは1項目20文字以内にする。
- 同じ境界・窓・設備を繰り返さない。画像上で確認できる実数だけを出力する。
- JSONを途中で切らない。情報量が多い場合は、confidenceをlowにして省略し、unreadable_itemsへ短く記録する。

注意:
- 中央の廊下・ホールから左右の居室や水回りへ開く扉を、居室同士の直結と誤認しない。
- 居室に入る扉が図面上どこにも描かれていない場合、図面の描き漏れもあり得る。通常あるはずという理由で扉・connectionを補わない。全周を明瞭に確認できた場合だけnegative_observationsに「屋内入口なし」を記録し、不鮮明ならunreadable_itemsへ記録する。
- 浴室の入口は通常、隣接する洗面所・脱衣所側を重点確認する。
- キッチン設備はLDK内ならfixtureであり、独立したconnectionを作らない。
- opening_idは1つのconnectionにだけ使う。
- floor_idが異なるspace同士を直接接続してはいけない。階段（vertical_circulation）を介する場合だけ、同じ階のspaceと階段を接続する。
- 窓と設備はこの段階で出力しない。後の専用工程で再確認する。

【確定済み空間一覧】
{json.dumps(inventory["spaces"], ensure_ascii=False, separators=(',', ':'))}

openings, connections, windows, fixtures, negative_observations, unreadable_itemsを含むJSONだけを返す。
""".strip()


def build_feature_prompt(inventory: dict[str, Any]) -> str:
    return f"""
この間取り図に実際に描かれた窓と住宅設備だけを、画像を再確認してJSONで列挙してください。
扉・接続・部屋の再判定や採点は不要です。窓の細線、開口と外壁、キッチンのシンクとコンロ、トイレ便器、浴槽、洗面台を重点確認してください。
見えないものは推測しません。画像で判別できない項目はunreadable_itemsに記録してください。
各窓はid, space_id, faces_exterior, position, confidence, evidenceを持ち、各設備はspace_id, fixture, confidence, evidenceを持ちます。
座標はこの画像の左上[0,0]、右下[1,1]です。IDは以下の空間一覧だけを使用してください。
【空間一覧】{json.dumps(inventory['spaces'], ensure_ascii=False, separators=(',', ':'))}
{{"windows":[],"fixtures":[],"unreadable_items":[]}} の形式のJSONだけ返してください。
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

    required = {"image_quality", "orientation", "spaces", "openings", "connections", "windows", "fixtures", "negative_observations", "unreadable_items"}
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

    for space in structure["spaces"]:
        bbox = space.get("bbox")
        if not _valid_normalized_coordinates(bbox, 4) or bbox[0] > bbox[2] or bbox[1] > bbox[3]:
            raise RuntimeError(f"空間{space.get('id')}のbboxが不正です。")

    opening_ids = [item.get("id") for item in structure["openings"] if isinstance(item, dict)]
    if len(opening_ids) != len(structure["openings"]) or any(not value for value in opening_ids):
        raise RuntimeError("すべての扉・開口にIDが必要です。")
    if len(opening_ids) != len(set(opening_ids)):
        raise RuntimeError("扉・開口IDが重複しています。")
    for opening in structure["openings"]:
        if not _valid_normalized_coordinates(opening.get("position"), 2):
            raise RuntimeError(f"扉・開口{opening.get('id')}のpositionが不正です。")

    known_ids = set(space_ids)
    for connection in structure["connections"]:
        if not isinstance(connection, dict):
            raise RuntimeError("connectionsの要素が不正です。")
        endpoints = {connection.get("space_a"), connection.get("space_b")}
        unknown_ids = endpoints.difference(known_ids)
        if unknown_ids:
            raise RuntimeError(f"接続関係が未定義の空間を参照しています: {', '.join(sorted(map(str, unknown_ids)))}")
        if connection.get("opening_id") not in set(opening_ids):
            raise RuntimeError(f"接続関係が未定義の扉・開口を参照しています: {connection.get('opening_id')}")
        if not _valid_normalized_coordinates(connection.get("position"), 2):
            raise RuntimeError(f"接続{connection.get('id')}のpositionが不正です。")
    for window in structure["windows"]:
        if not isinstance(window, dict) or window.get("space_id") not in known_ids:
            raise RuntimeError("窓が未定義の空間を参照しています。")
        if not window.get("id") or not _valid_normalized_coordinates(window.get("position"), 2):
            raise RuntimeError("窓のIDまたは座標が不正です。")
    for fixture in structure["fixtures"]:
        if not isinstance(fixture, dict) or fixture.get("space_id") not in known_ids:
            raise RuntimeError("設備が未定義の空間を参照しています。")
    return structure


def _valid_normalized_coordinates(value: Any, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) and 0 <= item <= 1 for item in value)
    )


def normalize_plan_regions(inventory: dict[str, Any]) -> dict[str, Any]:
    """Assign a floor only when a labelled drawing region contains the space."""
    result = deepcopy(inventory)
    regions = result.get("plan_regions")
    if not isinstance(regions, list) or not regions:
        raise RuntimeError("図面領域plan_regionsを読めません。階を特定してから採点してください。")
    def canonical(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        normalized = value.translate(str.maketrans("０１２３４５６７８９", "0123456789")).strip()
        match = re.fullmatch(r"([0-9]+)\s*(?:F|階)(?:平面図)?", normalized, re.IGNORECASE)
        return f"{match.group(1)}F" if match else normalized

    for region in regions:
        if isinstance(region, dict):
            region["floor_id"] = canonical(region.get("floor_id"))
        bbox = region.get("bbox") if isinstance(region, dict) else None
        if not _valid_normalized_coordinates(bbox, 4) or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise RuntimeError("plan_regionsのbboxが不正です。")
    multiple = len(regions) > 1
    if multiple and len({region.get("floor_id") for region in regions}) != len(regions):
        raise RuntimeError("複数の図面領域の階識別が重複または欠落しています。")
    for space in result.get("spaces", []):
        space["floor_id"] = canonical(space.get("floor_id"))
        bbox = space.get("bbox", [])
        if not _valid_normalized_coordinates(bbox, 4):
            continue
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        matching = [region for region in regions if region["bbox"][0] <= center[0] <= region["bbox"][2]
                    and region["bbox"][1] <= center[1] <= region["bbox"][3]]
        if multiple and len(matching) != 1:
            raise RuntimeError(f"{space.get('id')}の所属階が一意に決まりません。")
        if matching:
            identified = matching[0].get("floor_id")
            if multiple and (not identified or identified in {"unknown", "不明", "null"}):
                raise RuntimeError(f"{space.get('id')}の階を読めません。採点を停止します。")
            if space.get("floor_id") not in (None, "", "unknown", "不明", identified):
                raise RuntimeError(f"{space.get('id')}の階指定が図面領域と矛盾しています。")
            space["floor_id"] = identified
    return result


def _map_point(point: list[float], region: dict[str, Any], *, to_global: bool) -> list[float]:
    x1, y1, x2, y2 = region["bbox"]
    if to_global:
        return [round(x1 + point[0] * (x2 - x1), 5), round(y1 + point[1] * (y2 - y1), 5)]
    return [round((point[0] - x1) / (x2 - x1), 5), round((point[1] - y1) / (y2 - y1), 5)]


def _crop_floor_region(image: Any, region: dict[str, Any]) -> Any:
    width, height = image.size
    x1, y1, x2, y2 = region["bbox"]
    return image.crop((round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)))


def _local_inventory(inventory: dict[str, Any], region: dict[str, Any]) -> dict[str, Any]:
    local = deepcopy(inventory)
    local["spaces"] = [space for space in local["spaces"] if space.get("floor_id") == region["floor_id"]]
    for space in local["spaces"]:
        bbox = space["bbox"]
        space["bbox"] = _map_point(bbox[:2], region, to_global=False) + _map_point(bbox[2:], region, to_global=False)
        space["bbox"] = [max(0, min(1, value)) for value in space["bbox"]]
    return local


def _globalize_observations(observations: dict[str, Any], region: dict[str, Any], index: int, prefix: str = "R") -> dict[str, Any]:
    result = deepcopy(observations)
    opening_ids = {}
    for item in result.get("openings", []):
        original = item["id"]
        item["id"] = f"{prefix}{index}_{original}"
        opening_ids[original] = item["id"]
        if _valid_normalized_coordinates(item.get("position"), 2):
            item["position"] = _map_point(item["position"], region, to_global=True)
    for item in result.get("connections", []):
        item["id"] = f"{prefix}{index}_{item['id']}"
        item["opening_id"] = opening_ids.get(item.get("opening_id"), item.get("opening_id"))
        if _valid_normalized_coordinates(item.get("position"), 2):
            item["position"] = _map_point(item["position"], region, to_global=True)
    for item in result.get("windows", []):
        item["id"] = f"{prefix}{index}_{item.get('id', 'W')}"
        if _valid_normalized_coordinates(item.get("position"), 2):
            item["position"] = _map_point(item["position"], region, to_global=True)
    return result


def validate_connections(structure: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Reject structurally implausible edges before they can be used for scoring."""
    enriched = deepcopy(structure)
    spaces = {space["id"]: space for space in enriched["spaces"]}
    openings = {item["id"]: item for item in enriched.get("openings", [])}
    warnings: list[str] = []
    seen_pairs: set[tuple[str, str, str]] = set()
    used_openings: set[str] = set()
    for connection in enriched["connections"]:
        prior_status = connection.get("validation_status", "accepted")
        connection["validation_status"] = prior_status
        reasons: list[str] = list(connection.get("validation_reasons", []))
        if prior_status in {"reject", "rejected", "uncertain"}:
            connection["traversable"] = False
        space_a = spaces[connection["space_a"]]
        space_b = spaces[connection["space_b"]]
        relation = connection.get("boundary_relation")
        traversable = connection.get("traversable") is True
        opening = openings.get(connection.get("opening_id"))
        pair_key = tuple(sorted((space_a["id"], space_b["id"]))) + (str(relation),)

        if space_a["id"] == space_b["id"]:
            reasons.append("同じ空間同士の接続")
        if traversable and relation in {"shared_wall_only", "unknown"}:
            reasons.append("通行不可の境界種別なのにtraversable=true")
        if pair_key in seen_pairs:
            reasons.append("同一境界の重複")
        seen_pairs.add(pair_key)
        if connection.get("opening_id") in used_openings:
            reasons.append("同じ扉・開口を複数接続に使用")
        used_openings.add(connection.get("opening_id"))

        if traversable and (connection.get("confidence") == "low" or not opening or opening.get("confidence") == "low"):
            reasons.append("接続または扉のconfidenceがlow")
        if opening:
            opening_position = opening.get("position")
            connection_position = connection.get("position")
            if _point_distance(opening_position, connection_position) > 0.03:
                reasons.append("connectionとopeningの座標が不一致")
            if traversable and (
                _point_to_bbox_distance(opening_position, space_a.get("bbox")) > 0.06
                or _point_to_bbox_distance(opening_position, space_b.get("bbox")) > 0.06
            ):
                reasons.append("扉座標が両空間の境界付近にない")

        types = {space_a.get("space_type"), space_b.get("space_type")}
        labels = f"{space_a.get('label', '')} {space_b.get('label', '')}"
        floor_a, floor_b = space_a.get("floor_id"), space_b.get("floor_id")
        if traversable and (not floor_a or not floor_b):
            reasons.append("接続先の図面領域・階が不明")
        if traversable and floor_a and floor_b and floor_a != floor_b:
            reasons.append("別階の図面同士を直接接続している。階段間はvertical_linksで別管理")
        if traversable and types & {"room", "storage", "vertical_circulation"} and not types & {"hall", "sanitary", "exterior"}:
            first_box, second_box = space_a.get("bbox"), space_b.get("bbox")
            if _valid_normalized_coordinates(first_box, 4) and _valid_normalized_coordinates(second_box, 4):
                if _bbox_gap(first_box, second_box) > 0.025:
                    reasons.append("空間同士が離れている")
                elif opening and (
                    _point_to_bbox_edge_distance(opening.get("position"), first_box) > 0.035
                    or _point_to_bbox_edge_distance(opening.get("position"), second_box) > 0.035
                ):
                    reasons.append("扉が両空間の境界上にない")
        if traversable and "バルコニー" in labels and "hall" in types:
            reasons.append("玄関・廊下とバルコニーの直結は要画像再確認")
        if traversable and "exterior" in types and ("storage" in types or "sanitary" in types):
            reasons.append("収納・水回りと屋外の直結は要画像再確認")
        if traversable and types == {"storage"}:
            reasons.append("収納同士を通行経路にしている")
        if traversable and types == {"storage", "hall"} and any(
            space.get("id", "").endswith("_HALL") and space.get("floor_id") in enriched.get("floor_hall_recoveries", [])
            for space in (space_a, space_b)
        ):
            reasons.append("補完した廊下の位置が概略のため収納への扉は要画像確認")
        if traversable and types in ({"storage", "sanitary"}, {"storage", "vertical_circulation"}):
            reasons.append("収納と水回り・階段の直結は要画像再確認")
        if traversable and "storage" in types and "屋根裏" in labels and "vertical_circulation" not in types:
            reasons.append("屋根裏収納への通常扉による直結は要画像再確認")
        if traversable and types == {"sanitary"} and "トイレ" in labels and "浴室" in labels:
            reasons.append("トイレと浴室の直結は洗面所側の扉を要再確認")
        if traversable and types == {"room", "sanitary"}:
            room = space_a if space_a.get("space_type") == "room" else space_b
            sanitary = space_b if room is space_a else space_a
            room_label = str(room.get("label", "")).upper()
            sanitary_label = str(sanitary.get("label", ""))
            if "LDK" not in room_label and any(word in sanitary_label for word in ("トイレ", "浴室", "便所")):
                reasons.append("一般居室とトイレ・浴室の直結は要画像再確認")
        if traversable and types == {"hall", "sanitary"}:
            sanitary = space_a if space_a.get("space_type") == "sanitary" else space_b
            if "浴室" in str(sanitary.get("label", "")):
                reasons.append("浴室と廊下の直結は洗面所・脱衣所側を要再確認")

        if reasons:
            reasons = list(dict.fromkeys(reasons))
            if prior_status == "accepted":
                connection["validation_status"] = "rejected"
            connection["validation_reasons"] = reasons
            connection["traversable"] = False
            warnings.append(f"{connection.get('id')}: {' / '.join(reasons)}")
    return enriched, warnings


def deduplicate_topology_observations(topology: dict[str, Any]) -> dict[str, Any]:
    """Keep one strongest observation per pair of spaces before verification."""
    result = deepcopy(topology)
    confidence_rank = {"high": 3, "medium": 2, "low": 1}
    chosen: dict[tuple[str, str], dict[str, Any]] = {}
    for connection in result.get("connections", []):
        first, second = str(connection.get("space_a", "")), str(connection.get("space_b", ""))
        key = tuple(sorted((first, second)))
        previous = chosen.get(key)
        if previous is None or confidence_rank.get(connection.get("confidence"), 0) > confidence_rank.get(previous.get("confidence"), 0):
            chosen[key] = connection
    result["connections"] = list(chosen.values())
    used_openings = {item.get("opening_id") for item in result["connections"]}
    result["openings"] = [item for item in result.get("openings", []) if item.get("id") in used_openings]
    return result


def discard_invalid_observations(inventory: dict[str, Any], topology: dict[str, Any]) -> dict[str, Any]:
    """Keep a bad model reference from aborting the whole image analysis."""
    result = deepcopy(topology)
    known_spaces = {space["id"] for space in inventory["spaces"]}
    unreadable = result.get("unreadable_items")
    if not isinstance(unreadable, list):
        unreadable = []
        result["unreadable_items"] = unreadable
    seen_openings = set()
    valid_openings = []
    for item in result.get("openings") or []:
        if (not isinstance(item, dict) or not item.get("id") or item["id"] in seen_openings
                or not _valid_normalized_coordinates(item.get("position"), 2)):
            unreadable.append("扉・開口のIDまたは座標が不正のため除外")
            continue
        valid_openings.append(item)
        seen_openings.add(item["id"])
    result["openings"] = valid_openings
    opening_ids = seen_openings
    valid_connections = []
    for item in result.get("connections") or []:
        if (not isinstance(item, dict) or item.get("space_a") not in known_spaces
                or item.get("space_b") not in known_spaces or item.get("opening_id") not in opening_ids
                or not _valid_normalized_coordinates(item.get("position"), 2)):
            unreadable.append(f"{item.get('id', '接続') if isinstance(item, dict) else '接続'}: 未定義の部屋・開口IDを除外")
            continue
        valid_connections.append(item)
    result["connections"] = valid_connections
    for key in ("windows", "fixtures"):
        valid_items = []
        for item in result.get(key) or []:
            if (not isinstance(item, dict) or item.get("space_id") not in known_spaces
                    or key == "windows" and (not item.get("id") or not _valid_normalized_coordinates(item.get("position"), 2))):
                unreadable.append(f"{key}: 未定義の部屋IDを除外")
                continue
            valid_items.append(item)
        result[key] = valid_items
    result["negative_observations"] = [
        item for item in result.get("negative_observations") or []
        if isinstance(item, dict) and item.get("space_id") in known_spaces
    ]
    return result


def _point_distance(first: Any, second: Any) -> float:
    if not _valid_normalized_coordinates(first, 2) or not _valid_normalized_coordinates(second, 2):
        return float("inf")
    return ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5


def _point_to_bbox_distance(point: Any, bbox: Any) -> float:
    """Euclidean distance from a normalized point to a rectangle (zero inside)."""
    if not _valid_normalized_coordinates(point, 2) or not _valid_normalized_coordinates(bbox, 4):
        return float("inf")
    x, y = point
    dx = max(bbox[0] - x, 0, x - bbox[2])
    dy = max(bbox[1] - y, 0, y - bbox[3])
    return (dx * dx + dy * dy) ** 0.5


def _point_to_bbox_edge_distance(point: Any, bbox: Any) -> float:
    if not _valid_normalized_coordinates(point, 2) or not _valid_normalized_coordinates(bbox, 4):
        return float("inf")
    x, y = point
    vertical = min(
        ((x - edge_x) ** 2 + max(bbox[1] - y, 0, y - bbox[3]) ** 2) ** 0.5
        for edge_x in (bbox[0], bbox[2])
    )
    horizontal = min(
        ((y - edge_y) ** 2 + max(bbox[0] - x, 0, x - bbox[2]) ** 2) ** 0.5
        for edge_y in (bbox[1], bbox[3])
    )
    return min(vertical, horizontal)


def _bbox_gap(first: list[float], second: list[float]) -> float:
    dx = max(first[0] - second[2], second[0] - first[2], 0)
    dy = max(first[1] - second[3], second[1] - first[3], 0)
    return (dx * dx + dy * dy) ** 0.5


def add_verified_topology(structure: dict[str, Any]) -> dict[str, Any]:
    """Create adjacency only from connections accepted by deterministic validation."""
    enriched, warnings = validate_connections(structure)
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

    rejected_count = sum(
        1 for connection in enriched["connections"]
        if connection.get("validation_status") == "rejected"
    )
    connection_count = len(enriched["connections"])
    accepted_count = len(direct_connections)
    rejected_majority = connection_count >= 3 and rejected_count >= accepted_count and rejected_count >= 2
    essential_spaces = [space["id"] for space in enriched["spaces"]
                        if space.get("space_type") in {"room", "hall", "sanitary", "vertical_circulation"}]
    connected_essential = sum(bool(neighbors[space_id]) for space_id in essential_spaces)
    low_coverage = len(essential_spaces) >= 4 and connected_essential * 2 < len(essential_spaces)
    discarded_refs = sum("未定義" in str(item) or "座標が不正" in str(item)
                         for item in enriched.get("unreadable_items", []))
    low_discard = discarded_refs >= 2
    low_reliability = rejected_majority or low_coverage or low_discard

    space_types = {space["id"]: space.get("space_type") for space in enriched["spaces"]}
    stair_spaces = [space for space in enriched["spaces"] if space.get("space_type") == "vertical_circulation"]
    vertical_links = [
        {"space_a": first["id"], "space_b": second["id"], "status": "unverified"}
        for index, first in enumerate(stair_spaces)
        for second in stair_spaces[index + 1:]
        if first.get("floor_id") and second.get("floor_id") and first["floor_id"] != second["floor_id"]
    ]
    enriched["verified_topology"] = {
        "traversable_neighbors": {key: sorted(value) for key, value in neighbors.items()},
        "direct_connections": direct_connections,
        "vertical_links": vertical_links,
        "non_transit_spaces": sorted(
            space_id for space_id, space_type in space_types.items() if space_type == "storage"
        ),
        "confirmed_absences": [
            observation
            for observation in enriched["negative_observations"]
            if observation.get("confidence") == "high"
        ],
        "validation_warnings": warnings,
        "connection_quality": {
            "observed_count": connection_count,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "connected_essential_count": connected_essential,
            "essential_count": len(essential_spaces),
            "discarded_reference_count": discarded_refs,
            "reason": "接続候補の大半を拒否" if rejected_majority else "主要空間の半数以上に確認済み入口がない" if low_coverage else "未定義ID・座標の観測が複数" if low_discard else "",
            "reliability": "low" if low_reliability else "usable",
        },
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
            anomalies.append(f"{space_id}({space.get('label')})の入口が確認できない。図面の描き漏れまたは認識漏れを再確認")
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
        same_floor_living = [item for item in living_ids if spaces[item].get("floor_id") == spaces[hall_id].get("floor_id")]
        if same_floor_living and not any(item in reachable for item in same_floor_living):
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
    """Ask a fresh visual pass for verdicts; never regenerate the whole structure."""
    anomalies = find_topology_anomalies(draft)
    anomaly_text = "\n".join(f"- {item}" for item in anomalies) or "- 自動検出なし"
    return f"""
間取り画像と【暫定JSON】を照合し、connectionsの各項目だけを個別判定してください。
空間・扉・窓・設備を再生成してはいけません。評価や改善提案も行いません。

重点確認:
- spacesのbbox、openingsのposition、connectionsのpositionを元画像と再照合する。
- connectionごとにopening_idが実際の扉・開口を指し、その座標が両空間の境界付近にあるか確認する。
- 暫定JSONを正しいと仮定せず、すべての traversable=true を元画像の扉・開口記号と再照合する。
- 扉記号がある壁の両側の空間だけを接続する。近くにある別室と接続しない。
- 玄関ホールや廊下はL字など不整形になり得る。単純なbboxだけで接続先を決めない。
- 各居室に通常の屋内入口があるか確認する。バルコニーや収納を通らないと居室へ入れない結果は、画像に明白な根拠がない限り誤読として再確認する。
- 居室への扉がどこにも描かれていない場合もある。図面不備の可能性として扱い、常識で廊下との扉を補わない。
- トイレ、洗面所、浴室の扉が、玄関ホール・廊下・LDKのどれに実際に開くか、扉の円弧と壁の切れ目から確認する。
- shared_wall_only、door、glazed_door、windowを混同しない。
- 掃き出し窓は、通行可能ならglazed_doorとしてconnectionsに、採光可能ならwindowsにも記録する。
- 明確な不存在だけnegative_observationsへ残す。単に検出できなかった項目は削除し、unreadable_itemsへ移す。
- confidenceは再照合後の確信度に修正する。不鮮明ならlowにする。
- 暫定JSONに存在するconnection IDをそれぞれ1回ずつ判定する。
- acceptは扉記号と両側空間を明瞭に確認できる場合だけ。
- rejectは接続先が明らかに違う場合。判別できなければuncertainにする。
- JSONは {{"decisions": [{{"connection_id":"C1","verdict":"accept|reject|uncertain","confidence":"high|medium|low","reason":"20文字以内"}}]}} の形だけを返す。

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


def extract_floor_plan_structure(
    image: Any, client: Any, model: str = DEFAULT_MODEL,
    progress: Callable[[str], None] | None = None,
    checkpoint_path: Path | None = None,
    resume_checkpoint: bool = True,
) -> dict[str, Any]:
    """Stage 1: inventory spaces, then detect openings and associate topology."""
    stage_data: dict[str, Any] = {}
    if resume_checkpoint and checkpoint_path and checkpoint_path.is_file():
        try:
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if saved.get("pipeline_version") == PIPELINE_CACHE_VERSION:
                stage_data = saved.get("stages", {})
        except (OSError, ValueError, TypeError):
            pass

    def remember(stage: str, value: dict[str, Any]) -> None:
        if checkpoint_path is None:
            return
        stage_data[stage] = deepcopy(value)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"pipeline_version": PIPELINE_CACHE_VERSION, "stages": stage_data}, ensure_ascii=False), encoding="utf-8")
        temporary.replace(checkpoint_path)

    if progress:
        progress("1/4 図面領域を読み取り中")
    inventory = deepcopy(stage_data.get("inventory"))
    if inventory is None:
        inventory_response = _generate_with_retry(
            client, model=model,
            contents=build_visual_contents(build_space_inventory_prompt(), image),
            config={"response_mime_type": "application/json", "response_json_schema": SPACE_INVENTORY_SCHEMA, "temperature": 0},
        )
        inventory = normalize_plan_regions(_parse_json_object(_response_text(inventory_response, "空間一覧"), "空間一覧"))
        remember("inventory", inventory)
    regions = inventory["plan_regions"]
    split = len(regions) > 1 and hasattr(image, "crop") and hasattr(image, "size")
    if split:
        refined_spaces = []
        fallbacks = []
        hall_recoveries = []
        floor_repair_attempts = []
        for index, region in enumerate(regions, 1):
            if progress:
                progress(f"1/4 部屋の位置を再読取中（領域 {index}/{len(regions)}）")
            cropped = _crop_floor_region(image, region)
            stage = f"region_inventory_{index}"
            local = deepcopy(stage_data.get(stage))
            expected_count = sum(space.get("floor_id") == region["floor_id"] for space in inventory["spaces"])
            fallback = bool(stage_data.get(f"fallback_{index}"))
            if local is not None and not _valid_cropped_inventory(local, expected_count):
                stage_data.pop(stage, None)
                stage_data.pop(f"topology_{index}", None)
                stage_data.pop(f"features_{index}", None)
                local = None
                fallback = True
            if local is None and not fallback:
                response = _generate_with_retry(
                    client, model=model,
                    contents=build_visual_contents(build_cropped_inventory_prompt(region["floor_id"]), cropped),
                    config={"response_mime_type": "application/json", "response_json_schema": CROPPED_INVENTORY_SCHEMA, "temperature": 0},
                )
                local = _parse_or_repair_json_object(
                    _response_text(response, "階別空間一覧"), client, model,
                    "階別空間一覧", CROPPED_INVENTORY_SCHEMA,
                )
                if not _valid_cropped_inventory(local, expected_count):
                    fallback = True
                    stage_data.pop(f"topology_{index}", None)
                    stage_data.pop(f"features_{index}", None)
                else:
                    remember(stage, local)
            if fallback:
                floor_repair_attempts.append(region["floor_id"])
                originals = [deepcopy(space) for space in inventory["spaces"] if space.get("floor_id") == region["floor_id"]]
                if not originals:
                    raise RuntimeError(f"{region['floor_id']}の部屋一覧を読めません。")
                has_hall = any("廊下" in str(space.get("label", "")) or "ホール" in str(space.get("label", ""))
                               for space in originals)
                if not has_hall:
                    probe_stage = f"hall_probe_v2_{index}"
                    probe = deepcopy(stage_data.get(probe_stage))
                    if probe is None:
                        if progress:
                            progress(f"1/4 廊下候補を再確認中（領域 {index}/{len(regions)}）")
                        response = _generate_with_retry(
                            client, model=model,
                            contents=build_visual_contents(build_hall_probe_prompt(region["floor_id"], originals), cropped),
                            config={"response_mime_type": "application/json", "response_json_schema": HALL_PROBE_SCHEMA, "temperature": 0},
                        )
                        probe = _parse_or_repair_json_object(
                            _response_text(response, "廊下の再確認"), client, model,
                            "廊下の再確認", HALL_PROBE_SCHEMA,
                        )
                        remember(probe_stage, probe)
                    if _valid_hall_probe(probe):
                        hall_id = f"R{index}_HALL"
                        if hall_id not in {space["id"] for space in originals}:
                            bbox = probe["bbox"]
                            originals.append({
                                "id": hall_id, "label": "廊下", "space_type": "hall", "floor_id": region["floor_id"],
                                "bbox": _map_point(bbox[:2], region, to_global=True)
                                        + _map_point(bbox[2:], region, to_global=True),
                                "confidence": "high", "evidence": probe["evidence"],
                            })
                            hall_recoveries.append(region["floor_id"])
                            integration_stage = f"hall_probe_integrated_v2_{index}"
                            if not stage_data.get(integration_stage):
                                stage_data.pop(f"topology_{index}", None)
                                remember(integration_stage, {"status": "integrated"})
                refined_spaces.extend(originals)
                fallbacks.append(region["floor_id"])
                remember(f"fallback_{index}", {"reason": "階別の部屋再読取が不完全"})
                continue
            spaces = local.get("spaces")
            if not isinstance(spaces, list) or not spaces:
                raise RuntimeError(f"{region['floor_id']}の部屋を読めません。")
            for space in spaces:
                if not isinstance(space, dict) or not _valid_normalized_coordinates(space.get("bbox"), 4):
                    raise RuntimeError(f"{region['floor_id']}の部屋座標が不正です。")
                space["id"] = f"R{index}_{space.get('id', 'S')}"
                space["floor_id"] = region["floor_id"]
                space["bbox"] = _map_point(space["bbox"][:2], region, to_global=True) + _map_point(space["bbox"][2:], region, to_global=True)
                refined_spaces.append(space)
        inventory["spaces"] = refined_spaces
        inventory["floor_reading_fallbacks"] = fallbacks
        inventory["floor_hall_recoveries"] = hall_recoveries
        inventory["floor_repair_attempts"] = floor_repair_attempts
    provisional = {
        "image_quality": inventory.get("image_quality"),
        "orientation": inventory.get("orientation"),
        "plan_regions": inventory.get("plan_regions"),
        "floor_reading_fallbacks": inventory.get("floor_reading_fallbacks", []),
        "floor_hall_recoveries": inventory.get("floor_hall_recoveries", []),
        "floor_repair_attempts": inventory.get("floor_repair_attempts", []),
        "spaces": inventory.get("spaces"),
        "openings": [], "connections": [], "windows": [], "fixtures": [],
        "negative_observations": [], "unreadable_items": [],
    }
    parse_structure_response(json.dumps(provisional, ensure_ascii=False))

    stages = [(_local_inventory(inventory, region), _crop_floor_region(image, region), region, index)
              for index, region in enumerate(regions, 1)] if split else [(inventory, image, None, 0)]
    topology = {"openings": [], "connections": [], "windows": [], "fixtures": [],
                "negative_observations": [], "unreadable_items": []}
    for local_inventory, local_image, region, index in stages:
        if progress:
            progress(f"2/4 扉・接続を確認中（領域 {index or 1}/{len(stages)}）")
        stage = f"topology_{index}"
        observed = deepcopy(stage_data.get(stage))
        if observed is None:
            topology_response = _generate_with_retry(
                client, model=model,
                contents=build_visual_contents(build_topology_prompt(local_inventory), local_image),
                config={"response_mime_type": "application/json", "response_json_schema": TOPOLOGY_OBSERVATION_SCHEMA, "temperature": 0},
            )
            observed = _parse_or_repair_json_object(
                _response_text(topology_response, "扉・接続結果"), client, model,
                "扉・接続結果", TOPOLOGY_OBSERVATION_SCHEMA,
            )
            remember(stage, observed)
        observed = discard_invalid_observations(local_inventory, observed)
        if region:
            observed = _globalize_observations(observed, region, index)
        for key in topology:
            values = observed.get(key)
            if isinstance(values, list):
                topology[key].extend(values)

    if hasattr(image, "crop") and hasattr(image, "size"):
        for local_inventory, local_image, region, index in stages:
            if progress:
                progress(f"3/4 窓・設備を確認中（領域 {index or 1}/{len(stages)}）")
            stage = f"features_{index}"
            features = deepcopy(stage_data.get(stage))
            if features is None:
                feature_response = _generate_with_retry(
                    client, model=model,
                    contents=build_visual_contents(build_feature_prompt(local_inventory), local_image),
                    config={"response_mime_type": "application/json", "response_json_schema": FEATURE_SCHEMA, "temperature": 0},
                )
                features = _parse_or_repair_json_object(
                    _response_text(feature_response, "窓・設備結果"), client, model,
                    "窓・設備結果", FEATURE_SCHEMA,
                )
                remember(stage, features)
            features = discard_invalid_observations(local_inventory, features)
            if region:
                features = _globalize_observations(features, region, index, prefix="F")
            for window in features.get("windows", []):
                if not any(existing.get("space_id") == window.get("space_id")
                           and _point_distance(existing.get("position"), window.get("position")) < 0.025
                           for existing in topology["windows"]):
                    if any(existing.get("id") == window.get("id") for existing in topology["windows"]):
                        window["id"] = f"F{index}_{window['id']}"
                    topology["windows"].append(window)
            existing_fixtures = {(item.get("space_id"), item.get("fixture")) for item in topology["fixtures"]}
            topology["fixtures"].extend(item for item in features.get("fixtures", [])
                                        if (item.get("space_id"), item.get("fixture")) not in existing_fixtures)
            if isinstance(features.get("unreadable_items"), list):
                topology["unreadable_items"].extend(features["unreadable_items"])
    topology = discard_invalid_observations(inventory, topology)
    topology = deduplicate_topology_observations(topology)
    combined = {**inventory, **topology}
    return parse_structure_response(json.dumps(combined, ensure_ascii=False))


def _parse_json_object(text: str, stage: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
    elif "{" in candidate and "}" in candidate:
        candidate = candidate[candidate.find("{") : candidate.rfind("}") + 1]
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{stage}をJSONとして解釈できません: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{stage}がJSONオブジェクトではありません。")
    return value


def _parse_or_repair_json_object(
    text: str,
    client: Any,
    model: str,
    stage: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Repair a malformed stage JSON once, without re-reading the image."""
    try:
        return _parse_json_object(text, stage)
    except RuntimeError as original_error:
        repair_prompt = f"""
次のJSONは構文だけが壊れています。内容を追加・削除・推測・要約せず、JSON構文だけを修復してください。
Markdownや説明は出力せず、JSONオブジェクトだけを返してください。配列要素、ID、文字列内容は維持してください。

【壊れたJSON】
{text}
""".strip()
        response = _generate_with_retry(
            client,
            model=model,
            contents=[repair_prompt],
            config={"response_mime_type": "application/json", "response_json_schema": schema, "temperature": 0},
        )
        try:
            return _parse_json_object(_response_text(response, f"{stage}のJSON修復結果"), stage)
        except RuntimeError as repair_error:
            raise RuntimeError(
                f"{stage}のJSONが不正で、自動修復にも失敗しました。"
                f" 元のエラー: {original_error}; 修復後: {repair_error}"
            ) from repair_error


def verify_floor_plan_structure(
    image: Any,
    draft: dict[str, Any],
    client: Any,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Stage 1 verification: merge edge verdicts into the preserved draft."""
    response = _generate_with_retry(
        client,
        model=model,
        contents=build_visual_contents(build_verification_prompt(draft), image),
        config={"response_mime_type": "application/json", "response_json_schema": VERIFICATION_JSON_SCHEMA, "temperature": 0},
    )
    return merge_verification_decisions(
        draft, _response_text(response, "再照合結果")
    )


def merge_verification_decisions(draft: dict[str, Any], text: str) -> dict[str, Any]:
    """Preserve observations and apply only per-connection verification verdicts."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"再照合結果をJSONとして解釈できません: {exc}") from exc
    decisions = payload.get("decisions") if isinstance(payload, dict) else None
    if not isinstance(decisions, list):
        raise RuntimeError("再照合結果にdecisions配列がありません。")

    decision_map = {
        item.get("connection_id"): item
        for item in decisions
        if isinstance(item, dict) and item.get("connection_id")
    }
    merged = deepcopy(draft)
    for connection in merged["connections"]:
        decision = decision_map.get(connection["id"])
        verdict = decision.get("verdict") if decision else "uncertain"
        if verdict not in {"accept", "reject", "uncertain"}:
            verdict = "uncertain"
        connection["verification_verdict"] = verdict
        connection["verification_confidence"] = decision.get("confidence", "low") if decision else "low"
        connection["verification_reason"] = decision.get("reason", "再照合の回答なし") if decision else "再照合の回答なし"
        if verdict != "accept":
            connection["traversable"] = False
            connection["validation_status"] = verdict
            connection["validation_reasons"] = [connection["verification_reason"]]
    return add_verified_topology(merged)


def build_analysis_prompt(knowledge: str, structure: dict[str, Any]) -> str:
    """Stage 2 prompt: request score proposals as machine-checkable JSON."""
    if not knowledge.strip():
        raise ValueError("参考知識が空です。")

    rubric_lines = "\n".join(
        f"- {name}: {points}点 — {description}"
        for name, points, description in SCORE_CRITERIA
    )
    structure_json = json.dumps(compact_structure_for_scoring(structure), ensure_ascii=False, separators=(",", ":"))

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
- vertical_linksのstatus=unverifiedは上下階を通れる根拠ではない。階段の記号だけで上下階の移動経路が完成したと断定しない。
- bboxや部屋の広さだけで家具配置の余裕、安全性、収納容量を断定しない。設備や窓の未検出を好条件の根拠にしない。
- 不足を指摘できる対象は verified_topology.confirmed_absences に列挙されたものだけ。この配列にない収納・窓などを「ない」「不足」と書かない。

【固定採点表・合計100点】
{rubric_lines}

【出力ルール】
- MarkdownではなくJSONだけを返す。
- criteriaは固定採点表と同じ名称・順序で8件すべて返す。
- proposed_scoreは配点以内の整数。ただしPython側で根拠を検証し、最終点を決定する。
- statusは、十分な高・中confidence根拠がある場合confirmed、一部だけならpartial、判断材料がない場合unverifiable。
- evidence_idsは実在するspace ID、connection ID、window IDだけ。validation_status=rejectedのconnection IDは禁止。
- 確認不能を良い状態だと仮定しない。確認不能な項目はstatus=unverifiableとする。
- good_points、concerns、improvements、expert_checksは短い文の配列。改善案は「高: ...」のように優先度を付ける。

JSON形式:
{{
  "criteria": [{{"name": "生活動線", "proposed_score": 0, "status": "confirmed|partial|unverifiable", "evidence_ids": ["S1", "C1"], "reason": "...", "knowledge_basis": "..."}}],
  "good_points": ["..."], "concerns": ["..."],
  "improvements": ["高: ..."], "expert_checks": ["..."]
}}

【構造化された画像読取結果ここから】
{structure_json}
【構造化された画像読取結果ここまで】

【参考知識ここから】
{knowledge}
【参考知識ここまで】
""".strip()


def compact_structure_for_scoring(structure: dict[str, Any]) -> dict[str, Any]:
    """Remove vision-only geometry and rejected edge details from the scoring prompt."""
    accepted_connections = [
        item for item in structure.get("connections", [])
        if item.get("validation_status", "accepted") == "accepted"
    ]
    return {
        "image_quality": structure.get("image_quality", {}),
        "orientation": structure.get("orientation", {}),
        "plan_regions": [{"floor_id": region.get("floor_id")} for region in structure.get("plan_regions", [])],
        "spaces": [
            {key: item.get(key) for key in ("id", "label", "space_type", "floor_id", "area_text", "confidence") if key in item}
            for item in structure.get("spaces", [])
        ],
        "connections": [
            {key: item.get(key) for key in ("id", "space_a", "space_b", "boundary_relation", "traversable", "confidence")}
            for item in accepted_connections
        ],
        "windows": [
            {key: item.get(key) for key in ("id", "space_id", "faces_space_id", "faces_exterior", "confidence") if key in item}
            for item in structure.get("windows", [])
        ],
        "fixtures": [
            {key: item.get(key) for key in ("space_id", "fixture", "confidence") if key in item}
            for item in structure.get("fixtures", [])
        ],
        "negative_observations": structure.get("verified_topology", {}).get("confirmed_absences", []),
        "verified_topology": structure.get("verified_topology", {}),
    }


def analyze_from_structure(
    structure: dict[str, Any],
    client: Any,
    knowledge: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Stage 2: get JSON proposals, then enforce evidence-based caps in Python."""
    if len(structure.get("plan_regions", [])) > 1:
        floors = {region.get("floor_id") for region in structure["plan_regions"]}
        if not floors or floors & {None, "unknown", "不明", "null"} or any(space.get("floor_id") not in floors for space in structure.get("spaces", [])):
            raise RuntimeError("複数階の所属が未確定のため採点を停止しました。")
    quality = structure.get("verified_topology", {}).get("connection_quality", {})
    hold_reasons = []
    if structure.get("manual_correction_status") == "draft":
        hold_reasons.append("位置修正が未承認の下書きのため採点を保留")
    if structure.get("floor_reading_fallbacks"):
        if structure.get("floor_hall_recoveries"):
            hold_reasons.append(
                f"{', '.join(structure['floor_reading_fallbacks'])}の階別部屋再読取が失敗。"
                "廊下候補のみ追加したが、位置は概略で他の部屋は初回の一覧を使用"
            )
        else:
            hold_reasons.append(f"{', '.join(structure['floor_reading_fallbacks'])}の階別部屋再読取が失敗し、初回の部屋一覧を使用")
    if (structure.get("image_quality", {}).get("level") == "high"
            and len(structure.get("spaces", [])) >= 3
            and not structure.get("windows") and not structure.get("fixtures")):
        hold_reasons.append("判読性が高い図面なのに窓・設備を1件も抽出できず、認識漏れの可能性が高い")
    if quality.get("reliability") == "low":
        hold_reasons.append(f"{quality.get('reason') or '通行経路の信頼性が不足'}。接続候補{quality.get('observed_count', 0)}件中{quality.get('rejected_count', 0)}件を拒否、主要空間{quality.get('essential_count', 0)}件中{quality.get('connected_essential_count', 0)}件に接続を確認")
    if hold_reasons:
        return {
            "status": "held", "total": None, "criteria": [],
            "hold_reasons": hold_reasons,
            "validation_adjustments": [],
        }
    response = _generate_with_retry(
        client,
        model=model,
        contents=[build_analysis_prompt(knowledge, structure)],
        config={"response_mime_type": "application/json", "response_json_schema": SCORE_JSON_SCHEMA, "temperature": 0},
    )
    score_text = _response_text(response, "分析結果")
    try:
        return validate_scoring_json(score_text, structure)
    except RuntimeError as original_error:
        repair_response = _generate_with_retry(
            client,
            model=model,
            contents=[
                "次の壊れた採点JSONを、内容を推測で追加せずJSON構文だけ修復してください。"
                "不足したcriteriaはstatus=unverifiable、proposed_score=0、evidence_ids=[]で補い、固定8項目を返してください。\n\n"
                + score_text
            ],
            config={"response_mime_type": "application/json", "response_json_schema": SCORE_JSON_SCHEMA, "temperature": 0},
        )
        try:
            return validate_scoring_json(_response_text(repair_response, "採点JSON修復結果"), structure)
        except RuntimeError as repair_error:
            raise RuntimeError(
                f"採点JSONの自動修復に失敗しました。元のエラー: {original_error}; 修復後: {repair_error}"
            ) from repair_error


def _valid_evidence_ids(structure: dict[str, Any]) -> set[str]:
    ids = {space["id"] for space in structure["spaces"]}
    ids.update(window.get("id") for window in structure["windows"] if window.get("id"))
    ids.update(
        connection.get("id")
        for connection in structure["connections"]
        if connection.get("id") and connection.get("validation_status", "accepted") == "accepted"
    )
    return ids


def _criterion_cap(name: str, allocation: int, structure: dict[str, Any]) -> tuple[int, str | None]:
    """Return a deterministic evidence ceiling for each scoring category."""
    warnings = structure.get("verified_topology", {}).get("validation_warnings", [])
    windows = structure.get("windows", [])
    glazed = any(
        item.get("boundary_relation") == "glazed_door"
        and item.get("validation_status", "accepted") == "accepted"
        for item in structure.get("connections", [])
    )
    spaces = structure.get("spaces", [])
    fixtures = structure.get("fixtures", [])
    connection_quality = structure.get("verified_topology", {}).get("connection_quality", {})
    direct_connections = structure.get("verified_topology", {}).get("direct_connections", [])

    if not direct_connections and name in {"生活動線", "ゾーニング・プライバシー", "家事効率"}:
        return allocation // 2, "確認済みの通行経路がない"

    if len(structure.get("plan_regions", [])) > 1 and name in {
        "生活動線", "ゾーニング・プライバシー", "家事効率",
    } and not any(link.get("status") == "verified" for link in structure.get("verified_topology", {}).get("vertical_links", [])):
        return allocation // 2, "上下階の階段接続が未検証のため、住戸全体の評価は確認不能"

    if connection_quality.get("reliability") == "low" and name in {
        "生活動線", "ゾーニング・プライバシー", "家事効率", "安全性・バリアフリー",
    }:
        return allocation // 2, "接続の大半が機械検証で拒否されたため中立点以下"

    if name == "採光・通風":
        if not windows and not glazed:
            return allocation // 2, "窓・ガラス戸の確認根拠がないため中立点以下"
        if not any(window.get("confidence") in {"high", "medium"} for window in windows) and not glazed:
            return allocation // 2, "窓の読取確度が低い"
        return round(allocation * 0.6), "隣棟・日射・通風経路を画像だけで確認できない"
    if name == "家具配置・居住性":
        return allocation // 2, "家具寸法・有効幅・開閉干渉を確認できない"
    if name == "収納":
        storage = [item for item in spaces if item.get("space_type") == "storage"]
        if not storage:
            return allocation // 2, "収納の確認根拠がないため中立点以下"
        connected = structure.get("verified_topology", {}).get("traversable_neighbors", {})
        if any(not connected.get(item["id"]) for item in storage):
            return allocation // 2, "一部収納へのアクセスが確認できず、容量も不明"
        return round(allocation * 0.6), "収納容量・使いやすさを画像だけで確認できない"
    if name == "家事効率" and not fixtures:
        return allocation // 2, "設備の確認根拠がないため中立点以下"
    if name == "安全性・バリアフリー":
        return allocation // 2, "段差・寸法・扉干渉・避難条件を画像だけで確認できない"
    if name == "将来対応・可変性":
        return allocation // 2, "構造・家族条件・可変壁の情報がないため中立点以下"
    if warnings and name in {"生活動線", "ゾーニング・プライバシー", "家事効率"}:
        return round(allocation * 0.6), "不自然な接続が検出されたため上限を制限"
    return allocation, None


def validate_scoring_json(text: str, structure: dict[str, Any]) -> dict[str, Any]:
    """Parse model proposals and deterministically reject unsupported/full scores."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"採点結果をJSONとして解釈できません: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("criteria"), list):
        raise RuntimeError("採点JSONにcriteria配列がありません。")

    proposed = {item.get("name"): item for item in raw["criteria"] if isinstance(item, dict)}
    valid_ids = _valid_evidence_ids(structure)
    light_ids = {window.get("id") for window in structure.get("windows", [])
                 if window.get("id") and window.get("confidence") in {"high", "medium"}}
    light_ids.update(item.get("id") for item in structure.get("connections", [])
                     if item.get("boundary_relation") == "glazed_door" and item.get("validation_status", "accepted") == "accepted")
    route_ids = {item.get("id") for item in structure.get("connections", [])
                 if item.get("validation_status", "accepted") == "accepted" and item.get("traversable") is True}
    connection_ids = {item.get("id") for item in structure.get("connections", [])}
    results = []
    adjustments = []
    status_ratios = {"confirmed": 1.0, "partial": 0.7, "unverifiable": 0.5}
    for name, allocation, _ in SCORE_CRITERIA:
        item = proposed.get(name, {})
        status = item.get("status") if item.get("status") in status_ratios else "unverifiable"
        requested_ids = item.get("evidence_ids", [])
        evidence_ids = [value for value in requested_ids if value in valid_ids]
        invalid_ids = [value for value in requested_ids if value not in valid_ids]
        if name == "収納":
            storage_ids = {space["id"] for space in structure["spaces"] if space.get("space_type") == "storage"}
            non_storage_ids = [value for value in evidence_ids if value not in storage_ids]
            evidence_ids = [value for value in evidence_ids if value in storage_ids]
            if non_storage_ids:
                invalid_ids.extend(non_storage_ids)
        if name == "採光・通風":
            invalid_ids.extend(value for value in evidence_ids if value not in light_ids)
            evidence_ids = [value for value in evidence_ids if value in light_ids]
        if name in {"生活動線", "家事効率"}:
            invalid_ids.extend(value for value in evidence_ids if value in connection_ids and value not in route_ids)
            evidence_ids = [value for value in evidence_ids if value not in connection_ids or value in route_ids]
        if not evidence_ids:
            status = "unverifiable"
        proposed_score = item.get("proposed_score", 0)
        if not isinstance(proposed_score, int):
            proposed_score = 0
        proposed_score = max(0, min(proposed_score, allocation))
        status_cap = round(allocation * status_ratios[status])
        evidence_cap, cap_reason = _criterion_cap(name, allocation, structure)
        final_score = min(proposed_score, status_cap, evidence_cap)
        if evidence_cap < allocation and cap_reason and status == "confirmed":
            status = "partial" if evidence_ids else "unverifiable"
        reasons = []
        if invalid_ids:
            reasons.append(f"無効な根拠IDを除外: {', '.join(map(str, invalid_ids))}")
        if not evidence_ids:
            reasons.append("有効な根拠IDがないため確認不能扱い")
        if cap_reason and final_score < proposed_score:
            reasons.append(cap_reason)
        if final_score != proposed_score:
            adjustments.append(f"{name}: {proposed_score}→{final_score}（{' / '.join(reasons) or '確認状態による上限'}）")
        reason = str(item.get("reason", "確認不能"))
        knowledge_basis = str(item.get("knowledge_basis", "該当根拠なし"))
        if cap_reason and evidence_cap < allocation:
            reason = f"{cap_reason}。この項目の良否は断定できない。"
            knowledge_basis = "参考知識は確認済み条件に限って適用"
        results.append({
            "name": name, "score": final_score, "allocation": allocation,
            "status": status, "evidence_ids": evidence_ids,
            "reason": reason,
            "knowledge_basis": knowledge_basis,
        })

    return {
        "criteria": results,
        "total": sum(item["score"] for item in results),
        "good_points": raw.get("good_points", []),
        "concerns": raw.get("concerns", []),
        "improvements": raw.get("improvements", []),
        "expert_checks": raw.get("expert_checks", []),
        "validation_adjustments": adjustments,
    }


def render_analysis_report(scoring: dict[str, Any], structure: dict[str, Any]) -> str:
    """Render final Markdown deterministically; the model never controls totals or table shape."""
    if scoring.get("status") == "held":
        warnings = structure.get("verified_topology", {}).get("validation_warnings", [])
        lines = [
            "# 総合評価: 採点保留", "", "## 保留理由", "",
            *(f"- {reason}" for reason in scoring.get("hold_reasons", [])),
            "", "## 読み取り条件", "",
            f"- 登録空間数: {len(structure.get('spaces', []))}",
            f"- 窓: {len(structure.get('windows', []))}件、設備: {len(structure.get('fixtures', []))}件",
            f"- 採用した通行可能接続: {len(structure.get('verified_topology', {}).get('direct_connections', []))}",
            "", "## 接続検証警告", "",
            *(f"- {warning}" for warning in warnings),
            "", "構造の再読取または人手確認を終えてから採点してください。",
        ]
        return "\n".join(lines)
    quality = structure.get("image_quality", {}).get("level", "unknown")
    quality_label = {"high": "高", "medium": "中", "low": "低"}.get(quality, "不明")
    warnings = structure.get("verified_topology", {}).get("validation_warnings", [])

    lines = [
        f"# 総合評価: {scoring['total']} / 100点", "", "## 読み取り条件",
        f"- 画像の判読性: {quality_label}",
        f"- 図面領域: {', '.join(str(region.get('floor_id', '不明')) for region in structure.get('plan_regions', [])) or '未記録'}",
        f"- 登録空間数: {len(structure.get('spaces', []))}",
        f"- 採用した通行可能接続: {len(structure.get('verified_topology', {}).get('direct_connections', []))}",
        f"- 未検証の階段間候補: {len(structure.get('verified_topology', {}).get('vertical_links', []))}",
        "", "## 採点表", "",
        "| 評価項目 | 得点 | 配点 | 確認状態 | 有効な根拠ID | 判定理由 | 参考知識 |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for item in scoring["criteria"]:
        cells = [item["name"], str(item["score"]), str(item["allocation"]), item["status"], ", ".join(item["evidence_ids"]) or "なし", item["reason"], item["knowledge_basis"]]
        lines.append("| " + " | ".join(str(cell).replace("|", "／").replace("\n", " ") for cell in cells) + " |")

    def add_section(title: str, values: list[Any], empty: str) -> None:
        lines.extend(["", f"## {title}"])
        cleaned = [str(value).strip() for value in values if str(value).strip()]
        lines.extend(f"- {value}" for value in cleaned or [empty])

    verified_good_points = [
        f"{item['name']}: {item['reason']}（根拠: {', '.join(item['evidence_ids'])}）"
        for item in scoring["criteria"]
        if item["status"] == "confirmed"
        and item["evidence_ids"]
        and item["score"] >= round(item["allocation"] * 0.7)
    ]
    verified_concerns = list(warnings) + list(scoring["validation_adjustments"])
    verified_improvements = [
        f"{'高' if item['score'] < item['allocation'] * 0.5 else '中'}: {item['name']}は追加資料・現地確認後に再評価する"
        for item in scoring["criteria"]
        if item["status"] != "confirmed" or item["score"] < item["allocation"] * 0.7
    ]
    expert_checks = [
        "方位、隣棟、法令、構造、段差、扉干渉および有効寸法",
        "画像で確認不能または機械検証で拒否された接続関係",
    ]

    add_section("接続検証警告", warnings, "機械検出なし")
    add_section("採点の自動補正", scoring["validation_adjustments"], "補正なし")
    add_section("良い点", verified_good_points, "検証済み根拠から断定できる項目なし")
    add_section("気になる点", verified_concerns, "機械検出なし")
    add_section("改善案", verified_improvements, "追加確認後に検討")
    add_section("専門家に確認すべき事項", expert_checks, "法令・構造・現地条件")
    lines.extend(["", "本評価は画像から確認できた範囲の参考情報であり、建築士による法的・構造的確認の代替ではありません。"])
    return "\n".join(lines)


def analyze_floor_plan(
    image: Any,
    client: Any,
    knowledge: str,
    model: str = DEFAULT_MODEL,
    progress: Callable[[str], None] | None = None,
    checkpoint_path: Path | None = None,
    resume_checkpoint: bool = True,
) -> FloorPlanAnalysis:
    """Run topology extraction first, then evaluate the extracted JSON."""
    draft = extract_floor_plan_structure(image, client, model, progress, checkpoint_path, resume_checkpoint)
    if progress:
        progress("接続を再照合中")
    structure = verify_floor_plan_structure(image, draft, client, model)
    if progress:
        progress("採点中")
    scoring = analyze_from_structure(structure, client, knowledge, model)
    report = render_analysis_report(scoring, structure)
    return FloorPlanAnalysis(draft_structure=draft, structure=structure, scoring=scoring, report=report)


def analyze_cached_structure(
    draft: dict[str, Any],
    structure: dict[str, Any],
    client: Any,
    knowledge: str,
    model: str = DEFAULT_MODEL,
) -> FloorPlanAnalysis:
    """Reuse cached vision results and execute only the scoring stage."""
    parse_structure_response(json.dumps(draft, ensure_ascii=False))
    structure = add_verified_topology(structure)
    scoring = analyze_from_structure(structure, client, knowledge, model)
    report = render_analysis_report(scoring, structure)
    return FloorPlanAnalysis(draft_structure=draft, structure=structure, scoring=scoring, report=report)

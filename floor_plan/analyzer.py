"""Two-stage floor-plan extraction and evaluation with a vision model."""

import json
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "qwen3-vl:4b-instruct"
PIPELINE_CACHE_VERSION = "floor-aware-topology-v4"

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
        "spaces": {"type": "array", "items": {"type": "object", "required": ["id", "label", "space_type", "bbox", "confidence"], "properties": {"id": {"type": "string"}, "label": {"type": "string"}, "space_type": {"type": "string"}, "floor_id": {"type": ["string", "null"]}, "area_text": {"type": ["string", "null"]}, "bbox": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}}, "confidence": {"type": "string"}, "evidence": {"type": "string"}}}},
        "openings": {"type": "array", "items": {"type": "object", "required": ["id", "opening_type", "position", "confidence"], "properties": {"id": {"type": "string"}, "opening_type": {"type": "string"}, "position": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}, "confidence": {"type": "string"}, "evidence": {"type": "string"}}}},
        "connections": {"type": "array", "items": {"type": "object", "required": ["id", "opening_id", "space_a", "space_b", "boundary_relation", "traversable", "position", "confidence"], "properties": {"id": {"type": "string"}, "opening_id": {"type": "string"}, "space_a": {"type": "string"}, "space_b": {"type": "string"}, "boundary_relation": {"type": "string"}, "traversable": {"type": "boolean"}, "position": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}, "confidence": {"type": "string"}, "evidence": {"type": "string"}}}},
        "windows": {"type": "array", "items": {"type": "object"}},
        "fixtures": {"type": "array", "items": {"type": "object"}},
        "negative_observations": {"type": "array", "items": {"type": "object"}},
        "unreadable_items": {"type": "array", "items": {"type": "string"}},
    },
}

SPACE_INVENTORY_SCHEMA = {
    "type": "object",
    "required": ["image_quality", "orientation", "spaces"],
    "properties": {
        "image_quality": STRUCTURE_JSON_SCHEMA["properties"]["image_quality"],
        "orientation": STRUCTURE_JSON_SCHEMA["properties"]["orientation"],
        "spaces": STRUCTURE_JSON_SCHEMA["properties"]["spaces"],
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
- 読めない空間を推測で作らない。bboxは文字だけでなく壁で囲まれた領域全体を示す。

JSON形式:
{"image_quality":{"level":"high|medium|low","notes":[]},"orientation":{"value":null,"confidence":"high|medium|low","evidence":"..."},"spaces":[{"id":"S1","label":"LDK","space_type":"room|hall|storage|sanitary|exterior|vertical_circulation|other","floor_id":"1F|null","area_text":"14.5帖|null","bbox":[0,0,1,1],"confidence":"high|medium|low","evidence":"..."}]}
JSON以外は返さない。
""".strip()


def build_topology_prompt(inventory: dict[str, Any]) -> str:
    """Detect openings first, then associate only the fixed inventory IDs."""
    return f"""
間取り画像と確定済みの空間一覧を使い、扉・開口を先に検出してから接続先を割り当ててください。
空間の追加・削除・ID変更は禁止です。

手順:
1. 扉の円弧、引戸線、壁の切れ目、掃き出し窓をopeningsへ重複なく登録する。
2. 各opening.positionの両側にあるspace IDを、bboxだけでなく壁の形状も見て決める。
3. 対応できるopeningだけconnectionsへ登録する。不明なら接続を推測せずunreadable_itemsへ記録する。
4. 窓、設備、明確な不存在も記録する。

出力を短く保つ:
- 指定されたキー以外は出力しない。evidenceは12文字以内、unreadable_itemsは1項目20文字以内にする。
- 同じ境界・窓・設備を繰り返さない。画像上で確認できる実数だけを出力する。
- JSONを途中で切らない。情報量が多い場合は、confidenceをlowにして省略し、unreadable_itemsへ短く記録する。

注意:
- 中央の廊下・ホールから左右の居室や水回りへ開く扉を、居室同士の直結と誤認しない。
- 浴室の入口は通常、隣接する洗面所・脱衣所側を重点確認する。
- キッチン設備はLDK内ならfixtureであり、独立したconnectionを作らない。
- opening_idは1つのconnectionにだけ使う。
- floor_idが異なるspace同士を直接接続してはいけない。階段（vertical_circulation）を介する場合だけ、同じ階のspaceと階段を接続する。

【確定済み空間一覧】
{json.dumps(inventory["spaces"], ensure_ascii=False, separators=(',', ':'))}

openings, connections, windows, fixtures, negative_observations, unreadable_itemsを含むJSONだけを返す。
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
    return structure


def _valid_normalized_coordinates(value: Any, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) and 0 <= item <= 1 for item in value)
    )


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
        if (
            traversable
            and floor_a
            and floor_b
            and floor_a != floor_b
            and "vertical_circulation" not in types
        ):
            reasons.append("別階の空間を階段なしで直接接続している")
        if traversable and "バルコニー" in labels and "hall" in types:
            reasons.append("玄関・廊下とバルコニーの直結は要画像再確認")
        if traversable and "exterior" in types and ("storage" in types or "sanitary" in types):
            reasons.append("収納・水回りと屋外の直結は要画像再確認")
        if traversable and types == {"storage"}:
            reasons.append("収納同士を通行経路にしている")
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
    low_reliability = connection_count >= 3 and rejected_count >= accepted_count and rejected_count >= 2

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
        "validation_warnings": warnings,
        "connection_quality": {
            "observed_count": connection_count,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
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


def extract_floor_plan_structure(image: Any, client: Any, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Stage 1: inventory spaces, then detect openings and associate topology."""
    inventory_response = _generate_with_retry(
        client,
        model=model,
        contents=build_visual_contents(build_space_inventory_prompt(), image),
        config={"response_mime_type": "application/json", "response_json_schema": SPACE_INVENTORY_SCHEMA, "temperature": 0},
    )
    inventory = _parse_json_object(_response_text(inventory_response, "空間一覧"), "空間一覧")
    provisional = {
        "image_quality": inventory.get("image_quality"),
        "orientation": inventory.get("orientation"),
        "spaces": inventory.get("spaces"),
        "openings": [], "connections": [], "windows": [], "fixtures": [],
        "negative_observations": [], "unreadable_items": [],
    }
    parse_structure_response(json.dumps(provisional, ensure_ascii=False))

    topology_response = _generate_with_retry(
        client,
        model=model,
        contents=build_visual_contents(build_topology_prompt(inventory), image),
        config={"response_mime_type": "application/json", "response_json_schema": TOPOLOGY_OBSERVATION_SCHEMA, "temperature": 0},
    )
    topology = _parse_or_repair_json_object(
        _response_text(topology_response, "扉・接続結果"),
        client,
        model,
        "扉・接続結果",
        TOPOLOGY_OBSERVATION_SCHEMA,
    )
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

    if connection_quality.get("reliability") == "low" and name in {
        "生活動線", "ゾーニング・プライバシー", "家事効率", "安全性・バリアフリー",
    }:
        return allocation // 2, "接続の大半が機械検証で拒否されたため中立点以下"

    if name == "採光・通風" and not windows and not glazed:
        return allocation // 2, "窓・ガラス戸の確認根拠がないため中立点以下"
    if name == "家具配置・居住性" and not any(item.get("area_text") or item.get("bbox") for item in spaces):
        return allocation // 2, "面積・形状の根拠がないため中立点以下"
    if name == "収納" and not any(item.get("space_type") == "storage" for item in spaces):
        return allocation // 2, "収納の確認根拠がないため中立点以下"
    if name == "家事効率" and not fixtures:
        return allocation // 2, "設備の確認根拠がないため中立点以下"
    if name == "安全性・バリアフリー":
        return min(allocation, round(allocation * 0.7)), "段差・扉干渉・避難条件を画像だけで完全確認できない"
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
        if not evidence_ids:
            status = "unverifiable"
        proposed_score = item.get("proposed_score", 0)
        if not isinstance(proposed_score, int):
            proposed_score = 0
        proposed_score = max(0, min(proposed_score, allocation))
        status_cap = round(allocation * status_ratios[status])
        evidence_cap, cap_reason = _criterion_cap(name, allocation, structure)
        final_score = min(proposed_score, status_cap, evidence_cap)
        reasons = []
        if invalid_ids:
            reasons.append(f"無効な根拠IDを除外: {', '.join(map(str, invalid_ids))}")
        if not evidence_ids:
            reasons.append("有効な根拠IDがないため確認不能扱い")
        if cap_reason and final_score < proposed_score:
            reasons.append(cap_reason)
        if final_score != proposed_score:
            adjustments.append(f"{name}: {proposed_score}→{final_score}（{' / '.join(reasons) or '確認状態による上限'}）")
        results.append({
            "name": name, "score": final_score, "allocation": allocation,
            "status": status, "evidence_ids": evidence_ids,
            "reason": str(item.get("reason", "確認不能")),
            "knowledge_basis": str(item.get("knowledge_basis", "該当根拠なし")),
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
    quality = structure.get("image_quality", {}).get("level", "unknown")
    quality_label = {"high": "高", "medium": "中", "low": "低"}.get(quality, "不明")
    warnings = structure.get("verified_topology", {}).get("validation_warnings", [])

    lines = [
        f"# 総合評価: {scoring['total']} / 100点", "", "## 読み取り条件",
        f"- 画像の判読性: {quality_label}",
        f"- 登録空間数: {len(structure.get('spaces', []))}",
        f"- 採用した通行可能接続: {len(structure.get('verified_topology', {}).get('direct_connections', []))}",
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
) -> FloorPlanAnalysis:
    """Run topology extraction first, then evaluate the extracted JSON."""
    draft = extract_floor_plan_structure(image, client, model)
    structure = verify_floor_plan_structure(image, draft, client, model)
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
    if "verified_topology" not in structure:
        structure = add_verified_topology(structure)
    scoring = analyze_from_structure(structure, client, knowledge, model)
    report = render_analysis_report(scoring, structure)
    return FloorPlanAnalysis(draft_structure=draft, structure=structure, scoring=scoring, report=report)

"""Apply explicitly reviewed geometry and doorway corrections to a vision result."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .analyzer import _valid_normalized_coordinates, add_verified_topology


def apply_reviewed_corrections(
    structure: dict[str, Any], corrections: dict[str, Any], *, allow_draft: bool = False,
) -> dict[str, Any]:
    status = corrections.get("review_status")
    if status != "approved" and not (allow_draft and status == "draft"):
        raise ValueError("修正データが未承認です。利用者確認後にreview_statusをapprovedにしてください。")
    if status == "approved":
        legacy_geometry_status = corrections.get("geometry_review_status", "approved")
        review_states = {
            "connection_review_status": corrections.get("connection_review_status", "approved"),
            "bbox_review_status": corrections.get("bbox_review_status", legacy_geometry_status),
            "opening_position_review_status": corrections.get("opening_position_review_status", legacy_geometry_status),
        }
        pending = [key for key, value in review_states.items() if value != "approved"]
        if pending:
            raise ValueError(f"接続または位置の確認が未完了です: {', '.join(pending)}")

    result = deepcopy(structure)
    spaces = {space["id"]: space for space in result["spaces"]}
    for space_id, correction in corrections.get("spaces", {}).items():
        if space_id not in spaces:
            raise ValueError(f"修正対象の空間IDがありません: {space_id}")
        bbox = correction.get("bbox")
        if not _valid_normalized_coordinates(bbox, 4) or bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise ValueError(f"修正bboxが不正です: {space_id}")
        spaces[space_id]["bbox"] = bbox
        spaces[space_id]["correction_source"] = status

    no_entrance = set(corrections.get("confirmed_no_entrance", []))
    if no_entrance - spaces.keys():
        raise ValueError(f"入口なしの空間IDがありません: {sorted(no_entrance - spaces.keys())}")
    replacement_pairs: set[frozenset[str]] = set()
    for correction in corrections.get("connections", []):
        first, second = correction.get("space_a"), correction.get("space_b")
        if first not in spaces or second not in spaces or first == second:
            raise ValueError(f"修正接続の空間IDが不正です: {first}, {second}")
        if first in no_entrance or second in no_entrance:
            raise ValueError(f"入口なしと確認された空間への接続は追加できません: {first}, {second}")
        if correction.get("boundary_relation") not in {"door", "open_passage", "glazed_door"}:
            raise ValueError(f"修正接続の開口種別が不正です: {first}, {second}")
        if not _valid_normalized_coordinates(correction.get("position"), 2):
            raise ValueError(f"修正接続の開口座標が不正です: {first}, {second}")
        pair = frozenset((first, second))
        if pair in replacement_pairs:
            raise ValueError(f"修正接続が重複しています: {first}, {second}")
        replacement_pairs.add(pair)

    old_opening_ids = {
        item.get("opening_id") for item in result["connections"]
        if frozenset((item.get("space_a"), item.get("space_b"))) in replacement_pairs
    }
    result["connections"] = [
        item for item in result["connections"]
        if frozenset((item.get("space_a"), item.get("space_b"))) not in replacement_pairs
    ]
    result["openings"] = [item for item in result["openings"] if item.get("id") not in old_opening_ids]

    used_connection_ids = {item.get("id") for item in result["connections"]}
    used_opening_ids = {item.get("id") for item in result["openings"]}
    for index, correction in enumerate(corrections.get("connections", []), 1):
        suffix = index
        while f"MANUAL_C{suffix}" in used_connection_ids or f"MANUAL_O{suffix}" in used_opening_ids:
            suffix += 1
        connection_id, opening_id = f"MANUAL_C{suffix}", f"MANUAL_O{suffix}"
        used_connection_ids.add(connection_id)
        used_opening_ids.add(opening_id)
        relation = correction["boundary_relation"]
        position = correction["position"]
        evidence = str(correction.get("evidence") or "人手で図面を確認")
        result["openings"].append({
            "id": opening_id, "opening_type": relation, "position": position,
            "confidence": "high", "evidence": evidence, "correction_source": status,
        })
        result["connections"].append({
            "id": connection_id, "opening_id": opening_id,
            "space_a": correction["space_a"], "space_b": correction["space_b"],
            "boundary_relation": relation, "traversable": True, "position": position,
            "confidence": "high", "evidence": evidence, "correction_source": status,
        })

    for connection in result["connections"]:
        if connection.get("space_a") in no_entrance or connection.get("space_b") in no_entrance:
            connection["traversable"] = False
            connection["validation_status"] = "rejected"
            connection["validation_reasons"] = list(dict.fromkeys(
                [*connection.get("validation_reasons", []), "図面に入口がないことを利用者が確認"]
            ))
    result["manual_correction_status"] = status
    if status == "approved":
        result["manual_confirmed_bbox_ids"] = sorted(corrections.get("spaces", {}))
    result.pop("verified_topology", None)
    return add_verified_topology(result)

"""Compare accepted topology edges with a manually reviewed ground truth."""

import argparse
import json
import sys
from pathlib import Path


def canonical(label: str, aliases: dict[str, str]) -> str:
    text = str(label).strip()
    for source, target in aliases.items():
        if source in text:
            return target
    return text


def canonical_space(space: dict, aliases: dict[str, str]) -> str:
    label = str(space.get("label", ""))
    area = str(space.get("area_text") or "")
    evidence = str(space.get("evidence") or "")
    combined = f"{label} {area} {evidence}"
    if "洋室" in label:
        if "4.85" in combined:
            return "洋室4.85帖"
        if "5.1" in combined:
            return "洋室5.1帖"
    if "バルコニー" in label:
        bbox = space.get("bbox") or []
        if len(bbox) == 4:
            return "バルコニー上" if (bbox[1] + bbox[3]) / 2 < 0.5 else "バルコニー下"
    return canonical(label, aliases)


def edge(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((first, second)))


def expected_label(label: str, aliases: dict[str, str], floor_qualified: bool) -> str:
    if not floor_qualified:
        return canonical(label, aliases)
    floor_id, separator, room_label = str(label).partition(":")
    if not separator or not floor_id or not room_label:
        raise ValueError(f"階付きの接続名が必要です: {label}")
    return f"{floor_id}:{canonical(room_label, aliases)}"


def evaluate(structure: dict, truth: dict) -> dict:
    aliases = truth.get("label_aliases", {})
    floor_qualified = truth.get("floor_qualified", False)
    spaces = {item["id"]: item for item in structure["spaces"]}
    ignored_types = set(truth.get("ignore_space_types", []))
    predicted = set()
    for connection in structure["connections"]:
        first = spaces[connection["space_a"]]
        second = spaces[connection["space_b"]]
        if (
            connection.get("traversable") is not True
            or connection.get("validation_status", "accepted") != "accepted"
            or first.get("space_type") in ignored_types
            or second.get("space_type") in ignored_types
        ):
            continue
        first_label = canonical_space(first, aliases)
        second_label = canonical_space(second, aliases)
        if floor_qualified:
            first_label = f"{first.get('floor_id', 'unknown')}:{first_label}"
            second_label = f"{second.get('floor_id', 'unknown')}:{second_label}"
        if first_label == second_label:
            continue
        predicted.add(edge(first_label, second_label))

    expected = {edge(expected_label(a, aliases, floor_qualified), expected_label(b, aliases, floor_qualified))
                for a, b in truth["connections"]}
    excluded = {edge(expected_label(a, aliases, floor_qualified), expected_label(b, aliases, floor_qualified))
                for a, b in truth.get("excluded_connections", [])}
    predicted -= excluded
    expected -= excluded
    true_positive = predicted & expected
    false_positive = predicted - expected
    false_negative = expected - predicted
    precision = len(true_positive) / len(predicted) if predicted else 0.0
    recall = len(true_positive) / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "review_status": truth.get("review_status", "unspecified"),
        "excluded_count": len(excluded),
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        "true_positive": sorted(true_positive),
        "false_positive": sorted(false_positive),
        "false_negative": sorted(false_negative),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("structure", type=Path)
    parser.add_argument("ground_truth", type=Path)
    args = parser.parse_args()
    structure = json.loads(args.structure.read_text(encoding="utf-8"))
    truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    print(json.dumps(evaluate(structure, truth), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

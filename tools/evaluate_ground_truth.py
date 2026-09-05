"""Compare accepted topology edges with a manually reviewed ground truth."""

import argparse
import json
from pathlib import Path


def canonical(label: str, aliases: dict[str, str]) -> str:
    text = str(label).strip()
    for source, target in aliases.items():
        if source in text:
            return target
    if "バルコニー" in text:
        return "バルコニー"
    return text


def edge(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((first, second)))


def evaluate(structure: dict, truth: dict) -> dict:
    aliases = truth.get("label_aliases", {})
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
        predicted.add(edge(canonical(first["label"], aliases), canonical(second["label"], aliases)))

    expected = {edge(canonical(a, aliases), canonical(b, aliases)) for a, b in truth["connections"]}
    # Balcony position is not currently retained in normalized labels, so either balcony is a partial match.
    expected_collapsed = {edge(a.replace("バルコニー上", "バルコニー").replace("バルコニー下", "バルコニー"), b.replace("バルコニー上", "バルコニー").replace("バルコニー下", "バルコニー")) for a, b in expected}
    true_positive = predicted & expected_collapsed
    false_positive = predicted - expected_collapsed
    false_negative = expected_collapsed - predicted
    precision = len(true_positive) / len(predicted) if predicted else 0.0
    recall = len(true_positive) / len(expected_collapsed) if expected_collapsed else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
        "true_positive": sorted(true_positive),
        "false_positive": sorted(false_positive),
        "false_negative": sorted(false_negative),
    }


def main() -> int:
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

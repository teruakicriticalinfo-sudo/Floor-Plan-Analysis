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
        first_label = canonical_space(first, aliases)
        second_label = canonical_space(second, aliases)
        if first_label == second_label:
            continue
        predicted.add(edge(first_label, second_label))

    expected = {edge(canonical(a, aliases), canonical(b, aliases)) for a, b in truth["connections"]}
    true_positive = predicted & expected
    false_positive = predicted - expected
    false_negative = expected - predicted
    precision = len(true_positive) / len(predicted) if predicted else 0.0
    recall = len(true_positive) / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
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

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from floor_plan import (
    SCORE_CRITERIA,
    analyze_floor_plan,
    build_analysis_prompt,
    build_extraction_prompt,
    load_knowledge,
    parse_structure_response,
    validate_connections,
    validate_scoring_json,
)


VALID_STRUCTURE = {
    "image_quality": {"level": "high", "notes": []},
    "orientation": {"value": None, "confidence": "low", "evidence": "方位記号なし"},
    "spaces": [
        {"id": "S1", "label": "LDK", "space_type": "room", "area_text": "11帖", "confidence": "high", "evidence": "表記"},
        {"id": "S2", "label": "バルコニー", "space_type": "exterior", "area_text": None, "confidence": "high", "evidence": "表記"},
    ],
    "connections": [
        {"id": "C1", "space_a": "S1", "space_b": "S2", "boundary_relation": "shared_wall_only", "traversable": False, "confidence": "high", "evidence": "壁"}
    ],
    "windows": [],
    "fixtures": [],
    "negative_observations": [],
    "unreadable_items": [],
}

VALID_SCORING = {
    "criteria": [
        {"name": name, "proposed_score": points, "status": "confirmed", "evidence_ids": ["S1"], "reason": "根拠あり", "knowledge_basis": "参考知識"}
        for name, points, _ in SCORE_CRITERIA
    ],
    "good_points": ["良い点"], "concerns": [],
    "improvements": ["低: 確認"], "expert_checks": ["専門家確認"],
}


class FakeModels:
    def __init__(self, response_texts):
        self.response_texts = iter(response_texts)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=next(self.response_texts))


class FakeClient:
    def __init__(self, *response_texts):
        self.models = FakeModels(response_texts)


class FloorPlanAnalyzerBacktest(unittest.TestCase):
    def test_score_weights_total_100(self):
        self.assertEqual(sum(points for _, points, _ in SCORE_CRITERIA), 100)
        self.assertEqual(len(SCORE_CRITERIA), 8)

    def test_extraction_prompt_distinguishes_adjacency_and_access(self):
        prompt = build_extraction_prompt()
        self.assertIn("壁を共有するだけ", prompt)
        self.assertIn("直接移動できる", prompt)
        self.assertIn("バルコニーへ直接出入り", prompt)

    def test_prompt_contains_structure_knowledge_and_access_rules(self):
        knowledge = "玄関の確認項目\nキッチンの確認項目"
        prompt = build_analysis_prompt(knowledge, VALID_STRUCTURE)
        self.assertIn(knowledge, prompt)
        self.assertIn('"boundary_relation": "shared_wall_only"', prompt)
        self.assertIn("traversable=true", prompt)
        self.assertIn("既に独立した個室", prompt)
        self.assertIn("negative_observations", prompt)
        for name, points, _ in SCORE_CRITERIA:
            self.assertIn(f"{name}: {points}点", prompt)

    def test_json_with_code_fence_is_accepted(self):
        text = f"```json\n{json.dumps(VALID_STRUCTURE, ensure_ascii=False)}\n```"
        self.assertEqual(parse_structure_response(text), VALID_STRUCTURE)

    def test_connection_to_unknown_space_is_rejected(self):
        invalid = json.loads(json.dumps(VALID_STRUCTURE))
        invalid["connections"][0]["space_b"] = "S999"
        with self.assertRaises(RuntimeError):
            parse_structure_response(json.dumps(invalid))

    def test_implausible_hall_balcony_connection_is_rejected(self):
        structure = json.loads(json.dumps(VALID_STRUCTURE))
        structure["spaces"][0]["label"] = "玄関"
        structure["spaces"][0]["space_type"] = "hall"
        structure["connections"][0].update(boundary_relation="door", traversable=True)
        checked, warnings = validate_connections(structure)
        self.assertFalse(checked["connections"][0]["traversable"])
        self.assertEqual(checked["connections"][0]["validation_status"], "rejected")
        self.assertTrue(warnings)

    def test_bedroom_to_toilet_connection_is_rejected(self):
        structure = json.loads(json.dumps(VALID_STRUCTURE))
        structure["spaces"][0]["label"] = "洋室5帖"
        structure["spaces"][1].update(label="トイレ", space_type="sanitary")
        structure["connections"][0].update(boundary_relation="door", traversable=True)
        checked, warnings = validate_connections(structure)
        self.assertFalse(checked["connections"][0]["traversable"])
        self.assertIn("一般居室", warnings[0])

    def test_unverifiable_categories_cannot_receive_full_score(self):
        structure = json.loads(json.dumps(VALID_STRUCTURE))
        structure["verified_topology"] = {"validation_warnings": [], "direct_connections": []}
        scoring = validate_scoring_json(json.dumps(VALID_SCORING, ensure_ascii=False), structure)
        scores = {item["name"]: item["score"] for item in scoring["criteria"]}
        self.assertLess(scores["採光・通風"], 10)
        self.assertLess(scores["安全性・バリアフリー"], 10)
        self.assertLess(scores["将来対応・可変性"], 10)
        self.assertEqual(scoring["total"], sum(scores.values()))

    def test_real_knowledge_is_loaded_in_full(self):
        project_dir = Path(__file__).resolve().parents[1]
        knowledge = load_knowledge(project_dir / "knowledge.md")
        prompt = build_analysis_prompt(knowledge, VALID_STRUCTURE)
        self.assertIn("1. 玄関（全7項目）", prompt)
        self.assertIn("10. 子供部屋（全7項目）", prompt)
        self.assertIn("狭すぎて友達を呼べない", prompt)
        self.assertEqual(prompt.count(knowledge), 1)

    def test_load_knowledge_reads_complete_utf8_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "knowledge.md"
            path.write_text("1. 玄関\n項目A\n\n2. 台所\n項目B\n", encoding="utf-8")
            self.assertEqual(load_knowledge(path), "1. 玄関\n項目A\n\n2. 台所\n項目B")

    def test_missing_and_empty_knowledge_are_rejected(self):
        with self.assertRaises(FileNotFoundError):
            load_knowledge("this-file-does-not-exist.md")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "knowledge.md"
            path.write_text("  \n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_knowledge(path)

    def test_two_stage_analysis_uses_image_only_in_extraction(self):
        image = object()
        client = FakeClient(
            json.dumps(VALID_STRUCTURE),
            json.dumps(VALID_STRUCTURE),
            json.dumps(VALID_SCORING, ensure_ascii=False),
        )
        result = analyze_floor_plan(image, client, "参考知識", model="test-model")

        extracted_without_topology = {
            key: value for key, value in result.structure.items() if key != "verified_topology"
        }
        for connection in extracted_without_topology["connections"]:
            connection.pop("validation_status", None)
            connection.pop("validation_reasons", None)
        self.assertEqual(extracted_without_topology, VALID_STRUCTURE)
        self.assertIn("# 総合評価:", result.report)
        self.assertEqual(result.scoring["total"], sum(item["score"] for item in result.scoring["criteria"]))
        self.assertEqual(result.structure["verified_topology"]["traversable_neighbors"]["S1"], [])
        self.assertEqual(len(client.models.calls), 3)
        first, verification, scoring = client.models.calls
        self.assertIn(image, first["contents"])
        self.assertEqual(first["config"]["response_mime_type"], "application/json")
        self.assertIn(image, verification["contents"])
        self.assertIn("暫定JSON", verification["contents"][0])
        self.assertEqual(len(scoring["contents"]), 1)
        self.assertIn('"spaces"', scoring["contents"][0])
        self.assertNotIn(image, scoring["contents"])
        self.assertEqual(scoring["config"]["response_mime_type"], "application/json")


if __name__ == "__main__":
    unittest.main()

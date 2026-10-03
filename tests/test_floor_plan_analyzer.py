import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from floor_plan import (
    PIPELINE_CACHE_VERSION,
    SCORE_CRITERIA,
    analyze_cached_structure,
    analyze_floor_plan,
    build_analysis_prompt,
    build_extraction_prompt,
    compact_structure_for_scoring,
    deduplicate_topology_observations,
    load_knowledge,
    merge_verification_decisions,
    parse_structure_response,
    load_structure_cache,
    create_analysis_client,
    save_structure_cache,
    structure_cache_key,
    validate_connections,
    validate_scoring_json,
)
from floor_plan.analyzer import (
    TOPOLOGY_OBSERVATION_SCHEMA, _parse_or_repair_json_object,
    add_verified_topology, discard_invalid_observations, extract_floor_plan_structure, normalize_plan_regions,
    _recheck_empty_features, _valid_cropped_inventory, build_feature_prompt, CROPPED_INVENTORY_SCHEMA,
    _valid_hall_probe, sanitize_visual_features,
)
from floor_plan.providers import OllamaClient


VALID_STRUCTURE = {
    "image_quality": {"level": "high", "notes": []},
    "orientation": {"value": None, "confidence": "low", "evidence": "方位記号なし"},
    "plan_regions": [{"floor_id": "unknown", "bbox": [0.0, 0.0, 1.0, 1.0]}],
    "spaces": [
        {"id": "S1", "label": "LDK", "space_type": "room", "floor_id": "unknown", "area_text": "11帖", "bbox": [0.0, 0.0, 0.6, 1.0], "confidence": "high", "evidence": "表記"},
        {"id": "S2", "label": "バルコニー", "space_type": "exterior", "floor_id": "unknown", "area_text": None, "bbox": [0.6, 0.0, 1.0, 1.0], "confidence": "high", "evidence": "表記"},
    ],
    "openings": [
        {"id": "O1", "opening_type": "unknown", "position": [0.6, 0.5], "confidence": "high", "evidence": "境界"}
    ],
    "connections": [
        {"id": "C1", "opening_id": "O1", "space_a": "S1", "space_b": "S2", "boundary_relation": "shared_wall_only", "traversable": False, "position": [0.6, 0.5], "confidence": "high", "evidence": "壁"}
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
    def test_floor_regions_assign_floor_and_reject_ambiguous_space(self):
        inventory = {
            "plan_regions": [{"floor_id": "1階", "bbox": [0, 0, 0.48, 1]},
                             {"floor_id": "２階平面図", "bbox": [0.52, 0, 1, 1]}],
            "spaces": [{"id": "S1", "bbox": [0.1, 0.1, 0.2, 0.2], "floor_id": None},
                       {"id": "S2", "bbox": [0.6, 0.1, 0.8, 0.2], "floor_id": None}],
        }
        normalized = normalize_plan_regions(inventory)
        self.assertEqual([space["floor_id"] for space in normalized["spaces"]], ["1F", "2F"])
        inventory["spaces"][1]["bbox"] = [0.48, 0.1, 0.52, 0.2]
        with self.assertRaisesRegex(RuntimeError, "所属階"):
            normalize_plan_regions(inventory)

    def test_separate_or_unsupported_room_connections_are_rejected(self):
        structure = json.loads(json.dumps(VALID_STRUCTURE))
        structure["spaces"][1].update(label="主寝室", space_type="room", bbox=[0.7, 0, 1, 1])
        structure["connections"][0].update(boundary_relation="door", traversable=True)
        checked = add_verified_topology(structure)
        self.assertEqual(checked["verified_topology"]["direct_connections"], [])
        self.assertTrue(any("離れている" in value for value in checked["verified_topology"]["validation_warnings"]))

    def test_sample1_false_bedroom_living_door_is_rejected_by_edge_position(self):
        structure = json.loads(json.dumps(VALID_STRUCTURE))
        structure["spaces"][0]["bbox"] = [0.67, 0.12, 0.94, 0.48]
        structure["spaces"][1].update(label="主寝室", space_type="room", bbox=[0.62, 0.48, 0.87, 0.73])
        structure["openings"][0]["position"] = [0.62, 0.48]
        structure["connections"][0].update(boundary_relation="door", traversable=True, position=[0.62, 0.48])
        checked = add_verified_topology(structure)
        self.assertEqual(checked["verified_topology"]["direct_connections"], [])
        self.assertTrue(any("境界上にない" in value for value in checked["verified_topology"]["validation_warnings"]))

    def test_cross_floor_stairs_are_only_unverified_candidates(self):
        structure = json.loads(json.dumps(VALID_STRUCTURE))
        structure["spaces"][0].update(space_type="vertical_circulation", floor_id="1F")
        structure["spaces"][1].update(space_type="vertical_circulation", floor_id="2F")
        structure["connections"][0].update(boundary_relation="door", traversable=True)
        checked = add_verified_topology(structure)
        self.assertEqual(checked["verified_topology"]["direct_connections"], [])
        self.assertEqual(checked["verified_topology"]["vertical_links"][0]["status"], "unverified")

    def test_attic_storage_and_bath_to_toilet_are_not_accepted_as_direct_routes(self):
        attic = json.loads(json.dumps(VALID_STRUCTURE))
        attic["spaces"][1].update(label="屋根裏収納", space_type="storage")
        attic["connections"][0].update(boundary_relation="door", traversable=True)
        self.assertEqual(add_verified_topology(attic)["verified_topology"]["direct_connections"], [])

        sanitary = json.loads(json.dumps(VALID_STRUCTURE))
        sanitary["spaces"][0].update(label="トイレ", space_type="sanitary")
        sanitary["spaces"][1].update(label="浴室", space_type="sanitary")
        sanitary["connections"][0].update(boundary_relation="door", traversable=True)
        self.assertEqual(add_verified_topology(sanitary)["verified_topology"]["direct_connections"], [])

        stairs = json.loads(json.dumps(VALID_STRUCTURE))
        stairs["spaces"][0].update(label="階段", space_type="vertical_circulation")
        stairs["spaces"][1].update(label="クローゼット", space_type="storage")
        stairs["connections"][0].update(boundary_relation="door", traversable=True)
        self.assertEqual(add_verified_topology(stairs)["verified_topology"]["direct_connections"], [])

    def test_furniture_and_storage_cannot_receive_perfect_score_from_space_ids(self):
        structure = json.loads(json.dumps(VALID_STRUCTURE))
        structure["spaces"].append({"id": "S3", "label": "収納", "space_type": "storage", "floor_id": "unknown", "bbox": [0.1, 0.1, 0.2, 0.2]})
        structure = add_verified_topology(structure)
        proposal = json.loads(json.dumps(VALID_SCORING))
        for item in proposal["criteria"]:
            if item["name"] == "収納":
                item["evidence_ids"] = ["S3"]
        scoring = validate_scoring_json(json.dumps(proposal, ensure_ascii=False), structure)
        by_name = {item["name"]: item for item in scoring["criteria"]}
        self.assertLessEqual(by_name["家具配置・居住性"]["score"], 7)
        self.assertLessEqual(by_name["収納"]["score"], 5)
        self.assertNotEqual(by_name["家具配置・居住性"]["status"], "confirmed")

    def test_two_floor_crops_keep_global_window_positions_and_unique_ids(self):
        class FakeImage:
            def __init__(self, size=(100, 100)):
                self.size = size

            def crop(self, box):
                return FakeImage((box[2] - box[0], box[3] - box[1]))

            def resize(self, size, _resampling):
                return FakeImage(size)

        inventory = {
            "image_quality": {"level": "high", "notes": []},
            "orientation": {"value": None},
            "plan_regions": [{"floor_id": "1F", "bbox": [0, 0, 0.5, 1]},
                             {"floor_id": "2F", "bbox": [0.5, 0, 1, 1]}],
            "spaces": [
                {"id": "S1", "label": "洋室", "space_type": "room", "floor_id": None,
                 "bbox": [0.1, 0.2, 0.4, 0.8], "confidence": "high"},
                {"id": "S2", "label": "LDK", "space_type": "room", "floor_id": None,
                 "bbox": [0.6, 0.2, 0.9, 0.8], "confidence": "high"},
            ],
        }
        blank = {"openings": [], "connections": [], "windows": [], "fixtures": [],
                 "negative_observations": [], "unreadable_items": []}
        local1 = {"spaces": [{"id": "S1", "label": "洋室", "space_type": "room",
                              "bbox": [0.2, 0.2, 0.8, 0.8], "confidence": "high"}]}
        local2 = {"spaces": [{"id": "S1", "label": "LDK", "space_type": "room",
                              "bbox": [0.2, 0.2, 0.8, 0.8], "confidence": "high"}]}
        features1 = {"windows": [{"id": "W1", "space_id": "R1_S1", "position": [0.2, 0.5],
                                   "confidence": "high"}], "fixtures": [], "unreadable_items": []}
        features2 = {"windows": [{"id": "W1", "space_id": "R2_S1", "position": [0.2, 0.5],
                                   "confidence": "high"}], "fixtures": [], "unreadable_items": []}
        client = FakeClient(*(json.dumps(value, ensure_ascii=False) for value in
                              (inventory, local1, local2, blank, blank, features1, features2)))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stages.json"
            extracted = extract_floor_plan_structure(FakeImage(), client, "test-model", checkpoint_path=checkpoint)
            repeat_client = FakeClient()
            repeated = extract_floor_plan_structure(FakeImage(), repeat_client, "test-model", checkpoint_path=checkpoint)
        self.assertEqual(extracted, repeated)
        self.assertEqual(len(repeat_client.models.calls), 0)
        self.assertEqual(len(client.models.calls), 7)
        self.assertEqual([space["floor_id"] for space in extracted["spaces"]], ["1F", "2F"])
        self.assertEqual([item["position"] for item in extracted["windows"]], [[0.1, 0.5], [0.6, 0.5]])
        self.assertEqual(len({item["id"] for item in extracted["windows"]}), 2)

    def test_scoring_stops_if_a_multi_floor_space_has_no_floor(self):
        structure = json.loads(json.dumps(VALID_STRUCTURE))
        structure["plan_regions"] = [{"floor_id": "1F", "bbox": [0, 0, 0.5, 1]},
                                     {"floor_id": "2F", "bbox": [0.5, 0, 1, 1]}]
        structure["spaces"][0]["floor_id"] = None
        client = FakeClient(json.dumps(VALID_SCORING, ensure_ascii=False))
        with self.assertRaisesRegex(RuntimeError, "所属が未確定"):
            analyze_cached_structure(structure, structure, client, "知識", "test-model")
        self.assertEqual(len(client.models.calls), 0)

    def test_non_traversable_edge_is_not_route_evidence(self):
        structure = add_verified_topology(VALID_STRUCTURE)
        proposal = json.loads(json.dumps(VALID_SCORING))
        proposal["criteria"][0]["evidence_ids"] = ["C1"]
        scoring = validate_scoring_json(json.dumps(proposal, ensure_ascii=False), structure)
        self.assertEqual(scoring["criteria"][0]["evidence_ids"], [])
        self.assertEqual(scoring["criteria"][0]["status"], "unverifiable")

    def test_low_reliability_holds_scoring_without_model_call(self):
        structure = json.loads(json.dumps(VALID_STRUCTURE))
        for index in range(2, 5):
            opening = dict(structure["openings"][0], id=f"O{index}", position=[0.1, 0.1])
            connection = dict(structure["connections"][0], id=f"C{index}", opening_id=f"O{index}",
                              boundary_relation="door", traversable=True, position=[0.1, 0.1])
            structure["openings"].append(opening)
            structure["connections"].append(connection)
        client = FakeClient()
        result = analyze_cached_structure(structure, structure, client, "参考知識", "test-model")
        self.assertIsNone(result.scoring["total"])
        self.assertIn("# 総合評価: 採点保留", result.report)
        self.assertEqual(len(client.models.calls), 0)

    def test_missing_entrances_for_most_rooms_also_holds_score(self):
        structure = json.loads(json.dumps(VALID_STRUCTURE))
        for index in range(3, 7):
            structure["spaces"].append({"id": f"S{index}", "label": "洋室", "space_type": "room",
                                        "floor_id": "unknown", "bbox": [0.1, 0.1, 0.2, 0.2]})
        checked = add_verified_topology(structure)
        self.assertEqual(checked["verified_topology"]["connection_quality"]["reliability"], "low")
        self.assertEqual(checked["verified_topology"]["connection_quality"]["reason"], "主要空間の半数以上に確認済み入口がない")

    def test_unknown_model_space_id_drops_only_bad_connection(self):
        topology = {key: json.loads(json.dumps(VALID_STRUCTURE[key]))
                    for key in ("openings", "connections", "windows", "fixtures", "unreadable_items")}
        topology["connections"][0]["space_b"] = "R1_missing"
        cleaned = discard_invalid_observations(VALID_STRUCTURE, topology)
        self.assertEqual(cleaned["connections"], [])
        self.assertTrue(any("未定義" in value for value in cleaned["unreadable_items"]))

    def test_cropped_inventory_rejects_copied_json_example(self):
        copied = {"spaces": [{"id": "S1", "label": "LDK",
                              "space_type": "room|hall|sanitary|storage|exterior|vertical_circulation|other",
                              "bbox": [0, 0, 1, 1]}]}
        self.assertFalse(_valid_cropped_inventory(copied, expected_count=7))

    def test_cropped_inventory_rejects_missing_washroom_despite_valid_json(self):
        initial = [{"label": "洋室"}, {"label": "玄関"}, {"label": "浴室"},
                   {"label": "洗面所"}, {"label": "トイレ"}, {"label": "階段"}]
        cropped = {"spaces": [
            {"id": f"S{index}", "label": label, "space_type": space_type,
             "bbox": [0.1, 0.1, 0.3, 0.3], "confidence": "high"}
            for index, (label, space_type) in enumerate(
                [("洋室", "room"), ("玄関", "hall"), ("浴室", "sanitary"),
                 ("トイレ", "sanitary"), ("階段", "vertical_circulation")], 1
            )
        ]}
        self.assertFalse(_valid_cropped_inventory(cropped, 6, initial))
        cropped["spaces"].append({"id": "S6", "label": "洗面所", "space_type": "sanitary",
                                  "bbox": [0.3, 0.3, 0.5, 0.5], "confidence": "high"})
        self.assertTrue(_valid_cropped_inventory(cropped, 6, initial))

    def test_feature_empty_recheck_separates_fixtures_and_windows(self):
        inventory = {"spaces": [{"id": "S1", "label": "浴室", "space_type": "sanitary",
                                  "bbox": [0.2, 0.1, 0.6, 0.6]}]}
        client = FakeClient(
            json.dumps({"fixtures": [{"space_id": "S1", "fixture": "浴槽", "confidence": "high"}]}),
            json.dumps({"windows": [{"id": "W1", "space_id": "S1", "position": [0.2, 0.2],
                                      "confidence": "medium", "faces_exterior": True}]}),
        )
        features = _recheck_empty_features(inventory, object(), client, "test-model")
        self.assertEqual(len(features["fixtures"]), 1)
        self.assertEqual(len(features["windows"]), 1)
        self.assertEqual(len(client.models.calls), 2)
        self.assertEqual(client.models.calls[0]["config"]["max_output_tokens"], 768)
        self.assertNotIn('"windows":[]', build_feature_prompt(inventory))
        space_schema = CROPPED_INVENTORY_SCHEMA["properties"]["spaces"]["items"]["properties"]
        self.assertEqual(space_schema["confidence"]["enum"], ["high", "medium", "low"])
        self.assertEqual(space_schema["id"]["minLength"], 1)
        window_only = FakeClient(json.dumps({"windows": []}))
        _recheck_empty_features(inventory, object(), window_only, "test-model", {"windows"})
        self.assertEqual(len(window_only.models.calls), 1)
        self.assertIn("窓だけ", window_only.models.calls[0]["contents"][0])

    def test_low_confidence_feature_cannot_count_as_detected(self):
        inventory = {"spaces": [{"id": "S1", "label": "洋室", "space_type": "room"}]}
        features = {"windows": [], "fixtures": [{"space_id": "S1", "fixture": "シンク",
                                                  "confidence": "low", "evidence": "見えない"}],
                    "unreadable_items": []}
        cleaned = discard_invalid_observations(inventory, features)
        self.assertEqual(cleaned["fixtures"], [])
        self.assertTrue(any("低確信度" in value for value in cleaned["unreadable_items"]))

    def test_cached_visual_features_reject_interior_windows_and_wrong_room_fixtures(self):
        structure = json.loads(json.dumps(VALID_STRUCTURE))
        structure["spaces"][0].update(label="洋室", space_type="room", bbox=[0.1, 0.1, 0.5, 0.5])
        structure["spaces"][1].update(label="トイレ", space_type="sanitary")
        structure["windows"] = [
            {"id": "W1", "space_id": "S1", "position": [0.3, 0.3], "confidence": "high"},
            {"id": "W2", "space_id": "S1", "position": [0.1, 0.3], "confidence": "high"},
        ]
        structure["fixtures"] = [
            {"space_id": "S1", "fixture": "コンロ", "confidence": "high"},
            {"space_id": "S2", "fixture": "浴槽", "confidence": "high"},
            {"space_id": "S2", "fixture": "便器", "confidence": "high"},
        ]
        checked = sanitize_visual_features(structure)
        self.assertEqual([item["id"] for item in checked["windows"]], ["W2"])
        self.assertEqual([item["fixture"] for item in checked["fixtures"]], ["便器"])
        self.assertEqual(len(structure["windows"]), 2)

    def test_repair_retries_failed_floor_but_keeps_valid_other_floor(self):
        class FakeImage:
            size = (100, 100)

            def crop(self, box):
                return self

        inventory = {
            "image_quality": {"level": "medium", "notes": []},
            "orientation": {"value": None},
            "plan_regions": [{"floor_id": "1F", "bbox": [0, 0, 0.5, 1]},
                             {"floor_id": "2F", "bbox": [0.5, 0, 1, 1]}],
            "spaces": [
                {"id": "S1", "label": "洋室", "space_type": "room", "floor_id": "1F", "bbox": [0.1, 0.1, 0.4, 0.5]},
                {"id": "S2", "label": "玄関", "space_type": "hall", "floor_id": "1F", "bbox": [0.1, 0.5, 0.4, 0.9]},
                {"id": "S3", "label": "LDK", "space_type": "room", "floor_id": "2F", "bbox": [0.6, 0.1, 0.9, 0.9]},
            ],
        }
        blank = {"openings": [], "connections": [], "windows": [], "fixtures": [],
                 "negative_observations": [], "unreadable_items": []}
        stages = {
            "inventory": inventory, "fallback_1": {"reason": "old failure"},
            "region_inventory_2": {"spaces": [{"id": "S1", "label": "LDK", "space_type": "room",
                                               "bbox": [0.2, 0.1, 0.8, 0.9]}]},
            "topology_1": blank, "topology_2": blank,
            "features_1": {"windows": [], "fixtures": [], "unreadable_items": []},
            "features_2": {"windows": [{"id": "W2", "space_id": "R2_S1", "position": [0.2, 0.5],
                                        "confidence": "high"}],
                           "fixtures": [{"space_id": "R2_S1", "fixture": "シンク",
                                                        "confidence": "high"}], "unreadable_items": []},
        }
        repaired_floor = {"spaces": [
            {"id": "S1", "label": "洋室", "space_type": "room", "bbox": [0.2, 0.1, 0.8, 0.5],
             "confidence": "high"},
            {"id": "S2", "label": "玄関", "space_type": "hall", "bbox": [0.2, 0.5, 0.8, 0.9],
             "confidence": "high"},
        ]}
        repaired_features = {
            "windows": [{"id": "W1", "space_id": "R1_S1", "position": [0.2, 0.3], "confidence": "high"}],
            "fixtures": [{"space_id": "R1_S2", "fixture": "玄関収納", "confidence": "high"}],
            "unreadable_items": [],
        }
        client = FakeClient(json.dumps(repaired_floor, ensure_ascii=False), json.dumps(blank),
                            json.dumps(repaired_features, ensure_ascii=False))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stages.json"
            checkpoint.write_text(json.dumps({"pipeline_version": PIPELINE_CACHE_VERSION, "stages": stages}),
                                  encoding="utf-8")
            extracted = extract_floor_plan_structure(
                FakeImage(), client, "test-model", checkpoint_path=checkpoint, retry_incomplete_stages=True,
            )
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))["stages"]
        self.assertEqual(extracted["floor_reading_fallbacks"], [])
        self.assertNotIn("fallback_1", saved)
        self.assertEqual(len(client.models.calls), 3)
        self.assertEqual([space["id"] for space in extracted["spaces"]], ["R1_S1", "R1_S2", "R2_S1"])

    def test_failed_floor_inventory_recovers_hall_from_checkpoint(self):
        class FakeImage:
            size = (100, 100)

            def crop(self, box):
                return self

            def resize(self, size, _resampling):
                return self

        inventory = {
            "image_quality": {"level": "high", "notes": []}, "orientation": {"value": None},
            "plan_regions": [{"floor_id": "1F", "bbox": [0, 0, 0.5, 1]},
                             {"floor_id": "2F", "bbox": [0.5, 0, 1, 1]}],
            "spaces": [
                {"id": "S1", "label": "玄関", "space_type": "hall", "floor_id": "1F", "bbox": [0.2, 0.6, 0.3, 0.8]},
                {"id": "S2", "label": "洋室", "space_type": "room", "floor_id": "1F", "bbox": [0.2, 0.1, 0.45, 0.6]},
                {"id": "S3", "label": "階段", "space_type": "vertical_circulation", "floor_id": "1F", "bbox": [0.1, 0.2, 0.2, 0.6]},
                {"id": "S4", "label": "LDK", "space_type": "room", "floor_id": "2F", "bbox": [0.6, 0.1, 0.9, 0.8]},
            ],
        }
        blank = {"openings": [], "connections": [], "windows": [], "fixtures": [],
                 "negative_observations": [], "unreadable_items": []}
        features = {"windows": [], "fixtures": [], "unreadable_items": []}
        stages = {
            "inventory": inventory, "fallback_1": {"reason": "階別の部屋再読取が不完全"},
            "region_inventory_2": {"spaces": [{"id": "S1", "label": "LDK", "space_type": "room", "bbox": [0.2, 0.1, 0.8, 0.8]}]},
            "topology_1": blank, "topology_2": blank, "features_1": features, "features_2": features,
        }
        probe = {"separate_hall": True, "bbox": [0.3, 0.2, 0.5, 0.8],
                 "confidence": "high", "evidence": "玄関と階段の間の黄色い通路"}
        client = FakeClient(json.dumps(probe, ensure_ascii=False), json.dumps(blank, ensure_ascii=False))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stages.json"
            checkpoint.write_text(json.dumps({"pipeline_version": PIPELINE_CACHE_VERSION, "stages": stages}), encoding="utf-8")
            extracted = extract_floor_plan_structure(FakeImage(), client, "test-model", checkpoint_path=checkpoint)
            repeated_client = FakeClient()
            repeated = extract_floor_plan_structure(FakeImage(), repeated_client, "test-model", checkpoint_path=checkpoint)
        self.assertEqual(extracted, repeated)
        self.assertEqual(extracted["floor_hall_recoveries"], ["1F"])
        self.assertEqual(extracted["floor_repair_attempts"], ["1F"])
        self.assertEqual(extracted["floor_reading_fallbacks"], ["1F"])
        self.assertIn("R1_HALL", {space["id"] for space in extracted["spaces"]})
        self.assertEqual(len(client.models.calls), 2)
        self.assertEqual(len(repeated_client.models.calls), 0)

    def test_hall_probe_requires_clear_visual_evidence(self):
        self.assertFalse(_valid_hall_probe({"separate_hall": True, "bbox": [0.1, 0.1, 0.3, 0.3],
                                            "confidence": "low", "evidence": "maybe"}))
        self.assertFalse(_valid_hall_probe({"separate_hall": True, "bbox": [0, 0, 1, 1],
                                            "confidence": "high", "evidence": "whole plan"}))

    def test_recovered_hall_does_not_verify_storage_door_from_approximate_bbox(self):
        structure = json.loads(json.dumps(VALID_STRUCTURE))
        structure["spaces"] = [
            {"id": "R1_HALL", "label": "廊下", "space_type": "hall", "floor_id": "1F", "bbox": [0.1, 0.1, 0.3, 0.4]},
            {"id": "S2", "label": "クローゼット", "space_type": "storage", "floor_id": "1F", "bbox": [0.3, 0.1, 0.5, 0.4]},
        ]
        structure["openings"] = [{"id": "O1", "opening_type": "door", "position": [0.3, 0.2], "confidence": "high"}]
        structure["connections"] = [{"id": "C1", "opening_id": "O1", "space_a": "R1_HALL", "space_b": "S2",
                                     "boundary_relation": "door", "traversable": True, "position": [0.3, 0.2], "confidence": "high"}]
        self.assertEqual(len(add_verified_topology(structure)["verified_topology"]["direct_connections"]), 1)
        structure["floor_hall_recoveries"] = ["1F"]
        checked = add_verified_topology(structure)
        self.assertEqual(checked["verified_topology"]["direct_connections"], [])
        self.assertIn("位置が概略", checked["connections"][0]["validation_reasons"][0])

    def test_cropped_inventory_fallback_holds_score(self):
        structure = add_verified_topology(VALID_STRUCTURE)
        structure["floor_reading_fallbacks"] = ["1F"]
        client = FakeClient()
        result = analyze_cached_structure(structure, structure, client, "参考知識", "test-model")
        self.assertIsNone(result.scoring["total"])
        self.assertIn("1Fの階別部屋再読取", result.report)
        self.assertEqual(len(client.models.calls), 0)

    def test_high_quality_plan_with_no_windows_or_fixtures_holds_score(self):
        structure = json.loads(json.dumps(VALID_STRUCTURE))
        structure["spaces"].append({"id": "S3", "label": "収納", "space_type": "storage",
                                    "floor_id": "unknown", "bbox": [0.1, 0.1, 0.2, 0.2]})
        client = FakeClient()
        result = analyze_cached_structure(structure, structure, client, "参考知識", "test-model")
        self.assertIn("窓・設備を1件も抽出", result.report)
        self.assertIsNone(result.scoring["total"])

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
        self.assertIn('"boundary_relation":"shared_wall_only"', prompt)
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

    def test_bathroom_to_hall_connection_is_rejected(self):
        structure = json.loads(json.dumps(VALID_STRUCTURE))
        structure["spaces"][0].update(label="廊下", space_type="hall")
        structure["spaces"][1].update(label="浴室", space_type="sanitary")
        structure["connections"][0].update(boundary_relation="door", traversable=True)
        checked, warnings = validate_connections(structure)
        self.assertFalse(checked["connections"][0]["traversable"])
        self.assertTrue(any("洗面所・脱衣所" in warning for warning in warnings))

    def test_cross_floor_connection_without_stairs_is_rejected(self):
        structure = json.loads(json.dumps(VALID_STRUCTURE))
        structure["spaces"][0]["floor_id"] = "1F"
        structure["spaces"][1]["floor_id"] = "2F"
        structure["connections"][0].update(boundary_relation="door", traversable=True)
        checked, warnings = validate_connections(structure)
        self.assertFalse(checked["connections"][0]["traversable"])
        self.assertTrue(any("別階" in warning for warning in warnings))

    def test_duplicate_topology_pairs_are_collapsed_before_verification(self):
        topology = {
            "openings": [
                {"id": "O1", "position": [0.5, 0.5]},
                {"id": "O2", "position": [0.5, 0.55]},
            ],
            "connections": [
                {"id": "C1", "opening_id": "O1", "space_a": "S1", "space_b": "S2", "confidence": "high"},
                {"id": "C2", "opening_id": "O2", "space_a": "S2", "space_b": "S1", "confidence": "medium"},
            ],
        }
        collapsed = deduplicate_topology_observations(topology)
        self.assertEqual([item["id"] for item in collapsed["connections"]], ["C1"])
        self.assertEqual([item["id"] for item in collapsed["openings"]], ["O1"])

    def test_opening_outside_both_spaces_is_rejected(self):
        structure = json.loads(json.dumps(VALID_STRUCTURE))
        structure["connections"][0].update(boundary_relation="door", traversable=True, position=[0.05, 0.05])
        structure["openings"][0]["position"] = [0.05, 0.05]
        checked, warnings = validate_connections(structure)
        self.assertFalse(checked["connections"][0]["traversable"])
        self.assertTrue(any("境界付近にない" in warning for warning in warnings))

    def test_missing_verification_decision_preserves_structure_and_marks_uncertain(self):
        merged = merge_verification_decisions(VALID_STRUCTURE, '{"decisions": []}')
        self.assertEqual(merged["spaces"], VALID_STRUCTURE["spaces"])
        self.assertEqual(merged["openings"], VALID_STRUCTURE["openings"])
        connection = merged["connections"][0]
        self.assertEqual(connection["verification_verdict"], "uncertain")
        self.assertEqual(connection["validation_status"], "uncertain")

    def test_unverifiable_categories_cannot_receive_full_score(self):
        structure = json.loads(json.dumps(VALID_STRUCTURE))
        structure["verified_topology"] = {"validation_warnings": [], "direct_connections": []}
        scoring = validate_scoring_json(json.dumps(VALID_SCORING, ensure_ascii=False), structure)
        scores = {item["name"]: item["score"] for item in scoring["criteria"]}
        self.assertLess(scores["採光・通風"], 10)
        self.assertLess(scores["安全性・バリアフリー"], 10)
        self.assertLess(scores["将来対応・可変性"], 10)
        self.assertEqual(scoring["total"], sum(scores.values()))

    def test_low_connection_reliability_caps_route_dependent_scores(self):
        structure = json.loads(json.dumps(VALID_STRUCTURE))
        structure["verified_topology"] = {
            "validation_warnings": [], "direct_connections": [],
            "connection_quality": {"reliability": "low"},
        }
        scoring = validate_scoring_json(json.dumps(VALID_SCORING, ensure_ascii=False), structure)
        scores = {item["name"]: item["score"] for item in scoring["criteria"]}
        self.assertLessEqual(scores["生活動線"], 7)
        self.assertLessEqual(scores["安全性・バリアフリー"], 5)

    def test_exterior_cannot_be_storage_evidence(self):
        structure = json.loads(json.dumps(VALID_STRUCTURE))
        structure["verified_topology"] = {"validation_warnings": [], "direct_connections": []}
        scoring = validate_scoring_json(json.dumps(VALID_SCORING, ensure_ascii=False), structure)
        storage = next(item for item in scoring["criteria"] if item["name"] == "収納")
        self.assertEqual(storage["status"], "unverifiable")
        self.assertLessEqual(storage["score"], 5)

    def test_compact_scoring_structure_removes_geometry_and_rejected_edges(self):
        structure = json.loads(json.dumps(VALID_STRUCTURE))
        structure["connections"][0]["validation_status"] = "rejected"
        structure["verified_topology"] = {"direct_connections": [], "validation_warnings": ["C1"]}
        compact = compact_structure_for_scoring(structure)
        self.assertNotIn("bbox", compact["spaces"][0])
        self.assertNotIn("openings", compact)
        self.assertEqual(compact["connections"], [])

    def test_structure_cache_is_versioned_and_model_specific(self):
        first_key = structure_cache_key(b"image", "ollama", "4b", 8192, 1)
        second_key = structure_cache_key(b"image", "ollama", "8b", 8192, 1)
        self.assertNotEqual(first_key, second_key)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cache.json"
            save_structure_cache(path, VALID_STRUCTURE, VALID_STRUCTURE, {"model": "4b"})
            cached = load_structure_cache(path)
            self.assertIsNotNone(cached)
            self.assertEqual(cached[0], VALID_STRUCTURE)

    def test_ollama_timeout_is_configurable(self):
        client = create_analysis_client("ollama", ollama_timeout=1800)
        self.assertEqual(client.timeout, 1800)

    def test_ollama_feature_recheck_limits_generated_tokens(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"message":{"content":"{}"}}'

        with patch("floor_plan.providers.urlopen", return_value=FakeResponse()) as request_mock:
            OllamaClient().generate_content(
                model="test-model", contents=["prompt"], config={"max_output_tokens": 768},
            )
        payload = json.loads(request_mock.call_args.args[0].data)
        self.assertEqual(payload["options"]["num_predict"], 768)

    def test_malformed_topology_json_is_repaired_once(self):
        repaired = {
            "openings": [], "connections": [], "windows": [], "fixtures": [],
            "negative_observations": [], "unreadable_items": [],
        }
        client = FakeClient(json.dumps(repaired, ensure_ascii=False))
        value = _parse_or_repair_json_object(
            '{"openings": [}', client, "test-model", "扉・接続結果", TOPOLOGY_OBSERVATION_SCHEMA
        )
        self.assertEqual(value, repaired)
        self.assertEqual(len(client.models.calls), 1)

    def test_cached_analysis_calls_only_scoring_model(self):
        structure = json.loads(json.dumps(VALID_STRUCTURE))
        structure["verified_topology"] = {
            "traversable_neighbors": {"S1": [], "S2": []},
            "direct_connections": [], "non_transit_spaces": [],
            "confirmed_absences": [], "validation_warnings": [],
        }
        client = FakeClient(json.dumps(VALID_SCORING, ensure_ascii=False))
        result = analyze_cached_structure(VALID_STRUCTURE, structure, client, "参考知識", "test-model")
        self.assertEqual(len(client.models.calls), 1)
        self.assertIn("# 総合評価:", result.report)

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
        inventory = {key: VALID_STRUCTURE[key] for key in ("image_quality", "orientation", "plan_regions", "spaces")}
        topology = {
            key: VALID_STRUCTURE[key]
            for key in ("openings", "connections", "windows", "fixtures", "negative_observations", "unreadable_items")
        }
        client = FakeClient(
            json.dumps(inventory),
            json.dumps(topology),
            json.dumps({"decisions": [{"connection_id": "C1", "verdict": "accept", "confidence": "high", "reason": "壁のみ"}]}),
            json.dumps(VALID_SCORING, ensure_ascii=False),
        )
        result = analyze_floor_plan(image, client, "参考知識", model="test-model")

        extracted_without_topology = {
            key: value for key, value in result.structure.items() if key != "verified_topology"
        }
        for connection in extracted_without_topology["connections"]:
            connection.pop("validation_status", None)
            connection.pop("validation_reasons", None)
            connection.pop("verification_verdict", None)
            connection.pop("verification_confidence", None)
            connection.pop("verification_reason", None)
        self.assertEqual(extracted_without_topology, VALID_STRUCTURE)
        self.assertIn("# 総合評価:", result.report)
        self.assertEqual(result.scoring["total"], sum(item["score"] for item in result.scoring["criteria"]))
        self.assertEqual(result.structure["verified_topology"]["traversable_neighbors"]["S1"], [])
        self.assertEqual(len(client.models.calls), 4)
        first, topology_call, verification, scoring = client.models.calls
        self.assertIn(image, first["contents"])
        self.assertEqual(first["config"]["response_mime_type"], "application/json")
        self.assertIn(image, topology_call["contents"])
        self.assertIn("確定済み空間一覧", topology_call["contents"][0])
        self.assertIn(image, verification["contents"])
        self.assertIn("暫定JSON", verification["contents"][0])
        self.assertEqual(verification["config"]["response_json_schema"]["properties"]["decisions"]["type"], "array")
        self.assertEqual(len(scoring["contents"]), 1)
        self.assertIn('"spaces"', scoring["contents"][0])
        self.assertNotIn(image, scoring["contents"])
        self.assertEqual(scoring["config"]["response_mime_type"], "application/json")


if __name__ == "__main__":
    unittest.main()

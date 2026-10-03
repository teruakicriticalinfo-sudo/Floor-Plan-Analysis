import unittest

from floor_plan.corrections import apply_reviewed_corrections
from floor_plan.analyzer import analyze_from_structure


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        self.structure = {
            "spaces": [
                {"id": "hall", "label": "廊下", "space_type": "hall", "floor_id": "1F", "bbox": [0.1, 0.1, 0.8, 0.8]},
                {"id": "wash", "label": "洗面所", "space_type": "sanitary", "floor_id": "1F", "bbox": [0.1, 0.5, 0.4, 0.8]},
                {"id": "bed", "label": "主寝室", "space_type": "room", "floor_id": "2F", "bbox": [0.5, 0.5, 0.8, 0.8]},
            ],
            "openings": [
                {"id": "O1", "opening_type": "door", "position": [0.3, 0.6], "confidence": "high"},
                {"id": "O2", "opening_type": "door", "position": [0.6, 0.6], "confidence": "high"},
            ],
            "connections": [
                {"id": "C1", "opening_id": "O1", "space_a": "hall", "space_b": "wash",
                 "boundary_relation": "door", "traversable": True, "position": [0.3, 0.6], "confidence": "high"},
                {"id": "C2", "opening_id": "O2", "space_a": "hall", "space_b": "bed",
                 "boundary_relation": "door", "traversable": True, "position": [0.6, 0.6], "confidence": "high"},
            ],
            "negative_observations": [], "fixtures": [],
        }
        self.corrections = {
            "review_status": "draft", "spaces": {"hall": {"bbox": [0.25, 0.1, 0.4, 0.8]}},
            "connections": [{"space_a": "hall", "space_b": "wash", "boundary_relation": "door",
                             "position": [0.3, 0.6], "evidence": "利用者確認"}],
            "confirmed_no_entrance": ["bed"],
        }

    def test_draft_requires_explicit_preview_and_replaces_model_edge(self):
        with self.assertRaisesRegex(ValueError, "未承認"):
            apply_reviewed_corrections(self.structure, self.corrections)
        corrected = apply_reviewed_corrections(self.structure, self.corrections, allow_draft=True)
        self.assertEqual(corrected["manual_correction_status"], "draft")
        self.assertEqual(next(item for item in corrected["spaces"] if item["id"] == "hall")["bbox"], [0.25, 0.1, 0.4, 0.8])
        self.assertEqual([item["id"] for item in corrected["connections"] if item["traversable"]], ["MANUAL_C1"])
        self.assertEqual(corrected["verified_topology"]["direct_connections"],
                         [{"connection_id": "MANUAL_C1", "spaces": ["hall", "wash"]}])
        self.assertIn("入口がない", next(item for item in corrected["connections"] if item["id"] == "C2")["validation_reasons"][0])
        scoring = analyze_from_structure(corrected, object(), "知識")
        self.assertEqual(scoring["status"], "held")
        self.assertIn("位置修正が未承認", scoring["hold_reasons"][0])

    def test_approved_corrections_and_invalid_geometry(self):
        approved = {**self.corrections, "review_status": "approved"}
        self.assertEqual(apply_reviewed_corrections(self.structure, approved)["manual_correction_status"], "approved")
        with self.assertRaisesRegex(ValueError, "位置の確認"):
            apply_reviewed_corrections(self.structure, {**approved, "geometry_review_status": "draft"})
        with self.assertRaisesRegex(ValueError, "opening_position_review_status"):
            apply_reviewed_corrections(self.structure, {
                **approved, "bbox_review_status": "approved", "opening_position_review_status": "draft",
            })
        with self.assertRaisesRegex(ValueError, "bbox_review_status"):
            apply_reviewed_corrections(self.structure, {
                **approved, "bbox_review_status": "draft", "opening_position_review_status": "approved",
            })
        self.assertEqual(apply_reviewed_corrections(self.structure, {
            **approved, "bbox_review_status": "approved", "opening_position_review_status": "approved",
        })["manual_correction_status"], "approved")
        bad = {**approved, "spaces": {"hall": {"bbox": [0.4, 0.1, 0.25, 0.8]}}}
        with self.assertRaisesRegex(ValueError, "bbox"):
            apply_reviewed_corrections(self.structure, bad)


if __name__ == "__main__":
    unittest.main()

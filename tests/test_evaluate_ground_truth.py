import unittest

from tools.evaluate_ground_truth import evaluate, expected_label


class GroundTruthEvaluationTests(unittest.TestCase):
    def test_floor_qualified_labels_keep_same_named_rooms_separate(self):
        structure = {
            "spaces": [
                {"id": "a", "label": "LDK", "space_type": "room", "floor_id": "1F"},
                {"id": "b", "label": "トイレ", "space_type": "sanitary", "floor_id": "1F"},
                {"id": "c", "label": "LDK", "space_type": "room", "floor_id": "2F"},
                {"id": "d", "label": "トイレ", "space_type": "sanitary", "floor_id": "2F"},
            ],
            "connections": [
                {"space_a": "a", "space_b": "b", "traversable": True, "validation_status": "accepted"},
                {"space_a": "c", "space_b": "d", "traversable": True, "validation_status": "accepted"},
            ],
        }
        truth = {
            "floor_qualified": True,
            "connections": [["1F:LDK", "1F:トイレ"]],
            "excluded_connections": [["2F:LDK", "2F:トイレ"]],
        }

        result = evaluate(structure, truth)

        self.assertEqual(result["true_positive"], [("1F:LDK", "1F:トイレ")])
        self.assertEqual(result["false_positive"], [])
        self.assertEqual(result["f1"], 1.0)

    def test_existing_unqualified_truth_remains_supported(self):
        structure = {
            "spaces": [
                {"id": "a", "label": "LDK", "space_type": "room", "floor_id": "1F"},
                {"id": "b", "label": "トイレ", "space_type": "sanitary", "floor_id": "1F"},
            ],
            "connections": [
                {"space_a": "a", "space_b": "b", "traversable": True, "validation_status": "accepted"},
            ],
        }
        result = evaluate(structure, {"connections": [["LDK", "トイレ"]]})
        self.assertEqual(result["f1"], 1.0)

    def test_floor_qualified_alias_preserves_floor(self):
        self.assertEqual(expected_label("2F:リビング", {"リビング": "LDK"}, True), "2F:LDK")
        with self.assertRaises(ValueError):
            expected_label("LDK", {}, True)


if __name__ == "__main__":
    unittest.main()

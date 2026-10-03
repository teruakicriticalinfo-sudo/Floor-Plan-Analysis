import unittest

from tools.evaluate_window_annotations import evaluate


class WindowAnnotationEvaluationTests(unittest.TestCase):
    def test_only_windows_count_and_matches_are_one_to_one(self):
        annotation = {
            "classification_review_status": "user_confirmed",
            "coverage_review_status": "not_confirmed",
            "geometry_review_status": "assistant_interpolated_from_user_markers",
            "windows": [{"id": "W1", "position": [0.2, 0.1]}],
            "user_markers": [
                {"id": "M1", "kind": "window", "position": [0.21, 0.1]},
                {"id": "M2", "kind": "balcony_door", "position": [0.3, 0.1]},
                {"id": "M3", "kind": "garage_window_door", "position": [0.4, 0.1]},
            ],
        }
        structure = {
            "spaces": [{"id": "S1", "label": "洋室", "bbox": [0, 0.1, 0.5, 0.5]}],
            "windows": [{"id": "P1", "space_id": "S1", "position": [0.2, 0.1],
                         "confidence": "high", "faces_exterior": True}],
            "fixtures": [], "unreadable_items": [],
        }
        result = evaluate(annotation, structure, (100, 100), tolerance_px=12)
        self.assertEqual(result["annotated_window_markers"], 2)
        self.assertEqual(result["annotated_exterior_doors_excluded"], 2)
        self.assertEqual(result["matched_markers"], 1)
        self.assertEqual(result["recall_on_marked_positions"], 0.5)
        self.assertIsNone(result["precision"])

    def test_unconfirmed_annotations_cannot_be_measured(self):
        with self.assertRaises(ValueError):
            evaluate({"classification_review_status": "draft"}, {}, (100, 100))


if __name__ == "__main__":
    unittest.main()

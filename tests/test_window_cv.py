import io
import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from floor_plan.window_cv import detect_wall_gap_candidates
from window_cv_review import evaluate_candidate_coverage, migrate_review, render_review_page


class WallGapDetectionTests(unittest.TestCase):
    def test_straight_wall_gap_is_a_candidate_without_reference_labels(self):
        image = Image.new("RGB", (100, 100), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 5, 24, 94), fill="black")
        draw.rectangle((20, 40, 24, 51), fill="white")
        candidates = detect_wall_gap_candidates(image)
        self.assertTrue(any(item["orientation"] == "vertical"
                            and item["pixel_bbox"][0] <= 22 < item["pixel_bbox"][2]
                            and item["pixel_bbox"][1] <= 45 < item["pixel_bbox"][3]
                            for item in candidates))
        self.assertTrue(all(item["classification"] == "unreviewed" for item in candidates))

    def test_unbroken_wall_has_no_candidates(self):
        image = Image.new("RGB", (100, 100), "white")
        ImageDraw.Draw(image).rectangle((20, 5, 24, 94), fill="black")
        self.assertEqual(detect_wall_gap_candidates(image), [])

    def test_coverage_keeps_window_and_exterior_door_marks_separate(self):
        candidates = [{"id": "C001", "bbox": [0.2, 0.2, 0.23, 0.3]},
                      {"id": "C002", "bbox": [0.8, 0.2, 0.83, 0.3]}]
        annotation = {"classification_review_status": "user_confirmed",
                      "reference_rectangles": [
                          {"id": "R1", "kind": "window", "bbox": [0.19, 0.2, 0.24, 0.3]},
                          {"id": "R2", "kind": "balcony_door", "bbox": [0.79, 0.2, 0.84, 0.3]}]}
        report = evaluate_candidate_coverage(candidates, annotation, (100, 100))
        self.assertEqual(report["covered_marked_windows"], 1)
        self.assertEqual(report["marked_window_rectangles"], 1)
        self.assertEqual(report["covered_exterior_door_ids"], ["R2"])
        self.assertIsNone(report["precision"])

    def test_one_candidate_cannot_cover_two_door_rectangles(self):
        candidates = [{"id": "C001", "bbox": [0.4, 0.2, 0.5, 0.25]}]
        annotation = {"classification_review_status": "user_confirmed",
                      "reference_rectangles": [
                          {"id": "R1", "kind": "balcony_door", "bbox": [0.4, 0.2, 0.5, 0.25]},
                          {"id": "R2", "kind": "balcony_door", "bbox": [0.51, 0.2, 0.56, 0.25]}]}
        report = evaluate_candidate_coverage(candidates, annotation, (100, 100))
        self.assertEqual(len(report["covered_exterior_door_ids"]), 1)
        self.assertEqual(len(report["missed_exterior_door_ids"]), 1)

    def test_legacy_generic_doors_require_reclassification(self):
        image_bytes = b"review-image"
        candidates = [{"id": "C001", "bbox": [0.1, 0.2, 0.2, 0.3], "orientation": "vertical"},
                      {"id": "C002", "bbox": [0.5, 0.2, 0.6, 0.3], "orientation": "vertical"}]
        review = {"schema_version": 1, "image": "sample.webp",
                  "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                  "detector_version": "wall-gap-v1",
                  "decisions": [{"candidate_id": "C001", "bbox": candidates[0]["bbox"],
                                 "orientation": "vertical", "classification": "window"},
                                {"candidate_id": "C002", "bbox": candidates[1]["bbox"],
                                 "orientation": "vertical", "classification": "exterior_door"}]}
        migrated = migrate_review(review, candidates, "sample.webp", image_bytes)
        self.assertEqual(migrated["schema_version"], 2)
        self.assertEqual(migrated["review_status"], "draft")
        self.assertEqual([item["classification"] for item in migrated["decisions"]],
                         ["window", "door_unclassified"])
        review["decisions"][1]["bbox"] = [0, 0, 1, 1]
        with self.assertRaises(ValueError):
            migrate_review(review, candidates, "sample.webp", image_bytes)

    def test_review_page_exports_decisions_and_hides_reference_by_default(self):
        image = Image.new("RGB", (30, 30), "white")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        candidates = [{"id": "C001", "orientation": "vertical",
                       "pixel_bbox": [10, 10, 14, 20],
                       "bbox": [10 / 30, 10 / 30, 14 / 30, 20 / 30]}]
        evaluation = {"covered_marked_windows": 0, "marked_window_rectangles": 1}
        annotation = {"reference_rectangles": [
            {"id": "R1", "kind": "window", "bbox": [0.3, 0.3, 0.5, 0.7]}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.html"
            render_review_page(buffer.getvalue(), "sample.png", image.size,
                               candidates, evaluation, annotation, path)
            page = path.read_text(encoding="utf-8")
        self.assertIn('id="reference-layer" style="display:none"', page)
        self.assertIn('value="window"', page)
        self.assertIn('value="exterior_door"', page)
        self.assertIn('value="interior_door"', page)
        self.assertIn('value="door_unclassified"', page)
        self.assertIn('value="uncertain"', page)
        self.assertIn("確認結果JSONを保存", page)
        self.assertIn("image_sha256", page)
        self.assertIn("review_status", page)
        self.assertIn("data.schema_version===1", page)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.render_window_review import render


class WindowReviewTests(unittest.TestCase):
    def test_renders_numbered_draft_without_claiming_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.webp"
            Image.new("RGB", (100, 80), "white").save(image_path)
            document = render({"image": "sample.webp", "review_status": "draft", "windows": [
                {"id": "W1", "floor_id": "1F", "room": "洋室", "position": [0.5, 0.25],
                 "description": "<上側>"},
            ], "user_markers": [
                {"id": "M1", "comment": 1, "kind": "window", "position": [0.2, 0.3]},
                {"id": "M2", "comment": 2, "kind": "balcony_door", "position": [0.7, 0.3]},
            ]}, image_path)
            self.assertIn('cx="50.0" cy="20.0"', document)
            self.assertIn("&lt;上側&gt;", document)
            self.assertIn("未確認", document)
            self.assertIn("ベランダに出る扉", document)
            confirmed = render({"image": "sample.webp", "review_status": "draft",
                                "classification_review_status": "user_confirmed",
                                "windows": [{"id": "W1", "position": [0.5, 0.25]}]}, image_path)
            self.assertIn('fill="#16803c"', confirmed)

    def test_rejects_invalid_or_duplicate_points(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.webp"
            Image.new("RGB", (100, 80), "white").save(image_path)
            base = {"image": "sample.webp", "review_status": "draft", "windows": [
                {"id": "W1", "position": [0.2, 0.5]},
                {"id": "W1", "position": [0.3, 0.6]},
            ]}
            with self.assertRaises(ValueError):
                render(base, image_path)
            base["windows"][1]["id"] = "W2"
            base["windows"][1]["position"] = [1.2, 0.6]
            with self.assertRaises(ValueError):
                render(base, image_path)
            base["review_status"] = "approved"
            with self.assertRaises(ValueError):
                render(base, image_path)

    def test_rejects_unrecognized_user_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.webp"
            Image.new("RGB", (100, 80), "white").save(image_path)
            annotation = {"image": "sample.webp", "review_status": "draft", "user_markers": [
                {"id": "M1", "kind": "unknown", "position": [0.2, 0.3]},
            ]}
            with self.assertRaises(ValueError):
                render(annotation, image_path)


if __name__ == "__main__":
    unittest.main()

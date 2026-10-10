import hashlib
import io
import unittest

from PIL import Image

from manual_window_labeler import render_label_page, validate_reference


class ManualWindowLabelerTests(unittest.TestCase):
    def setUp(self):
        image = Image.new("RGB", (40, 30), "white")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        self.image_bytes = buffer.getvalue()
        self.image_size = image.size

    def test_page_has_raw_image_and_no_detector_overlay(self):
        page = render_label_page(self.image_bytes, "sample3.png", self.image_size)
        self.assertIn("data:image/png;base64,", page)
        self.assertIn("窓記号を一つずつドラッグ", page)
        self.assertIn("候補は表示しません", page)
        self.assertIn("pointerdown", page)
        self.assertIn("preview.style.display='none'", page)
        self.assertIn("確認JSONを保存", page)
        self.assertNotIn("wall-symbol-v2", page)

    def test_approved_reference_requires_complete_valid_rectangles(self):
        data = {"schema_version": 1, "image": "sample3.png",
                "image_sha256": hashlib.sha256(self.image_bytes).hexdigest(),
                "image_size": [40, 30], "review_status": "approved",
                "classification_review_status": "user_confirmed",
                "geometry_review_status": "user_confirmed",
                "coverage_review_status": "user_confirmed_no_missing_windows",
                "reference_rectangles": [
                    {"id": "R01", "kind": "window", "bbox": [0.1, 0.2, 0.2, 0.5]},
                    {"id": "R02", "kind": "window_door", "bbox": [0.7, 0.2, 0.8, 0.5]}]}
        validate_reference(data, "sample3.png", self.image_bytes, self.image_size)
        data["reference_rectangles"][1]["id"] = "R01"
        with self.assertRaises(ValueError):
            validate_reference(data, "sample3.png", self.image_bytes, self.image_size)
        data["reference_rectangles"][1]["id"] = "R02"
        data["reference_rectangles"][1]["kind"] = "uncertain"
        with self.assertRaises(ValueError):
            validate_reference(data, "sample3.png", self.image_bytes, self.image_size)

    def test_reference_cannot_approve_missing_windows_or_wrong_image(self):
        data = {"schema_version": 1, "image": "sample3.png",
                "image_sha256": hashlib.sha256(self.image_bytes).hexdigest(),
                "image_size": [40, 30], "review_status": "approved",
                "coverage_review_status": "not_confirmed", "reference_rectangles": [
                    {"id": "R01", "kind": "window", "bbox": [0.1, 0.2, 0.2, 0.5]}]}
        with self.assertRaises(ValueError):
            validate_reference(data, "sample3.png", self.image_bytes, self.image_size)
        data["review_status"] = "draft"
        validate_reference(data, "sample3.png", self.image_bytes, self.image_size)
        data["image_sha256"] = "bad"
        with self.assertRaises(ValueError):
            validate_reference(data, "sample3.png", self.image_bytes, self.image_size)


if __name__ == "__main__":
    unittest.main()

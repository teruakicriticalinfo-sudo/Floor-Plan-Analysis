import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from floor_plan.wall_window import _rejection_reason, build_patches, scan_patches
from window_patch_backtest import evaluate_detections


class FakeClient:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = 0

    def generate_content(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(text=json.dumps(next(self.answers)))


class WallWindowTests(unittest.TestCase):
    def test_patches_cover_both_floors_without_annotations(self):
        regions = [{"floor_id": "1F", "bbox": [0, 0, 0.5, 1]},
                   {"floor_id": "2F", "bbox": [0.5, 0, 1, 1]}]
        patches = build_patches((640, 461), regions)
        self.assertEqual(len(patches), 30)
        for point in ((220, 49), (285, 170), (72, 292), (595, 182), (551, 277)):
            self.assertTrue(any(p["pixel_bbox"][0] <= point[0] <= p["pixel_bbox"][2]
                                and p["pixel_bbox"][1] <= point[1] <= p["pixel_bbox"][3]
                                for p in patches))

    def test_patch_results_are_checkpointed_and_overlaps_deduplicated(self):
        image = Image.new("RGB", (200, 100), "white")
        patches = build_patches(image.size, [], size_px=100, stride_px=100)
        client = FakeClient([
            {"detections": [{"kind": "window", "bbox": [0.9, 0.4, 1, 0.6],
                             "confidence": "high", "evidence": "line"}]},
            {"detections": [{"kind": "window", "bbox": [0, 0.4, 0.1, 0.6],
                             "confidence": "medium", "evidence": "line"}]},
        ])
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "patch.json"
            result = scan_patches(image, patches, client, "test-model", checkpoint_path=checkpoint)
            replay = FakeClient([])
            cached = scan_patches(image, patches, replay, "test-model", checkpoint_path=checkpoint)
        self.assertEqual(client.calls, 2)
        self.assertEqual(replay.calls, 0)
        self.assertEqual(len(result["accepted_detections"]), 1)
        self.assertEqual(cached["accepted_detections"], result["accepted_detections"])

    def test_evaluation_excludes_doors_and_flags_door_false_positive(self):
        result = {"accepted_detections": [
            {"kind": "window", "bbox": [0.19, 0.1, 0.21, 0.12], "patch_id": "P1"},
            {"kind": "window", "bbox": [0.79, 0.1, 0.81, 0.12], "patch_id": "P2"},
        ]}
        annotation = {"reference_rectangles": [
            {"id": "R1", "kind": "window", "bbox": [0.19, 0.1, 0.22, 0.13]},
            {"id": "R2", "kind": "balcony_door", "bbox": [0.79, 0.1, 0.82, 0.13]},
        ]}
        report = evaluate_detections(result, annotation, (100, 100))
        self.assertEqual(report["matched_window_rectangles"], 1)
        self.assertEqual(report["reference_window_rectangles"], 1)
        self.assertEqual(report["reference_exterior_door_rectangles"], 1)
        self.assertEqual(len(report["door_misclassified_as_window"]), 1)
        self.assertIsNone(report["precision"])

    def test_square_kitchen_like_symbol_is_rejected_as_window(self):
        self.assertEqual(_rejection_reason({"kind": "window", "confidence": "high",
                                            "bbox": [0.1, 0.1, 0.14, 0.15]}, (640, 461)),
                         "細長い窓記号の形状ではない")
        self.assertIsNone(_rejection_reason({"kind": "window", "confidence": "medium",
                                              "bbox": [0.1, 0.1, 0.108, 0.16]}, (640, 461)))


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from floor_plan.targeted_vision import map_box, map_point, render_review_html, run_probes, select_probes


class FakeImage:
    size = (1000, 600)

    def crop(self, box):
        return self


class FakeModels:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=json.dumps(next(self.answers), ensure_ascii=False))


class FakeClient:
    def __init__(self, answers):
        self.models = FakeModels(answers)


STRUCTURE = {
    "plan_regions": [{"floor_id": "1F", "bbox": [0, 0, 0.5, 1]}],
    "spaces": [
        {"id": "S1", "label": "洗面所", "floor_id": "1F", "space_type": "sanitary",
         "bbox": [0.12, 0.5, 0.23, 0.68]},
        {"id": "S2", "label": "浴室", "floor_id": "1F", "space_type": "sanitary",
         "bbox": [0.12, 0.68, 0.23, 0.82]},
        {"id": "S3", "label": "洋室", "floor_id": "1F", "space_type": "room",
         "bbox": [0.24, 0.12, 0.44, 0.48]},
    ],
}


class TargetedVisionTests(unittest.TestCase):
    def test_crop_is_bounded_by_floor_and_coordinates_map_back(self):
        probes = select_probes(STRUCTURE, max_window_rooms=1)
        self.assertEqual([p["kind"] for p in probes], ["washroom", "window"])
        self.assertLessEqual(probes[0]["crop_bbox"][2], 0.5)
        wall_probes = select_probes(STRUCTURE, max_window_rooms=1, window_wall_strips=True)
        self.assertEqual([p["kind"] for p in wall_probes], ["washroom", "window", "window"])
        self.assertEqual({p["edge"] for p in wall_probes[1:]}, {"top", "right"})
        self.assertEqual(map_point([0, 0], [0.1, 0.2, 0.3, 0.6]), [0.1, 0.2])
        self.assertEqual(map_box([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.6]), [0.1, 0.2, 0.3, 0.6])

    def test_candidates_require_visual_response_and_window_geometry(self):
        client = FakeClient([
            {"found": True, "bbox": [0.1, 0.1, 0.6, 0.5], "confidence": "high", "evidence": "洗面台と境界"},
            {"windows": [
                {"position": [0.5, 0.05], "confidence": "high", "evidence": "外壁の細線"},
                {"position": [0.5, 0.5], "confidence": "high", "evidence": "部屋中央"},
            ]},
        ])
        result = run_probes(FakeImage(), STRUCTURE, client, "test", max_window_rooms=1)
        self.assertEqual(result["probes"][0]["candidates"][0]["review_status"], "candidate")
        window_candidates = result["probes"][1]["candidates"]
        self.assertEqual(window_candidates[0]["review_status"], "candidate")
        self.assertEqual(window_candidates[1]["review_status"], "rejected_geometry_or_confidence")
        self.assertEqual(len(client.models.calls), 2)
        self.assertIn("自動反映しない", result["note"])

    def test_unknown_response_is_not_auto_confirmed_and_html_escapes_evidence(self):
        client = FakeClient([{"found": False, "bbox": None, "confidence": "low", "evidence": "<不明>"}])
        result = run_probes(FakeImage(), STRUCTURE, client, "test", max_window_rooms=0)
        self.assertEqual(result["probes"][0]["candidates"][0]["review_status"], "not_confirmed")
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.webp"
            output = Path(directory) / "review.html"
            render_review_html(image_path, result, output)
            page = output.read_text(encoding="utf-8")
            self.assertIn("&lt;不明&gt;", page)
            self.assertIn("svg", page)


if __name__ == "__main__":
    unittest.main()

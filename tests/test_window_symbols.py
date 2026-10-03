import unittest

from PIL import Image, ImageDraw

from floor_plan.window_symbols import detect_window_symbol_candidates


class ParallelWindowSymbolTests(unittest.TestCase):
    def test_three_parallel_strokes_make_unreviewed_candidate(self):
        image = Image.new("RGB", (80, 80), "white")
        draw = ImageDraw.Draw(image)
        for x in (20, 22, 24):
            draw.line((x, 10, x, 39), fill="black")
        candidates = detect_window_symbol_candidates(image)
        self.assertTrue(any(item["source"] == "parallel_strokes"
                            and item["orientation"] == "vertical"
                            and item["pixel_bbox"][0] <= 22 < item["pixel_bbox"][2]
                            and item["pixel_bbox"][1] <= 20 < item["pixel_bbox"][3]
                            and item["classification"] == "unreviewed"
                            for item in candidates))

    def test_broken_parallel_strokes_merge_small_occlusion(self):
        image = Image.new("RGB", (80, 80), "white")
        draw = ImageDraw.Draw(image)
        for y1, y2 in ((10, 25), (30, 49)):
            for x in (20, 22, 24):
                draw.line((x, y1, x, y2), fill="black")
        candidates = [item for item in detect_window_symbol_candidates(image)
                      if item["source"] == "parallel_strokes"]
        self.assertEqual(len(candidates), 1)
        self.assertLessEqual(candidates[0]["pixel_bbox"][1], 11)
        self.assertGreaterEqual(candidates[0]["pixel_bbox"][3], 49)

    def test_single_stroke_is_not_parallel_symbol(self):
        image = Image.new("RGB", (80, 80), "white")
        ImageDraw.Draw(image).line((20, 10, 20, 49), fill="black")
        self.assertFalse(any(item["source"] == "parallel_strokes"
                             for item in detect_window_symbol_candidates(image)))


if __name__ == "__main__":
    unittest.main()

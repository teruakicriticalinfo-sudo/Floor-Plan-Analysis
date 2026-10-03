import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from tools.inspect_red_boxes import red_rectangles


class RedBoxInspectionTests(unittest.TestCase):
    def test_finds_drawn_rectangles_and_ignores_right_legend(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotated.png"
            image = Image.new("RGB", (100, 60), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((10, 10, 25, 20), outline="red", width=2)
            draw.rectangle((80, 10, 90, 20), outline="red", width=2)
            image.save(path)
            self.assertEqual(red_rectangles(path, max_x=50), [[10, 10, 25, 20]])


if __name__ == "__main__":
    unittest.main()

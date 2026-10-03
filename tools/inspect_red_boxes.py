"""List red annotation rectangles in a supplied floor-plan screenshot."""

import argparse
import json
from collections import deque
from pathlib import Path

from PIL import Image


def red_rectangles(path: Path, max_x: int | None = None) -> list[list[int]]:
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    right = min(width, max_x) if max_x is not None else width
    pixels = image.load()
    red = {(x, y) for y in range(height) for x in range(right)
           if (lambda color: color[0] >= 200 and color[1] <= 80 and color[2] <= 80)(pixels[x, y])}
    boxes = []
    while red:
        start = red.pop()
        queue = deque([start])
        left = right_edge = start[0]
        top = bottom = start[1]
        count = 1
        while queue:
            x, y = queue.popleft()
            for neighbor in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if neighbor not in red:
                    continue
                red.remove(neighbor)
                queue.append(neighbor)
                left = min(left, neighbor[0])
                right_edge = max(right_edge, neighbor[0])
                top = min(top, neighbor[1])
                bottom = max(bottom, neighbor[1])
                count += 1
        if count >= 8 and right_edge > left and bottom > top:
            boxes.append([left, top, right_edge, bottom])
    return sorted(boxes, key=lambda box: (box[1], box[0]))


def register_reference(reference_path: Path, annotated_path: Path) -> dict[str, float]:
    """Fit a uniform scale and translation using dark pixels of the original plan."""
    with Image.open(reference_path) as source, Image.open(annotated_path) as annotated:
        original = source.convert("RGB")
        target = annotated.convert("RGB")
    source_pixels = original.load()
    target_pixels = target.load()
    samples = [(x, y, source_pixels[x, y])
               for y in range(8, original.height - 8, 7)
               for x in range(8, original.width - 8, 7)
               if min(source_pixels[x, y]) < 210]

    def score(scale: float, offset_x: int, offset_y: int) -> float:
        total = 0
        count = 0
        for x, y, color in samples:
            tx, ty = round(x * scale + offset_x), round(y * scale + offset_y)
            if not 0 <= tx < target.width or not 0 <= ty < target.height:
                continue
            observed = target_pixels[tx, ty]
            if observed[0] >= 200 and observed[1] <= 80 and observed[2] <= 80:
                continue
            total += sum(abs(first - second) for first, second in zip(color, observed))
            count += 1
        return total / count if count else float("inf")

    best = (float("inf"), 0.0, 0, 0)
    expected_scale = min(target.width / original.width, target.height / original.height)
    for index in range(21):
        scale = round(expected_scale - 0.1 + index * 0.01, 4)
        for offset_x in range(-24, 25, 6):
            for offset_y in range(-24, 25, 6):
                error = score(scale, offset_x, offset_y)
                if error < best[0]:
                    best = (error, scale, offset_x, offset_y)
    _, coarse_scale, coarse_x, coarse_y = best
    for index in range(-10, 11):
        scale = round(coarse_scale + index * 0.001, 4)
        for offset_x in range(coarse_x - 4, coarse_x + 5):
            for offset_y in range(coarse_y - 4, coarse_y + 5):
                error = score(scale, offset_x, offset_y)
                if error < best[0]:
                    best = (error, scale, offset_x, offset_y)
    return {"error": round(best[0], 2), "scale": best[1],
            "offset_x": best[2], "offset_y": best[3]}


def main() -> None:
    parser = argparse.ArgumentParser(description="赤い矩形注釈の画像ピクセル座標を列挙")
    parser.add_argument("image", type=Path)
    parser.add_argument("--max-x", type=int, help="右側の凡例を除外するための切断位置")
    parser.add_argument("--reference", type=Path, help="元画像への倍率・平行移動を推定")
    args = parser.parse_args()
    boxes = red_rectangles(args.image, args.max_x)
    if args.reference:
        alignment = register_reference(args.reference, args.image)
        scale = alignment["scale"]
        offset_x = alignment["offset_x"]
        offset_y = alignment["offset_y"]
        with Image.open(args.reference) as reference:
            source_width, source_height = reference.size
        mapped = [[round((box[0] + box[2]) / 2 / scale - alignment["offset_x"] / scale, 2),
                   round((box[1] + box[3]) / 2 / scale - alignment["offset_y"] / scale, 2)]
                  for box in boxes]
        normalized_boxes = [[round((box[0] - offset_x) / scale / source_width, 5),
                             round((box[1] - offset_y) / scale / source_height, 5),
                             round((box[2] - offset_x) / scale / source_width, 5),
                             round((box[3] - offset_y) / scale / source_height, 5)]
                            for box in boxes]
        print(json.dumps({"alignment": alignment, "screenshot_boxes": boxes,
                          "reference_centers": mapped, "reference_boxes_normalized": normalized_boxes}, indent=2))
    else:
        print(json.dumps(boxes, indent=2))


if __name__ == "__main__":
    main()

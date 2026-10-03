"""Find candidate window symbols made of parallel dark strokes.

This detects a drawing pattern, not a verified window.  It deliberately does
not read reference annotations; those are used only after candidate extraction.
Its stroke thresholds were adjusted while inspecting sample2, so sample2 is
development data rather than an independent benchmark.
"""

from __future__ import annotations

from PIL import Image, ImageOps

from .window_cv import detect_wall_gap_candidates


SYMBOL_DETECTOR_VERSION = "wall-symbol-v2"


def _true_runs(values: list[bool]) -> list[tuple[int, int]]:
    starts = []
    beginning = None
    for index, value in enumerate([*values, False]):
        if value and beginning is None:
            beginning = index
        elif not value and beginning is not None:
            starts.append((beginning, index))
            beginning = None
    return starts


def _majority_five(values: list[bool]) -> list[bool]:
    prefix = [0]
    for value in values:
        prefix.append(prefix[-1] + int(value))
    return [prefix[min(len(values), index + 3)] - prefix[max(0, index - 2)] >= 3
            for index in range(len(values))]


def _parallel_stroke_observations(brightness: list[list[int]], orientation: str) -> list[dict]:
    # Each opening symbol has three dark strokes with bright gaps between
    # them.  Allow 2- or 3-pixel stroke spacing for different rasterizations.
    bands = [list(column) for column in zip(*brightness)] if orientation == "vertical" else brightness
    observations = []
    for first_spacing in (2, 3):
        for second_spacing in (2, 3):
            band_width = first_spacing + second_spacing + 1
            for position in range(len(bands) - band_width + 1):
                stripes = bands[position:position + band_width]
                raw = []
                for along in range(len(stripes[0])):
                    matched = (stripes[0][along] < 180
                               and stripes[first_spacing][along] < 130
                               and stripes[-1][along] < 180
                               and all(stripes[offset][along] > 185
                                       for offset in range(1, first_spacing))
                               and all(stripes[first_spacing + offset][along] > 185
                                       for offset in range(1, second_spacing)))
                    raw.append(matched)
                for start, end in _true_runs(_majority_five(raw)):
                    if end - start < 8 or sum(raw[start:end]) < 6:
                        continue
                    box = ([position, start, position + band_width, end]
                           if orientation == "vertical" else
                           [start, position, end, position + band_width])
                    observations.append({"orientation": orientation,
                                         "pixel_bbox": box,
                                         "support_pixels": sum(raw[start:end])})
    return observations


def _close_observations(left: dict, right: dict) -> bool:
    if left["orientation"] != right["orientation"]:
        return False
    a, b = left["pixel_bbox"], right["pixel_bbox"]
    vertical = left["orientation"] == "vertical"
    perp = 0 if vertical else 1
    along = 1 if vertical else 0
    perp_distance = abs((a[perp] + a[perp + 2]) / 2 - (b[perp] + b[perp + 2]) / 2)
    along_gap = max(a[along], b[along]) - min(a[along + 2], b[along + 2])
    return perp_distance <= 4 and along_gap <= 5


def _merge_observations(observations: list[dict], size: tuple[int, int]) -> list[dict]:
    width, height = size
    clusters: list[dict] = []
    for observation in sorted(observations, key=lambda item: (
            item["orientation"], item["pixel_bbox"][0], item["pixel_bbox"][1])):
        for cluster in clusters:
            if _close_observations(cluster, observation):
                a, b = cluster["pixel_bbox"], observation["pixel_bbox"]
                cluster["pixel_bbox"] = [min(a[0], b[0]), min(a[1], b[1]),
                                         max(a[2], b[2]), max(a[3], b[3])]
                cluster["support_pixels"] = max(cluster["support_pixels"], observation["support_pixels"])
                break
        else:
            clusters.append({**observation})
    return [{"orientation": item["orientation"],
             "bbox": [round(item["pixel_bbox"][0] / width, 5),
                      round(item["pixel_bbox"][1] / height, 5),
                      round(item["pixel_bbox"][2] / width, 5),
                      round(item["pixel_bbox"][3] / height, 5)],
             "pixel_bbox": item["pixel_bbox"],
             "support_pixels": item["support_pixels"],
             "source": "parallel_strokes", "classification": "unreviewed"}
            for item in clusters]


def _same_candidate(left: dict, right: dict) -> bool:
    if left["orientation"] != right["orientation"]:
        return False
    a, b = left["pixel_bbox"], right["pixel_bbox"]
    vertical = left["orientation"] == "vertical"
    perp = 0 if vertical else 1
    along = 1 if vertical else 0
    if abs((a[perp] + a[perp + 2]) / 2 - (b[perp] + b[perp + 2]) / 2) > 5:
        return False
    overlap = min(a[along + 2], b[along + 2]) - max(a[along], b[along])
    shorter = min(a[along + 2] - a[along], b[along + 2] - b[along])
    longer = max(a[along + 2] - a[along], b[along + 2] - b[along])
    return overlap >= max(2, shorter * 0.55) and overlap >= longer * 0.5


def detect_window_symbol_candidates(image: Image.Image) -> list[dict]:
    """Combine wall interruptions and parallel-stroke symbols as unlabeled candidates."""
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    pixels = rgb.load()
    width, height = rgb.size
    brightness = [[max(pixels[x, y]) for x in range(width)] for y in range(height)]
    observations = (_parallel_stroke_observations(brightness, "vertical")
                    + _parallel_stroke_observations(brightness, "horizontal"))
    gap_candidates = detect_wall_gap_candidates(rgb)
    parallel_candidates = _merge_observations(observations, rgb.size)
    candidates = [*gap_candidates]
    for item in parallel_candidates:
        if not any(_same_candidate(item, existing) for existing in candidates):
            candidates.append(item)
    candidates.sort(key=lambda item: (item["pixel_bbox"][1], item["pixel_bbox"][0]))
    for number, candidate in enumerate(candidates, 1):
        candidate["id"] = f"C{number:03d}"
    return candidates

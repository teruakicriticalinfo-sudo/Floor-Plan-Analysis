"""Find visually plausible gaps in straight wall strokes without reference labels."""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageOps


DETECTOR_VERSION = "wall-gap-v1"


def _runs(values: list[bool]) -> list[tuple[bool, int, int]]:
    if not values:
        return []
    result = []
    start = 0
    previous = values[0]
    for index in range(1, len(values) + 1):
        current = values[index] if index < len(values) else not previous
        if current != previous:
            result.append((previous, start, index))
            start = index
            previous = current
    return result


def _wall_gap_observations(dark: list[list[bool]], width: int, height: int,
                           orientation: str, min_gap: int, max_gap: int) -> list[dict[str, Any]]:
    vertical = orientation == "vertical"
    line_count = width if vertical else height
    observations = []
    for line_index in range(line_count):
        values = ([dark[y][line_index] for y in range(height)] if vertical
                  else dark[line_index])
        runs = _runs(values)
        for run_index in range(1, len(runs) - 1):
            is_dark, start, end = runs[run_index]
            before, after = runs[run_index - 1], runs[run_index + 1]
            gap_length = end - start
            if (is_dark or not before[0] or not after[0]
                    or not min_gap <= gap_length <= max_gap
                    or min(before[2] - before[1], after[2] - after[1]) < 3
                    or max(before[2] - before[1], after[2] - after[1]) < 6):
                continue
            # The two uninterrupted dark runs must have enough wall pixels near
            # the opening. Text and furniture strokes are usually much thinner.
            if vertical:
                box = [line_index, start, line_index + 1, end]
            else:
                box = [start, line_index, end, line_index + 1]
            observations.append({"orientation": orientation, "pixel_bbox": box,
                                 "wall_before_px": before[2] - before[1],
                                 "wall_after_px": after[2] - after[1]})
    return observations


def _same_opening(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["orientation"] != right["orientation"]:
        return False
    a = left["pixel_bbox"]
    b = right["pixel_bbox"]
    vertical = left["orientation"] == "vertical"
    perp = 0 if vertical else 1
    along = 1 if vertical else 0
    if abs((a[perp] + a[perp + 2]) / 2 - (b[perp] + b[perp + 2]) / 2) > 5:
        return False
    overlap = min(a[along + 2], b[along + 2]) - max(a[along], b[along])
    shorter = min(a[along + 2] - a[along], b[along + 2] - b[along])
    return overlap >= max(2, shorter * 0.55)


def _merge_observations(observations: list[dict[str, Any]], image_size: tuple[int, int]) -> list[dict[str, Any]]:
    width, height = image_size
    clusters: list[dict[str, Any]] = []
    for observation in sorted(observations, key=lambda item: (
            item["orientation"], item["pixel_bbox"][0], item["pixel_bbox"][1])):
        for cluster in clusters:
            if _same_opening(cluster, observation):
                a, b = cluster["pixel_bbox"], observation["pixel_bbox"]
                cluster["pixel_bbox"] = [min(a[0], b[0]), min(a[1], b[1]),
                                         max(a[2], b[2]), max(a[3], b[3])]
                cluster["support_lines"] += 1
                cluster["wall_before_px"] = max(cluster["wall_before_px"], observation["wall_before_px"])
                cluster["wall_after_px"] = max(cluster["wall_after_px"], observation["wall_after_px"])
                break
        else:
            clusters.append({**observation, "support_lines": 1})

    candidates = []
    for cluster in clusters:
        x1, y1, x2, y2 = cluster["pixel_bbox"]
        # Keep only wall-sized strokes, not isolated one-pixel lettering.
        if cluster["support_lines"] < 3:
            continue
        candidates.append({"orientation": cluster["orientation"],
                           "bbox": [round(x1 / width, 5), round(y1 / height, 5),
                                    round(x2 / width, 5), round(y2 / height, 5)],
                           "pixel_bbox": [x1, y1, x2, y2],
                           "support_lines": cluster["support_lines"],
                           "wall_before_px": cluster["wall_before_px"],
                           "wall_after_px": cluster["wall_after_px"],
                           "source": "straight_wall_gap", "classification": "unreviewed"})
    candidates.sort(key=lambda item: (item["pixel_bbox"][1], item["pixel_bbox"][0]))
    for number, candidate in enumerate(candidates, 1):
        candidate["id"] = f"C{number:03d}"
    return candidates


def detect_wall_gap_candidates(image: Image.Image, *, dark_threshold: int = 105,
                               min_gap: int = 4, max_gap: int = 42) -> list[dict[str, Any]]:
    """Detect straight wall interruptions; a candidate is not a verified window."""
    if not 0 < dark_threshold < 256 or min_gap < 2 or max_gap < min_gap:
        raise ValueError("検出閾値が不正です")
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    dark = [[max(pixels[x, y]) < dark_threshold for x in range(width)] for y in range(height)]
    observations = _wall_gap_observations(dark, width, height, "vertical", min_gap, max_gap)
    observations.extend(_wall_gap_observations(dark, width, height, "horizontal", min_gap, max_gap))
    return _merge_observations(observations, rgb.size)

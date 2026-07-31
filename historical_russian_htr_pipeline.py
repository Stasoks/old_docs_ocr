#!/usr/bin/env python3
"""
Полный локальный HTR-пайплайн для русских исторических документов.

Этапы:
1. PDF -> изображения страниц.
2. Deskew и безопасная прямоугольная обрезка внешних полей.
3. Деление страницы на перекрывающиеся горизонтальные тайлы.
4. RF-DETR: детекция/сегментация отдельных строк.
5. Геометрическая дедупликация строк между соседними тайлами.
6. ATH-MaaS/OvisOCR2: OCR каждой строки.
7. Детерминированная сборка строк в JSON-блоки.
8. Опционально: Qwen через LM Studio классифицирует блоки, но НЕ переписывает OCR-текст.

Модели по умолчанию:
- detector: Kansallisarkisto/rfdetr_textline_textregion_detection_model
- recognizer: ATH-MaaS/OvisOCR2

Python: 3.10+
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import re
import shutil
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageOps

DEFAULT_DETECTOR_REPO = "Kansallisarkisto/rfdetr_textline_textregion_detection_model"
DEFAULT_DETECTOR_FILENAME = "rfdetr_text_seg_model_202510.pth"
DEFAULT_OCR_MODEL = "ATH-MaaS/OvisOCR2"
DEFAULT_QWEN_MODEL = "qwen/qwen2.5-vl-7b"
DEFAULT_LMSTUDIO_URL = "http://localhost:1234/v1"

OVIS_LINE_OCR_PROMPT = """
Transcribe every readable character in this single cropped text line from a
historical document. Return only the exact transcription as plain text.
Preserve the original language, spelling, punctuation, capitalization, and
word order. Do not translate, paraphrase, correct, explain, or add Markdown
formatting. If the line is unreadable, return [НЕРАЗБОРЧИВО].
""".strip()

BLOCK_TYPES = [
    "header",
    "footer",
    "page_number",
    "title",
    "subtitle",
    "author",
    "affiliation",
    "date",
    "text",
    "abstract",
    "keywords",
    "section_title",
    "list_item",
    "formula",
    "table",
    "caption",
    "footnote",
    "handwritten_note",
    "stamp",
    "signature",
    "unknown",
]

SOURCE_TYPES = ["printed", "handwritten", "mixed"]


@dataclass
class DetectionRecord:
    kind: str
    bbox: list[int]
    polygon: list[list[int]]
    confidence: float
    tile_ids: list[int] = field(default_factory=list)
    crop_path: str | None = None
    line_id: str | None = None
    region_id: str | None = None
    text: str = ""
    raw_text: str = ""
    quality_flags: list[str] = field(default_factory=list)

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    def to_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": self.line_id,
            "type": self.kind,
            "bbox": self.bbox,
            "polygon": self.polygon,
            "detector_confidence": round(float(self.confidence), 6),
            "tile_ids": sorted(set(self.tile_ids)),
        }
        if self.region_id is not None:
            result["region_id"] = self.region_id
        if self.crop_path is not None:
            result["crop_path"] = self.crop_path
        if self.kind == "text_line":
            result["text"] = self.text
            result["raw_text"] = self.raw_text
            result["quality_flags"] = self.quality_flags
        return result


# ---------------------------------------------------------------------------
# Общие функции
# ---------------------------------------------------------------------------


def log(message: str) -> None:
    print(message, flush=True)


def warn(message: str) -> None:
    print(f"[WARNING] {message}", file=sys.stderr, flush=True)


def save_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    json.loads(text)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def parse_page_spec(spec: str | None, page_count: int) -> list[int]:
    """Возвращает нулевые индексы страниц."""
    if not spec:
        return list(range(page_count))

    selected: set[int] = set()
    for raw_chunk in spec.split(","):
        chunk = raw_chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            left, right = chunk.split("-", 1)
            start = int(left)
            end = int(right)
            if start > end:
                start, end = end, start
            selected.update(range(start - 1, end))
        else:
            selected.add(int(chunk) - 1)

    result = sorted(index for index in selected if 0 <= index < page_count)
    if not result:
        raise ValueError("После разбора --pages не осталось допустимых страниц.")
    return result


def union_bbox(boxes: Iterable[Sequence[int]]) -> list[int]:
    boxes_list = [list(map(int, box)) for box in boxes]
    if not boxes_list:
        return [0, 0, 0, 0]
    return [
        min(box[0] for box in boxes_list),
        min(box[1] for box in boxes_list),
        max(box[2] for box in boxes_list),
        max(box[3] for box in boxes_list),
    ]


def bbox_iou(first: Sequence[int], second: Sequence[int]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection <= 0:
        return 0.0
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def bbox_intersection_over_smaller(
    first: Sequence[int],
    second: Sequence[int],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection <= 0:
        return 0.0
    first_area = max(1, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / min(first_area, second_area)


def bbox_overlap_fraction_x(first: Sequence[int], second: Sequence[int]) -> float:
    overlap = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    minimum = max(1, min(first[2] - first[0], second[2] - second[0]))
    return overlap / minimum


# ---------------------------------------------------------------------------
# Рендер, deskew, crop, тайлы
# ---------------------------------------------------------------------------


def render_page_image(page: pymupdf.Page, dpi: int) -> Image.Image:
    zoom = dpi / 72
    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)


def _normalize_line_angle(angle: float) -> float:
    while angle <= -90:
        angle += 180
    while angle > 90:
        angle -= 180
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    return angle


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    index = int(np.searchsorted(cumulative, weights.sum() / 2))
    return float(values[min(index, len(values) - 1)])


def estimate_document_skew(image_bgr: np.ndarray) -> float:
    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]

    margin = max(3, int(min(width, height) * 0.015))
    binary[:margin, :] = 0
    binary[-margin:, :] = 0
    binary[:, :margin] = 0
    binary[:, -margin:] = 0

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(20, width // 35), max(1, height // 1200)),
    )
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        connected,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    angles: list[float] = []
    weights: list[float] = []
    for contour in contours:
        _, _, box_width, box_height = cv2.boundingRect(contour)
        if box_width < width * 0.10:
            continue
        if box_width < box_height * 3:
            continue
        if box_height > height * 0.12:
            continue

        rectangle = cv2.minAreaRect(contour)
        points = cv2.boxPoints(rectangle)
        longest_length = 0.0
        longest_angle = 0.0
        for index in range(4):
            point1 = points[index]
            point2 = points[(index + 1) % 4]
            dx = float(point2[0] - point1[0])
            dy = float(point2[1] - point1[1])
            length = math.hypot(dx, dy)
            if length > longest_length:
                longest_length = length
                longest_angle = math.degrees(math.atan2(dy, dx))

        angle = _normalize_line_angle(longest_angle)
        if abs(angle) <= 12:
            angles.append(angle)
            weights.append(longest_length)

    if len(angles) < 3:
        return 0.0

    angle_array = np.asarray(angles, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    initial = _weighted_median(angle_array, weight_array)
    reliable = np.abs(angle_array - initial) <= 2.5
    if reliable.sum() >= 3:
        angle_array = angle_array[reliable]
        weight_array = weight_array[reliable]
    return _weighted_median(angle_array, weight_array)


def rotate_without_crop(image_bgr: np.ndarray, angle: float) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine = abs(matrix[0, 0])
    sine = abs(matrix[0, 1])
    new_width = int(height * sine + width * cosine)
    new_height = int(height * cosine + width * sine)
    matrix[0, 2] += new_width / 2 - center[0]
    matrix[1, 2] += new_height / 2 - center[1]
    return cv2.warpAffine(
        image_bgr,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def deskew_document(
    image: Image.Image,
    minimum_angle: float = 0.15,
    maximum_angle: float = 12.0,
) -> tuple[Image.Image, float]:
    rgb = image.convert("RGB")
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    angle = estimate_document_skew(bgr)
    if abs(angle) < minimum_angle or abs(angle) > maximum_angle:
        return rgb, 0.0
    corrected = rotate_without_crop(bgr, angle)
    corrected_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
    return Image.fromarray(corrected_rgb), angle


def make_ink_mask(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    sigma = max(8.0, min(width, height) / 120)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalized = cv2.divide(gray, background, scale=255)
    local = cv2.threshold(
        normalized,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]
    strong = np.where(gray < 205, 255, 0).astype(np.uint8)
    mask = cv2.bitwise_or(local, strong)

    x_border = max(2, int(width * 0.006))
    y_border = max(2, int(height * 0.006))
    mask[:y_border, :] = 0
    mask[-y_border:, :] = 0
    mask[:, :x_border] = 0
    mask[:, -x_border:] = 0
    return mask


def safe_outer_crop(
    image: Image.Image,
    padding_x_ratio: float = 0.04,
    padding_y_ratio: float = 0.02,
) -> tuple[Image.Image, tuple[int, int], dict[str, Any]]:
    """Обрезает только внешние поля. Контент внутри страницы не переставляет."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    mask = make_ink_mask(rgb)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8,
    )
    useful = np.zeros_like(mask)
    minimum_area = max(14, int(width * height * 0.000003))
    for label in range(1, component_count):
        _, _, box_width, box_height, area = stats[label]
        if area < minimum_area:
            continue
        if box_width <= 1 or box_height <= 2:
            continue
        useful[labels == label] = 255

    ys, xs = np.where(useful > 0)
    if len(xs) == 0:
        return (
            rgb,
            (0, 0),
            {
                "applied": False,
                "reason": "content_not_found",
                "original_size": [width, height],
                "result_size": [width, height],
            },
        )

    padding_x = max(24, int(width * padding_x_ratio))
    padding_y = max(20, int(height * padding_y_ratio))
    left = max(0, int(xs.min()) - padding_x)
    right = min(width, int(xs.max()) + 1 + padding_x)
    top = max(0, int(ys.min()) - padding_y)
    bottom = min(height, int(ys.max()) + 1 + padding_y)

    result_width = right - left
    result_height = bottom - top
    retained = result_width * result_height / max(1, width * height)
    if result_width < width * 0.30 or result_height < height * 0.20 or retained < 0.10:
        return (
            rgb,
            (0, 0),
            {
                "applied": False,
                "reason": "crop_too_aggressive",
                "original_size": [width, height],
                "result_size": [width, height],
            },
        )

    result = rgb.crop((left, top, right, bottom))
    return (
        result,
        (left, top),
        {
            "applied": True,
            "offset": [left, top],
            "original_size": [width, height],
            "result_size": [result.width, result.height],
            "retained_area_ratio": retained,
        },
    )


def horizontal_ink_profile(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"))
    height, width = gray.shape
    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]
    x_margin = max(1, int(width * 0.06))
    roi = binary[:, x_margin : max(x_margin + 1, width - x_margin)]
    if roi.size == 0:
        roi = binary
    profile = np.count_nonzero(roi, axis=1).astype(np.float64)
    profile /= max(1, roi.shape[1])
    smooth_window = max(5, int(height * 0.004))
    if smooth_window % 2 == 0:
        smooth_window += 1
    kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
    return np.convolve(profile, kernel, mode="same")


def find_line_safe_cuts(
    image: Image.Image,
    tile_count: int,
    search_ratio: float,
) -> list[int]:
    if tile_count <= 1:
        return []
    _, height = image.size
    nominal_height = height / tile_count
    profile = horizontal_ink_profile(image)
    band_half = max(4, int(height * 0.003))
    kernel = np.ones(2 * band_half + 1, dtype=np.float64)
    kernel /= kernel.size
    band_profile = np.convolve(profile, kernel, mode="same")

    cuts: list[int] = []
    minimum_core = max(40, int(nominal_height * 0.55))
    for cut_index in range(1, tile_count):
        nominal = round(cut_index * nominal_height)
        radius = max(20, int(nominal_height * search_ratio))
        remaining = tile_count - cut_index
        lower = nominal - radius
        upper = nominal + radius
        lower = max(lower, (cuts[-1] + minimum_core) if cuts else minimum_core)
        upper = min(upper, height - remaining * minimum_core)
        if lower >= upper:
            cuts.append(nominal)
            continue
        candidates = np.arange(lower, upper + 1)
        local_ink = band_profile[candidates]
        distance = np.abs(candidates - nominal) / max(1, radius)
        ink_scale = max(float(np.percentile(local_ink, 75)), 1e-6)
        score = local_ink + distance * ink_scale * 0.08
        cuts.append(int(candidates[int(np.argmin(score))]))
    return cuts


def make_vertical_tiles(
    image: Image.Image,
    tile_count: int,
    overlap_ratio: float,
    cut_search_ratio: float,
) -> list[tuple[int, int, int, Image.Image]]:
    width, height = image.size
    if tile_count <= 1:
        return [(0, 0, height, image.copy())]
    cuts = find_line_safe_cuts(image, tile_count, cut_search_ratio)
    boundaries = [0, *cuts, height]
    overlap = int((height / tile_count) * overlap_ratio)
    tiles: list[tuple[int, int, int, Image.Image]] = []
    for index in range(tile_count):
        top = boundaries[index]
        bottom = boundaries[index + 1]
        if index > 0:
            top = max(0, top - overlap)
        if index < tile_count - 1:
            bottom = min(height, bottom + overlap)
        tiles.append((index, top, bottom, image.crop((0, top, width, bottom))))
    return tiles


def resize_long_side(image: Image.Image, max_size: int) -> Image.Image:
    width, height = image.size
    scale = max_size / max(width, height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# RF-DETR
# ---------------------------------------------------------------------------


def resolve_detector_weights(args: argparse.Namespace) -> Path:
    if args.detector_weights:
        path = Path(args.detector_weights).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Не найден detector checkpoint: {path}")
        return path

    from huggingface_hub import hf_hub_download

    log(
        "Скачивание/поиск detector checkpoint: "
        f"{args.detector_repo}/{args.detector_filename}"
    )
    return Path(
        hf_hub_download(
            repo_id=args.detector_repo,
            filename=args.detector_filename,
        )
    )


def load_detector(weights: Path, cpu: bool = False) -> Any:
    if cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        from rfdetr import RFDETRSegPreview
    except ImportError as error:
        raise RuntimeError(
            "В установленном rfdetr отсутствует RFDETRSegPreview. "
            "Установи версию из requirements.txt, например rfdetr==1.6.2."
        ) from error

    log(f"Загрузка RF-DETR: {weights}")
    model = RFDETRSegPreview(pretrain_weights=str(weights))
    try:
        model.optimize_for_inference()
    except Exception as error:
        warn(f"optimize_for_inference не сработал, продолжаю без него: {error}")
    return model


def largest_contour_polygon(mask: np.ndarray) -> list[list[int]]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon = max(1.0, 0.002 * cv2.arcLength(contour, True))
    simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return [[int(x), int(y)] for x, y in simplified]


def build_masked_line_crop(
    tile_rgb: np.ndarray,
    mask: np.ndarray | None,
    bbox: Sequence[int],
    pad_y_ratio: float,
    pad_x_ratio: float,
) -> Image.Image:
    tile_height, tile_width = tile_rgb.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    line_height = max(1, y2 - y1)
    line_width = max(1, x2 - x1)
    pad_y = max(4, int(line_height * pad_y_ratio))
    pad_x = max(8, int(max(line_height * 0.55, line_width * pad_x_ratio)))
    x1p = max(0, x1 - pad_x)
    y1p = max(0, y1 - pad_y)
    x2p = min(tile_width, x2 + pad_x)
    y2p = min(tile_height, y2 + pad_y)

    crop = tile_rgb[y1p:y2p, x1p:x2p].copy()
    if mask is not None:
        crop_mask = mask[y1p:y2p, x1p:x2p].astype(np.uint8)
        kernel_size = max(3, round(line_height * 0.08))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        crop_mask = cv2.dilate(crop_mask, kernel, iterations=1)
        white = np.full_like(crop, 255)
        crop = np.where(crop_mask[..., None] > 0, crop, white)

    return Image.fromarray(crop).convert("RGB")


def predict_tile(
    detector: Any,
    tile: Image.Image,
    tile_id: int,
    tile_top: int,
    detector_max_size: int,
    confidence_threshold: float,
    line_class_id: int,
    region_class_id: int,
    line_area_threshold: float,
    region_area_threshold: float,
    line_pad_y: float,
    line_pad_x: float,
) -> tuple[list[tuple[DetectionRecord, Image.Image]], list[DetectionRecord]]:
    original_width, original_height = tile.size
    model_input = resize_long_side(tile.convert("RGB"), detector_max_size)
    detections = detector.predict(
        np.asarray(model_input),
        threshold=confidence_threshold,
    )

    xyxy = np.asarray(getattr(detections, "xyxy", []))
    class_ids = np.asarray(getattr(detections, "class_id", []))
    confidences = np.asarray(getattr(detections, "confidence", []))
    masks = getattr(detections, "mask", None)

    tile_rgb = np.asarray(tile.convert("RGB"))
    lines: list[tuple[DetectionRecord, Image.Image]] = []
    regions: list[DetectionRecord] = []

    if len(class_ids) == 0:
        return lines, regions

    for index, class_id_raw in enumerate(class_ids):
        class_id = int(class_id_raw)
        if class_id not in (line_class_id, region_class_id):
            continue
        confidence = (
            float(confidences[index])
            if len(confidences) > index and confidences[index] is not None
            else 0.0
        )

        mask_original: np.ndarray | None = None
        polygon_local: list[list[int]] = []
        if masks is not None and len(masks) > index:
            mask_small = np.asarray(masks[index]).astype(np.uint8)
            mask_original = cv2.resize(
                mask_small,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_original = (mask_original > 0).astype(np.uint8)
            ys, xs = np.where(mask_original > 0)
            if len(xs) == 0:
                continue
            bbox_local = [
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1,
            ]
            polygon_local = largest_contour_polygon(mask_original)
        else:
            if len(xyxy) <= index:
                continue
            box = xyxy[index]
            scale_x = original_width / model_input.width
            scale_y = original_height / model_input.height
            bbox_local = [
                round(float(box[0]) * scale_x),
                round(float(box[1]) * scale_y),
                round(float(box[2]) * scale_x),
                round(float(box[3]) * scale_y),
            ]
            polygon_local = [
                [bbox_local[0], bbox_local[1]],
                [bbox_local[2], bbox_local[1]],
                [bbox_local[2], bbox_local[3]],
                [bbox_local[0], bbox_local[3]],
            ]

        x1, y1, x2, y2 = bbox_local
        x1 = max(0, min(original_width - 1, x1))
        y1 = max(0, min(original_height - 1, y1))
        x2 = max(x1 + 1, min(original_width, x2))
        y2 = max(y1 + 1, min(original_height, y2))
        bbox_local = [x1, y1, x2, y2]

        area_ratio = ((x2 - x1) * (y2 - y1)) / max(
            1,
            original_width * original_height,
        )
        threshold = (
            line_area_threshold if class_id == line_class_id else region_area_threshold
        )
        if area_ratio < threshold:
            continue

        bbox_page = [x1, y1 + tile_top, x2, y2 + tile_top]
        polygon_page = [[int(x), int(y + tile_top)] for x, y in polygon_local]

        if class_id == line_class_id:
            crop = build_masked_line_crop(
                tile_rgb=tile_rgb,
                mask=mask_original,
                bbox=bbox_local,
                pad_y_ratio=line_pad_y,
                pad_x_ratio=line_pad_x,
            )
            record = DetectionRecord(
                kind="text_line",
                bbox=bbox_page,
                polygon=polygon_page,
                confidence=confidence,
                tile_ids=[tile_id + 1],
            )
            lines.append((record, crop))
        else:
            regions.append(
                DetectionRecord(
                    kind="text_region",
                    bbox=bbox_page,
                    polygon=polygon_page,
                    confidence=confidence,
                    tile_ids=[tile_id + 1],
                )
            )

    return lines, regions


def choose_better_duplicate(
    first: tuple[DetectionRecord, Image.Image],
    second: tuple[DetectionRecord, Image.Image],
) -> tuple[DetectionRecord, Image.Image]:
    first_record, _ = first
    second_record, _ = second
    if abs(first_record.confidence - second_record.confidence) <= 0.08:
        chosen = first if first_record.area >= second_record.area else second
    else:
        chosen = (
            first if first_record.confidence >= second_record.confidence else second
        )
    chosen[0].tile_ids = sorted(set(first_record.tile_ids + second_record.tile_ids))
    return chosen


def deduplicate_lines(
    records: list[tuple[DetectionRecord, Image.Image]],
    iou_threshold: float,
    overlap_threshold: float,
) -> list[tuple[DetectionRecord, Image.Image]]:
    ordered = sorted(
        records,
        key=lambda item: (
            item[0].bbox[1],
            item[0].bbox[0],
            -item[0].confidence,
        ),
    )
    result: list[tuple[DetectionRecord, Image.Image]] = []
    for candidate in ordered:
        candidate_record = candidate[0]
        duplicate_index: int | None = None
        for index, existing in enumerate(result):
            existing_record = existing[0]
            if abs(candidate_record.center[1] - existing_record.center[1]) > max(
                candidate_record.height,
                existing_record.height,
            ):
                continue
            iou = bbox_iou(candidate_record.bbox, existing_record.bbox)
            overlap = bbox_intersection_over_smaller(
                candidate_record.bbox,
                existing_record.bbox,
            )
            if iou >= iou_threshold or overlap >= overlap_threshold:
                duplicate_index = index
                break
        if duplicate_index is None:
            result.append(candidate)
        else:
            result[duplicate_index] = choose_better_duplicate(
                result[duplicate_index],
                candidate,
            )
    return result


def deduplicate_regions(
    records: list[DetectionRecord],
    iou_threshold: float = 0.45,
) -> list[DetectionRecord]:
    ordered = sorted(records, key=lambda item: -item.confidence)
    result: list[DetectionRecord] = []
    for candidate in ordered:
        match: DetectionRecord | None = None
        for existing in result:
            if bbox_iou(candidate.bbox, existing.bbox) >= iou_threshold:
                match = existing
                break
        if match is None:
            result.append(candidate)
        else:
            match.tile_ids = sorted(set(match.tile_ids + candidate.tile_ids))
    result.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    for index, region in enumerate(result, start=1):
        region.region_id = f"r{index:03d}"
    return result


def assign_regions(
    lines: list[tuple[DetectionRecord, Image.Image]],
    regions: list[DetectionRecord],
) -> None:
    for line, _ in lines:
        center_x, center_y = line.center
        candidates: list[tuple[float, int, DetectionRecord]] = []
        for region in regions:
            contains = (
                region.bbox[0] <= center_x <= region.bbox[2]
                and region.bbox[1] <= center_y <= region.bbox[3]
            )
            overlap = bbox_intersection_over_smaller(line.bbox, region.bbox)
            if contains or overlap >= 0.35:
                candidates.append((overlap, -region.area, region))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            line.region_id = candidates[0][2].region_id


def reading_order(
    lines: list[tuple[DetectionRecord, Image.Image]],
) -> list[tuple[DetectionRecord, Image.Image]]:
    if not lines:
        return []
    median_height = float(np.median([item[0].height for item in lines]))
    tolerance = max(8.0, median_height * 0.45)
    candidates = sorted(lines, key=lambda item: (item[0].center[1], item[0].bbox[0]))
    rows: list[list[tuple[DetectionRecord, Image.Image]]] = []
    row_centers: list[float] = []

    for item in candidates:
        y_center = item[0].center[1]
        best_index: int | None = None
        best_distance = float("inf")
        for index, center in enumerate(row_centers):
            distance = abs(y_center - center)
            if distance <= tolerance and distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index is None:
            rows.append([item])
            row_centers.append(y_center)
        else:
            rows[best_index].append(item)
            row_centers[best_index] = float(
                np.mean([member[0].center[1] for member in rows[best_index]])
            )

    order = sorted(range(len(rows)), key=lambda index: row_centers[index])
    result: list[tuple[DetectionRecord, Image.Image]] = []
    for row_index in order:
        row = sorted(rows[row_index], key=lambda item: item[0].bbox[0])
        result.extend(row)
    return result


def save_detection_overlay(
    page_image: Image.Image,
    lines: Sequence[DetectionRecord],
    regions: Sequence[DetectionRecord],
    output_path: Path,
) -> None:
    overlay = page_image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    for region in regions:
        draw.rectangle(region.bbox, outline=(90, 90, 90), width=2)
        if region.region_id:
            draw.text(
                (region.bbox[0], region.bbox[1]), region.region_id, fill=(90, 90, 90)
            )
    for line in lines:
        draw.rectangle(line.bbox, outline=(0, 0, 0), width=2)
        if line.line_id:
            draw.text(
                (line.bbox[0], max(0, line.bbox[1] - 12)), line.line_id, fill=(0, 0, 0)
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)


# ---------------------------------------------------------------------------
# OvisOCR2
# ---------------------------------------------------------------------------


def resolve_torch_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Запрошен CUDA, но torch.cuda.is_available() == False.")
    return requested


def resolve_torch_dtype(name: str, device: str) -> Any:
    import torch

    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_ocr_model(model_name: str, device: str, dtype_name: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    resolved_device = resolve_torch_device(device)
    dtype = resolve_torch_dtype(dtype_name, resolved_device)
    log(f"Загрузка OCR-модели {model_name} на {resolved_device}, dtype={dtype}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_name,
        dtype=dtype,
    )
    model.to(resolved_device)
    model.eval()
    tokenizer = processor.tokenizer
    stop_token_ids = {
        int(tokenizer.eos_token_id),
        int(tokenizer.convert_tokens_to_ids("<|im_end|>")),
    }
    model.generation_config.eos_token_id = sorted(stop_token_ids)
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.eos_token_id
    if resolved_device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    return processor, model


def trim_line_whitespace(image: Image.Image, threshold: int = 248) -> Image.Image:
    rgb = image.convert("RGB")
    gray = np.asarray(rgb.convert("L"))
    ys, xs = np.where(gray < threshold)
    if len(xs) == 0:
        return rgb
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pad_x = max(4, int((x2 - x1) * 0.02))
    pad_y = max(3, int((y2 - y1) * 0.12))
    return rgb.crop(
        (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(rgb.width, x2 + pad_x),
            min(rgb.height, y2 + pad_y),
        )
    )


def fit_line_to_canvas(
    image: Image.Image,
    width: int = 1024,
    height: int = 256,
    autocontrast: bool = True,
) -> Image.Image:
    line = trim_line_whitespace(image.convert("RGB"))
    if autocontrast:
        line = ImageOps.autocontrast(line, cutoff=1)
    margin_x = max(8, int(width * 0.015))
    margin_y = max(6, int(height * 0.08))
    available_width = width - 2 * margin_x
    available_height = height - 2 * margin_y
    scale = min(
        available_width / max(1, line.width),
        available_height / max(1, line.height),
    )
    new_width = max(1, round(line.width * scale))
    new_height = max(1, round(line.height * scale))
    resized = line.resize((new_width, new_height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    x = (width - new_width) // 2
    y = (height - new_height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def repeated_pattern(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 24:
        return False
    if re.search(r"(.)\1{11,}", compact):
        return True
    for period in range(1, min(12, len(compact) // 6 + 1)):
        pattern = compact[:period]
        repeated = (pattern * (len(compact) // period + 1))[: len(compact)]
        similarity = sum(a == b for a, b in zip(compact, repeated)) / len(compact)
        if similarity >= 0.88:
            return True
    return False


def truncate_token_repetition(
    text: str,
    minimum_repeats: int = 4,
    maximum_period_tokens: int = 12,
) -> tuple[str, bool]:
    """Cuts an exact repeated token tail while retaining its first copy."""
    matches = list(re.finditer(r"\S+", text))
    tokens = [match.group(0) for match in matches]
    token_count = len(tokens)

    for start in range(token_count):
        remaining = token_count - start
        for period in range(1, min(maximum_period_tokens, remaining) + 1):
            if remaining < period * minimum_repeats or remaining % period:
                continue
            unit = tokens[start : start + period]
            if all(
                tokens[index] == unit[(index - start) % period]
                for index in range(start, token_count)
            ):
                end = matches[start + period - 1].end()
                return text[:end].rstrip(), True

    return text, False


def inspect_ocr_text(text: str) -> tuple[str, list[str]]:
    raw = re.sub(r"\s+", " ", text).strip()
    raw = re.sub(r"^#{1,6}\s+", "", raw)
    flags: list[str] = []
    if not raw:
        return "[НЕРАЗБОРЧИВО]", ["empty_output"]
    raw, repetition_was_truncated = truncate_token_repetition(raw)
    if repetition_was_truncated:
        flags.append("repetition_truncated")
    if repeated_pattern(raw):
        return "[НЕРАЗБОРЧИВО]", [*flags, "repetition_loop"]

    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", raw)
    if letters:
        latin = sum(letter.isascii() and letter.isalpha() for letter in letters)
        cyrillic = sum(bool(re.fullmatch(r"[А-Яа-яЁё]", letter)) for letter in letters)
        if latin > cyrillic * 1.5 and latin >= 5:
            flags.append("suspicious_latin_output")
    if len(raw) >= 240:
        flags.append("very_long_line")
    return raw, flags


def batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def recognize_line_images(
    processor: Any,
    model: Any,
    device: str,
    line_records: list[DetectionRecord],
    page_dir: Path,
    batch_size: int,
    max_new_tokens: int,
    canvas_width: int,
    canvas_height: int,
    autocontrast: bool,
    save_prepared_lines: bool,
) -> None:
    import torch

    resolved_device = resolve_torch_device(device)
    for batch_number, batch in enumerate(batched(line_records, batch_size), start=1):
        images: list[Image.Image] = []
        for line in batch:
            if not line.crop_path:
                raise RuntimeError(f"У строки {line.line_id} отсутствует crop_path.")
            crop_path = page_dir / line.crop_path
            crop = Image.open(crop_path).convert("RGB")
            prepared = fit_line_to_canvas(
                crop,
                width=canvas_width,
                height=canvas_height,
                autocontrast=autocontrast,
            )
            if save_prepared_lines:
                prepared_dir = page_dir / "lines" / "prepared"
                prepared_dir.mkdir(parents=True, exist_ok=True)
                prepared.save(prepared_dir / f"{line.line_id}.png")
            images.append(prepared)

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": OVIS_LINE_OCR_PROMPT},
                    ],
                }
            ]
            for image in images
        ]
        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"text_kwargs": {"padding": True}},
        )
        inputs = inputs.to(resolved_device)
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
            )
        generated_only = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        texts = processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for line, raw_text in zip(batch, texts):
            cleaned, flags = inspect_ocr_text(raw_text)
            line.raw_text = re.sub(r"\s+", " ", raw_text).strip()
            line.text = cleaned
            line.quality_flags = flags
        log(f"  OCR batch {batch_number}: {len(batch)} строк")


# ---------------------------------------------------------------------------
# Сборка JSON
# ---------------------------------------------------------------------------


def classify_line_type(line: DetectionRecord, page_width: int, page_height: int) -> str:
    text = line.text.strip()
    if not text:
        return "unknown"
    x1, y1, x2, y2 = line.bbox
    center_x = (x1 + x2) / 2
    near_top_or_bottom = y1 < page_height * 0.14 or y2 > page_height * 0.86
    near_side = center_x < page_width * 0.25 or center_x > page_width * 0.75

    if (
        re.fullmatch(r"[-–— ]*\d{1,4}[-–— ]*", text)
        and near_top_or_bottom
        and near_side
    ):
        return "page_number"
    if (
        re.search(
            r"\b(?:[0-3]?\d[./-][01]?\d[./-](?:\d{2}|\d{4})|"
            r"[0-3]?\d\s+[а-яё]+\s+\d{4})\b",
            text.lower(),
        )
        and len(text) <= 80
    ):
        return "date"
    if re.match(r"^\s*(?:[-•·]|\d+[.)]|[а-яё][.)])\s+", text.lower()):
        return "list_item"

    letters = re.findall(r"[А-Яа-яЁёA-Za-z]", text)
    uppercase_ratio = (
        sum(letter.isupper() for letter in letters) / len(letters) if letters else 0.0
    )
    centered = abs(center_x - page_width / 2) <= page_width * 0.16
    if len(text) <= 100 and centered and uppercase_ratio >= 0.65:
        return "title"
    if len(text) <= 100 and centered and y1 < page_height * 0.35:
        return "subtitle"
    return "text"


def group_lines_deterministically(
    lines: list[DetectionRecord],
    page_width: int,
    page_height: int,
    default_source: str,
) -> list[dict[str, Any]]:
    if not lines:
        return []
    median_height = float(np.median([max(1, line.height) for line in lines]))
    blocks: list[dict[str, Any]] = []
    current: list[DetectionRecord] = []
    current_type: str | None = None

    def flush() -> None:
        nonlocal current, current_type
        if not current:
            return
        blocks.append(
            {
                "type": current_type or "text",
                "source": default_source,
                "content": "\n".join(line.text for line in current),
                "line_ids": [line.line_id for line in current],
                "bbox": union_bbox(line.bbox for line in current),
            }
        )
        current = []
        current_type = None

    previous: DetectionRecord | None = None
    for line in lines:
        line_type = classify_line_type(line, page_width, page_height)
        separate_type = line_type in {
            "page_number",
            "date",
            "title",
            "subtitle",
            "list_item",
            "formula",
        }
        if separate_type:
            flush()
            current = [line]
            current_type = line_type
            flush()
            previous = line
            continue

        new_block = False
        if current_type not in (None, line_type):
            new_block = True
        if previous is not None:
            gap = line.bbox[1] - previous.bbox[3]
            horizontal_overlap = bbox_overlap_fraction_x(line.bbox, previous.bbox)
            indentation_change = abs(line.bbox[0] - previous.bbox[0]) / max(
                1, page_width
            )
            if (
                gap > median_height * 1.45
                or horizontal_overlap < 0.15
                and gap > median_height * 0.25
                or indentation_change > 0.16
                and gap > median_height * 0.45
            ):
                new_block = True

        if new_block:
            flush()
        if not current:
            current_type = line_type
        current.append(line)
        previous = line
    flush()
    return blocks


STRUCTURE_SYSTEM_PROMPT = """
Ты структурируешь уже распознанные OCR-строки архивного документа.

Тебе передаются строки с неизменяемыми id, текстом и координатами bbox.
Ты НЕ выполняешь OCR и НЕ переписываешь текст.
Ты должен только:
1. сгруппировать line_id в логические блоки;
2. назначить каждому блоку type и source;
3. сохранить порядок чтения.

Нельзя исправлять, дополнять, сокращать или повторно печатать текст строк.
В ответе вообще нет поля content: передавай только line_ids.
Не выдумывай line_id. Каждую содержательную строку используй не более одного раза.
Если источник неясен, используй source=mixed.
Обязательно вызови save_ocr_structure ровно один раз.
"""


def structure_with_qwen(
    lines: list[DetectionRecord],
    base_url: str,
    model_name: str,
    timeout: float,
    default_source: str,
) -> tuple[list[str], list[dict[str, Any]]] | None:
    from openai import OpenAI

    tool = {
        "type": "function",
        "function": {
            "name": "save_ocr_structure",
            "description": "Группировка OCR-строк по line_id без переписывания текста.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "properties": {
                            "languages": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "blocks": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "enum": BLOCK_TYPES,
                                        },
                                        "source": {
                                            "type": "string",
                                            "enum": SOURCE_TYPES,
                                        },
                                        "line_ids": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                        },
                                    },
                                    "required": ["type", "source", "line_ids"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["languages", "blocks"],
                        "additionalProperties": False,
                    }
                },
                "required": ["data"],
                "additionalProperties": False,
            },
        },
    }

    payload = [
        {
            "id": line.line_id,
            "text": line.text,
            "bbox": line.bbox,
            "quality_flags": line.quality_flags,
        }
        for line in lines
    ]
    client = OpenAI(base_url=base_url, api_key="lm-studio", timeout=timeout)
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        max_tokens=2500,
        messages=[
            {"role": "system", "content": STRUCTURE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Структурируй эти OCR-строки. Не переписывай их текст:\n"
                    + json.dumps(payload, ensure_ascii=False)
                ),
            },
        ],
        tools=[tool],
        tool_choice="required",
    )
    if not response.choices:
        return None
    message = response.choices[0].message
    if not message.tool_calls:
        return None
    matching = next(
        (
            call
            for call in message.tool_calls
            if call.function.name == "save_ocr_structure"
        ),
        None,
    )
    if matching is None:
        return None
    arguments = json.loads(matching.function.arguments)
    data = arguments.get("data")
    if not isinstance(data, dict):
        return None

    line_map = {line.line_id: line for line in lines if line.line_id}
    used: set[str] = set()
    blocks: list[dict[str, Any]] = []
    raw_blocks = data.get("blocks", [])
    if not isinstance(raw_blocks, list):
        return None

    for raw_block in raw_blocks:
        if not isinstance(raw_block, dict):
            continue
        block_type = raw_block.get("type", "unknown")
        source = raw_block.get("source", default_source)
        if block_type not in BLOCK_TYPES:
            block_type = "unknown"
        if source not in SOURCE_TYPES:
            source = default_source
        raw_ids = raw_block.get("line_ids", [])
        if not isinstance(raw_ids, list):
            continue
        valid_ids: list[str] = []
        for raw_id in raw_ids:
            line_id = str(raw_id)
            if line_id not in line_map or line_id in used:
                continue
            valid_ids.append(line_id)
            used.add(line_id)
        if not valid_ids:
            continue
        block_lines = [line_map[line_id] for line_id in valid_ids]
        blocks.append(
            {
                "type": block_type,
                "source": source,
                "content": "\n".join(line.text for line in block_lines),
                "line_ids": valid_ids,
                "bbox": union_bbox(line.bbox for line in block_lines),
            }
        )

    # Qwen не имеет права терять строки. Всё неприсвоенное возвращается назад.
    for line in lines:
        if line.line_id in used:
            continue
        blocks.append(
            {
                "type": classify_line_type(
                    line,
                    max(item.bbox[2] for item in lines),
                    max(item.bbox[3] for item in lines),
                ),
                "source": default_source,
                "content": line.text,
                "line_ids": [line.line_id],
                "bbox": line.bbox,
            }
        )

    order_index = {
        line.line_id: index for index, line in enumerate(lines) if line.line_id
    }
    blocks.sort(
        key=lambda block: min(
            order_index.get(line_id, 10**9) for line_id in block["line_ids"]
        )
    )
    languages = data.get("languages", ["ru"])
    if not isinstance(languages, list) or not languages:
        languages = ["ru"]
    return [str(item) for item in languages], blocks


def build_page_json(
    page_number: int,
    page_image: Image.Image,
    lines: list[DetectionRecord],
    regions: list[DetectionRecord],
    preprocessing: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    blocks = group_lines_deterministically(
        lines,
        page_image.width,
        page_image.height,
        args.default_source,
    )
    languages = ["ru"]
    structure_method = "deterministic"

    if args.structure_with_qwen:
        try:
            structured = structure_with_qwen(
                lines=lines,
                base_url=args.lmstudio_url,
                model_name=args.qwen_model,
                timeout=args.qwen_timeout,
                default_source=args.default_source,
            )
            if structured is not None:
                languages, blocks = structured
                structure_method = "qwen_line_id_grouping"
            else:
                warn(
                    f"Страница {page_number}: Qwen не вернул структуру, "
                    "использована детерминированная сборка."
                )
        except Exception as error:
            warn(
                f"Страница {page_number}: ошибка Qwen-структурирования: "
                f"{error}. Использована детерминированная сборка."
            )

    return {
        "languages": languages,
        "blocks": blocks,
        "lines": [line.to_json() for line in lines],
        "regions": [region.to_json() for region in regions],
        "page": page_number,
        "page_size": {
            "width": page_image.width,
            "height": page_image.height,
        },
        "coordinate_space": "prepared_page_after_deskew_and_outer_crop",
        "preprocessing": preprocessing,
        "models": {
            "line_detector": args.detector_repo,
            "line_recognizer": args.ocr_model,
            "structurer": args.qwen_model if args.structure_with_qwen else None,
        },
        "structure_method": structure_method,
    }


# ---------------------------------------------------------------------------
# Двухфазный pipeline
# ---------------------------------------------------------------------------


def input_page_count(input_path: Path) -> int:
    if input_path.suffix.lower() == ".pdf":
        with pymupdf.open(input_path) as document:
            return document.page_count
    return 1


def load_input_page(
    input_path: Path,
    page_index: int,
    dpi: int,
) -> Image.Image:
    if input_path.suffix.lower() == ".pdf":
        with pymupdf.open(input_path) as document:
            return render_page_image(document[page_index], dpi)
    if page_index != 0:
        raise IndexError("Для изображения доступна только первая страница.")
    return Image.open(input_path).convert("RGB")


def detect_document_pages(
    input_path: Path,
    page_indices: list[int],
    output_root: Path,
    detector: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    page_manifests: list[dict[str, Any]] = []

    for current, page_index in enumerate(page_indices, start=1):
        page_number = page_index + 1
        page_dir = output_root / f"page_{page_number:03d}"
        lines_dir = page_dir / "lines" / "raw"
        tiles_dir = page_dir / "tiles"
        page_dir.mkdir(parents=True, exist_ok=True)
        lines_dir.mkdir(parents=True, exist_ok=True)
        if args.save_debug:
            tiles_dir.mkdir(parents=True, exist_ok=True)

        log(f"[{current}/{len(page_indices)}] Детекция страницы {page_number}")
        original = load_input_page(input_path, page_index, args.dpi)
        original_size = original.size

        if args.no_deskew:
            deskewed = original.convert("RGB")
            deskew_angle = 0.0
        else:
            deskewed, deskew_angle = deskew_document(original)

        if args.no_crop:
            prepared = deskewed
            crop_offset = (0, 0)
            crop_info = {
                "applied": False,
                "reason": "disabled",
                "original_size": list(deskewed.size),
                "result_size": list(deskewed.size),
            }
        else:
            prepared, crop_offset, crop_info = safe_outer_crop(
                deskewed,
                padding_x_ratio=args.crop_padding_x,
                padding_y_ratio=args.crop_padding_y,
            )

        prepared_path = page_dir / "prepared.png"
        prepared.save(prepared_path)
        if args.save_debug:
            original.save(page_dir / "original.png")

        tiles = make_vertical_tiles(
            prepared,
            tile_count=args.tile_count,
            overlap_ratio=args.tile_overlap,
            cut_search_ratio=args.tile_cut_search,
        )
        all_lines: list[tuple[DetectionRecord, Image.Image]] = []
        all_regions: list[DetectionRecord] = []

        for tile_index, tile_top, tile_bottom, tile_image in tiles:
            log(
                f"  tile {tile_index + 1}/{len(tiles)}: "
                f"y={tile_top}:{tile_bottom}, size={tile_image.width}x{tile_image.height}"
            )
            if args.save_debug:
                tile_image.save(tiles_dir / f"tile_{tile_index + 1:02d}.png")
            lines, regions = predict_tile(
                detector=detector,
                tile=tile_image,
                tile_id=tile_index,
                tile_top=tile_top,
                detector_max_size=args.detector_max_size,
                confidence_threshold=args.detector_threshold,
                line_class_id=args.line_class_id,
                region_class_id=args.region_class_id,
                line_area_threshold=args.line_area_threshold,
                region_area_threshold=args.region_area_threshold,
                line_pad_y=args.line_pad_y,
                line_pad_x=args.line_pad_x,
            )
            all_lines.extend(lines)
            all_regions.extend(regions)

        deduplicated = deduplicate_lines(
            all_lines,
            iou_threshold=args.line_duplicate_iou,
            overlap_threshold=args.line_duplicate_overlap,
        )
        regions = deduplicate_regions(all_regions)
        assign_regions(deduplicated, regions)
        ordered = reading_order(deduplicated)

        line_records: list[DetectionRecord] = []
        for line_index, (record, crop) in enumerate(ordered, start=1):
            record.line_id = f"p{page_number:03d}_l{line_index:04d}"
            relative_crop = Path("lines") / "raw" / f"{record.line_id}.png"
            crop.save(page_dir / relative_crop)
            record.crop_path = relative_crop.as_posix()
            line_records.append(record)

        if args.save_debug:
            save_detection_overlay(
                prepared,
                line_records,
                regions,
                page_dir / "detections.png",
            )

        preprocessing = {
            "dpi": args.dpi,
            "original_size": list(original_size),
            "deskew_angle": deskew_angle,
            "outer_crop": crop_info,
            "crop_offset_after_deskew": list(crop_offset),
            "tile_count": args.tile_count,
            "tile_overlap": args.tile_overlap,
            "tile_cut_search": args.tile_cut_search,
            "detector_max_size": args.detector_max_size,
        }
        detection_manifest = {
            "page": page_number,
            "prepared_image": prepared_path.name,
            "page_size": [prepared.width, prepared.height],
            "preprocessing": preprocessing,
            "lines": [line.to_json() for line in line_records],
            "regions": [region.to_json() for region in regions],
        }
        save_json_atomic(page_dir / "detections.json", detection_manifest)
        page_manifests.append(detection_manifest)
        log(
            f"  Найдено строк: raw={len(all_lines)}, "
            f"after_dedup={len(line_records)}, regions={len(regions)}"
        )

    return page_manifests


def records_from_detection_manifest(
    manifest: dict[str, Any],
) -> tuple[list[DetectionRecord], list[DetectionRecord]]:
    lines: list[DetectionRecord] = []
    for raw in manifest.get("lines", []):
        lines.append(
            DetectionRecord(
                kind="text_line",
                bbox=[int(value) for value in raw["bbox"]],
                polygon=[[int(x), int(y)] for x, y in raw.get("polygon", [])],
                confidence=float(raw.get("detector_confidence", 0.0)),
                tile_ids=[int(value) for value in raw.get("tile_ids", [])],
                crop_path=raw.get("crop_path"),
                line_id=raw.get("id"),
                region_id=raw.get("region_id"),
            )
        )
    regions: list[DetectionRecord] = []
    for raw in manifest.get("regions", []):
        regions.append(
            DetectionRecord(
                kind="text_region",
                bbox=[int(value) for value in raw["bbox"]],
                polygon=[[int(x), int(y)] for x, y in raw.get("polygon", [])],
                confidence=float(raw.get("detector_confidence", 0.0)),
                tile_ids=[int(value) for value in raw.get("tile_ids", [])],
                region_id=raw.get("region_id"),
            )
        )
    return lines, regions


def ocr_and_assemble_pages(
    output_root: Path,
    page_manifests: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    processor, model = load_ocr_model(
        args.ocr_model,
        args.ocr_device,
        args.ocr_dtype,
    )
    pages: list[dict[str, Any]] = []
    all_text_parts: list[str] = []

    for current, manifest in enumerate(page_manifests, start=1):
        page_number = int(manifest["page"])
        page_dir = output_root / f"page_{page_number:03d}"
        prepared_image = Image.open(page_dir / manifest["prepared_image"]).convert(
            "RGB"
        )
        lines, regions = records_from_detection_manifest(manifest)
        log(
            f"[{current}/{len(page_manifests)}] OCR страницы {page_number}: {len(lines)} строк"
        )

        recognize_line_images(
            processor=processor,
            model=model,
            device=args.ocr_device,
            line_records=lines,
            page_dir=page_dir,
            batch_size=args.ocr_batch_size,
            max_new_tokens=args.max_new_tokens,
            canvas_width=args.ocr_canvas_width,
            canvas_height=args.ocr_canvas_height,
            autocontrast=not args.no_line_autocontrast,
            save_prepared_lines=args.save_debug,
        )

        page_json = build_page_json(
            page_number=page_number,
            page_image=prepared_image,
            lines=lines,
            regions=regions,
            preprocessing=manifest["preprocessing"],
            args=args,
        )
        save_json_atomic(page_dir / f"ocr_json_{page_number:03d}.json", page_json)
        page_text = "\n".join(line.text for line in lines)
        (page_dir / f"page_{page_number:03d}.txt").write_text(
            page_text + "\n",
            encoding="utf-8",
        )
        pages.append(page_json)
        all_text_parts.append(f"=== PAGE {page_number} ===\n{page_text}")

    (output_root / "all_text.txt").write_text(
        "\n\n".join(all_text_parts) + "\n",
        encoding="utf-8",
    )
    return pages


def unload_detector(detector: Any) -> None:
    del detector
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def package_result(output_root: Path, archive_path: Path) -> Path:
    archive_path = archive_path.expanduser().resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    base_name = archive_path
    if archive_path.suffix.lower() == ".zip":
        base_name = archive_path.with_suffix("")
    result = shutil.make_archive(
        str(base_name),
        "zip",
        root_dir=output_root.parent,
        base_dir=output_root.name,
    )
    return Path(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("PDF -> RF-DETR line detection -> OvisOCR2 -> structured JSON")
    )
    parser.add_argument("input", help="PDF или изображение страницы")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Каталог результата. По умолчанию ./htr_output/<имя файла>",
    )
    parser.add_argument("--pages", default=None, help="Например: 1-10 или 1,3,5-7")
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--archive", default=None, help="Опциональный путь к ZIP")

    parser.add_argument("--no-deskew", action="store_true")
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--crop-padding-x", type=float, default=0.04)
    parser.add_argument("--crop-padding-y", type=float, default=0.02)
    parser.add_argument("--tile-count", type=int, default=4)
    parser.add_argument("--tile-overlap", type=float, default=0.15)
    parser.add_argument("--tile-cut-search", type=float, default=0.25)

    parser.add_argument("--detector-repo", default=DEFAULT_DETECTOR_REPO)
    parser.add_argument("--detector-filename", default=DEFAULT_DETECTOR_FILENAME)
    parser.add_argument("--detector-weights", default=None)
    parser.add_argument("--detector-max-size", type=int, default=768)
    parser.add_argument("--detector-threshold", type=float, default=0.15)
    parser.add_argument("--detector-cpu", action="store_true")
    parser.add_argument("--line-class-id", type=int, default=2)
    parser.add_argument("--region-class-id", type=int, default=1)
    parser.add_argument("--line-area-threshold", type=float, default=7e-05)
    parser.add_argument("--region-area-threshold", type=float, default=7e-05)
    parser.add_argument("--line-duplicate-iou", type=float, default=0.38)
    parser.add_argument("--line-duplicate-overlap", type=float, default=0.72)
    parser.add_argument("--line-pad-y", type=float, default=0.30)
    parser.add_argument("--line-pad-x", type=float, default=0.018)

    parser.add_argument(
        "--ocr-model",
        default=DEFAULT_OCR_MODEL,
        help=(
            "OvisOCR2-compatible checkpoint в формате Transformers. "
            f"По умолчанию: {DEFAULT_OCR_MODEL}"
        ),
    )
    parser.add_argument(
        "--ocr-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--ocr-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument("--ocr-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--ocr-canvas-width", type=int, default=1024)
    parser.add_argument("--ocr-canvas-height", type=int, default=256)
    parser.add_argument("--no-line-autocontrast", action="store_true")

    parser.add_argument("--default-source", choices=SOURCE_TYPES, default="mixed")
    parser.add_argument("--structure-with-qwen", action="store_true")
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--lmstudio-url", default=DEFAULT_LMSTUDIO_URL)
    parser.add_argument("--qwen-timeout", type=float, default=180.0)

    parser.add_argument(
        "--skip-detection",
        action="store_true",
        help="Использовать уже сохранённые detections.json и crops.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.dpi < 72:
        raise ValueError("--dpi должен быть не меньше 72.")
    if args.tile_count < 1:
        raise ValueError("--tile-count должен быть не меньше 1.")
    if not 0 <= args.tile_overlap < 0.45:
        raise ValueError("--tile-overlap должен быть в диапазоне [0, 0.45).")
    if not 0 <= args.tile_cut_search < 0.45:
        raise ValueError("--tile-cut-search должен быть в диапазоне [0, 0.45).")
    if args.ocr_batch_size < 1:
        raise ValueError("--ocr-batch-size должен быть не меньше 1.")
    if args.default_source not in SOURCE_TYPES:
        raise ValueError("Некорректный --default-source.")


def load_saved_manifests(
    output_root: Path,
    page_indices: list[int],
) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for page_index in page_indices:
        page_number = page_index + 1
        path = output_root / f"page_{page_number:03d}" / "detections.json"
        if not path.exists():
            raise FileNotFoundError(f"Для --skip-detection не найден файл: {path}")
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    return manifests


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Не найден входной файл: {input_path}")

    output_root = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path.cwd() / "htr_output" / input_path.stem).resolve()
    )
    if output_root.exists() and args.overwrite and not args.skip_detection:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    page_count = input_page_count(input_path)
    page_indices = parse_page_spec(args.pages, page_count)
    log(f"Вход: {input_path}")
    log(f"Страниц: {page_count}; обрабатывается: {[i + 1 for i in page_indices]}")
    log(f"Результат: {output_root}")

    started = time.time()
    if args.skip_detection:
        page_manifests = load_saved_manifests(output_root, page_indices)
    else:
        weights = resolve_detector_weights(args)
        detector = load_detector(weights, cpu=args.detector_cpu)
        page_manifests = detect_document_pages(
            input_path=input_path,
            page_indices=page_indices,
            output_root=output_root,
            detector=detector,
            args=args,
        )
        unload_detector(detector)

    pages = ocr_and_assemble_pages(
        output_root=output_root,
        page_manifests=page_manifests,
        args=args,
    )

    document_json = {
        "source_file": str(input_path),
        "page_count_in_source": page_count,
        "processed_pages": [index + 1 for index in page_indices],
        "pages": pages,
        "models": {
            "line_detector": args.detector_repo,
            "line_recognizer": args.ocr_model,
            "structurer": args.qwen_model if args.structure_with_qwen else None,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    save_json_atomic(output_root / "document.json", document_json)

    if args.archive:
        archive = package_result(output_root, Path(args.archive))
        log(f"Архив: {archive}")

    log(f"Готово за {time.time() - started:.1f} с.")
    log(f"Главный JSON: {output_root / 'document.json'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as error:
        print(f"[FATAL] {type(error).__name__}: {error}", file=sys.stderr)
        raise

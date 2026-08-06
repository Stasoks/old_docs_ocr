#!/usr/bin/env python3
"""
Hybrid HTR pipeline for historical documents and ZIP collections.

RF-DETR detects text lines and text regions. Large, sufficiently rectangular
regions are treated as possible tables, deduplicated, and sent once to a VLM
served by LM Studio. The VLM must classify every candidate as either `table`
or `text` and transcribe it. Lines outside successfully recognized VLM regions
are read by Kansallisarkisto/cyrillic-large-handwritten (TrOCR). Both streams
are then merged by page coordinates into one JSON document.

After OCR, the pipeline writes a content-only JSON without geometry and sends
that compact document to Qwen3.6-27B through LM Studio's OpenAI-compatible API.
The resulting well parameters are constrained by a JSON Schema and checked
against literal evidence on the cited page before they are saved.

The RF-DETR and TrOCR instances are cached in one Python process and remain on
CUDA until it exits. With --keep-alive, additional PDF/image/ZIP paths can be
entered without reloading either model. LM Studio receives a long TTL with
each request so Qwen remains loaded on its side as well.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw

import historical_russian_htr_pipeline as htr
from strip_ocr_for_llm import clean_document

DEFAULT_OCR_MODEL = "Kansallisarkisto/cyrillic-large-handwritten"
DEFAULT_QWEN_MODEL = "qwen/qwen3-vl-4b"
DEFAULT_EXTRACTION_MODEL = "qwen/qwen3.6-27b"
DEFAULT_LMSTUDIO_URL = "http://localhost:1234/v1"
DEFAULT_EXTRACTION_PROMPT = (
    Path(__file__).resolve().with_name("well_extraction_prompt.txt")
)
MAX_ARCHIVE_MEMBERS = 10_000
MAX_ARCHIVE_MEMBER_BYTES = 2 * 1024**3
MAX_ARCHIVE_TOTAL_BYTES = 20 * 1024**3

CANONICAL_WELL_PARAMETERS = (
    "Номер скважины",
    "Расположение скважины",
    "Дата начала бурения",
    "Дата завершения бурения",
    "Статус скважины",
    "Проектная глубина",
    "Фактическая глубина (забой)",
    "Максимальный угол отклонения",
    "Проектное назначение",
)

VLM_SYSTEM_PROMPT = """
Ты получаешь прямоугольный фрагмент исторического документа. Скрипт
предположил, что фрагмент может быть таблицей, но это лишь геометрическая
эвристика.

Твоя задача:
1. Выбрать type="table", если видны строки/столбцы, ячейки, табличная сетка или
   форма с регулярными полями. В остальных случаях выбрать type="text".
2. Дословно распознать всё видимое. Не переводить, не исправлять и не дополнять.
3. Для table передать rows как двумерный массив. Для text rows можно не передавать.
4. Вызвать save_region_ocr ровно один раз. Не отвечать обычным текстом.
""".strip()

TABLE_LOCALIZATION_PROMPT = """
Ты получаешь полную страницу исторического документа. Найди ВСЕ таблицы,
ведомости, журнальные сетки и регулярные формы со строками и колонками.

Для каждой таблицы верни один bbox [x1, y1, x2, y2] в системе
normalized_1000: левый верх изображения — [0, 0], правый низ — [1000, 1000].
Включай заголовки колонок, левую и правую половины и последние строки таблицы.
Не включай номер страницы, поля, внешние пометки и обычные абзацы.
На этом шаге не распознавай содержимое таблиц.

Вызови locate_tables ровно один раз. Если таблиц нет, передай tables=[].
""".strip()

STRICT_TABLE_OCR_PROMPT = """
Ты получаешь вырезанное изображение одной полной таблицы.
Верни type="table" и rows, сохранив все строки сверху вниз и колонки слева
направо. Не объединяй строки, не исправляй числа, не добавляй значения.
Пустые ячейки сохраняй как "", неясные — как "[НЕРАЗБОРЧИВО]".
Вызови save_table_ocr ровно один раз.
""".strip()

SUSPICIOUS_LINE_OCR_PROMPT = """
Ты получаешь crop одной строки исторического русского документа. Предыдущий
OCR посчитал строку сомнительной. Независимо и дословно распознай только то,
что видно на изображении: не переводи, не исправляй по смыслу и не дополняй.
Если строка действительно не читается, верни [НЕРАЗБОРЧИВО].
Вызови save_line_ocr ровно один раз.
""".strip()

SUSPICIOUS_OCR_FLAGS = {
    "empty_output",
    "repetition_truncated",
    "repetition_loop",
    "suspicious_latin_output",
    "very_long_line",
}

REGION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "save_region_ocr",
        "description": "Классификация и OCR одного кандидата в таблицу.",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["table", "text"]},
                        "source": {
                            "type": "string",
                            "enum": ["printed", "handwritten", "mixed"],
                        },
                        "languages": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "content": {"type": "string"},
                        "rows": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                    "required": ["type", "source", "languages", "content"],
                    "additionalProperties": False,
                }
            },
            "required": ["data"],
            "additionalProperties": False,
        },
    },
}

TABLE_LOCALIZATION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "locate_tables",
        "description": "Координаты всех таблиц на полной странице.",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "coordinate_space": {
                            "type": "string",
                            "enum": ["normalized_1000"],
                        },
                        "tables": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "bbox": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "minItems": 4,
                                        "maxItems": 4,
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                    },
                                },
                                "required": ["bbox", "confidence"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["coordinate_space", "tables"],
                    "additionalProperties": False,
                }
            },
            "required": ["data"],
            "additionalProperties": False,
        },
    },
}

TABLE_OCR_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "save_table_ocr",
        "description": "Строгий OCR одной полной таблицы.",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["table"]},
                        "rows": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    },
                    "required": ["type", "rows"],
                    "additionalProperties": False,
                }
            },
            "required": ["data"],
            "additionalProperties": False,
        },
    },
}

LINE_OCR_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "save_line_ocr",
        "description": "Повторный OCR одной сомнительной строки.",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                }
            },
            "required": ["data"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class TableCandidate:
    region_id: str
    bbox: list[int]
    detector_confidence: float
    area_ratio: float
    aspect_ratio: float
    squareness: float
    score: float
    line_ids: list[str]
    candidate_id: str = ""
    crop_bbox: list[int] = field(default_factory=list)
    crop_path: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    localization_source: str = "rfdetr_bbox_heuristic"

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.candidate_id,
            "detector_region_id": self.region_id,
            "bbox": self.bbox,
            "crop_bbox": self.crop_bbox or self.bbox,
            "crop_path": self.crop_path,
            "detector_confidence": round(self.detector_confidence, 6),
            "area_ratio": round(self.area_ratio, 6),
            "aspect_ratio": round(self.aspect_ratio, 6),
            "squareness": round(self.squareness, 6),
            "candidate_score": round(self.score, 6),
            "covered_line_ids": self.line_ids,
            "localization_source": self.localization_source,
            "vlm_result": self.result,
            "error": self.error,
        }


class CudaModelPool:
    """Lazily loads local models once and never moves them off CUDA."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.detector: Any | None = None
        self.ocr_processor: Any | None = None
        self.ocr_model: Any | None = None
        self.ocr_dtype: Any | None = None
        self.qwen_client: Any | None = None

    def get_detector(self) -> Any:
        if self.detector is None:
            weights = htr.resolve_detector_weights(self.args)
            self.detector = htr.load_detector(
                weights,
                cpu=self.args.detector_cpu,
            )
        return self.detector

    def get_trocr(self) -> tuple[Any, Any, Any]:
        if self.ocr_model is None:
            (
                self.ocr_processor,
                self.ocr_model,
                self.ocr_dtype,
            ) = load_trocr_model(
                self.args.ocr_model,
                self.args.ocr_device,
                self.args.ocr_dtype,
            )
        return self.ocr_processor, self.ocr_model, self.ocr_dtype

    def get_qwen_client(self) -> Any:
        if self.qwen_client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError(
                    "Для Qwen нужен openai: pip install 'openai>=1.50.0'"
                ) from error
            self.qwen_client = OpenAI(
                base_url=self.args.lmstudio_url,
                api_key="lm-studio",
                timeout=self.args.qwen_timeout,
                max_retries=0,
            )
        return self.qwen_client


def load_trocr_model(
    model_name: str,
    device: str,
    dtype_name: str,
) -> tuple[Any, Any, Any]:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    class TrOCRProcessorCustom(TrOCRProcessor):
        def __init__(self, image_processor: Any, tokenizer: Any):
            self.image_processor = image_processor
            self.tokenizer = tokenizer
            self.current_processor = self.image_processor
            self.chat_template = None

    resolved_device = htr.resolve_torch_device(device)
    if dtype_name == "float32":
        dtype = torch.float32
    elif dtype_name == "float16":
        dtype = torch.float16
    elif dtype_name == "bfloat16":
        dtype = torch.bfloat16
    else:
        # This is the original TrOCR pipeline default. OvisOCR2 uses BF16,
        # while this recognizer was previously run in FP16 on CUDA.
        dtype = torch.float16 if resolved_device == "cuda" else torch.float32
    htr.log(f"Загрузка TrOCR {model_name} на {resolved_device}, dtype={dtype}")
    processor = TrOCRProcessorCustom.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name, dtype=dtype)
    model.to(resolved_device)
    model.eval()
    if resolved_device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    return processor, model, dtype


def recognize_trocr_lines(
    processor: Any,
    model: Any,
    dtype: Any,
    device: str,
    line_records: list[htr.DetectionRecord],
    page_dir: Path,
    batch_size: int,
    max_new_tokens: int,
    canvas_width: int,
    canvas_height: int,
    autocontrast: bool,
    save_prepared_lines: bool,
) -> None:
    import torch

    resolved_device = htr.resolve_torch_device(device)
    for batch_number, batch in enumerate(
        htr.batched(line_records, batch_size),
        start=1,
    ):
        images: list[Image.Image] = []
        for line in batch:
            if not line.crop_path:
                raise RuntimeError(f"У строки {line.line_id} отсутствует crop_path.")
            with Image.open(page_dir / line.crop_path) as source:
                prepared = htr.fit_line_to_canvas(
                    source.convert("RGB"),
                    width=canvas_width,
                    height=canvas_height,
                    autocontrast=autocontrast,
                )
            if save_prepared_lines:
                prepared_dir = page_dir / "lines" / "prepared_trocr"
                prepared_dir.mkdir(parents=True, exist_ok=True)
                prepared.save(prepared_dir / f"{line.line_id}.png")
            images.append(prepared)

        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(
            device=resolved_device,
            dtype=dtype,
        )
        with torch.inference_mode():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                do_sample=False,
            )
        texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        for line, raw_text in zip(batch, texts):
            cleaned, flags = htr.inspect_ocr_text(raw_text)
            line.raw_text = re.sub(r"\s+", " ", raw_text).strip()
            line.text = cleaned
            line.quality_flags = flags
        htr.log(f"  TrOCR batch {batch_number}: {len(batch)} строк")


def _intersection_over_smaller(
    first: Sequence[int],
    second: Sequence[int],
) -> float:
    return htr.bbox_intersection_over_smaller(first, second)


def _line_belongs_to_region(
    line: htr.DetectionRecord,
    bbox: Sequence[int],
    overlap_threshold: float,
) -> bool:
    center_x, center_y = line.center
    center_inside = bbox[0] <= center_x <= bbox[2] and bbox[1] <= center_y <= bbox[3]
    return center_inside or (
        _intersection_over_smaller(line.bbox, bbox) >= overlap_threshold
    )


def select_table_candidates(
    lines: list[htr.DetectionRecord],
    regions: list[htr.DetectionRecord],
    page_width: int,
    page_height: int,
    args: argparse.Namespace,
) -> list[TableCandidate]:
    page_area = max(1, page_width * page_height)
    raw_candidates: list[TableCandidate] = []

    for region in regions:
        width = region.width
        height = region.height
        if width <= 0 or height <= 0:
            continue
        area_ratio = region.area / page_area
        aspect_ratio = width / height
        squareness = min(width, height) / max(width, height)
        crop_bbox = _expanded_bbox(
            region.bbox,
            page_width,
            page_height,
            args.table_crop_padding,
        )
        # Account for the exact crop sent to the VLM. A line caught by the
        # padding band must not be recognized a second time by TrOCR.
        covered_lines = [
            line
            for line in lines
            if _line_belongs_to_region(
                line,
                crop_bbox,
                args.table_line_overlap,
            )
        ]
        line_ids = [
            str(line.line_id) for line in covered_lines if line.line_id is not None
        ]

        if not args.table_min_area_ratio <= area_ratio <= args.table_max_area_ratio:
            continue
        if squareness < args.table_min_squareness:
            continue
        if len(line_ids) < args.table_min_lines:
            continue

        score = (
            squareness * 2.0
            + min(1.0, area_ratio / max(args.table_min_area_ratio * 3, 1e-6))
            + min(1.0, len(line_ids) / 8)
            + region.confidence * 0.25
        )
        raw_candidates.append(
            TableCandidate(
                region_id=region.region_id or "unknown",
                bbox=list(region.bbox),
                detector_confidence=region.confidence,
                area_ratio=area_ratio,
                aspect_ratio=aspect_ratio,
                squareness=squareness,
                score=score,
                line_ids=line_ids,
                crop_bbox=crop_bbox,
            )
        )

    selected: list[TableCandidate] = []
    for candidate in sorted(raw_candidates, key=lambda item: item.score, reverse=True):
        candidate_lines = set(candidate.line_ids)
        duplicate = False
        for existing in selected:
            geometry_overlap = _intersection_over_smaller(
                candidate.crop_bbox,
                existing.crop_bbox,
            )
            shared_lines = candidate_lines.intersection(existing.line_ids)
            if geometry_overlap >= args.table_candidate_overlap or bool(shared_lines):
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)

    selected.sort(key=lambda item: (item.bbox[1], item.bbox[0]))
    for index, candidate in enumerate(selected, start=1):
        candidate.candidate_id = f"table_candidate_{index:03d}"
    return selected


def _expanded_bbox(
    bbox: Sequence[int],
    page_width: int,
    page_height: int,
    padding_ratio: float,
) -> list[int]:
    padding_x = max(4, int(page_width * padding_ratio))
    padding_y = max(4, int(page_height * padding_ratio))
    return [
        max(0, int(bbox[0]) - padding_x),
        max(0, int(bbox[1]) - padding_y),
        min(page_width, int(bbox[2]) + padding_x),
        min(page_height, int(bbox[3]) + padding_y),
    ]


def image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _json_from_text(content: Any) -> dict[str, Any] | None:
    text = str(content or "").strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    tool_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    candidates = [tool_match.group(1)] if tool_match else []
    candidates.append(text)
    first, last = text.find("{"), text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first : last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def clean_document_for_extraction(
    document: dict[str, Any],
) -> dict[str, Any]:
    """Keep only file/page plus block type/content for the final LLM."""

    return clean_document(document)


def load_well_extraction_prompt(path: Path) -> tuple[str, str]:
    text = path.expanduser().resolve().read_text(encoding="utf-8").strip()
    system_marker = "[SYSTEM]"
    user_marker = "[USER]"
    if system_marker not in text and user_marker not in text:
        if not text:
            raise ValueError("Промпт экстракции пуст.")
        return (
            text,
            "Извлеки параметры скважины из OCR JSON по системным правилам. "
            "Сначала проанализируй весь документ, затем верни только JSON.",
        )
    if system_marker not in text or user_marker not in text:
        raise ValueError(
            "В промпте должны быть либо обе секции [SYSTEM]/[USER], либо ни одной."
        )
    system_start = text.index(system_marker) + len(system_marker)
    user_start = text.index(user_marker, system_start)
    system_prompt = text[system_start:user_start].strip()
    user_prompt = text[user_start + len(user_marker) :].strip()
    if not system_prompt or not user_prompt:
        raise ValueError("Секции [SYSTEM]/[USER] в промпте не могут быть пустыми.")
    return system_prompt, user_prompt


def well_extraction_response_format() -> dict[str, Any]:
    parameters = list(CANONICAL_WELL_PARAMETERS)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "well_parameter_extraction",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "records": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "row_number": {"type": "integer", "minimum": 1},
                                "parameter": {"type": "string", "enum": parameters},
                                "value": {"type": "string", "minLength": 1},
                                "raw_value": {"type": "string"},
                                "file": {
                                    "anyOf": [
                                        {"type": "string"},
                                        {"type": "null"},
                                    ]
                                },
                                "page": {"type": "integer", "minimum": 1},
                                "evidence": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 300,
                                },
                                "confidence": {
                                    "type": "string",
                                    "enum": ["high", "medium", "low"],
                                },
                                "notes": {"type": "string"},
                            },
                            "required": [
                                "row_number",
                                "parameter",
                                "value",
                                "raw_value",
                                "file",
                                "page",
                                "evidence",
                                "confidence",
                                "notes",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "conflicts": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "parameter": {"type": "string", "enum": parameters},
                                "conflict_type": {
                                    "type": "string",
                                    "enum": [
                                        "ocr_error",
                                        "possible_document_inconsistency",
                                        "possible_rounding",
                                        "technical_stage_difference",
                                        "temporal_change",
                                        "unresolved",
                                    ],
                                },
                                "raw_value": {"type": "string"},
                                "file": {
                                    "anyOf": [
                                        {"type": "string"},
                                        {"type": "null"},
                                    ]
                                },
                                "page": {"type": "integer", "minimum": 1},
                                "evidence": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 300,
                                },
                                "reason": {"type": "string", "minLength": 1},
                            },
                            "required": [
                                "parameter",
                                "conflict_type",
                                "raw_value",
                                "file",
                                "page",
                                "evidence",
                                "reason",
                            ],
                            "additionalProperties": False,
                        },
                    },
                    "missing_parameters": {
                        "type": "array",
                        "items": {"type": "string", "enum": parameters},
                    },
                    "warnings": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "records",
                    "conflicts",
                    "missing_parameters",
                    "warnings",
                ],
                "additionalProperties": False,
            },
        },
    }


def _normalized_literal(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


SourcePageKey = tuple[str, int]


def _clean_file_name(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    return text.rsplit("/", 1)[-1] if text else ""


def _page_search_texts(
    cleaned_document: dict[str, Any],
) -> dict[SourcePageKey, list[str]]:
    fragments_by_page: dict[SourcePageKey, list[str]] = {}
    for page in cleaned_document.get("pages", []):
        if not isinstance(page, dict):
            continue
        try:
            page_number = int(page.get("page"))
        except (TypeError, ValueError):
            continue
        source_file = _clean_file_name(page.get("file"))
        page_key = (source_file, page_number)
        fragments: list[str] = []
        for block in page.get("blocks", []):
            if not isinstance(block, dict):
                continue
            content = str(block.get("content", "")).strip()
            if content:
                fragments.append(content)
        fragments_by_page.setdefault(page_key, []).extend(fragments)

    search_texts: dict[SourcePageKey, list[str]] = {}
    for page_key, fragments in fragments_by_page.items():
        whole_page = "\n".join(fragments)
        variants = [whole_page, whole_page.replace(" | ", " ")]
        search_texts[page_key] = [_normalized_literal(variant) for variant in variants]
    return search_texts


def _literal_exists_on_page(
    value: str,
    page_key: SourcePageKey,
    search_texts: dict[SourcePageKey, list[str]],
) -> bool:
    needle = _normalized_literal(value)
    return bool(needle) and any(
        needle in haystack for haystack in search_texts.get(page_key, [])
    )


def _resolve_evidence_page(
    source_file: Any,
    page_number: int,
    evidence: str,
    search_texts: dict[SourcePageKey, list[str]],
) -> SourcePageKey | None:
    candidates = [key for key in search_texts if key[1] == page_number]
    requested_file = _clean_file_name(source_file)
    if requested_file:
        file_candidates = [key for key in candidates if key[0] == requested_file]
        if file_candidates:
            candidates = file_candidates
        elif len(candidates) != 1:
            return None
    matching = [
        key
        for key in candidates
        if _literal_exists_on_page(evidence, key, search_texts)
    ]
    return matching[0] if len(matching) == 1 else None


def sanitize_well_extraction(
    raw_result: dict[str, Any],
    cleaned_document: dict[str, Any],
) -> dict[str, Any]:
    """Apply deterministic checks after the model's structured response."""

    for field_name in ("records", "conflicts", "missing_parameters", "warnings"):
        if not isinstance(raw_result.get(field_name), list):
            raise ValueError(f"Qwen вернул некорректное поле {field_name}.")

    valid_parameters = set(CANONICAL_WELL_PARAMETERS)
    search_texts = _page_search_texts(cleaned_document)
    local_warnings: list[str] = []
    records: list[dict[str, Any]] = []

    for index, raw_record in enumerate(raw_result["records"], start=1):
        if not isinstance(raw_record, dict):
            local_warnings.append(f"Удалена невалидная запись records[{index}].")
            continue
        parameter = str(raw_record.get("parameter") or "").strip()
        value = str(raw_record.get("value") or "").strip()
        raw_value = str(raw_record.get("raw_value") or "").strip()
        evidence = str(raw_record.get("evidence") or "").strip()
        confidence = str(raw_record.get("confidence") or "").strip().lower()
        notes = str(raw_record.get("notes") or "").strip()
        try:
            page_number = int(raw_record.get("page"))
        except (TypeError, ValueError):
            page_number = -1
        if parameter not in valid_parameters or not value or not raw_value:
            local_warnings.append(
                f"Удалена неполная запись records[{index}] ({parameter or 'без параметра'})."
            )
            continue
        if confidence not in {"high", "medium", "low"}:
            local_warnings.append(
                f"Удалена запись records[{index}] с некорректным confidence."
            )
            continue
        page_key = _resolve_evidence_page(
            raw_record.get("file"),
            page_number,
            evidence,
            search_texts,
        )
        if len(evidence) > 300 or page_key is None:
            local_warnings.append(
                f"Удалена запись records[{index}]: evidence не найдена "
                f"однозначно для файла/страницы {page_number}."
            )
            continue
        if _normalized_literal(raw_value) not in _normalized_literal(evidence):
            local_warnings.append(
                f"Удалена запись records[{index}]: raw_value отсутствует в evidence."
            )
            continue
        records.append(
            {
                "row_number": len(records) + 1,
                "parameter": parameter,
                "value": value,
                "raw_value": raw_value,
                "file": page_key[0] or None,
                "page": page_number,
                "evidence": evidence,
                "confidence": confidence,
                "notes": notes,
            }
        )

    conflicts: list[dict[str, Any]] = []
    valid_conflict_types = {
        "ocr_error",
        "possible_document_inconsistency",
        "possible_rounding",
        "technical_stage_difference",
        "temporal_change",
        "unresolved",
    }
    for index, raw_conflict in enumerate(raw_result["conflicts"], start=1):
        if not isinstance(raw_conflict, dict):
            local_warnings.append(f"Удалён невалидный conflicts[{index}].")
            continue
        parameter = str(raw_conflict.get("parameter") or "").strip()
        conflict_type = str(raw_conflict.get("conflict_type") or "").strip()
        raw_value = str(raw_conflict.get("raw_value") or "").strip()
        evidence = str(raw_conflict.get("evidence") or "").strip()
        reason = str(raw_conflict.get("reason") or "").strip()
        try:
            page_number = int(raw_conflict.get("page"))
        except (TypeError, ValueError):
            page_number = -1
        page_key = _resolve_evidence_page(
            raw_conflict.get("file"),
            page_number,
            evidence,
            search_texts,
        )
        if (
            parameter not in valid_parameters
            or conflict_type not in valid_conflict_types
            or not raw_value
            or not reason
            or len(evidence) > 300
            or page_key is None
            or _normalized_literal(raw_value) not in _normalized_literal(evidence)
        ):
            local_warnings.append(
                f"Удалён conflicts[{index}]: поля или evidence не прошли проверку."
            )
            continue
        conflicts.append(
            {
                "parameter": parameter,
                "conflict_type": conflict_type,
                "raw_value": raw_value,
                "file": page_key[0] or None,
                "page": page_number,
                "evidence": evidence,
                "reason": reason,
            }
        )

    present_parameters = {record["parameter"] for record in records}
    missing_parameters = [
        parameter
        for parameter in CANONICAL_WELL_PARAMETERS
        if parameter not in present_parameters
    ]
    model_warnings = [
        str(item or "").strip()
        for item in raw_result["warnings"]
        if str(item or "").strip()
    ]
    return {
        "records": records,
        "conflicts": conflicts,
        "missing_parameters": missing_parameters,
        "warnings": list(dict.fromkeys([*model_warnings, *local_warnings])),
    }


def extract_well_data_with_qwen(
    cleaned_document: dict[str, Any],
    pool: "CudaModelPool",
    args: argparse.Namespace,
) -> dict[str, Any]:
    system_prompt, user_prompt = load_well_extraction_prompt(args.extraction_prompt)
    compact_document = json.dumps(
        cleaned_document,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    client = pool.get_qwen_client()
    last_error: Exception | None = None
    for attempt in range(1, args.qwen_retries + 2):
        try:
            htr.log(
                f"Извлечение параметров: {args.extraction_model}, "
                f"попытка {attempt}/{args.qwen_retries + 1}"
            )
            response = client.chat.completions.create(
                model=args.extraction_model,
                temperature=0,
                max_tokens=args.extraction_max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"{user_prompt}\n\n"
                            "<OCR_JSON>\n"
                            f"{compact_document}\n"
                            "</OCR_JSON>"
                        ),
                    },
                ],
                response_format=well_extraction_response_format(),
                timeout=args.extraction_timeout,
                extra_body={"ttl": args.qwen_ttl},
            )
            if not response.choices:
                raise RuntimeError("LM Studio вернул пустой choices.")
            parsed = _json_from_text(response.choices[0].message.content)
            if parsed is None:
                raise RuntimeError("Qwen3.6-27B не вернул валидный JSON.")
            return sanitize_well_extraction(parsed, cleaned_document)
        except KeyboardInterrupt:
            raise
        except Exception as error:
            last_error = error
            htr.warn(
                f"Document extraction attempt "
                f"{attempt}/{args.qwen_retries + 1}: {error}"
            )
            if attempt <= args.qwen_retries:
                time.sleep(args.qwen_retry_delay)
    raise RuntimeError(
        f"Не удалось получить итоговый JSON из {args.extraction_model}: {last_error}"
    )


def _unwrap_tool_payload(
    value: dict[str, Any],
    tool_name: str,
) -> dict[str, Any]:
    if value.get("name") == tool_name and isinstance(value.get("arguments"), dict):
        value = value["arguments"]
    arguments = value.get("arguments")
    if isinstance(arguments, str):
        parsed = _json_from_text(arguments)
        if parsed is not None:
            value = parsed
    elif isinstance(arguments, dict):
        value = arguments
    data = value.get("data", value)
    if not isinstance(data, dict):
        raise TypeError(f"{tool_name} data должен быть объектом.")
    return data


def _extract_tool_data(response: Any, tool_name: str) -> dict[str, Any]:
    if not response.choices:
        raise RuntimeError("LM Studio вернул пустой choices.")
    message = response.choices[0].message
    tool_calls = list(message.tool_calls or [])
    preferred = [
        call for call in tool_calls if call.function.name == tool_name
    ] or tool_calls
    for call in preferred:
        parsed = _json_from_text(call.function.arguments)
        if parsed is not None:
            return _unwrap_tool_payload(parsed, tool_name)
    parsed_content = _json_from_text(message.content)
    if parsed_content is not None:
        return _unwrap_tool_payload(parsed_content, tool_name)
    raise RuntimeError(f"Модель не вернула JSON для {tool_name}.")


def _extract_region_data(response: Any) -> dict[str, Any]:
    return _extract_tool_data(response, "save_region_ocr")


def _sanitize_region_data(data: dict[str, Any]) -> dict[str, Any]:
    block_type = str(data.get("type", "text")).strip().lower()
    if block_type not in {"table", "text"}:
        block_type = "text"
    source = str(data.get("source", "mixed")).strip().lower()
    if source not in htr.SOURCE_TYPES:
        source = "mixed"

    languages = data.get("languages", ["ru"])
    if isinstance(languages, str):
        languages = [languages]
    if not isinstance(languages, list):
        languages = ["ru"]
    languages = [str(item).strip() for item in languages if str(item).strip()]

    rows: list[list[str]] = []
    raw_rows = data.get("rows")
    if isinstance(raw_rows, list):
        for raw_row in raw_rows:
            if isinstance(raw_row, list):
                rows.append([str(cell).strip() for cell in raw_row])
            else:
                rows.append([str(raw_row).strip()])

    content = data.get("content", "")
    if isinstance(content, list):
        content = "\n".join(str(item) for item in content)
    elif not isinstance(content, str):
        content = str(content)
    content = content.strip()
    if not content and rows:
        content = "\n".join(" | ".join(row) for row in rows)
    if not content:
        raise ValueError("Qwen вернул пустой content.")

    result: dict[str, Any] = {
        "type": block_type,
        "source": source,
        "languages": languages or ["ru"],
        "content": content,
    }
    if block_type == "table" and rows:
        result["rows"] = rows
    return result


def recognize_candidate_with_qwen(
    client: Any,
    image: Image.Image,
    candidate: TableCandidate,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if candidate.localization_source == "qwen_full_page":
        response = client.chat.completions.create(
            model=args.qwen_model,
            temperature=0,
            max_tokens=args.qwen_max_tokens,
            messages=[
                {"role": "system", "content": STRICT_TABLE_OCR_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Таблица {candidate.candidate_id}, "
                                f"bbox={candidate.bbox}. Распознай все ячейки."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image_to_data_url(image)},
                        },
                    ],
                },
            ],
            tools=[TABLE_OCR_TOOL],
            tool_choice="required",
            timeout=args.qwen_timeout,
            extra_body={"ttl": args.qwen_ttl},
        )
        return _sanitize_region_data(_extract_tool_data(response, "save_table_ocr"))

    response = client.chat.completions.create(
        model=args.qwen_model,
        temperature=0,
        max_tokens=args.qwen_max_tokens,
        messages=[
            {"role": "system", "content": VLM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Кандидат {candidate.candidate_id}, bbox={candidate.bbox}. "
                            "Определи, table это или text, и распознай всё без потерь."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image)},
                    },
                ],
            },
        ],
        tools=[REGION_TOOL],
        tool_choice="required",
        timeout=args.qwen_timeout,
        extra_body={"ttl": args.qwen_ttl},
    )
    return _sanitize_region_data(_extract_region_data(response))


def _normalized_bbox_to_pixels(
    bbox: Sequence[Any],
    page_width: int,
    page_height: int,
) -> list[int] | None:
    if len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((max(0.0, min(1000.0, x1)), max(0.0, min(1000.0, x2))))
    y1, y2 = sorted((max(0.0, min(1000.0, y1)), max(0.0, min(1000.0, y2))))
    result = [
        round(x1 * page_width / 1000),
        round(y1 * page_height / 1000),
        round(x2 * page_width / 1000),
        round(y2 * page_height / 1000),
    ]
    width, height = result[2] - result[0], result[3] - result[1]
    if width < 8 or height < 8:
        return None
    if width * height / max(1, page_width * page_height) < 0.002:
        return None
    return result


def _deduplicate_localized_tables(
    tables: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for table in sorted(tables, key=lambda item: item["confidence"], reverse=True):
        duplicate = next(
            (
                existing
                for existing in selected
                if _intersection_over_smaller(
                    table["pixel_bbox"], existing["pixel_bbox"]
                )
                >= 0.70
            ),
            None,
        )
        if duplicate is None:
            selected.append(table)
            continue
        duplicate["pixel_bbox"] = htr.union_bbox(
            [duplicate["pixel_bbox"], table["pixel_bbox"]]
        )
        duplicate["confidence"] = max(duplicate["confidence"], table["confidence"])
    selected.sort(key=lambda item: (item["pixel_bbox"][1], item["pixel_bbox"][0]))
    return selected


def _localized_table_candidates(
    tables: list[dict[str, Any]],
    lines: list[htr.DetectionRecord],
    page_width: int,
    page_height: int,
    args: argparse.Namespace,
) -> list[TableCandidate]:
    page_area = max(1, page_width * page_height)
    candidates: list[TableCandidate] = []
    for index, table in enumerate(tables, start=1):
        bbox = list(table["pixel_bbox"])
        crop_bbox = _expanded_bbox(
            bbox, page_width, page_height, args.table_crop_padding
        )
        covered_lines = [
            line
            for line in lines
            if _line_belongs_to_region(line, crop_bbox, args.table_line_overlap)
        ]
        width = max(1, bbox[2] - bbox[0])
        height = max(1, bbox[3] - bbox[1])
        candidates.append(
            TableCandidate(
                region_id=f"qwen_full_page_table_{index:03d}",
                candidate_id=f"table_candidate_{index:03d}",
                bbox=bbox,
                crop_bbox=crop_bbox,
                detector_confidence=float(table["confidence"]),
                area_ratio=width * height / page_area,
                aspect_ratio=width / height,
                squareness=min(width, height) / max(width, height),
                score=1.0 + float(table["confidence"]),
                line_ids=[
                    str(line.line_id)
                    for line in covered_lines
                    if line.line_id is not None
                ],
                localization_source="qwen_full_page",
            )
        )
    return candidates


def _save_localization_overlay(
    image: Image.Image,
    tables: list[dict[str, Any]],
    path: Path,
) -> None:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    stroke = max(2, round(max(image.size) / 700))
    for index, table in enumerate(tables, start=1):
        bbox = table["pixel_bbox"]
        draw.rectangle(bbox, outline=(190, 30, 220), width=stroke)
        draw.text(
            (bbox[0] + 4, bbox[1] + 4),
            f"Qwen table {index} ({table['confidence']:.2f})",
            fill=(190, 30, 220),
        )
    canvas.save(path)


def localize_tables_with_qwen(
    page_image: Image.Image,
    lines: list[htr.DetectionRecord],
    page_dir: Path,
    pool: CudaModelPool,
    args: argparse.Namespace,
) -> tuple[list[TableCandidate] | None, dict[str, Any]]:
    """Return candidates, or None when RF-DETR fallback must be used."""

    if args.qwen_table_localization == "off":
        return None, {
            "status": "off",
            "table_count": 0,
            "tables": [],
            "fallback": "rfdetr_bbox_heuristic",
            "error": None,
        }
    client = pool.get_qwen_client()
    last_error: Exception | None = None
    for attempt in range(1, args.qwen_retries + 2):
        try:
            response = client.chat.completions.create(
                model=args.qwen_model,
                temperature=0,
                max_tokens=args.qwen_localization_max_tokens,
                messages=[
                    {"role": "system", "content": TABLE_LOCALIZATION_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Найди координаты всех таблиц на полной странице. "
                                    "Верни bbox в normalized_1000."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_to_data_url(page_image)},
                            },
                        ],
                    },
                ],
                tools=[TABLE_LOCALIZATION_TOOL],
                tool_choice="required",
                timeout=args.qwen_timeout,
                extra_body={"ttl": args.qwen_ttl},
            )
            data = _extract_tool_data(response, "locate_tables")
            raw_tables = data.get("tables", [])
            if not isinstance(raw_tables, list):
                raise TypeError("locate_tables.tables должен быть массивом.")
            validated: list[dict[str, Any]] = []
            for raw in raw_tables:
                if not isinstance(raw, dict):
                    continue
                pixel_bbox = _normalized_bbox_to_pixels(
                    raw.get("bbox", []), page_image.width, page_image.height
                )
                if pixel_bbox is None:
                    continue
                try:
                    confidence = max(0.0, min(1.0, float(raw.get("confidence", 0.5))))
                except (TypeError, ValueError):
                    confidence = 0.5
                validated.append(
                    {
                        "normalized_bbox": list(raw.get("bbox", [])),
                        "pixel_bbox": pixel_bbox,
                        "confidence": confidence,
                    }
                )
            tables = _deduplicate_localized_tables(validated)
            candidates = _localized_table_candidates(
                tables,
                lines,
                page_image.width,
                page_image.height,
                args,
            )
            result = {
                "status": "ok",
                "model": args.qwen_model,
                "source_coordinate_space": "normalized_1000",
                "coordinate_space": "prepared_page_pixels",
                "table_count": len(tables),
                "tables": tables,
                "fallback": None,
                "error": None,
            }
            if args.save_debug:
                _save_localization_overlay(
                    page_image, tables, page_dir / "qwen_table_localization.png"
                )
            return candidates, result
        except KeyboardInterrupt:
            raise
        except Exception as error:
            last_error = error
            htr.warn(
                f"Full-page table localization attempt "
                f"{attempt}/{args.qwen_retries + 1}: {error}"
            )
            if attempt <= args.qwen_retries:
                time.sleep(args.qwen_retry_delay)
    return None, {
        "status": "error",
        "model": args.qwen_model,
        "table_count": 0,
        "tables": [],
        "fallback": "rfdetr_bbox_heuristic",
        "error": str(last_error),
    }


def recognize_vlm_candidates(
    candidates: list[TableCandidate],
    page_image: Image.Image,
    page_dir: Path,
    pool: CudaModelPool,
    args: argparse.Namespace,
) -> list[TableCandidate]:
    if not candidates:
        return candidates
    client = pool.get_qwen_client()
    regions_dir = page_dir / "vlm_regions"
    regions_dir.mkdir(parents=True, exist_ok=True)

    for candidate in candidates:
        if not candidate.crop_bbox:
            candidate.crop_bbox = _expanded_bbox(
                candidate.bbox,
                page_image.width,
                page_image.height,
                args.table_crop_padding,
            )
        crop = page_image.crop(tuple(candidate.crop_bbox)).convert("RGB")
        relative_crop = Path("vlm_regions") / f"{candidate.candidate_id}.png"
        crop.save(page_dir / relative_crop)
        candidate.crop_path = relative_crop.as_posix()

        last_error: Exception | None = None
        for attempt in range(1, args.qwen_retries + 2):
            try:
                candidate.result = recognize_candidate_with_qwen(
                    client,
                    crop,
                    candidate,
                    args,
                )
                candidate.error = None
                htr.log(
                    f"  {candidate.candidate_id}: Qwen -> {candidate.result['type']}"
                )
                break
            except KeyboardInterrupt:
                raise
            except Exception as error:
                last_error = error
                htr.warn(
                    f"{candidate.candidate_id}: Qwen attempt "
                    f"{attempt}/{args.qwen_retries + 1}: {error}"
                )
                if attempt <= args.qwen_retries:
                    time.sleep(args.qwen_retry_delay)
        if candidate.result is None:
            candidate.error = str(last_error)
            htr.warn(f"{candidate.candidate_id}: fallback на TrOCR для всех строк.")

        htr.save_json_atomic(
            regions_dir / f"{candidate.candidate_id}.json",
            candidate.to_json(),
        )
    return candidates


def _line_ocr_quality_score(text: str, flags: Sequence[str]) -> float:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized or normalized == "[НЕРАЗБОРЧИВО]":
        return -100.0
    cyrillic = len(re.findall(r"[А-Яа-яЁё]", normalized))
    latin = len(re.findall(r"[A-Za-z]", normalized))
    digits = len(re.findall(r"\d", normalized))
    score = min(40.0, (cyrillic + digits) * 0.7 + len(normalized) * 0.08)
    if cyrillic:
        score += 8.0
    if latin > cyrillic * 1.5 and latin >= 5:
        score -= 18.0
    penalties = {
        "empty_output": 100.0,
        "repetition_loop": 80.0,
        "repetition_truncated": 24.0,
        "suspicious_latin_output": 18.0,
        "very_long_line": 8.0,
    }
    score -= sum(penalties.get(flag, 0.0) for flag in flags)
    return score


def recognize_suspicious_line_with_qwen(
    client: Any,
    image: Image.Image,
    line: htr.DetectionRecord,
    trigger_flags: Sequence[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=args.qwen_model,
        temperature=0,
        max_tokens=args.qwen_line_review_max_tokens,
        messages=[
            {"role": "system", "content": SUSPICIOUS_LINE_OCR_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Строка {line.line_id}. Причины повторной проверки: "
                            f"{', '.join(trigger_flags)}. Выполни независимый OCR."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(image)},
                    },
                ],
            },
        ],
        tools=[LINE_OCR_TOOL],
        tool_choice="required",
        timeout=args.qwen_timeout,
        extra_body={"ttl": args.qwen_ttl},
    )
    data = _extract_tool_data(response, "save_line_ocr")
    raw_text = re.sub(r"\s+", " ", str(data.get("text", ""))).strip()
    cleaned, quality_flags = htr.inspect_ocr_text(raw_text)
    return {
        "raw_text": raw_text,
        "text": cleaned,
        "quality_flags": quality_flags,
        "quality_score": _line_ocr_quality_score(cleaned, quality_flags),
    }


def review_suspicious_trocr_lines(
    lines: list[htr.DetectionRecord],
    page_dir: Path,
    pool: CudaModelPool,
    args: argparse.Namespace,
) -> dict[str, int]:
    if not args.qwen_review_suspicious_lines:
        return {"candidates": 0, "accepted": 0, "rejected": 0, "failed": 0}
    suspicious = [
        line for line in lines if SUSPICIOUS_OCR_FLAGS.intersection(line.quality_flags)
    ]
    selected = suspicious[: args.qwen_line_review_limit]
    if not selected:
        return {"candidates": 0, "accepted": 0, "rejected": 0, "failed": 0}

    client = pool.get_qwen_client()
    reviews_dir = page_dir / "qwen_line_reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    accepted = rejected = failed = 0

    for line in selected:
        trigger_flags = [
            flag for flag in line.quality_flags if flag in SUSPICIOUS_OCR_FLAGS
        ]
        trocr_result = {
            "raw_text": line.raw_text,
            "text": line.text,
            "quality_flags": list(line.quality_flags),
            "quality_score": _line_ocr_quality_score(line.text, line.quality_flags),
        }
        review: dict[str, Any] = {
            "line_id": line.line_id,
            "bbox": line.bbox,
            "trigger_flags": trigger_flags,
            "status": "failed",
            "primary_recognizer": "trocr",
            "trocr": trocr_result,
            "qwen3_vl": None,
            "error": None,
        }
        try:
            if not line.crop_path:
                raise RuntimeError("У строки отсутствует crop_path.")
            with Image.open(page_dir / line.crop_path) as source:
                crop = source.convert("RGB")
            last_error: Exception | None = None
            qwen_result: dict[str, Any] | None = None
            for attempt in range(1, args.qwen_retries + 2):
                try:
                    qwen_result = recognize_suspicious_line_with_qwen(
                        client, crop, line, trigger_flags, args
                    )
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as error:
                    last_error = error
                    if attempt <= args.qwen_retries:
                        time.sleep(args.qwen_retry_delay)
            if qwen_result is None:
                raise RuntimeError(str(last_error))

            review["qwen3_vl"] = qwen_result
            if qwen_result["quality_score"] > trocr_result["quality_score"]:
                accepted += 1
                review["status"] = "accepted"
                review["primary_recognizer"] = "qwen3-vl-line-review"
                line.text = qwen_result["text"]
                line.quality_flags = list(
                    dict.fromkeys(
                        [
                            "reviewed_by_qwen",
                            *qwen_result["quality_flags"],
                            *(f"trocr_flag:{flag}" for flag in trigger_flags),
                        ]
                    )
                )
                line.primary_recognizer = "qwen3-vl-line-review"
            else:
                rejected += 1
                review["status"] = "rejected"
                line.quality_flags.append("qwen_review_rejected")
                line.primary_recognizer = "trocr"
        except KeyboardInterrupt:
            raise
        except Exception as error:
            failed += 1
            review["error"] = str(error)
            line.quality_flags.append("qwen_review_failed")
            line.primary_recognizer = "trocr"

        line.qwen_review = review
        htr.save_json_atomic(
            reviews_dir / f"{line.line_id}.json",
            review,
        )

    return {
        "candidates": len(suspicious),
        "reviewed": len(selected),
        "accepted": accepted,
        "rejected": rejected,
        "failed": failed,
        "skipped_by_limit": max(0, len(suspicious) - len(selected)),
    }


def line_to_json(line: htr.DetectionRecord) -> dict[str, Any]:
    data = line.to_json()
    review = getattr(line, "qwen_review", None)
    data["primary_recognizer"] = getattr(line, "primary_recognizer", "trocr")
    if isinstance(review, dict):
        data["qwen_review"] = review
        data["alternatives"] = {
            "trocr": review["trocr"],
            "qwen3-vl": review.get("qwen3_vl"),
        }
    return data


def _merge_blocks(
    ocr_blocks: list[dict[str, Any]],
    candidates: list[TableCandidate],
    lines: list[htr.DetectionRecord],
) -> list[dict[str, Any]]:
    lines_by_id = {
        str(line.line_id): line for line in lines if line.line_id is not None
    }
    blocks: list[dict[str, Any]] = []
    for block in ocr_blocks:
        item = dict(block)
        recognizers = {
            getattr(lines_by_id[line_id], "primary_recognizer", "trocr")
            for line_id in item.get("line_ids", [])
            if line_id in lines_by_id
        }
        item["recognizer"] = (
            next(iter(recognizers))
            if len(recognizers) == 1
            else "hybrid"
            if recognizers
            else "trocr"
        )
        blocks.append(item)

    for candidate in candidates:
        if candidate.result is None:
            continue
        item: dict[str, Any] = {
            "type": candidate.result["type"],
            "source": candidate.result["source"],
            "content": candidate.result["content"],
            "line_ids": candidate.line_ids,
            "bbox": candidate.bbox,
            "recognizer": "qwen3-vl",
            "vlm_region_id": candidate.candidate_id,
        }
        if "rows" in candidate.result:
            item["rows"] = candidate.result["rows"]
        blocks.append(item)

    blocks.sort(
        key=lambda block: (
            block.get("bbox", [0, 10**9, 0, 0])[1],
            block.get("bbox", [10**9, 0, 0, 0])[0],
        )
    )
    for index, block in enumerate(blocks, start=1):
        block["id"] = f"b{index:04d}"
    return blocks


def assemble_hybrid_pages(
    output_root: Path,
    manifests: list[dict[str, Any]],
    pool: CudaModelPool,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    processor, model, dtype = pool.get_trocr()
    pages: list[dict[str, Any]] = []
    all_text_parts: list[str] = []

    for current, manifest in enumerate(manifests, start=1):
        page_number = int(manifest["page"])
        page_dir = output_root / f"page_{page_number:03d}"
        with Image.open(page_dir / manifest["prepared_image"]) as source:
            page_image = source.convert("RGB")
        lines, regions = htr.records_from_detection_manifest(manifest)
        localized_candidates, table_localization = localize_tables_with_qwen(
            page_image=page_image,
            lines=lines,
            page_dir=page_dir,
            pool=pool,
            args=args,
        )
        if localized_candidates is None:
            candidates = select_table_candidates(
                lines,
                regions,
                page_image.width,
                page_image.height,
                args,
            )
        else:
            candidates = localized_candidates
        localization_dir = page_dir / "vlm_regions"
        localization_dir.mkdir(parents=True, exist_ok=True)
        htr.save_json_atomic(
            localization_dir / "full_page_table_localization.json",
            table_localization,
        )
        htr.log(
            f"[{current}/{len(manifests)}] Страница {page_number}: "
            f"lines={len(lines)}, VLM candidates={len(candidates)}, "
            f"localizer={table_localization['status']}"
        )
        recognize_vlm_candidates(
            candidates,
            page_image,
            page_dir,
            pool,
            args,
        )

        successful_candidates = [
            candidate for candidate in candidates if candidate.result is not None
        ]
        routed_line_ids = {
            line_id
            for candidate in successful_candidates
            for line_id in candidate.line_ids
        }
        ocr_lines = [line for line in lines if line.line_id not in routed_line_ids]
        for line in lines:
            if line.line_id in routed_line_ids:
                owner = next(
                    candidate.candidate_id
                    for candidate in successful_candidates
                    if line.line_id in candidate.line_ids
                )
                line.quality_flags = [f"routed_to_vlm:{owner}"]
                line.primary_recognizer = "qwen3-vl"

        recognize_trocr_lines(
            processor=processor,
            model=model,
            dtype=dtype,
            device=args.ocr_device,
            line_records=ocr_lines,
            page_dir=page_dir,
            batch_size=args.ocr_batch_size,
            max_new_tokens=args.max_new_tokens,
            canvas_width=args.ocr_canvas_width,
            canvas_height=args.ocr_canvas_height,
            autocontrast=not args.no_line_autocontrast,
            save_prepared_lines=args.save_debug,
        )
        line_review_stats = review_suspicious_trocr_lines(
            lines=ocr_lines,
            page_dir=page_dir,
            pool=pool,
            args=args,
        )
        ocr_blocks = htr.group_lines_deterministically(
            ocr_lines,
            page_image.width,
            page_image.height,
            args.default_source,
        )
        blocks = _merge_blocks(ocr_blocks, successful_candidates, lines)

        languages = ["ru"]
        for candidate in successful_candidates:
            for language in candidate.result.get("languages", []):
                if language not in languages:
                    languages.append(language)

        page_json = {
            "languages": languages,
            "blocks": blocks,
            "lines": [line_to_json(line) for line in lines],
            "regions": [region.to_json() for region in regions],
            "vlm_regions": [candidate.to_json() for candidate in candidates],
            "table_localization": table_localization,
            "page": page_number,
            "page_size": {
                "width": page_image.width,
                "height": page_image.height,
            },
            "coordinate_space": "prepared_page_after_deskew_and_outer_crop",
            "preprocessing": manifest["preprocessing"],
            "models": {
                "line_detector": args.detector_repo,
                "line_recognizer": args.ocr_model,
                "region_recognizer": args.qwen_model,
            },
            "routing": {
                "method": "qwen_full_page_bbox_then_qwen_table_ocr",
                "table_localizer": "qwen3-vl-full-page",
                "rfdetr_table_fallback_used": localized_candidates is None,
                "ocr_line_count": len(ocr_lines),
                "vlm_candidate_count": len(candidates),
                "vlm_success_count": len(successful_candidates),
                "suppressed_duplicate_line_count": len(routed_line_ids),
                "qwen_line_review": line_review_stats,
                "thresholds": {
                    "min_area_ratio": args.table_min_area_ratio,
                    "max_area_ratio": args.table_max_area_ratio,
                    "min_squareness": args.table_min_squareness,
                    "min_lines": args.table_min_lines,
                    "candidate_overlap": args.table_candidate_overlap,
                    "line_overlap": args.table_line_overlap,
                },
            },
        }
        htr.save_json_atomic(
            page_dir / f"ocr_json_{page_number:03d}.json",
            page_json,
        )
        page_text = "\n".join(str(block.get("content", "")) for block in blocks)
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


def output_root_for_input(
    input_path: Path,
    output_parent: Path,
) -> Path:
    return output_parent / input_path.stem


def _document_models(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "line_detector": args.detector_repo,
        "line_recognizer": args.ocr_model,
        "region_recognizer": args.qwen_model,
        "document_extractor": (
            args.extraction_model if args.extract_well_data else None
        ),
    }


def _model_residency(args: argparse.Namespace) -> dict[str, str]:
    return {
        "detector": "cpu" if args.detector_cpu else "cuda_until_process_exit",
        "line_recognizer": (
            "cuda_until_process_exit"
            if htr.resolve_torch_device(args.ocr_device) == "cuda"
            else "cpu"
        ),
        "region_recognizer": "lm_studio_ttl",
        "document_extractor": "lm_studio_ttl" if args.extract_well_data else "off",
    }


def finalize_document(
    output_root: Path,
    document_json: dict[str, Any],
    pool: CudaModelPool,
    args: argparse.Namespace,
    started: float,
) -> Path:
    document_json["elapsed_seconds"] = round(time.time() - started, 3)
    cleaned_document = clean_document_for_extraction(document_json)
    cleaned_path = output_root / "cleaned_document.json"
    htr.save_json_atomic(cleaned_path, cleaned_document)
    document_json["postprocessing"] = {
        "cleaned_json": cleaned_path.name,
        "well_extraction": {
            "status": "pending" if args.extract_well_data else "off",
            "model": args.extraction_model if args.extract_well_data else None,
            "prompt": (
                str(args.extraction_prompt.expanduser().resolve())
                if args.extract_well_data
                else None
            ),
            "result_json": None,
            "error": None,
        },
    }
    document_path = output_root / "document.json"
    htr.save_json_atomic(document_path, document_json)
    htr.log(f"Очищенный JSON: {cleaned_path}")

    if args.extract_well_data:
        extraction_metadata = document_json["postprocessing"]["well_extraction"]
        try:
            extraction_result = extract_well_data_with_qwen(
                cleaned_document,
                pool,
                args,
            )
            extraction_path = output_root / "well_extraction.json"
            htr.save_json_atomic(extraction_path, extraction_result)
            extraction_metadata.update(
                {
                    "status": "ok",
                    "result_json": extraction_path.name,
                    "error": None,
                }
            )
            htr.log(f"Итоговый JSON: {extraction_path}")
        except KeyboardInterrupt:
            raise
        except Exception as error:
            extraction_metadata.update(
                {
                    "status": "error",
                    "result_json": None,
                    "error": str(error),
                }
            )
            document_json["elapsed_seconds"] = round(time.time() - started, 3)
            htr.save_json_atomic(document_path, document_json)
            raise

    document_json["elapsed_seconds"] = round(time.time() - started, 3)
    htr.save_json_atomic(document_path, document_json)
    htr.log(f"Готово: {document_path}")
    return output_root


def _archive_member_filename(index: int, member_path: str) -> str:
    basename = member_path.replace("\\", "/").rsplit("/", 1)[-1]
    path = Path(basename)
    stem = re.sub(r"[^\w.-]+", "_", path.stem, flags=re.UNICODE).strip("._")
    suffix = path.suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
        suffix = ".bin"
    return f"{index:04d}_{stem or 'member'}{suffix}"


def extract_zip_members(
    archive_path: Path,
    temporary_root: Path,
) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        file_infos = [info for info in archive.infolist() if not info.is_dir()]
        if len(file_infos) > MAX_ARCHIVE_MEMBERS:
            raise ValueError(
                f"В ZIP слишком много файлов: {len(file_infos)} > "
                f"{MAX_ARCHIVE_MEMBERS}."
            )
        total_size = sum(max(0, info.file_size) for info in file_infos)
        if total_size > MAX_ARCHIVE_TOTAL_BYTES:
            raise ValueError(
                f"Распакованный размер ZIP превышает {MAX_ARCHIVE_TOTAL_BYTES} байт."
            )

        for index, info in enumerate(file_infos, start=1):
            member: dict[str, Any] = {
                "archive_path": info.filename,
                "size_bytes": int(info.file_size),
                "compressed_size_bytes": int(info.compress_size),
                "input_type": None,
                "temporary_path": None,
                "error": None,
            }
            if info.flag_bits & 0x1:
                member["error"] = "Зашифрованные элементы ZIP не поддерживаются."
                members.append(member)
                continue
            if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                member["error"] = (
                    f"Размер элемента превышает {MAX_ARCHIVE_MEMBER_BYTES} байт."
                )
                members.append(member)
                continue

            destination = temporary_root / _archive_member_filename(
                index, info.filename
            )
            try:
                with archive.open(info) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                member["temporary_path"] = str(destination)
                member["input_type"] = htr.detect_input_file_type(destination)
            except Exception as error:
                member["error"] = f"{type(error).__name__}: {error}"
            members.append(member)
    return members


def process_archive(
    archive_path: Path,
    pool: CudaModelPool,
    args: argparse.Namespace,
) -> Path:
    output_parent = args.output_dir.expanduser().resolve()
    output_root = output_root_for_input(archive_path, output_parent)
    if output_root.exists() and args.overwrite and not args.skip_detection:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    members_output = output_root / "members"
    members_output.mkdir(parents=True, exist_ok=True)

    started = time.time()
    htr.log(f"Вход: {archive_path}")
    htr.log(f"Тип входного файла: {htr.INPUT_TYPE_ZIP}")
    htr.log(f"Результат: {output_root}")

    archive_members: list[dict[str, Any]] = []
    combined_pages: list[dict[str, Any]] = []
    all_text_parts: list[str] = []
    source_page_count = 0
    processed_file_count = 0

    with tempfile.TemporaryDirectory(prefix="hybrid_htr_zip_") as temporary:
        extracted_members = extract_zip_members(archive_path, Path(temporary))
        for current, member in enumerate(extracted_members, start=1):
            archive_member = {
                "archive_path": member["archive_path"],
                "input_type": member["input_type"],
                "size_bytes": member["size_bytes"],
                "status": "skipped",
                "page_count": 0,
                "processed_pages": [],
                "output_dir": None,
                "error": member["error"],
            }
            input_type = member.get("input_type")
            temporary_path = member.get("temporary_path")
            if member.get("error"):
                htr.warn(f"ZIP {member['archive_path']}: пропущен: {member['error']}")
                archive_members.append(archive_member)
                continue
            if input_type == htr.INPUT_TYPE_ZIP:
                archive_member["error"] = "Вложенные ZIP-архивы не поддерживаются."
                htr.warn(f"ZIP {member['archive_path']}: вложенный архив пропущен.")
                archive_members.append(archive_member)
                continue
            if not temporary_path:
                archive_member["error"] = "Временный файл не создан."
                archive_members.append(archive_member)
                continue

            htr.log(
                f"[{current}/{len(extracted_members)}] ZIP member: "
                f"{member['archive_path']} ({input_type})"
            )
            member_args = argparse.Namespace(**vars(args))
            member_args.output_dir = members_output
            member_args.extract_well_data = False
            member_args.keep_alive = False
            try:
                member_output = process_input(temporary_path, pool, member_args)
                child_document_path = member_output / "document.json"
                child_document = json.loads(
                    child_document_path.read_text(encoding="utf-8")
                )
                child_document["source_file"] = member["archive_path"]
                child_document["archive_source"] = archive_path.name
                child_document_path.unlink()
                (member_output / "cleaned_document.json").unlink(missing_ok=True)

                member_relative = member_output.relative_to(output_root).as_posix()
                local_processed_pages = [
                    int(page) for page in child_document.get("processed_pages", [])
                ]
                page_count = int(child_document.get("page_count_in_source", 0))
                source_page_count += page_count
                processed_file_count += 1
                archive_member.update(
                    {
                        "status": "ok",
                        "page_count": page_count,
                        "processed_pages": local_processed_pages,
                        "output_dir": member_relative,
                        "error": None,
                    }
                )

                for raw_page in child_document.get("pages", []):
                    if not isinstance(raw_page, dict):
                        continue
                    source_page = int(raw_page.get("page", 0))
                    global_page_number = len(combined_pages) + 1
                    page = dict(raw_page)
                    page.update(
                        {
                            "page": global_page_number,
                            "source_file": member["archive_path"],
                            "source_page": source_page,
                            "artifact_root": (
                                f"{member_relative}/page_{source_page:03d}"
                            ),
                        }
                    )
                    combined_pages.append(page)
                    page_text = "\n".join(
                        str(block.get("content", ""))
                        for block in page.get("blocks", [])
                        if isinstance(block, dict)
                    )
                    all_text_parts.append(
                        f"=== PAGE {global_page_number} | "
                        f"{member['archive_path']}:{source_page} ===\n{page_text}"
                    )
            except KeyboardInterrupt:
                raise
            except Exception as error:
                archive_member["status"] = "error"
                archive_member["error"] = f"{type(error).__name__}: {error}"
                htr.warn(
                    f"ZIP {member['archive_path']}: "
                    f"ошибка обработки: {archive_member['error']}"
                )
                if not args.continue_on_error:
                    archive_members.append(archive_member)
                    raise
            archive_members.append(archive_member)

    if not combined_pages:
        raise RuntimeError("В ZIP не найдено ни одной успешно обработанной страницы.")

    (output_root / "all_text.txt").write_text(
        "\n\n".join(all_text_parts) + "\n",
        encoding="utf-8",
    )
    document_json = {
        "source_file": str(archive_path),
        "input_type": htr.INPUT_TYPE_ZIP,
        "page_count_in_source": source_page_count,
        "processed_pages": list(range(1, len(combined_pages) + 1)),
        "processed_files": processed_file_count,
        "pages": combined_pages,
        "archive": {
            "format": "zip",
            "member_count": len(archive_members),
            "processed_member_count": processed_file_count,
            "members": archive_members,
        },
        "models": _document_models(args),
        "model_residency": _model_residency(args),
        "qwen_lmstudio_ttl_seconds": args.qwen_ttl,
    }
    return finalize_document(output_root, document_json, pool, args, started)


def process_input(
    input_value: str | Path,
    pool: CudaModelPool,
    args: argparse.Namespace,
) -> Path:
    input_path = Path(input_value).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Не найден входной файл: {input_path}")
    input_type = htr.detect_input_file_type(input_path)
    if input_type == htr.INPUT_TYPE_ZIP:
        return process_archive(input_path, pool, args)
    output_parent = args.output_dir.expanduser().resolve()
    output_root = output_root_for_input(input_path, output_parent)
    if output_root.exists() and args.overwrite and not args.skip_detection:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    page_count = htr.input_page_count(input_path, input_type)
    page_indices = htr.parse_page_spec(args.pages, page_count)
    htr.log(f"Вход: {input_path}")
    htr.log(f"Тип входного файла: {input_type}")
    htr.log(f"Страницы: {[index + 1 for index in page_indices]}")
    htr.log(f"Результат: {output_root}")
    started = time.time()

    if args.skip_detection:
        manifests = htr.load_saved_manifests(output_root, page_indices)
    else:
        manifests = htr.detect_document_pages(
            input_path=input_path,
            page_indices=page_indices,
            output_root=output_root,
            detector=pool.get_detector(),
            args=args,
            input_type=input_type,
        )

    pages = assemble_hybrid_pages(output_root, manifests, pool, args)
    document_json = {
        "source_file": str(input_path),
        "input_type": input_type,
        "page_count_in_source": page_count,
        "processed_pages": [index + 1 for index in page_indices],
        "pages": pages,
        "models": _document_models(args),
        "model_residency": _model_residency(args),
        "qwen_lmstudio_ttl_seconds": args.qwen_ttl,
    }
    return finalize_document(output_root, document_json, pool, args, started)


def validate_args(args: argparse.Namespace) -> None:
    if not args.inputs and not args.keep_alive:
        raise ValueError("Укажи хотя бы один PDF/JPEG/TIFF/ZIP или --keep-alive.")
    if args.dpi < 72:
        raise ValueError("--dpi должен быть >= 72.")
    if args.tile_count < 1 or args.ocr_batch_size < 1:
        raise ValueError("tile-count и ocr-batch-size должны быть >= 1.")
    for name in ("tile_overlap", "tile_cut_search"):
        value = getattr(args, name)
        if not 0 <= value < 0.45:
            raise ValueError(f"--{name.replace('_', '-')} должен быть в [0, 0.45).")
    if not 0 < args.table_min_area_ratio < args.table_max_area_ratio <= 1:
        raise ValueError("Некорректные table area thresholds.")
    if not 0 < args.table_min_squareness <= 1:
        raise ValueError("--table-min-squareness должен быть в (0, 1].")
    if args.table_min_lines < 1:
        raise ValueError("--table-min-lines должен быть >= 1.")
    if args.qwen_ttl < 1 or args.qwen_timeout <= 0:
        raise ValueError("Qwen TTL/timeout должны быть положительными.")
    if args.qwen_localization_max_tokens < 64:
        raise ValueError("--qwen-localization-max-tokens должен быть >= 64.")
    if args.qwen_line_review_max_tokens < 32:
        raise ValueError("--qwen-line-review-max-tokens должен быть >= 32.")
    if args.qwen_line_review_limit < 1:
        raise ValueError("--qwen-line-review-limit должен быть >= 1.")
    for name in ("table_candidate_overlap", "table_line_overlap"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} должен быть в [0, 1].")
    if not 0 <= args.table_crop_padding < 0.25:
        raise ValueError("--table-crop-padding должен быть в [0, 0.25).")
    if args.qwen_retries < 0 or args.qwen_retry_delay < 0:
        raise ValueError("Qwen retries/delay не могут быть отрицательными.")
    if args.extraction_max_tokens < 256 or args.extraction_timeout <= 0:
        raise ValueError(
            "extraction-max-tokens должен быть >= 256, timeout — положительным."
        )
    if args.extract_well_data and not args.extraction_prompt.expanduser().is_file():
        raise FileNotFoundError(
            f"Не найден промпт экстракции: {args.extraction_prompt}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "RF-DETR -> TrOCR lines + Qwen3-VL tables -> cleaned JSON -> "
            "Qwen3.6-27B well extraction"
        )
    )
    parser.add_argument("inputs", nargs="*", help="PDF, JPEG, TIFF или ZIP")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hybrid_htr_output"),
        help="Корневая папка; для каждого input создаётся подпапка.",
    )
    parser.add_argument("--pages", default=None)
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--skip-detection", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="После inputs читать новые пути из stdin, не выгружая модели.",
    )

    parser.add_argument("--no-deskew", action="store_true")
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--crop-padding-x", type=float, default=0.04)
    parser.add_argument("--crop-padding-y", type=float, default=0.02)
    parser.add_argument("--tile-count", type=int, default=4)
    parser.add_argument("--tile-overlap", type=float, default=0.15)
    parser.add_argument("--tile-cut-search", type=float, default=0.25)

    parser.add_argument("--detector-repo", default=htr.DEFAULT_DETECTOR_REPO)
    parser.add_argument("--detector-filename", default=htr.DEFAULT_DETECTOR_FILENAME)
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

    parser.add_argument("--ocr-model", default=DEFAULT_OCR_MODEL)
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
    parser.add_argument("--ocr-canvas-width", type=int, default=1022)
    parser.add_argument("--ocr-canvas-height", type=int, default=182)
    parser.add_argument("--no-line-autocontrast", action="store_true")
    parser.add_argument("--default-source", choices=htr.SOURCE_TYPES, default="mixed")

    parser.add_argument("--table-min-area-ratio", type=float, default=0.05)
    parser.add_argument("--table-max-area-ratio", type=float, default=0.65)
    parser.add_argument("--table-min-squareness", type=float, default=0.25)
    parser.add_argument("--table-min-lines", type=int, default=2)
    parser.add_argument("--table-candidate-overlap", type=float, default=0.30)
    parser.add_argument("--table-line-overlap", type=float, default=0.55)
    parser.add_argument("--table-crop-padding", type=float, default=0.008)

    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--lmstudio-url", default=DEFAULT_LMSTUDIO_URL)
    parser.add_argument(
        "--qwen-table-localization",
        choices=["always", "off"],
        default="always",
        help="Полностраничный поиск bbox таблиц через Qwen; default always.",
    )
    parser.add_argument(
        "--qwen-localization-max-tokens",
        type=int,
        default=1200,
    )
    line_review = parser.add_mutually_exclusive_group()
    line_review.add_argument(
        "--qwen-review-suspicious-lines",
        dest="qwen_review_suspicious_lines",
        action="store_true",
    )
    line_review.add_argument(
        "--no-qwen-review-suspicious-lines",
        dest="qwen_review_suspicious_lines",
        action="store_false",
    )
    # Disabled by default: one VLM request per doubtful TrOCR line is too
    # expensive for routine batch processing. It remains available explicitly.
    parser.set_defaults(qwen_review_suspicious_lines=False)
    parser.add_argument("--qwen-line-review-limit", type=int, default=24)
    parser.add_argument("--qwen-line-review-max-tokens", type=int, default=512)
    parser.add_argument("--qwen-timeout", type=float, default=240.0)
    parser.add_argument("--qwen-max-tokens", type=int, default=3000)
    parser.add_argument("--qwen-retries", type=int, default=2)
    parser.add_argument("--qwen-retry-delay", type=float, default=1.0)
    parser.add_argument(
        "--qwen-ttl",
        type=int,
        default=86400,
        help="LM Studio idle TTL в секундах; default 24 часа.",
    )
    extraction = parser.add_mutually_exclusive_group()
    extraction.add_argument(
        "--extract-well-data",
        dest="extract_well_data",
        action="store_true",
        help="Получить well_extraction.json после OCR (включено по умолчанию).",
    )
    extraction.add_argument(
        "--no-extract-well-data",
        dest="extract_well_data",
        action="store_false",
        help="Сохранить cleaned_document.json, но не вызывать текстовую LLM.",
    )
    parser.set_defaults(extract_well_data=True)
    parser.add_argument(
        "--extraction-model",
        default=DEFAULT_EXTRACTION_MODEL,
        help="Текстовая модель LM Studio для итогового извлечения.",
    )
    parser.add_argument(
        "--extraction-prompt",
        type=Path,
        default=DEFAULT_EXTRACTION_PROMPT,
        help="Файл с секциями [SYSTEM] и [USER].",
    )
    parser.add_argument("--extraction-max-tokens", type=int, default=8192)
    parser.add_argument("--extraction-timeout", type=float, default=900.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    pool = CudaModelPool(args)

    # Preload once. These objects stay referenced (and on CUDA) for the whole
    # process, including the interactive --keep-alive session.
    pool.get_trocr()
    if not args.skip_detection or args.keep_alive:
        pool.get_detector()

    failures = 0
    for input_value in args.inputs:
        try:
            process_input(input_value, pool, args)
        except KeyboardInterrupt:
            raise
        except Exception as error:
            failures += 1
            htr.warn(f"{input_value}: {type(error).__name__}: {error}")
            if not args.continue_on_error:
                return 1

    if args.keep_alive:
        htr.log(
            "\nCUDA models are resident. Enter the next PDF/JPEG/TIFF/ZIP path; "
            "type 'exit' to finish."
        )
        while True:
            try:
                value = input("hybrid-htr> ").strip()
            except EOFError:
                break
            if value.lower() in {"exit", "quit", "q"}:
                break
            if not value:
                continue
            try:
                process_input(value, pool, args)
            except KeyboardInterrupt:
                raise
            except Exception as error:
                failures += 1
                htr.warn(f"{value}: {type(error).__name__}: {error}")

    htr.log("Завершение процесса: CUDA-модели будут выгружены.")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nОстановлено пользователем.", file=sys.stderr)
        raise SystemExit(130) from None

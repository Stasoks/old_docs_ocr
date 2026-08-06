#!/usr/bin/env python3
"""Build a landscape PDF report from the pipeline's well_extraction.json."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import tempfile
from collections import Counter
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Iterable

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


DISPLAY_PARAMETER_NAMES = {"Статус скважины": "Этап жизненного цикла"}
CONFIDENCE_LABELS = {"high": "высокая", "medium": "средняя", "low": "низкая"}
CONFLICT_TYPE_LABELS = {
    "ocr_error": "вероятная ошибка OCR",
    "possible_document_inconsistency": "возможное противоречие документа",
    "possible_rounding": "возможное округление",
    "technical_stage_difference": "разные технические этапы",
    "temporal_change": "изменение во времени",
    "unresolved": "неразрешённый конфликт",
}

GRID_COLOR = colors.HexColor("#AEB7C2")
HEADER_BG = colors.HexColor("#F2F4F7")
LIGHT_BG = colors.HexColor("#FAFBFC")
TEXT_COLOR = colors.HexColor("#111827")
MUTED_COLOR = colors.HexColor("#4B5563")


def safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def escape(value: Any) -> str:
    return html.escape(safe_text(value), quote=False).replace("\n", "<br/>")


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", safe_text(value)).strip()


def basename(value: Any) -> str:
    text = safe_text(value).replace("\\", "/")
    return text.rsplit("/", 1)[-1] if text else ""


def validate_well_json(data: dict[str, Any], source: str = "JSON") -> None:
    if not isinstance(data, dict):
        raise TypeError(f"{source}: корень JSON должен быть объектом.")
    for field in ("records", "conflicts", "missing_parameters", "warnings"):
        value = data.get(field, [])
        if value is not None and not isinstance(value, list):
            raise TypeError(f"{source}: поле {field!r} должно быть массивом.")
    for index, record in enumerate(data.get("records") or [], start=1):
        if not isinstance(record, dict):
            raise TypeError(f"{source}: records[{index}] должен быть объектом.")
        if not safe_text(record.get("parameter")) or not safe_text(record.get("value")):
            raise ValueError(f"{source}: records[{index}] не содержит parameter/value.")


def load_json(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8-sig") as source:
        data = json.load(source)
    validate_well_json(data, str(path))
    return data


def _font_candidates() -> list[tuple[Path, Path]]:
    candidates: list[tuple[Path, Path]] = []
    regular_env = os.environ.get("WELL_REPORT_FONT_REGULAR")
    bold_env = os.environ.get("WELL_REPORT_FONT_BOLD")
    if regular_env and bold_env:
        candidates.append((Path(regular_env), Path(bold_env)))
    candidates.extend(
        [
            (
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            ),
            (
                Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
                Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
            ),
            (
                Path(r"C:\Windows\Fonts\arial.ttf"),
                Path(r"C:\Windows\Fonts\arialbd.ttf"),
            ),
            (
                Path("/Library/Fonts/Arial.ttf"),
                Path("/Library/Fonts/Arial Bold.ttf"),
            ),
        ]
    )
    return candidates


@lru_cache(maxsize=1)
def register_fonts() -> tuple[str, str]:
    for regular_path, bold_path in _font_candidates():
        if regular_path.is_file() and bold_path.is_file():
            pdfmetrics.registerFont(TTFont("WellReportRegular", str(regular_path)))
            pdfmetrics.registerFont(TTFont("WellReportBold", str(bold_path)))
            return "WellReportRegular", "WellReportBold"
    raise FileNotFoundError(
        "Не найден Unicode TTF-шрифт. Установи DejaVu Sans или задай "
        "WELL_REPORT_FONT_REGULAR/WELL_REPORT_FONT_BOLD."
    )


def make_styles() -> tuple[dict[str, ParagraphStyle], str, str]:
    regular, bold = register_fonts()
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "WellTitle",
            parent=base["Title"],
            fontName=bold,
            fontSize=17,
            leading=21,
            textColor=TEXT_COLOR,
            alignment=TA_LEFT,
            spaceAfter=5 * mm,
        ),
        "subtitle": ParagraphStyle(
            "WellSubtitle",
            parent=base["Normal"],
            fontName=regular,
            fontSize=8.5,
            leading=11,
            textColor=MUTED_COLOR,
            spaceAfter=3 * mm,
        ),
        "section": ParagraphStyle(
            "WellSection",
            parent=base["Heading2"],
            fontName=bold,
            fontSize=11.5,
            leading=14,
            textColor=TEXT_COLOR,
            spaceBefore=2 * mm,
            spaceAfter=2.5 * mm,
        ),
        "cell": ParagraphStyle(
            "WellCell",
            parent=base["Normal"],
            fontName=regular,
            fontSize=7.4,
            leading=9.2,
            textColor=TEXT_COLOR,
        ),
        "cell_center": ParagraphStyle(
            "WellCellCenter",
            parent=base["Normal"],
            fontName=regular,
            fontSize=7.4,
            leading=9.2,
            textColor=TEXT_COLOR,
            alignment=TA_CENTER,
        ),
        "small": ParagraphStyle(
            "WellSmall",
            parent=base["Normal"],
            fontName=regular,
            fontSize=6.8,
            leading=8.3,
            textColor=TEXT_COLOR,
        ),
        "header": ParagraphStyle(
            "WellHeader",
            parent=base["Normal"],
            fontName=bold,
            fontSize=7.2,
            leading=8.5,
            textColor=TEXT_COLOR,
        ),
        "header_center": ParagraphStyle(
            "WellHeaderCenter",
            parent=base["Normal"],
            fontName=bold,
            fontSize=7.2,
            leading=8.5,
            textColor=TEXT_COLOR,
            alignment=TA_CENTER,
        ),
        "note": ParagraphStyle(
            "WellNote",
            parent=base["Normal"],
            fontName=regular,
            fontSize=7.6,
            leading=9.5,
            textColor=TEXT_COLOR,
            leftIndent=3 * mm,
            bulletIndent=0,
            spaceAfter=1.2 * mm,
        ),
    }
    return styles, regular, bold


def _p(value: Any, styles: dict[str, ParagraphStyle], style: str = "cell") -> Paragraph:
    return Paragraph(escape(value) or " ", styles[style])


def _document_title(data: dict[str, Any], fallback: str) -> str:
    well_number = ""
    location = ""
    for record in data.get("records") or []:
        parameter = safe_text(record.get("parameter"))
        if parameter == "Номер скважины" and not well_number:
            well_number = safe_text(record.get("value"))
        elif parameter == "Расположение скважины" and not location:
            location = safe_text(record.get("value"))
    title = "Параметры скважины"
    if well_number:
        title += f" № {well_number}"
    return f"{title} — {location or fallback}" if location or fallback else title


def _dominant_file(data: dict[str, Any], fallback: str) -> str:
    files = [
        basename(record.get("file"))
        for record in data.get("records") or []
        if basename(record.get("file"))
    ]
    return Counter(files).most_common(1)[0][0] if files else fallback


def _record_notes(record: dict[str, Any]) -> str:
    parts: list[str] = []
    notes = safe_text(record.get("notes"))
    evidence = safe_text(record.get("evidence"))
    raw_value = safe_text(record.get("raw_value"))
    value = safe_text(record.get("value"))
    confidence = safe_text(record.get("confidence")).lower()
    if notes:
        parts.append(f"<b>Примечание:</b> {escape(notes)}")
    if (
        raw_value
        and normalize_space(raw_value).casefold() != normalize_space(value).casefold()
    ):
        parts.append(f"<b>OCR:</b> {escape(raw_value)}")
    if confidence and confidence != "high":
        parts.append(
            f"<b>Уверенность:</b> "
            f"{escape(CONFIDENCE_LABELS.get(confidence, confidence))}"
        )
    if evidence:
        parts.append(f"<b>Фрагмент:</b> «{escape(evidence)}»")
    return "<br/>".join(parts) or " "


def _base_table_style() -> TableStyle:
    return TableStyle(
        [
            ("GRID", (0, 0), (-1, -1), 0.55, GRID_COLOR),
            ("BOX", (0, 0), (-1, -1), 0.75, GRID_COLOR),
            ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3.2),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3.2),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ]
    )


def _records_table(
    data: dict[str, Any], styles: dict[str, ParagraphStyle]
) -> LongTable:
    rows: list[list[Any]] = [
        [
            _p("№\nп/п", styles, "header_center"),
            _p("Параметр\n(показатель)", styles, "header"),
            _p("Значение", styles, "header"),
            _p("Файл", styles, "header"),
            _p("Стр.", styles, "header_center"),
            _p("Особые примечания", styles, "header"),
        ]
    ]
    for fallback_number, record in enumerate(data.get("records") or [], start=1):
        try:
            row_number = int(record.get("row_number"))
        except (TypeError, ValueError):
            row_number = fallback_number
        parameter = DISPLAY_PARAMETER_NAMES.get(
            safe_text(record.get("parameter")), safe_text(record.get("parameter"))
        )
        rows.append(
            [
                _p(f"{row_number}.", styles, "cell_center"),
                _p(parameter, styles),
                _p(record.get("value"), styles),
                _p(basename(record.get("file")), styles),
                _p(record.get("page"), styles, "cell_center"),
                Paragraph(_record_notes(record), styles["small"]),
            ]
        )
    table = LongTable(
        rows,
        colWidths=[10 * mm, 38 * mm, 40 * mm, 40 * mm, 14 * mm, 119 * mm],
        repeatRows=1,
        splitByRow=1,
        hAlign="LEFT",
    )
    table.setStyle(_base_table_style())
    return table


def _conflicts_table(
    data: dict[str, Any],
    default_file: str,
    styles: dict[str, ParagraphStyle],
) -> LongTable:
    rows: list[list[Any]] = [
        [
            _p("№", styles, "header_center"),
            _p("Параметр", styles, "header"),
            _p("Распознанное\nзначение", styles, "header"),
            _p("Файл", styles, "header"),
            _p("Стр.", styles, "header_center"),
            _p("Фрагмент OCR", styles, "header"),
            _p("Причина конфликта", styles, "header"),
        ]
    ]
    for index, conflict in enumerate(data.get("conflicts") or [], start=1):
        conflict_type = safe_text(conflict.get("conflict_type"))
        reason = safe_text(conflict.get("reason"))
        if conflict_type:
            label = CONFLICT_TYPE_LABELS.get(conflict_type, conflict_type)
            reason = f"{label}. {reason}" if reason else label
        rows.append(
            [
                _p(f"{index}.", styles, "cell_center"),
                _p(
                    DISPLAY_PARAMETER_NAMES.get(
                        safe_text(conflict.get("parameter")),
                        safe_text(conflict.get("parameter")),
                    ),
                    styles,
                ),
                _p(conflict.get("raw_value"), styles),
                _p(basename(conflict.get("file")) or default_file, styles),
                _p(conflict.get("page"), styles, "cell_center"),
                _p(conflict.get("evidence"), styles, "small"),
                _p(reason, styles, "small"),
            ]
        )
    table = LongTable(
        rows,
        colWidths=[8 * mm, 31 * mm, 31 * mm, 36 * mm, 14 * mm, 55 * mm, 86 * mm],
        repeatRows=1,
        splitByRow=1,
        hAlign="LEFT",
    )
    table.setStyle(_base_table_style())
    return table


def _story(
    named_datasets: list[tuple[str, dict[str, Any]]],
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    story: list[Any] = []
    usable_width = landscape(A4)[0] - 36 * mm
    for document_index, (source_name, data) in enumerate(named_datasets):
        if document_index:
            story.append(PageBreak())
        default_file = _dominant_file(data, source_name)
        fallback = Path(source_name).stem if source_name else ""
        story.append(
            Paragraph(escape(_document_title(data, fallback)), styles["title"])
        )
        story.append(
            Paragraph(
                f"Источник результата: {escape(source_name)}"
                f"&nbsp;&nbsp;|&nbsp;&nbsp;Исходный файл: {escape(default_file)}",
                styles["subtitle"],
            )
        )
        story.append(Paragraph("Извлечённые параметры", styles["section"]))
        if data.get("records"):
            story.append(_records_table(data, styles))
        else:
            empty = Table(
                [[_p("Извлечённые параметры не найдены.", styles)]],
                colWidths=[usable_width],
            )
            empty.setStyle(_base_table_style())
            story.append(empty)

        story.extend(
            [Spacer(1, 5 * mm), Paragraph("Конфликты распознавания", styles["section"])]
        )
        if data.get("conflicts"):
            story.append(_conflicts_table(data, default_file, styles))
        else:
            empty = Table(
                [[_p("Конфликты не обнаружены.", styles)]],
                colWidths=[usable_width],
            )
            empty.setStyle(_base_table_style())
            story.append(empty)

        missing = [
            safe_text(item)
            for item in data.get("missing_parameters") or []
            if safe_text(item)
        ]
        warnings = [
            safe_text(item) for item in data.get("warnings") or [] if safe_text(item)
        ]
        if missing or warnings:
            details: list[Any] = [
                Spacer(1, 5 * mm),
                Paragraph("Дополнительная информация", styles["section"]),
            ]
            if missing:
                details.append(
                    Paragraph(
                        "<b>Не найдены:</b> "
                        + ", ".join(escape(item) for item in missing),
                        styles["note"],
                    )
                )
            details.extend(
                Paragraph(f"<b>Предупреждение:</b> {escape(item)}", styles["note"])
                for item in warnings
            )
            story.append(KeepTogether(details))
    return story


class NumberedDocTemplate(BaseDocTemplate):
    def __init__(
        self, filename: str | Path | BinaryIO, footer_font: str, **kwargs: Any
    ):
        self.footer_font = footer_font
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin, self.bottomMargin, self.width, self.height, id="normal"
        )
        self.addPageTemplates(
            [PageTemplate(id="well_report", frames=[frame], onPage=self._draw_footer)]
        )

    def _draw_footer(self, canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont(self.footer_font, 7)
        canvas.setFillColor(MUTED_COLOR)
        canvas.drawCentredString(self.pagesize[0] / 2, 7 * mm, f"Страница {doc.page}")
        canvas.restoreState()


def render_pdf(
    named_datasets: Iterable[tuple[str, dict[str, Any]]],
    title: str = "Параметры скважины",
) -> bytes:
    datasets = list(named_datasets)
    if not datasets:
        raise ValueError("Не передано ни одного JSON.")
    for source_name, data in datasets:
        validate_well_json(data, source_name)
    styles, regular_font, _ = make_styles()
    buffer = BytesIO()
    document = NumberedDocTemplate(
        buffer,
        footer_font=regular_font,
        pagesize=landscape(A4),
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=title,
        author="json_to_well_pdf.py",
        subject="Таблица параметров скважины и конфликтов OCR",
    )
    document.build(_story(datasets, styles))
    return buffer.getvalue()


def create_pdf(
    json_paths: Iterable[Path], output_path: Path, title: str | None = None
) -> None:
    paths = [Path(path).expanduser().resolve() for path in json_paths]
    datasets = [(path.name, load_json(path)) for path in paths]
    pdf_bytes = render_pdf(datasets, title or "Параметры скважины")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=output_path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(pdf_bytes)
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("well_report.pdf"))
    parser.add_argument("--separate-dir", type=Path)
    parser.add_argument("--title")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    create_pdf(args.inputs, args.output, args.title)
    print(f"Создан PDF: {args.output.expanduser().resolve()}")
    if args.separate_dir:
        output_dir = args.separate_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        for input_path in args.inputs:
            output_path = output_dir / f"{input_path.stem}_отчёт.pdf"
            create_pdf([input_path], output_path, args.title)
            print(f"Создан PDF: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

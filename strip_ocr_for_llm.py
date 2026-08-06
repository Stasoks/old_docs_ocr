#!/usr/bin/env python3
"""Remove OCR geometry and keep only content needed by the document LLM.

Output schema::

    {
      "pages": [
        {
          "file": "document.pdf",
          "page": 1,
          "blocks": [
            {"type": "text", "content": "..."}
          ]
        }
      ]
    }

For ZIP input, ``file`` is the archive member name and ``page`` is its original
page number. Empty OCR blocks and every geometry/diagnostic field are removed.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ALLOWED_BLOCK_TYPES = {"text", "title", "subtitle", "table"}


def _basename(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    return text.rsplit("/", 1)[-1] if text else ""


def _positive_int(value: Any, fallback: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return fallback
    return number if number > 0 else fallback


def _block_content(block: dict[str, Any]) -> str:
    value = block.get("content", "")
    if isinstance(value, list):
        content = "\n".join(str(item) for item in value).strip()
    else:
        content = str(value or "").strip()

    # Some VLM responses contain only structured table rows. Preserve their
    # text while still removing the rows structure from the compact JSON.
    rows = block.get("rows")
    if not content and isinstance(rows, list):
        rendered_rows: list[str] = []
        for row in rows:
            cells = row if isinstance(row, list) else [row]
            rendered_rows.append(" | ".join(str(cell).strip() for cell in cells))
        content = "\n".join(rendered_rows).strip()
    return content


def clean_document(document: dict[str, Any]) -> dict[str, Any]:
    """Return a geometry-free JSON with only file/page and block type/content."""

    if not isinstance(document, dict):
        raise TypeError("Корень OCR JSON должен быть объектом.")

    root_file = _basename(document.get("source_file"))
    raw_pages = document.get("pages", [])
    if not isinstance(raw_pages, list):
        raise TypeError("Поле pages должно быть массивом.")

    pages: list[dict[str, Any]] = []
    for page_index, raw_page in enumerate(raw_pages, start=1):
        if not isinstance(raw_page, dict):
            continue
        global_page = _positive_int(raw_page.get("page"), page_index)
        page_number = _positive_int(raw_page.get("source_page"), global_page)
        page_file = _basename(raw_page.get("source_file")) or root_file

        raw_blocks = raw_page.get("blocks", [])
        if not isinstance(raw_blocks, list):
            raw_blocks = []
        blocks: list[dict[str, str]] = []
        for raw_block in raw_blocks:
            if not isinstance(raw_block, dict):
                continue
            content = _block_content(raw_block)
            if not content:
                continue
            raw_type = str(raw_block.get("type", "text")).strip().lower()
            block_type = raw_type if raw_type in ALLOWED_BLOCK_TYPES else "text"
            blocks.append({"type": block_type, "content": content})

        pages.append(
            {
                "file": page_file,
                "page": page_number,
                "blocks": blocks,
            }
        )

    return {"pages": pages}


def stats(document: dict[str, Any]) -> dict[str, int]:
    pages = document.get("pages", [])
    blocks = [
        block
        for page in pages
        if isinstance(page, dict)
        for block in page.get("blocks", [])
        if isinstance(block, dict)
    ]
    return {
        "pages": len(pages),
        "blocks": len(blocks),
        "tables": sum(block.get("type") == "table" for block in blocks),
        "content_chars": sum(len(str(block.get("content", ""))) for block in blocks),
    }


def validate(original: dict[str, Any], cleaned: dict[str, Any]) -> None:
    """Ensure the result has the exact schema and unchanged textual content."""

    expected = clean_document(original)
    if cleaned != expected:
        raise ValueError("Очищенный JSON не совпадает с каноническим результатом.")
    if set(cleaned) != {"pages"}:
        raise ValueError("В корне очищенного JSON разрешено только pages.")
    for page in cleaned["pages"]:
        if set(page) != {"file", "page", "blocks"}:
            raise ValueError("Страница содержит лишние поля.")
        for block in page["blocks"]:
            if set(block) != {"type", "content"}:
                raise ValueError("Блок содержит лишние поля.")


def save_json(path: Path, data: dict[str, Any], pretty: bool = False) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        json.dump(
            data,
            temporary,
            ensure_ascii=False,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Исходный document.json")
    parser.add_argument("output", type=Path, help="Очищенный JSON")
    parser.add_argument("--pretty", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    with args.input.expanduser().open("r", encoding="utf-8-sig") as source:
        original = json.load(source)
    cleaned = clean_document(original)
    validate(original, cleaned)
    save_json(args.output, cleaned, pretty=args.pretty)

    old_size = args.input.expanduser().stat().st_size
    new_size = args.output.expanduser().stat().st_size
    reduction = 100 * (1 - new_size / old_size) if old_size else 0.0
    print(f"Готово: {args.output.expanduser().resolve()}")
    print(f"Размер: {old_size:,} -> {new_size:,} байт ({reduction:.1f}% меньше)")
    print(stats(cleaned))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

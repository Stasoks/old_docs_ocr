#!/usr/bin/env python3
"""Streamlit UI for the hybrid OCR -> well extraction -> PDF workflow."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

import hybrid_htr_table_pipeline as pipeline
from json_to_well_pdf import render_pdf


SUPPORTED_SUFFIXES = {".pdf", ".tif", ".tiff", ".jpg", ".jpeg", ".zip"}


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class RuntimeConfig:
    detector_weights: str | None
    detector_cpu: bool
    ocr_model: str
    ocr_device: str
    ocr_dtype: str
    lmstudio_url: str
    qwen_model: str
    extraction_model: str
    extraction_prompt: str

    @classmethod
    def from_environment(cls) -> "RuntimeConfig":
        detector_weights = os.environ.get("HTR_DETECTOR_WEIGHTS", "").strip()
        return cls(
            detector_weights=detector_weights or None,
            detector_cpu=_env_bool("HTR_DETECTOR_CPU"),
            ocr_model=os.environ.get("HTR_OCR_MODEL", pipeline.DEFAULT_OCR_MODEL),
            ocr_device=os.environ.get("HTR_OCR_DEVICE", "auto"),
            ocr_dtype=os.environ.get("HTR_OCR_DTYPE", "auto"),
            lmstudio_url=os.environ.get("LMSTUDIO_URL", pipeline.DEFAULT_LMSTUDIO_URL),
            qwen_model=os.environ.get("QWEN_VL_MODEL", pipeline.DEFAULT_QWEN_MODEL),
            extraction_model=os.environ.get(
                "QWEN_EXTRACTION_MODEL", pipeline.DEFAULT_EXTRACTION_MODEL
            ),
            extraction_prompt=os.environ.get(
                "WELL_EXTRACTION_PROMPT", str(pipeline.DEFAULT_EXTRACTION_PROMPT)
            ),
        )


@dataclass
class ModelRuntime:
    pool: pipeline.CudaModelPool
    base_args: argparse.Namespace
    lock: threading.Lock


def _base_args(config: RuntimeConfig) -> argparse.Namespace:
    # --keep-alive permits construction without a positional input. Streamlit
    # itself owns the long-lived process and supplies one input per job.
    args = pipeline.build_parser().parse_args(["--keep-alive"])
    args.detector_weights = config.detector_weights
    args.detector_cpu = config.detector_cpu
    args.ocr_model = config.ocr_model
    args.ocr_device = config.ocr_device
    args.ocr_dtype = config.ocr_dtype
    args.lmstudio_url = config.lmstudio_url
    args.qwen_model = config.qwen_model
    args.extraction_model = config.extraction_model
    args.extraction_prompt = Path(config.extraction_prompt)
    return args


@st.cache_resource(show_spinner=False)
def get_runtime(config: RuntimeConfig) -> ModelRuntime:
    args = _base_args(config)
    return ModelRuntime(
        pool=pipeline.CudaModelPool(args),
        base_args=args,
        lock=threading.Lock(),
    )


def _safe_upload_name(name: str) -> str:
    basename = name.replace("\\", "/").rsplit("/", 1)[-1]
    path = Path(basename)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError("Поддерживаются PDF, TIFF, JPEG и ZIP.")
    stem = re.sub(r"[^\w .()-]+", "_", path.stem, flags=re.UNICODE).strip(" ._")
    return f"{stem or 'document'}{suffix}"


def _job_args(
    runtime: ModelRuntime,
    input_path: Path,
    output_dir: Path,
) -> argparse.Namespace:
    args = argparse.Namespace(**vars(runtime.base_args))
    args.inputs = [str(input_path)]
    args.output_dir = output_dir
    args.keep_alive = False
    args.overwrite = True
    args.skip_detection = False
    args.continue_on_error = True
    pipeline.validate_args(args)
    return args


def _json_bytes(data: dict[str, Any]) -> bytes:
    return (
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")


def process_upload(
    uploaded_name: str,
    uploaded_bytes: bytes,
    config: RuntimeConfig,
    status: Any,
) -> dict[str, Any]:
    safe_name = _safe_upload_name(uploaded_name)
    runtime = get_runtime(config)
    logs = io.StringIO()
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="hybrid_htr_web_") as temporary:
        root = Path(temporary)
        input_path = root / safe_name
        input_path.write_bytes(uploaded_bytes)
        output_parent = root / "output"
        args = _job_args(runtime, input_path, output_parent)

        with runtime.lock:
            with contextlib.redirect_stdout(logs), contextlib.redirect_stderr(logs):
                status.update(label="Загрузка OCR-моделей…", state="running")
                runtime.pool.get_trocr()
                runtime.pool.get_detector()
                status.update(label="Распознавание документа…", state="running")
                result_dir = pipeline.process_input(input_path, runtime.pool, args)

        cleaned_path = result_dir / "cleaned_document.json"
        extraction_path = result_dir / "well_extraction.json"
        if not cleaned_path.is_file():
            raise RuntimeError("Пайплайн не создал cleaned_document.json.")
        if not extraction_path.is_file():
            raise RuntimeError("Пайплайн не создал well_extraction.json.")

        cleaned = json.loads(cleaned_path.read_text(encoding="utf-8"))
        extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
        status.update(label="Формирование PDF-отчёта…", state="running")
        pdf_bytes = render_pdf([("well_extraction.json", extraction)])

    report_stem = Path(safe_name).stem
    return {
        "source_name": safe_name,
        "pdf_name": f"{report_stem}_отчёт.pdf",
        "pdf_bytes": pdf_bytes,
        "cleaned_name": f"{report_stem}_cleaned.json",
        "cleaned_bytes": _json_bytes(cleaned),
        "extraction_name": f"{report_stem}_parameters.json",
        "extraction_bytes": _json_bytes(extraction),
        "records": len(extraction.get("records", [])),
        "conflicts": len(extraction.get("conflicts", [])),
        "missing": len(extraction.get("missing_parameters", [])),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "logs": logs.getvalue(),
    }


def _render_configuration(config: RuntimeConfig) -> None:
    with st.sidebar:
        st.header("Конфигурация")
        st.caption("Задаётся администратором через environment variables.")
        st.code(
            "\n".join(
                [
                    f"RF-DETR: {config.detector_weights or 'Hugging Face cache'}",
                    f"TrOCR: {config.ocr_model}",
                    f"Qwen VL: {config.qwen_model}",
                    f"Qwen extraction: {config.extraction_model}",
                    f"API: {config.lmstudio_url}",
                ]
            ),
            language=None,
        )


def main() -> None:
    st.set_page_config(
        page_title="OCR архивных документов",
        page_icon="📄",
        layout="centered",
    )
    config = RuntimeConfig.from_environment()
    _render_configuration(config)

    st.title("OCR архивных документов")
    st.write(
        "Загрузите PDF, TIFF, JPEG или ZIP. Сервис распознает документ, "
        "извлечёт параметры скважины и сформирует PDF-отчёт."
    )

    uploaded = st.file_uploader(
        "Выберите файл",
        type=["pdf", "tif", "tiff", "jpg", "jpeg", "zip"],
        accept_multiple_files=False,
    )
    process_clicked = st.button(
        "Распознать и создать отчёт",
        type="primary",
        disabled=uploaded is None,
        use_container_width=True,
    )

    if process_clicked and uploaded is not None:
        st.session_state.pop("job_result", None)
        try:
            with st.status("Подготовка…", expanded=True) as status:
                result = process_upload(
                    uploaded.name,
                    uploaded.getvalue(),
                    config,
                    status,
                )
                status.update(label="Отчёт готов", state="complete")
            st.session_state["job_result"] = result
        except Exception as error:
            st.error(f"Обработка не завершена: {type(error).__name__}: {error}")
            st.exception(error)

    result = st.session_state.get("job_result")
    if not result:
        return

    st.success(f"Обработан {result['source_name']} за {result['elapsed_seconds']} с.")
    columns = st.columns(3)
    columns[0].metric("Параметров", result["records"])
    columns[1].metric("Конфликтов", result["conflicts"])
    columns[2].metric("Не найдено", result["missing"])

    st.download_button(
        "Скачать PDF-отчёт",
        data=result["pdf_bytes"],
        file_name=result["pdf_name"],
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )
    with st.expander("Дополнительные файлы"):
        st.download_button(
            "Скачать очищенный OCR JSON",
            data=result["cleaned_bytes"],
            file_name=result["cleaned_name"],
            mime="application/json",
            use_container_width=True,
        )
        st.download_button(
            "Скачать JSON с параметрами",
            data=result["extraction_bytes"],
            file_name=result["extraction_name"],
            mime="application/json",
            use_container_width=True,
        )
        if result["logs"]:
            st.text_area("Журнал обработки", result["logs"], height=240)


if __name__ == "__main__":
    main()

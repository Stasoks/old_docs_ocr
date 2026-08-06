#!/usr/bin/env python3
"""Convert an RF-DETR PTH/PT checkpoint to native safetensors weights."""

from __future__ import annotations

import argparse
from pathlib import Path

from historical_russian_htr_pipeline import (
    convert_rfdetr_checkpoint_to_safetensors,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Trusted RF-DETR .pth/.pt file")
    parser.add_argument(
        "destination",
        type=Path,
        nargs="?",
        help="Output .safetensors; default: next to source",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip exact tensor-by-tensor verification after writing.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = args.source.expanduser().resolve()
    destination = (
        args.destination.expanduser().resolve()
        if args.destination
        else source.with_suffix(".safetensors")
    )
    result = convert_rfdetr_checkpoint_to_safetensors(
        source,
        destination,
        overwrite=args.overwrite,
        verify=not args.no_verify,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

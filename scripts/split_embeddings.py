#!/usr/bin/env python3
"""Split JSON sample files into NDJSON documents and embeddings."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract embeddings from sample JSON files and emit paired "
            ".vec.ndjson and .doc.ndjson files."
        )
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="samples",
        help="Directory containing source *.json files (default: samples).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated NDJSON files (default: same as input).",
    )
    return parser.parse_args()


def extract_embeddings(record: Dict[str, Any]) -> Tuple[Dict[str, Any], Any]:
    """Return a copy of the record without embeddings and the extracted vector."""
    doc_record = deepcopy(record)
    embeddings = None

    metadata = doc_record.get("metadata")
    if isinstance(metadata, dict):
        if "embedding" in metadata:
            embeddings = metadata.pop("embedding")
        elif "embeddings" in metadata:
            embeddings = metadata.pop("embeddings")

    if embeddings is None:
        if "embedding" in doc_record:
            embeddings = doc_record.pop("embedding")
        elif "embeddings" in doc_record:
            embeddings = doc_record.pop("embeddings")

    if embeddings is None:
        raise ValueError("record is missing an embedding field")

    return doc_record, embeddings


def iter_source_files(input_dir: Path) -> Iterable[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"input directory not found: {input_dir}")
    yield from sorted(p for p in input_dir.glob("*.json") if p.is_file())


def process_file(source_path: Path, output_dir: Path) -> None:
    with source_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if not isinstance(payload, list):
        raise ValueError(f"{source_path} does not contain a JSON array")

    vec_path = output_dir / f"{source_path.stem}.vec.ndjson"
    doc_path = output_dir / f"{source_path.stem}.doc.ndjson"

    with vec_path.open("w", encoding="utf-8") as vec_fh, doc_path.open(
        "w", encoding="utf-8"
    ) as doc_fh:
        for record in payload:
            if not isinstance(record, dict):
                raise ValueError("expected every record to be a JSON object")

            record_id = record.get("id")
            if record_id is None:
                raise ValueError("record is missing an id field")

            doc_record, embeddings = extract_embeddings(record)

            vec_entry = {"id": record_id, "embeddings": embeddings}
            vec_fh.write(json.dumps(vec_entry) + "\n")
            doc_fh.write(json.dumps(doc_record) + "\n")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir else input_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for source_file in iter_source_files(input_dir):
        process_file(source_file, output_dir)


if __name__ == "__main__":
    main()

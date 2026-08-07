"""Bounded JSON, JSONL, CSV and optional Parquet exporters."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _row(item: Any) -> dict[str, Any]:
    if isinstance(item, BaseModel):
        return item.model_dump(mode="json")
    if isinstance(item, dict):
        return item
    raise TypeError(f"Cannot export {type(item).__name__}; expected a model or mapping")


def _batches(records: Iterable[Any], size: int) -> Iterator[list[dict[str, Any]]]:
    if size <= 0:
        raise ValueError("batch_size must be positive")
    batch: list[dict[str, Any]] = []
    for record in records:
        batch.append(_row(record))
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def export_records(
    records: Iterable[Any],
    output: str | Path,
    *,
    format_name: str,
    batch_size: int = 1000,
) -> tuple[Path, ...]:
    """Write bounded, physical shards and atomically publish the output directory."""

    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent))
    created: list[Path] = []
    normalized_format = format_name.casefold()
    if normalized_format not in {"json", "jsonl", "csv", "parquet"}:
        shutil.rmtree(staging)
        raise ValueError(f"Unsupported export format: {format_name}")
    try:
        for index, batch in enumerate(_batches(records, batch_size)):
            suffix = "parquet" if normalized_format == "parquet" else normalized_format
            target = staging / f"part-{index:05d}.{suffix}"
            if normalized_format == "json":
                target.write_text(
                    json.dumps(batch, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8",
                )
            elif normalized_format == "jsonl":
                with target.open("w", encoding="utf-8", newline="\n") as handle:
                    for row in batch:
                        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            elif normalized_format == "csv":
                fields = sorted({str(key) for row in batch for key in row})
                with target.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                    writer.writeheader()
                    for row in batch:
                        writer.writerow(
                            {
                                key: json.dumps(value, ensure_ascii=False, default=str)
                                if isinstance(value, (dict, list, tuple))
                                else value
                                for key, value in row.items()
                            }
                        )
            else:
                try:
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                except ImportError as exc:
                    raise RuntimeError(
                        "Install congreso-open-data[parquet] for Parquet export"
                    ) from exc
                table = pa.Table.from_pylist(batch)
                pq.write_table(table, target, compression="zstd")
            created.append(target)
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite existing export directory: {output_path}")
        os.replace(staging, output_path)
        return tuple(output_path / path.name for path in created)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

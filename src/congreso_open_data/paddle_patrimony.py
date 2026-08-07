"""Provenance-preserving normalization of PaddleOCR document results.

Paddle is deliberately an evidence producer here, not an authority.  This
module converts the native PP-StructureV3 JSON into bounded JSONL-friendly
records while retaining the original labels, coordinates and raw table HTML.
Excluded layout objects (stamps, page numbers and footnotes) are retained with
``field_eligible=False`` so a later reviewer can audit every decision.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PADDLE_PIPELINE_VERSION = "ppstructurev3-normalizer-v1"
FIELD_ELIGIBLE_BLOCKS = frozenset(
    {"table", "text", "figure_title", "paragraph_title", "abstract", "content"}
)
NON_FIELD_BLOCKS = frozenset({"aside_text", "number", "footnote", "header", "footer"})
_AMOUNT_RE = re.compile(r"(?<![\d.,])(?:\d{1,3}(?:\.\d{3})+|\d+)(?:[,.]\d{1,2})?(?:\s*€)?")
_PERCENT_RE = re.compile(r"\b\d{1,3}(?:[,.]\d{1,4})?\s*%")
# OCR often inserts a space after the decimal separator (``4446, 23``).
# Override the legacy expression while retaining it above for provenance.
_AMOUNT_RE = re.compile(
    r"(?<![\d.,])(?:\d{1,3}(?:\.\d{3})+|\d+)(?:[,.]\s*\d{1,4})?(?:\s*(?:\u20ac|EUR|euros?))?",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(r"(?:EUR|euros?)", re.IGNORECASE)
_FINANCIAL_CONTEXT_RE = re.compile(
    r"(?:renta|ingreso|percepci|sueldo|salario|honorario|dividendo|inter[eé]s|dep[oó]sito|"
    r"cuenta|ahorro|activo|vivienda|inmueble|veh[ií]culo|acciones?|participaci[oó]n|"
    r"deuda|pr[eé]stamo|hipoteca|irpf|pensi[oó]n|seguro|valor|importe|euros?|%|porcentaje)",
    re.IGNORECASE,
)
_NON_FINANCIAL_CONTEXT_RE = re.compile(
    r"(?:fecha|acuerdo plenario|nacimiento|ejercicio|a[nñ]o|mes|trimestre|art[ií]culo|"
    r"reglamento|legislatura|libro|p[aá]gina|hect[aá]rea|kil[oó]metro)",
    re.IGNORECASE,
)
_MOJIBAKE_MARKERS = ("Ã", "Â", "â", "ð", "�")


def repair_mojibake(value: str) -> str:
    """Repair only demonstrable UTF-8-as-Latin-1 text corruption.

    Paddle outputs and the legacy manifest contain strings such as
    ``CÃ³rdoba`` although the source pixels contain ``Córdoba``.  Keep the
    original in ``text_raw`` and apply the round-trip only when it strictly
    reduces known corruption markers; otherwise return the input unchanged.
    """
    current = str(value)

    def score(text: str) -> int:
        return sum(text.count(marker) for marker in _MOJIBAKE_MARKERS)

    for _ in range(2):
        if score(current) == 0:
            break
        try:
            candidate = current.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if candidate == current or score(candidate) >= score(current):
            break
        current = candidate
    return current


def _bbox(values: Iterable[Any]) -> list[float]:
    values = list(values)
    if len(values) != 4:
        raise ValueError(f"expected four bbox coordinates, got {values!r}")
    result = [float(value) for value in values]
    if result[2] < result[0] or result[3] < result[1]:
        raise ValueError(f"inverted bbox: {result!r}")
    return result


def _normal_bbox(box: list[float], width: int, height: int) -> list[float]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return [
        round(box[0] / width, 8),
        round(box[1] / height, 8),
        round(box[2] / width, 8),
        round(box[3] / height, 8),
    ]


def _stable_id(*parts: Any) -> str:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def normalize_paddle_result(
    raw: dict[str, Any],
    *,
    source_sha256: str,
    page_number: int,
    image_path: str | Path,
    model_versions: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Normalize one PP-StructureV3 result without inventing text.

    The function is pure apart from reading the input image hash.  It accepts
    the JSON emitted by ``save_to_json`` and validates that every OCR box is
    inside the declared page dimensions.  ``ocr_lines`` are the direct OCR
    observations and ``blocks``/``tables`` preserve the structure proposals.
    """

    if not source_sha256 or len(source_sha256) != 64:
        raise ValueError("source_sha256 must be a SHA-256 hex digest")
    width = int(raw.get("width") or 0)
    height = int(raw.get("height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError("Paddle result has invalid width/height")
    page_number = int(page_number)
    if page_number < 1:
        raise ValueError("page_number must be positive")
    image_path = Path(image_path).resolve()
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    overall = raw.get("overall_ocr_res") or {}
    texts = list(overall.get("rec_texts") or [])
    scores = list(overall.get("rec_scores") or [])
    boxes = list(overall.get("rec_boxes") or [])
    if not (len(texts) == len(scores) == len(boxes)):
        raise ValueError("Paddle OCR text/score/box arrays have different lengths")

    ocr_lines: list[dict[str, Any]] = []
    for index, (text, score, box) in enumerate(zip(texts, scores, boxes)):
        raw_value = str(text or "").strip()
        value = repair_mojibake(raw_value)
        if not value:
            continue
        pixel_box = _bbox(box)
        if pixel_box[2] > width + 2 or pixel_box[3] > height + 2:
            raise ValueError(f"OCR bbox outside page at index {index}: {pixel_box!r}")
        ocr_lines.append(
            {
                "observation_id": _stable_id(
                    source_sha256, page_number, "ocr", index, value, pixel_box
                ),
                "source_sha256": source_sha256,
                "page_number": page_number,
                "image_path": str(image_path),
                "image_sha256": image_sha256,
                "text": value,
                "text_raw": raw_value,
                "score": round(float(score or 0.0), 8),
                "bbox_px": pixel_box,
                "bbox_norm": _normal_bbox(pixel_box, width, height),
                "field_eligible": True,
                "method": "paddle_ppstructurev3_overall_ocr",
            }
        )

    blocks: list[dict[str, Any]] = []
    for index, block in enumerate(raw.get("parsing_res_list") or []):
        label = str(block.get("block_label") or "unknown")
        pixel_box = _bbox(block.get("block_bbox") or [])
        if pixel_box[2] > width + 2 or pixel_box[3] > height + 2:
            raise ValueError(f"layout bbox outside page at index {index}: {pixel_box!r}")
        blocks.append(
            {
                "block_id": _stable_id(
                    source_sha256, page_number, "block", index, label, pixel_box
                ),
                "source_sha256": source_sha256,
                "page_number": page_number,
                "image_path": str(image_path),
                "image_sha256": image_sha256,
                "block_index": index,
                "block_label": label,
                "block_content": repair_mojibake(str(block.get("block_content") or "")),
                "block_content_raw": str(block.get("block_content") or ""),
                "block_order": block.get("block_order"),
                "bbox_px": pixel_box,
                "bbox_norm": _normal_bbox(pixel_box, width, height),
                "field_eligible": label in FIELD_ELIGIBLE_BLOCKS and label not in NON_FIELD_BLOCKS,
                "method": "paddle_ppstructurev3_layout",
            }
        )

    # Attach OCR observations to the smallest containing layout block.  This
    # is deliberately geometric: a stamp or page number may have perfectly
    # plausible OCR text, but it must stay outside field extraction.
    for line in ocr_lines:
        x0, y0, x1, y1 = line["bbox_px"]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        containing = [
            block
            for block in blocks
            if block["bbox_px"][0] <= cx <= block["bbox_px"][2]
            and block["bbox_px"][1] <= cy <= block["bbox_px"][3]
        ]
        if containing:
            chosen = min(
                containing,
                key=lambda block: (
                    (block["bbox_px"][2] - block["bbox_px"][0])
                    * (block["bbox_px"][3] - block["bbox_px"][1])
                ),
            )
            line["layout_block_id"] = chosen["block_id"]
            line["layout_block_label"] = chosen["block_label"]
            line["field_eligible"] = bool(chosen["field_eligible"])
        else:
            line["layout_block_id"] = None
            line["layout_block_label"] = None
            # Unclassified text is retained, but is not eligible for automatic
            # field extraction until a template assigns it to a region.
            line["field_eligible"] = False

    tables: list[dict[str, Any]] = []
    for index, table in enumerate(raw.get("table_res_list") or []):
        cells: list[dict[str, Any]] = []
        for cell_index, cell in enumerate(table.get("cell_box_list") or []):
            pixel_box = _bbox(cell)
            bbox_valid = (
                pixel_box[0] >= -2
                and pixel_box[1] >= -2
                and pixel_box[2] <= width + 2
                and pixel_box[3] <= height + 2
            )
            cells.append(
                {
                    "cell_index": cell_index,
                    "bbox_px": pixel_box,
                    "bbox_norm": _normal_bbox(pixel_box, width, height),
                    "bbox_valid": bbox_valid,
                }
            )
        tables.append(
            {
                "table_id": _stable_id(source_sha256, page_number, "table", index),
                "source_sha256": source_sha256,
                "page_number": page_number,
                "image_path": str(image_path),
                "image_sha256": image_sha256,
                "table_index": index,
                "pred_html": repair_mojibake(str(table.get("pred_html") or "")),
                "pred_html_raw": str(table.get("pred_html") or ""),
                "cells": cells,
                "method": "paddle_ppstructurev3_table",
            }
        )

    # Attach each OCR line to a table cell when the center falls inside one.
    # Cell association is evidence for a template parser; it is not a value
    # decision and therefore remains nullable.
    for line in ocr_lines:
        x0, y0, x1, y1 = line["bbox_px"]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for table in tables:
            for cell in table["cells"]:
                bx = cell["bbox_px"]
                if bx[0] <= cx <= bx[2] and bx[1] <= cy <= bx[3]:
                    matches.append((table, cell))
        if matches:
            table, cell = min(
                matches,
                key=lambda pair: (
                    (pair[1]["bbox_px"][2] - pair[1]["bbox_px"][0])
                    * (pair[1]["bbox_px"][3] - pair[1]["bbox_px"][1])
                ),
            )
            line["table_id"] = table["table_id"]
            line["table_index"] = table["table_index"]
            line["cell_index"] = cell["cell_index"]
            line["cell_bbox_valid"] = bool(cell.get("bbox_valid"))
        else:
            line["table_id"] = None
            line["table_index"] = None
            line["cell_index"] = None
            line["cell_bbox_valid"] = None

    settings = raw.get("model_settings") or {}
    return {
        "pipeline_version": PADDLE_PIPELINE_VERSION,
        "source_sha256": source_sha256,
        "page_number": page_number,
        "image_path": str(image_path),
        "image_sha256": image_sha256,
        "width": width,
        "height": height,
        "model_settings": settings,
        "model_versions": model_versions or {},
        "ocr_lines": ocr_lines,
        "blocks": blocks,
        "tables": tables,
        "excluded_block_labels": sorted(NON_FIELD_BLOCKS),
        "raw_result_keys": sorted(raw.keys()),
    }


def paddle_numeric_candidates(
    normalized: dict[str, Any], *, min_score: float = 0.95
) -> list[dict[str, Any]]:
    """Return review-gated numeric candidates from eligible table/text boxes.

    The function only transcribes characters already returned by Paddle. It
    never parses a value into a number and never promotes a candidate. Values
    below the score gate, outside a template table cell, or with no clear
    financial shape are retained as ``quarantined`` diagnostics. Even a
    high-scoring candidate remains ``review_required`` until a template and
    visual adjudication identify the field.
    """

    if not 0.0 < min_score <= 1.0:
        raise ValueError("min_score must be in (0, 1]")
    candidates: list[dict[str, Any]] = []
    for line in normalized.get("ocr_lines") or []:
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        matches = list(_AMOUNT_RE.finditer(text)) + list(_PERCENT_RE.finditer(text))
        if not matches:
            continue
        score = float(line.get("score") or 0.0)
        status = "review_required"
        reason = "requires_template_and_visual_review"
        if not line.get("field_eligible"):
            status, reason = "quarantined", "layout_block_excluded"
        elif score < min_score:
            status, reason = "quarantined", "ocr_score_below_gate"
        elif line.get("cell_index") is None:
            status, reason = "quarantined", "no_table_cell_assignment"
        elif line.get("cell_bbox_valid") is False:
            status, reason = "quarantined", "invalid_table_cell_bbox"
        for ordinal, match in enumerate(matches):
            raw = match.group(0).strip()
            is_percent = bool(_PERCENT_RE.fullmatch(raw))
            explicit_currency = bool(_CURRENCY_RE.search(raw)) or "\u20ac" in raw
            decimal_shape = bool(re.search(r"\d[,.]\d{2}(?:\D|$)", raw))
            financial_context = bool(_FINANCIAL_CONTEXT_RE.search(text))
            non_financial_context = bool(_NON_FINANCIAL_CONTEXT_RE.search(text))
            prediction_class = "percentage" if is_percent else "financial_amount"
            candidate_status = status
            review_reason = reason
            if (
                not is_percent
                and not explicit_currency
                and (non_financial_context or (not decimal_shape and not financial_context))
            ):
                # Bare years, dates, article numbers and footnote markers are
                # retained as evidence, but must not masquerade as financial
                # predictions. They are never auto-promotable.
                prediction_class = "non_financial_numeric"
                candidate_status = "quarantined"
                review_reason = "non_financial_numeric_context"
            candidates.append(
                {
                    "candidate_id": _stable_id(
                        normalized["source_sha256"],
                        normalized["page_number"],
                        line["observation_id"],
                        ordinal,
                        raw,
                    ),
                    "source_sha256": normalized["source_sha256"],
                    "page_number": normalized["page_number"],
                    "image_path": normalized["image_path"],
                    "image_sha256": normalized["image_sha256"],
                    "observation_id": line["observation_id"],
                    "raw_text": raw,
                    "context_text": text,
                    "prediction_class": prediction_class,
                    "bbox_px": line["bbox_px"],
                    "bbox_norm": line["bbox_norm"],
                    "layout_block_label": line.get("layout_block_label"),
                    "table_id": line.get("table_id"),
                    "table_index": line.get("table_index"),
                    "cell_index": line.get("cell_index"),
                    "cell_bbox_valid": line.get("cell_bbox_valid"),
                    "ocr_score": score,
                    "candidate_status": candidate_status,
                    "review_reason": review_reason,
                    "method": "paddle_ppstructurev3_numeric_gate",
                }
            )
    return candidates


def json_line(record: dict[str, Any]) -> str:
    """Stable UTF-8 serialization for append-only audit logs."""

    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

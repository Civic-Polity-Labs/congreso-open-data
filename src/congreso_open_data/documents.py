from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any

from congreso_open_data.extractors.patrimony_nlp import (
    classify_patrimony_line,
    has_patrimony_evidence_signal,
    is_patrimony_boilerplate,
)
from congreso_open_data.normalization import stable_id
from congreso_open_data.rapidocr_compat import (
    create_rapidocr_engine,
    rapidocr_lines,
    rapidocr_runtime_kwargs,
)
from congreso_open_data.runtime import (
    desired_ocr_onnx_providers,
    model_name_from_contract,
    ocr_runtime_config,
    require_model_contract,
    review_threshold_for_model,
    selected_ocr_onnx_providers,
)

_RAPID_OCR_ENGINE: Any | None = None
_RAPID_OCR_ENGINE_LOCK = Lock()
INTERVENTION_PDF_LAYOUT_VERSION = "repeated_geometric_margins_reversible_v6_five_line_issue_header"

_OFFICIAL_RUNNING_HEADER_COMPONENTS = {
    "cortes generales",
    "congreso de los diputados",
    "diario de sesiones del",
    "diario de sesiones de las",
    "diario de sesiones del congreso de los diputados",
    "comisiones",
    "comisiones mixtas",
    "pleno y diputación permanente",
}
_OFFICIAL_ISSUE_HEADER_RE = re.compile(
    r"^N(?:ú|u)m\.\s+\d+\s+\d{1,2}\s+de\s+[^\r\n]+\s+de\s+\d{4}"
    r"\s+P(?:á|a)g\.\s+\d+$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PdfTextExtraction:
    page_texts: list[str]
    page_methods: list[str]
    page_confidences: list[float]
    status: str
    error: str | None = None
    page_diagnostics: list[str] | None = None


def pdf_document_text_row(
    content: bytes,
    *,
    source_url: str,
    source_sha256: str,
    snapshot_date: str,
    use_ocr: bool = False,
    strict_validation: bool = False,
) -> dict[str, Any]:
    try:
        extraction = _extract_pdf_text(
            content,
            use_ocr=use_ocr,
            strict_validation=strict_validation,
        )
        text = "\n\n".join(part for part in extraction.page_texts if part)
        extraction_method = _document_extraction_method(extraction, use_ocr=use_ocr)
        return {
            "document_sha256": source_sha256,
            "document_kind": _document_kind_from_url(source_url),
            "source_url": source_url,
            "mime_type": "application/pdf",
            "page_count": len(extraction.page_texts),
            "text": text,
            "page_texts": extraction.page_texts,
            "extraction_method": extraction_method,
            "model_name": _page_model_name(extraction_method),
            "extraction_status": extraction.status,
            "extraction_error": extraction.error,
            "snapshot_date": snapshot_date,
            "_page_methods": extraction.page_methods,
            "_page_confidences": extraction.page_confidences,
            "_page_diagnostics": extraction.page_diagnostics or [],
        }
    except Exception as exc:
        return {
            "document_sha256": source_sha256,
            "document_kind": _document_kind_from_url(source_url),
            "source_url": source_url,
            "mime_type": "application/pdf",
            "page_count": None,
            "text": None,
            "page_texts": [],
            "extraction_method": "pypdf_text",
            "model_name": _page_model_name("pypdf_text"),
            "extraction_status": "error",
            "extraction_error": f"{type(exc).__name__}: {exc}",
            "snapshot_date": snapshot_date,
            "page_diagnostics": [],
        }


def pdf_document_text_row_from_pages(
    page_texts: list[str],
    *,
    source_url: str,
    source_sha256: str,
    snapshot_date: str,
    extraction_method: str = "pypdf_text",
    extraction_error: str | None = None,
) -> dict[str, Any]:
    """Build the public text-layer row from pages already parsed from a PDF."""

    text = "\n\n".join(part for part in page_texts if part)
    return {
        "document_sha256": source_sha256,
        "document_kind": _document_kind_from_url(source_url),
        "source_url": source_url,
        "mime_type": "application/pdf",
        "page_count": len(page_texts),
        "text": text,
        "page_texts": page_texts,
        "page_methods": [extraction_method] * len(page_texts),
        "page_confidences": [
            _text_layer_confidence(extraction_method, bool(page.strip())) for page in page_texts
        ],
        "page_diagnostics": [
            _page_diagnostic_json(
                page_number=index,
                selected_method=extraction_method,
                confidence=_text_layer_confidence(extraction_method, bool(page.strip())),
                agreement=None,
                issues=[] if page.strip() else ["empty_page"],
            )
            for index, page in enumerate(page_texts, start=1)
        ],
        "extraction_method": extraction_method,
        "model_name": _page_model_name(extraction_method),
        "extraction_status": "ok" if text.strip() else "empty_text",
        "extraction_error": extraction_error,
        "snapshot_date": snapshot_date,
    }


def pdf_document_normalized_rows(
    content: bytes,
    *,
    source_url: str,
    source_sha256: str,
    snapshot_date: str,
    use_ocr: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    text_row = pdf_document_text_row(
        content,
        source_url=source_url,
        source_sha256=source_sha256,
        snapshot_date=snapshot_date,
        use_ocr=use_ocr,
    )
    page_rows = _page_rows(text_row)
    extraction_rows = _extraction_rows(
        pages=page_rows,
        source_url=source_url,
        source_sha256=source_sha256,
        snapshot_date=snapshot_date,
        document_kind=text_row["document_kind"],
    )
    return {
        "document_texts": [_public_text_row(text_row)],
        "document_pages": page_rows,
        "document_tables": _table_rows(page_rows),
        "document_entities": _entity_rows(
            pages=page_rows,
            source_url=source_url,
            source_sha256=source_sha256,
            snapshot_date=snapshot_date,
            document_kind=text_row["document_kind"],
        ),
        "document_extractions": extraction_rows,
    }


def deputy_document_categories() -> dict[str, tuple[str, ...]]:
    return {
        "income": INCOME_CATEGORIES,
        "asset": ASSET_CATEGORIES,
        "liability": LIABILITY_CATEGORIES,
        "position": POSITION_CATEGORIES,
        "company_interest": COMPANY_INTEREST_CATEGORIES,
        "entity": ENTITY_TYPES,
        "document_kind": DOCUMENT_KINDS,
        "declaration_phase": DECLARATION_PHASES,
    }


def _document_kind_from_url(url: str) -> str:
    lower = url.lower()
    if "/docbienes/" in lower:
        return "assets_income"
    if "/docacteco/" in lower:
        return "economic_interests"
    if "/docinte/" in lower or "registro_intereses" in lower:
        return "activities"
    if "/publicaciones/" in lower or "bocg" in lower:
        return "publication"
    return "pdf"


def _public_text_row(text_row: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in text_row.items() if not key.startswith("_")}
    if "_page_methods" in text_row:
        public["page_methods"] = text_row["_page_methods"]
    if "_page_confidences" in text_row:
        public["page_confidences"] = text_row["_page_confidences"]
    if "_page_diagnostics" in text_row:
        public["page_diagnostics"] = text_row["_page_diagnostics"]
    return public


DOCUMENT_KINDS = (
    "activities",
    "assets_income",
    "economic_interests",
    "publication",
    "pdf",
)
DECLARATION_PHASES = ("initial", "modification", "final", "unknown")
INCOME_CATEGORIES = (
    "public_salary",
    "private_activity",
    "rental_income",
    "dividends",
    "interest",
    "pension",
    "teaching",
    "speaking",
    "royalties",
    "other_income",
)
ASSET_CATEGORIES = (
    "real_estate",
    "bank_account",
    "deposit",
    "shares",
    "fund",
    "pension_plan",
    "vehicle",
    "company_interest",
    "insurance",
    "crypto",
    "other_asset",
)
LIABILITY_CATEGORIES = ("mortgage", "loan", "credit", "guarantee", "other_liability")
POSITION_CATEGORIES = (
    "public_office",
    "party",
    "private_company",
    "foundation",
    "university",
    "board",
    "advisory",
    "media",
    "association",
    "other_position",
)
COMPANY_INTEREST_CATEGORIES = (
    "shareholding",
    "director",
    "board_member",
    "advisor",
    "employee",
    "other_company_interest",
)
ENTITY_TYPES = (
    "amount",
    "date",
    "percentage",
    "iban_partial",
    "company",
    "location",
    "registry_id",
)

AMOUNT_RE = re.compile(
    r"(?<!\d)(\d+(?:[.\s]\d{3})*,\d{1,4})\s*(?:\u20ac|eur(?:os?)?)?",
    re.I,
)
DATE_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b")
PERCENT_RE = re.compile(r"\b(\d{1,3}(?:,\d{1,4})?)\s*%")
IBAN_RE = re.compile(r"\bES\d{2}(?:[\s*Xx]{2,}\d*){1,6}\b")
COMPANY_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&.,' -]{2,80}\s+(?:S\.?L\.?|S\.?A\.?|SLU|SAU|S\.?Coop\.?))\b"
)
REGISTRY_RE = re.compile(r"\b(?:CIF|NIF|DNI)\s*[:.-]?\s*([A-Z0-9*X-]{4,16})\b", re.I)


def _extract_pdf_page_texts(content: bytes) -> list[str]:
    page_texts, _, _ = _extract_pdf_page_texts_with_method(content)
    return page_texts


def _extract_pdf_page_texts_with_method(content: bytes) -> tuple[list[str], str, str | None]:
    pages, methods, _, warning, _ = _extract_pdf_page_texts_with_diagnostics(content)
    method = methods[0] if methods and len(set(methods)) == 1 else "mixed"
    return pages, method, warning


def _extract_pdf_page_texts_with_diagnostics(
    content: bytes,
) -> tuple[list[str], list[str], list[float], str | None, list[str]]:
    """Run independent native parsers and select text page by page."""

    pypdf_pages: list[str] | None = None
    pymupdf_pages: list[str] | None = None
    pymupdf_geometry: list[dict[str, Any]] = []
    errors: list[str] = []
    pypdf_error: Exception | None = None
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content), strict=False)
        pypdf_pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except Exception as exc:
        pypdf_error = exc
        errors.append(f"pypdf={type(exc).__name__}: {exc}")

    try:
        import pymupdf as fitz

        display_errors = bool(fitz.TOOLS.mupdf_display_errors())
        fitz.TOOLS.reset_mupdf_warnings()
        fitz.TOOLS.mupdf_display_errors(False)
        try:
            with fitz.open(stream=content, filetype="pdf") as document:
                pymupdf_pages = []
                for page in document:
                    page_text, geometry = _extract_pymupdf_page_blocks(page)
                    pymupdf_pages.append(page_text)
                    pymupdf_geometry.append(geometry)
            warnings = fitz.TOOLS.mupdf_warnings().strip()
        finally:
            fitz.TOOLS.mupdf_display_errors(display_errors)
            fitz.TOOLS.reset_mupdf_warnings()
        if warnings:
            errors.append(f"pymupdf diagnostics: {warnings[:2000]}")
    except Exception as exc:
        errors.append(f"pymupdf={type(exc).__name__}: {exc}")

    if pypdf_pages is None and pymupdf_pages is None:
        recovery_error = "; ".join(errors)
        if content.startswith(b"%PDF"):
            fallback = _fallback_pdf_text(content)
            if fallback:
                diagnostic = _page_diagnostic_json(
                    page_number=1,
                    selected_method="pdf_literal_fallback",
                    confidence=0.0,
                    agreement=None,
                    issues=["unsafe_literal_fallback", "native_parsers_failed"],
                )
                return [fallback], ["pdf_literal_fallback"], [0.0], recovery_error, [diagnostic]
        raise RuntimeError(f"PDF text extraction failed: {recovery_error}") from pypdf_error

    if pypdf_pages is None:
        warning = f"pypdf recovery: {errors[0].split('=', 1)[1]}" if errors else None
        pages = pymupdf_pages or []
        diagnostics = [
            _page_diagnostic_json(
                page_number=index,
                selected_method="pymupdf_text",
                confidence=_text_layer_confidence("pymupdf_text", bool(page.strip())),
                agreement=None,
                issues=["pypdf_failed"] if page.strip() else ["pypdf_failed", "empty_page"],
                pymupdf_geometry=(
                    pymupdf_geometry[index - 1] if index - 1 < len(pymupdf_geometry) else None
                ),
            )
            for index, page in enumerate(pages, start=1)
        ]
        return (
            pages,
            ["pymupdf_text"] * len(pages),
            [_text_layer_confidence("pymupdf_text", bool(page.strip())) for page in pages],
            warning,
            diagnostics,
        )
    if pymupdf_pages is None:
        pages = pypdf_pages
        diagnostics = [
            _page_diagnostic_json(
                page_number=index,
                selected_method="pypdf_text",
                confidence=_text_layer_confidence("pypdf_text", bool(page.strip())),
                agreement=None,
                issues=["pymupdf_failed"] if page.strip() else ["pymupdf_failed", "empty_page"],
            )
            for index, page in enumerate(pages, start=1)
        ]
        return (
            pages,
            ["pypdf_text"] * len(pages),
            [_text_layer_confidence("pypdf_text", bool(page.strip())) for page in pages],
            "; ".join(errors) or None,
            diagnostics,
        )

    page_count = max(len(pypdf_pages), len(pymupdf_pages))
    selected_pages: list[str] = []
    methods: list[str] = []
    confidences: list[float] = []
    diagnostics: list[str] = []
    if len(pypdf_pages) != len(pymupdf_pages):
        errors.append(
            f"page_count_disagreement:pypdf={len(pypdf_pages)}:pymupdf={len(pymupdf_pages)}"
        )
    for index in range(page_count):
        pypdf_text = pypdf_pages[index] if index < len(pypdf_pages) else ""
        pymupdf_text = pymupdf_pages[index] if index < len(pymupdf_pages) else ""
        selected, method, confidence, agreement, issues = _select_native_page_text(
            pypdf_text=pypdf_text,
            pymupdf_text=pymupdf_text,
        )
        selected_pages.append(selected)
        methods.append(method)
        confidences.append(confidence)
        diagnostics.append(
            _page_diagnostic_json(
                page_number=index + 1,
                selected_method=method,
                confidence=confidence,
                agreement=agreement,
                issues=issues,
                pymupdf_geometry=(
                    pymupdf_geometry[index] if index < len(pymupdf_geometry) else None
                ),
            )
        )
    return selected_pages, methods, confidences, "; ".join(errors) or None, diagnostics


def _select_native_page_text(
    *,
    pypdf_text: str,
    pymupdf_text: str,
) -> tuple[str, str, float, float | None, list[str]]:
    pypdf_score = _page_text_quality(pypdf_text)
    pymupdf_score = _page_text_quality(pymupdf_text)
    agreement = _page_text_agreement(pypdf_text, pymupdf_text)
    issues: list[str] = []
    if not pypdf_text.strip() and not pymupdf_text.strip():
        return "", "pypdf_text", 0.0, agreement, ["empty_page"]
    if agreement is not None and agreement < 0.85:
        issues.append("native_parser_disagreement")
    if pypdf_score >= pymupdf_score:
        selected, method, score = pypdf_text, "pypdf_text", pypdf_score
    else:
        selected, method, score = pymupdf_text, "pymupdf_text", pymupdf_score
    if not pypdf_text.strip():
        issues.append("pypdf_empty")
    if not pymupdf_text.strip():
        issues.append("pymupdf_empty")
    confidence = min(0.99, max(0.0, score))
    if agreement is not None:
        confidence = min(confidence, 0.55 + 0.45 * agreement)
        if agreement >= 0.98:
            confidence = max(confidence, 0.92)
    return selected, method, confidence, agreement, issues


def _extract_pymupdf_page_blocks(page: Any) -> tuple[str, dict[str, Any]]:
    """Extract deterministic top-to-bottom blocks and retain coordinate lineage."""

    try:
        try:
            raw_blocks = page.get_text("blocks", sort=True)
        except TypeError:
            raw_blocks = page.get_text("blocks")
        blocks = [block for block in raw_blocks if len(block) >= 5 and str(block[4]).strip()]
    except Exception:
        text = str(page.get_text("text") or "").strip()
        return text, {"reading_order": "native_text_fallback", "blocks": []}
    text = "\n".join(str(block[4]).strip() for block in blocks).strip()
    rect = getattr(page, "rect", None)
    geometry = {
        "reading_order": "pymupdf_blocks_sort_true",
        "rotation": int(getattr(page, "rotation", 0) or 0),
        "page_width": round(float(getattr(rect, "width", 0.0) or 0.0), 3),
        "page_height": round(float(getattr(rect, "height", 0.0) or 0.0), 3),
        "blocks": [
            {
                "order": index,
                "bbox": [round(float(value), 3) for value in block[:4]],
                "text_sha256": hashlib.sha256(str(block[4]).strip().encode("utf-8")).hexdigest(),
                "lines": [
                    re.sub(r"\s+", " ", line).strip()
                    for line in str(block[4]).splitlines()
                    if re.sub(r"\s+", " ", line).strip()
                ],
            }
            for index, block in enumerate(blocks)
        ],
    }
    return text, geometry


def _page_text_quality(text: str) -> float:
    value = str(text or "")
    if not value.strip():
        return 0.0
    characters = len(value)
    replacement_ratio = value.count("\ufffd") / max(1, characters)
    control_ratio = sum(ord(char) < 32 and char not in "\n\r\t" for char in value) / max(
        1, characters
    )
    tokens = re.findall(r"[^\W\d_]{2,}", value, flags=re.UNICODE)
    token_score = min(1.0, len(tokens) / 30)
    length_score = min(1.0, characters / 400)
    return max(
        0.0,
        0.55 * token_score + 0.45 * length_score - 4 * replacement_ratio - 4 * control_ratio,
    )


def _page_text_agreement(left: str, right: str) -> float | None:
    left_normalized = re.sub(r"\s+", " ", left).strip()
    right_normalized = re.sub(r"\s+", " ", right).strip()
    if not left_normalized or not right_normalized:
        return None
    return SequenceMatcher(None, left_normalized, right_normalized, autojunk=False).ratio()


def _page_diagnostic_json(
    *,
    page_number: int,
    selected_method: str,
    confidence: float,
    agreement: float | None,
    issues: list[str],
    pymupdf_geometry: dict[str, Any] | None = None,
) -> str:
    import json

    return json.dumps(
        {
            "page_number": page_number,
            "selected_method": selected_method,
            "confidence": round(float(confidence), 6),
            "native_agreement": None if agreement is None else round(float(agreement), 6),
            "issues": issues,
            "pymupdf_geometry": pymupdf_geometry,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _extract_pdf_text(
    content: bytes,
    *,
    use_ocr: bool = False,
    strict_validation: bool = False,
) -> PdfTextExtraction:
    (
        page_texts,
        page_methods,
        page_confidences,
        extraction_error,
        page_diagnostics,
    ) = _extract_pdf_page_texts_with_diagnostics(content)
    text = "\n\n".join(part for part in page_texts if part)
    unsafe_pages = [
        index
        for index, (page, confidence, diagnostic) in enumerate(
            zip(page_texts, page_confidences, page_diagnostics, strict=True)
        )
        if (
            pypdf_text_needs_ocr(page)
            and (
                _pdf_page_is_stamp_only(page)
                or (_diagnostic_native_agreement(diagnostic) or 0.0) < 0.95
            )
        )
        or confidence < 0.7
        or _page_diagnostic_is_unsafe(diagnostic)
    ]
    if text.strip() and (not use_ocr or not unsafe_pages):
        return PdfTextExtraction(
            page_texts=page_texts,
            page_methods=page_methods,
            page_confidences=page_confidences,
            status=(
                "needs_review"
                if strict_validation
                and (
                    unsafe_pages
                    or any(_page_diagnostic_is_unsafe(item) for item in page_diagnostics)
                    or "pdf_literal_fallback" in page_methods
                )
                else "ok"
            ),
            error=extraction_error,
            page_diagnostics=page_diagnostics,
        )
    if not use_ocr:
        return PdfTextExtraction(
            page_texts=page_texts,
            page_methods=page_methods,
            page_confidences=page_confidences,
            status=(
                "needs_review"
                if strict_validation
                and (
                    unsafe_pages
                    or any(_page_diagnostic_is_unsafe(item) for item in page_diagnostics)
                    or "pdf_literal_fallback" in page_methods
                )
                else ("ok" if text.strip() else "empty_text")
            ),
            error=extraction_error,
            page_diagnostics=page_diagnostics,
        )

    ocr = _ocr_pdf_page_texts(content)
    selected_pages = list(page_texts)
    selected_methods = list(page_methods)
    selected_confidences = list(page_confidences)
    selected_diagnostics = list(page_diagnostics)
    for index in unsafe_pages:
        if index >= len(ocr.page_texts) or not ocr.page_texts[index].strip():
            continue
        native_quality = _page_text_quality(page_texts[index])
        ocr_quality = _page_text_quality(ocr.page_texts[index])
        if ocr_quality >= native_quality:
            selected_pages[index] = ocr.page_texts[index]
            selected_methods[index] = ocr.page_methods[index]
            selected_confidences[index] = ocr.page_confidences[index]
            selected_diagnostics[index] = _page_diagnostic_json(
                page_number=index + 1,
                selected_method=ocr.page_methods[index],
                confidence=ocr.page_confidences[index],
                agreement=_page_text_agreement(page_texts[index], ocr.page_texts[index]),
                issues=["page_level_ocr_selected"],
            )
    ocr_text = "\n\n".join(part for part in selected_pages if part)
    if not ocr_text.strip() and text.strip():
        return PdfTextExtraction(
            page_texts=page_texts,
            page_methods=page_methods,
            page_confidences=page_confidences,
            status="needs_review" if unsafe_pages else "ok",
            error=ocr.error or extraction_error,
            page_diagnostics=page_diagnostics,
        )
    return PdfTextExtraction(
        page_texts=selected_pages,
        page_methods=selected_methods,
        page_confidences=selected_confidences,
        status=(
            "needs_review"
            if strict_validation
            and ocr_text.strip()
            and any(
                (pypdf_text_needs_ocr(page) and "ocr" not in method.casefold())
                or confidence < 0.7
                or _page_diagnostic_is_unsafe(diagnostic, allow_selected_ocr=True)
                for page, method, confidence, diagnostic in zip(
                    selected_pages,
                    selected_methods,
                    selected_confidences,
                    selected_diagnostics,
                    strict=True,
                )
            )
            else ("ok" if ocr_text.strip() else "empty_text")
        ),
        error=ocr.error,
        page_diagnostics=selected_diagnostics,
    )


def _page_diagnostic_is_unsafe(
    diagnostic: str,
    *,
    allow_selected_ocr: bool = False,
) -> bool:
    try:
        issues = set(json.loads(diagnostic).get("issues") or [])
    except (json.JSONDecodeError, AttributeError, TypeError):
        return True
    if allow_selected_ocr and issues == {"page_level_ocr_selected"}:
        return False
    return bool(issues)


def _diagnostic_native_agreement(diagnostic: str) -> float | None:
    try:
        value = json.loads(diagnostic).get("native_agreement")
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def pypdf_text_needs_ocr(text: str) -> bool:
    meaningful_lines = [
        _clean_line(line)
        for line in text.splitlines()
        if _clean_line(line) and not _is_pdf_stamp_line(_clean_line(line))
    ]
    if not meaningful_lines:
        return True
    meaningful_text = " ".join(meaningful_lines)
    alpha_tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{3,}", meaningful_text)
    return len(meaningful_text) < 80 or len(alpha_tokens) < 8


def _pdf_page_is_stamp_only(text: str) -> bool:
    """Return true when native extraction contains only an official side stamp.

    Some scanned patrimony forms have a perfectly self-consistent native text
    layer containing only the registration stamp.  Parser agreement must not
    suppress OCR in that case: agreement proves only that both native parsers
    saw the same stamp, not that they recovered the form.
    """
    return not any(
        _clean_line(line) and not _is_pdf_stamp_line(_clean_line(line))
        for line in text.splitlines()
    )


def clean_repeated_page_margins(
    page_texts: list[str],
    page_diagnostics: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Remove only statistically demonstrated repeated PDF margin lines.

    The selected native/OCR page text remains untouched in ``page_texts``. This
    function creates the content projection used by the turn parser and records
    every removed line with its page-local source offsets.
    """

    if len(page_texts) < 3 and not page_diagnostics:
        return list(page_texts), []
    candidates: dict[tuple[str, str], set[int]] = {}
    page_lines: list[list[tuple[str, int, int]]] = []
    geometric_margin_lines: list[dict[str, set[str]]] = []
    for page_index, page_text in enumerate(page_texts):
        lines: list[tuple[str, int, int]] = []
        for match in re.finditer(r"[^\r\n]+", str(page_text or "")):
            raw = match.group(0)
            normalized = re.sub(r"\s+", " ", raw).strip()
            if normalized:
                lines.append((normalized, match.start(), match.end()))
        page_lines.append(lines)
        geometric_lines = _geometric_margin_line_signatures(
            page_index=page_index,
            page_diagnostics=page_diagnostics,
        )
        geometric_margin_lines.append(geometric_lines)
        header_window = 6 if page_diagnostics else 3
        for position, subset in (
            ("header", lines[:header_window]),
            ("footer", lines[-3:]),
        ):
            for normalized, _, _ in subset:
                if 3 <= len(normalized) <= 160:
                    signature = _margin_line_signature(normalized)
                    if page_diagnostics and signature not in geometric_lines[position]:
                        continue
                    key = (position, signature)
                    candidates.setdefault(key, set()).add(page_index)
    has_geometric_evidence = bool(page_diagnostics)
    minimum_pages = max(2 if has_geometric_evidence else 3, (len(page_texts) + 1) // 2)
    repeated = {key for key, pages in candidates.items() if len(pages) >= minimum_pages}
    if not repeated and not has_geometric_evidence:
        return list(page_texts), []

    cleaned_pages: list[str] = []
    removals: list[str] = []
    for page_index, (page_text, lines) in enumerate(zip(page_texts, page_lines, strict=True)):
        spans: list[tuple[int, int, str, str]] = []
        geometric_header, geometric_footer, geometric_side = _geometric_margin_evidence(
            page_index=page_index,
            page_diagnostics=page_diagnostics,
        )
        # The official header can be five physical lines: publication title,
        # chamber, issue number, date and page.  Geometry is mandatory for the
        # expanded window, so a repeated discourse line cannot be removed merely
        # because it appears near the start of a page.
        header_window = 6 if page_diagnostics else 3
        header_ids = {id(item) for item in lines[:header_window]}
        footer_ids = {id(item) for item in lines[-3:]}
        for line_index, item in enumerate(lines):
            normalized, start, end = item
            positions: list[str] = []
            if id(item) in header_ids:
                positions.append("header")
            if id(item) in footer_ids:
                positions.append("footer")
            repeated_positions = [
                position
                for position in positions
                if (position, _margin_line_signature(normalized)) in repeated
                and (
                    not page_diagnostics
                    or _margin_line_signature(normalized)
                    in geometric_margin_lines[page_index][position]
                )
            ]
            geometric_position: str | None = None
            if (
                "header" in positions
                and not repeated_positions
                and geometric_header
                and _margin_line_signature(normalized)
                in geometric_margin_lines[page_index]["header"]
                and _looks_like_official_running_header_component(normalized)
            ):
                geometric_position = "header"
            elif (
                line_index >= max(0, len(lines) - 6)
                and geometric_footer
                and _looks_like_official_legal_footer(normalized)
            ):
                geometric_position = "footer"
            elif geometric_side and _looks_like_official_side_mark(normalized):
                geometric_position = "side_margin"
            if geometric_position is not None:
                spans.append((start, end, f"geometric_{geometric_position}", normalized))
                continue
            if repeated_positions:
                # On short pages the first/last three-line windows overlap. Preserve
                # the physically nearest margin so provenance does not mislabel a
                # footer as a header (or vice versa).
                position = min(
                    repeated_positions,
                    key=lambda value: (
                        line_index if value == "header" else len(lines) - line_index - 1
                    ),
                )
                spans.append((start, end, position, normalized))
        cleaned = str(page_text or "")
        for start, end, position, _normalized in reversed(spans):
            cleaned = cleaned[:start] + cleaned[end:]
            removals.append(
                json.dumps(
                    {
                        "page_number": page_index + 1,
                        "source_start": start,
                        "source_end": end,
                        "source_page_length": len(str(page_text or "")),
                        "offset_basis": "selected_page_text_unicode_codepoints",
                        "kind": (
                            position
                            if position.startswith("geometric_")
                            else f"repeated_{position}"
                        ),
                        "text": str(page_text or "")[start:end],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        # Do not strip or collapse residual whitespace here: the parser can ignore
        # layout whitespace, while retaining it makes this projection exactly
        # reconstructable from ``page_texts`` plus the recorded removals.
        cleaned_pages.append(cleaned)
    return cleaned_pages, removals


def _margin_line_signature(value: str) -> str:
    # Independent PDF parsers can expose the same accented label either as proper
    # Unicode or as the common UTF-8-as-Latin-1 mojibake. Canonicalize only this
    # known layout token before replacing the dynamic page number.
    repaired_value = value
    for _ in range(2):
        try:
            repaired = repaired_value.encode("latin-1").decode("utf-8")
        except (UnicodeDecodeError, UnicodeEncodeError):
            break
        if repaired == repaired_value:
            break
        repaired_value = repaired
    folded = repaired_value.casefold()
    return re.sub(r"(p(?:\u00e1|a)g\.?\s*)\d+", r"\1{page}", folded)


def _looks_like_official_running_header_component(value: str) -> bool:
    normalized = re.sub(r"\s+", " ", value).strip()
    return (
        normalized.casefold() in _OFFICIAL_RUNNING_HEADER_COMPONENTS
        or _OFFICIAL_ISSUE_HEADER_RE.fullmatch(normalized) is not None
    )


def _geometric_margin_evidence(
    *,
    page_index: int,
    page_diagnostics: list[str] | None,
) -> tuple[bool, bool, bool]:
    if not page_diagnostics or page_index >= len(page_diagnostics):
        return False, False, False
    try:
        diagnostic = json.loads(page_diagnostics[page_index])
        geometry = diagnostic.get("pymupdf_geometry") or {}
        height = float(geometry.get("page_height") or 0.0)
        width = float(geometry.get("page_width") or 0.0)
        blocks = geometry.get("blocks") or []
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        return False, False, False
    if height <= 0 or width <= 0:
        return False, False, False
    has_header = any(float(block["bbox"][3]) <= height * 0.115 for block in blocks)
    has_footer = any(float(block["bbox"][1]) >= height * 0.93 for block in blocks)
    has_side = any(float(block["bbox"][0]) >= width * 0.9 for block in blocks)
    return has_header, has_footer, has_side


def _geometric_margin_line_signatures(
    *,
    page_index: int,
    page_diagnostics: list[str] | None,
) -> dict[str, set[str]]:
    positions = {"header": set(), "footer": set()}
    if not page_diagnostics or page_index >= len(page_diagnostics):
        return positions
    try:
        diagnostic = json.loads(page_diagnostics[page_index])
        geometry = diagnostic.get("pymupdf_geometry") or {}
        height = float(geometry.get("page_height") or 0.0)
        blocks = geometry.get("blocks") or []
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        return positions
    if height <= 0:
        return positions
    position_lines: dict[str, list[str]] = {"header": [], "footer": []}
    for block in blocks:
        try:
            top = float(block["bbox"][1])
            bottom = float(block["bbox"][3])
            lines = block.get("lines") or []
        except (KeyError, TypeError, ValueError, IndexError, AttributeError):
            continue
        block_positions = []
        if bottom <= height * 0.115:
            block_positions.append("header")
        if top >= height * 0.93:
            block_positions.append("footer")
        for line in lines:
            normalized = re.sub(r"\s+", " ", str(line)).strip()
            signature = _margin_line_signature(normalized)
            for position in block_positions:
                positions[position].add(signature)
                if normalized:
                    position_lines[position].append(normalized)
    # PyMuPDF commonly reports the three horizontally separated running-header
    # fields as distinct lines while pypdf emits one flattened line. Add every
    # contiguous geometric combination so both independent representations can
    # identify the same layout-only span without a generic text regex.
    for position, lines in position_lines.items():
        for start in range(len(lines)):
            for end in range(start + 2, len(lines) + 1):
                positions[position].add(_margin_line_signature(" ".join(lines[start:end])))
    return positions


def _looks_like_official_legal_footer(value: str) -> bool:
    folded = value.casefold()
    return any(
        marker in folded
        for marker in (
            "http://www.congreso.es",
            "calle floridablanca",
            "d. l.:",
            "m-12.580/1961",
            "congreso de los diputados tel",
            "edición electrónica preparada",
            "edici�n electr�nica preparada",
            "http://boe.es",
        )
    )


def _looks_like_official_side_mark(value: str) -> bool:
    return value.casefold().startswith("cve:")


def _pypdf_text_needs_ocr(text: str) -> bool:
    return pypdf_text_needs_ocr(text)


def _is_pdf_stamp_line(line: str) -> bool:
    return bool(
        re.fullmatch(r"C\.?\s*DIP\s+\d+\s+\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}", line, re.I)
    )


def _ocr_pdf_page_texts(content: bytes) -> PdfTextExtraction:
    try:
        import pymupdf as fitz
    except ImportError as exc:
        return PdfTextExtraction(
            page_texts=[],
            page_methods=[],
            page_confidences=[],
            status="ocr_unavailable",
            error=f"PyMuPDF not installed: {exc}",
        )

    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        return PdfTextExtraction(
            page_texts=[],
            page_methods=[],
            page_confidences=[],
            status="error",
            error=f"PyMuPDF open failed: {type(exc).__name__}: {exc}",
        )

    page_texts: list[str] = []
    page_confidences: list[float] = []
    with tempfile.TemporaryDirectory(prefix="congreso_ocr_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        image_paths: list[Path] = []
        for page_number, page in enumerate(document, start=1):
            image_path = tmp_path / f"page_{page_number:04d}.png"
            try:
                pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                pixmap.save(image_path)
            except Exception:
                image_path = Path()
            image_paths.append(image_path)
        valid_images = [path for path in image_paths if path.name]
        workers = min(_ocr_worker_count(), len(valid_images)) if valid_images else 1
        rapidocr_engine()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_rapidocr_image_text, valid_images))
        result_by_path = dict(zip(valid_images, results, strict=True))
        for image_path in image_paths:
            text, confidence = result_by_path.get(image_path, ("", 0.0))
            page_texts.append(text)
            page_confidences.append(confidence)
    return PdfTextExtraction(
        page_texts=page_texts,
        page_methods=["rapidocr_onnxruntime"] * len(page_texts),
        page_confidences=page_confidences,
        status="ok" if any(text.strip() for text in page_texts) else "empty_text",
    )


def _ocr_worker_count() -> int:
    raw = os.getenv("CONGRESO_OCR_WORKERS", "4")
    try:
        return max(1, min(4, int(raw)))
    except ValueError:
        return 4


def _rapidocr_image_text(image_path: Path) -> tuple[str, float]:
    return rapidocr_image_text_with_engine(image_path, rapidocr_engine())


def rapidocr_image_text_with_engine(image_path: Path, engine: Any) -> tuple[str, float]:
    """Extract ordered text from one image using an injected RapidOCR engine."""

    lines: list[Any] = []
    use_angle_classifier = bool(ocr_runtime_config().get("use_angle_classifier", False))
    for image_input in _ocr_image_inputs(image_path):
        result = rapidocr_lines(engine(image_input, use_cls=use_angle_classifier))
        if result:
            lines.extend(_ocr_lines_in_reading_order(result))
    text = "\n".join(str(item[1]).strip() for item in lines if str(item[1]).strip())
    scores = [float(item[2]) for item in lines if len(item) >= 3 and item[2] is not None]
    confidence = sum(scores) / len(scores) if scores else 0.0
    return text, confidence


def _ocr_image_inputs(image_path: Path) -> list[Any]:
    """Return OCR regions in page reading order.

    Most historical pages are either fully single-column or fully two-column. Some
    cover pages switch from a full-width masthead to two columns near the bottom;
    those must be read as ``header -> left -> right`` instead of globally sorting
    all OCR boxes or splitting the masthead in half.
    """

    try:
        import cv2
    except ImportError:
        return [str(image_path)]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3:
        return [str(image_path)]
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    split = _two_column_split(grayscale)
    if split is not None:
        return [image[:, :split], image[:, split + 1 :]]
    mixed_layout = _mixed_two_column_layout(grayscale)
    if mixed_layout is not None:
        split, column_start = mixed_layout
        return [
            image[:column_start, :],
            image[column_start:, :split],
            image[column_start:, split + 1 :],
        ]
    return [image]


def _mixed_two_column_layout(grayscale: Any) -> tuple[int, int] | None:
    """Detect a late page transition from a full-width header to two columns.

    A short but continuous central rule is strong evidence for a mixed layout even
    when it covers too little of the whole page for ``_two_column_split``. The gate
    deliberately searches only the lower half, requires a locally prominent run,
    and verifies ink on both sides. This avoids treating large masthead glyphs or
    procedural prose as a column separator.
    """

    try:
        import numpy as np
    except ImportError:
        return None
    if getattr(grayscale, "ndim", 0) != 2:
        return None
    height, width = grayscale.shape
    if height < 200 or width < 200:
        return None
    y_search_start, y_search_end = int(height * 0.45), int(height * 0.97)
    x_start, x_end = int(width * 0.35), int(width * 0.65)
    band_half_width = max(4, int(width * 0.004))
    ink = grayscale < 200
    candidates: list[tuple[int, int, int]] = []
    run_lengths: list[int] = []
    for center in range(x_start + band_half_width, x_end - band_half_width):
        occupied_rows = ink[
            y_search_start:y_search_end,
            center - band_half_width : center + band_half_width + 1,
        ].any(axis=1)
        run_start, run_end = _longest_true_run(occupied_rows)
        run_length = run_end - run_start
        run_lengths.append(run_length)
        candidates.append((run_length, run_start, center))
    if not candidates:
        return None
    best_length = max(item[0] for item in candidates)
    typical_length = float(np.median(np.asarray(run_lengths, dtype=float)))
    minimum_length = max(48, int(height * 0.03))
    if best_length < minimum_length or best_length < max(typical_length * 2.0, 1.0):
        return None
    near_best = [item for item in candidates if item[0] >= best_length * 0.98]
    split = int(round(float(np.mean([item[2] for item in near_best]))))
    run_start = min(item[1] for item in near_best)
    rule_start = y_search_start + run_start
    sample_end = min(height, rule_start + max(best_length, int(height * 0.08)))
    left_density = float(ink[rule_start:sample_end, int(width * 0.05) : split].mean())
    right_density = float(ink[rule_start:sample_end, split + 1 : int(width * 0.95)].mean())
    if left_density < 0.005 or right_density < 0.005:
        return None
    column_start = max(1, rule_start - max(8, int(height * 0.012)))
    if column_start >= height - 1:
        return None
    return split, column_start


def _longest_true_run(values: Any) -> tuple[int, int]:
    best_start = best_end = current_start = 0
    in_run = False
    for index, value in enumerate(values):
        if bool(value) and not in_run:
            current_start = index
            in_run = True
        elif not bool(value) and in_run:
            if index - current_start > best_end - best_start:
                best_start, best_end = current_start, index
            in_run = False
    if in_run and len(values) - current_start > best_end - best_start:
        best_start, best_end = current_start, len(values)
    return best_start, best_end


def _two_column_split(grayscale: Any) -> int | None:
    """Detect a central whitespace gutter or vertical rule in scanned proceedings."""

    try:
        import numpy as np
    except ImportError:
        return None
    if getattr(grayscale, "ndim", 0) != 2:
        return None
    height, width = grayscale.shape
    if height < 100 or width < 100:
        return None
    y_start, y_end = int(height * 0.10), int(height * 0.92)
    x_start, x_end = int(width * 0.35), int(width * 0.65)
    central_ink = grayscale[y_start:y_end] < 200
    # Historical scans often skew a one-pixel separator across 10-15 pixels from
    # top to bottom. No single x coordinate is then dark enough to look like a
    # rule. Measure whether a narrow moving band contains ink on most rows and
    # require a strong local prominence; ordinary single-column prose has ink on
    # only the text-line rows and remains below this gate.
    band_half_width = max(4, int(width * 0.004))
    candidate_centers = range(
        x_start + band_half_width,
        x_end - band_half_width,
    )
    band_coverages = np.asarray(
        [
            central_ink[
                :,
                center - band_half_width : center + band_half_width + 1,
            ]
            .any(axis=1)
            .mean()
            for center in candidate_centers
        ],
        dtype=float,
    )
    if len(band_coverages):
        rule_offset = int(np.argmax(band_coverages))
        rule_coverage = float(band_coverages[rule_offset])
        typical_coverage = float(np.median(band_coverages))
        if rule_coverage >= 0.58 and rule_coverage >= typical_coverage * 1.35:
            near_peak = np.flatnonzero(band_coverages >= rule_coverage * 0.995)
            centered_offset = int(round(float(np.mean(near_peak))))
            return x_start + band_half_width + centered_offset

    ink_density = (grayscale[y_start:y_end] < 180).mean(axis=0)
    central = ink_density[x_start:x_end]
    if not len(central):
        return None
    whitespace = central < 0.03
    best_start = best_end = current_start = None
    for index, is_space in enumerate(whitespace):
        if is_space and current_start is None:
            current_start = index
        if not is_space and current_start is not None:
            if best_start is None or index - current_start > best_end - best_start:
                best_start, best_end = current_start, index
            current_start = None
    if current_start is not None and (
        best_start is None or len(whitespace) - current_start > best_end - best_start
    ):
        best_start, best_end = current_start, len(whitespace)
    if best_start is None or best_end - best_start < max(4, int(width * 0.004)):
        return None
    return x_start + (best_start + best_end) // 2


def _ocr_lines_in_reading_order(result: list[Any]) -> list[Any]:
    """Order one already classified image region from top to bottom.

    Column classification belongs exclusively to ``_two_column_split``. Applying a
    second column heuristic to OCR boxes can split ordinary full-width prose whenever
    the recognizer emits several short boxes on one line.
    """

    lines = [item for item in result if len(item) >= 2 and str(item[1]).strip()]
    return sorted(lines, key=_ocr_line_sort_key)


def _ocr_line_sort_key(item: Any) -> tuple[float, float]:
    return _box_top(item[0]), _box_left(item[0])


def rapidocr_engine() -> Any:
    """Return the process-local RapidOCR engine configured by package contracts."""

    global _RAPID_OCR_ENGINE
    if _RAPID_OCR_ENGINE is None:
        with _RAPID_OCR_ENGINE_LOCK:
            if _RAPID_OCR_ENGINE is None:
                require_model_contract("document_ocr")
                _RAPID_OCR_ENGINE = create_rapidocr_engine(
                    providers=tuple(selected_ocr_onnx_providers())
                )
    return _RAPID_OCR_ENGINE


def _rapidocr_runtime_kwargs(rapidocr_cls: Any) -> dict[str, Any]:
    return rapidocr_runtime_kwargs(
        rapidocr_cls,
        providers=tuple(desired_ocr_onnx_providers()),
    )


def _page_model_name(extraction_method: str) -> str | None:
    if extraction_method == "rapidocr_onnxruntime":
        return ocr_runtime_config()["pipeline_version"]
    if extraction_method == "pypdf_text":
        return model_name_from_contract("pdf_text_layer", "pypdf")
    if extraction_method == "pymupdf_text":
        return "pymupdf"
    return None


def _text_layer_confidence(extraction_method: str, nonempty: bool) -> float:
    if not nonempty:
        return 0.2
    if extraction_method == "pymupdf_text":
        return 0.90
    if extraction_method == "pdf_literal_fallback":
        return 0.60
    return 0.92


def _document_extraction_method(
    extraction: PdfTextExtraction,
    *,
    use_ocr: bool = False,
) -> str:
    methods = [method for method in extraction.page_methods if method]
    unique_methods = sorted(set(methods))
    if len(unique_methods) == 1:
        return unique_methods[0]
    if len(unique_methods) > 1:
        return "mixed"
    if use_ocr and extraction.status == "ocr_unavailable":
        return "rapidocr_onnxruntime"
    return "pypdf_text"


def _box_left(points: list[list[float]]) -> float:
    return min(point[0] for point in points)


def _box_top(points: list[list[float]]) -> float:
    return min(point[1] for point in points)


def _fallback_pdf_text(content: bytes) -> str:
    text = content.decode("latin-1", errors="ignore")
    chunks = []
    for match in re.finditer(r"\(([^()]*)\)\s*Tj", text):
        chunks.append(match.group(1).replace(r"\(", "(").replace(r"\)", ")"))
    return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())


def _page_rows(text_row: dict[str, Any]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    page_methods = text_row.get("_page_methods") or text_row.get("page_methods") or []
    if not page_methods and text_row.get("extraction_method") != "mixed":
        page_methods = [text_row.get("extraction_method") or "pypdf_text"] * len(
            text_row.get("page_texts") or []
        )
    page_confidences = text_row.get("_page_confidences") or text_row.get("page_confidences") or []
    for index, text in enumerate(text_row.get("page_texts") or [], start=1):
        confidence = (
            float(page_confidences[index - 1])
            if index - 1 < len(page_confidences)
            else 0.92
            if text.strip()
            else 0.2
        )
        pages.append(
            {
                "page_id": stable_id(text_row["document_sha256"], index),
                "document_sha256": text_row["document_sha256"],
                "source_url": text_row["source_url"],
                "document_kind": text_row["document_kind"],
                "page_number": index,
                "text": text,
                "char_count": len(text),
                "extraction_method": page_methods[index - 1]
                if index - 1 < len(page_methods)
                else "pypdf_text",
                "model_name": _page_model_name(
                    page_methods[index - 1] if index - 1 < len(page_methods) else "pypdf_text"
                ),
                "confidence": confidence,
                "source_file_sha256": text_row["document_sha256"],
                "snapshot_date": text_row["snapshot_date"],
            }
        )
    return pages


def _table_rows(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in pages:
        blocks = _candidate_table_blocks(page["text"])
        for index, block in enumerate(blocks, start=1):
            lines = [line for line in block.splitlines() if line.strip()]
            rows.append(
                {
                    "table_id": stable_id(page["document_sha256"], page["page_number"], index),
                    "document_sha256": page["document_sha256"],
                    "source_url": page["source_url"],
                    "document_kind": page["document_kind"],
                    "page_number": page["page_number"],
                    "table_index": index,
                    "raw_text": block,
                    "row_count": len(lines),
                    "column_count": _rough_column_count(lines),
                    "extraction_method": "text_layout_heuristic",
                    "model_name": model_name_from_contract(
                        "deputy_document_table_detector",
                        "text_layout_heuristic",
                    ),
                    "confidence": 0.45,
                    "source_file_sha256": page["document_sha256"],
                    "snapshot_date": page["snapshot_date"],
                }
            )
    return rows


def _entity_rows(
    *,
    pages: list[dict[str, Any]],
    source_url: str,
    source_sha256: str,
    snapshot_date: str,
    document_kind: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in pages:
        text = page["text"]
        rows.extend(
            _regex_entity_rows(
                pattern=AMOUNT_RE,
                entity_type="amount",
                page=page,
                source_url=source_url,
                source_sha256=source_sha256,
                snapshot_date=snapshot_date,
                document_kind=document_kind,
                normalizer=_amount_to_float,
            )
        )
        rows.extend(
            _regex_entity_rows(
                pattern=DATE_RE,
                entity_type="date",
                page=page,
                source_url=source_url,
                source_sha256=source_sha256,
                snapshot_date=snapshot_date,
                document_kind=document_kind,
            )
        )
        rows.extend(
            _regex_entity_rows(
                pattern=PERCENT_RE,
                entity_type="percentage",
                page=page,
                source_url=source_url,
                source_sha256=source_sha256,
                snapshot_date=snapshot_date,
                document_kind=document_kind,
                normalizer=lambda value: value.replace(",", "."),
            )
        )
        for pattern, entity_type in (
            (IBAN_RE, "iban_partial"),
            (COMPANY_RE, "company"),
            (REGISTRY_RE, "registry_id"),
        ):
            rows.extend(
                _regex_entity_rows(
                    pattern=pattern,
                    entity_type=entity_type,
                    page=page,
                    source_url=source_url,
                    source_sha256=source_sha256,
                    snapshot_date=snapshot_date,
                    document_kind=document_kind,
                )
            )
        for match in _location_matches(text):
            rows.append(
                _entity_row(
                    page=page,
                    source_url=source_url,
                    source_sha256=source_sha256,
                    snapshot_date=snapshot_date,
                    document_kind=document_kind,
                    entity_type="location",
                    value=match["value"],
                    normalized_value=match["value"],
                    start_char=match["start"],
                    end_char=match["end"],
                )
            )
    return _dedupe_rows(rows, "entity_id")


def _regex_entity_rows(
    *,
    pattern: re.Pattern[str],
    entity_type: str,
    page: dict[str, Any],
    source_url: str,
    source_sha256: str,
    snapshot_date: str,
    document_kind: str,
    normalizer: Any | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for match in pattern.finditer(page["text"]):
        value = match.group(1) if match.groups() else match.group(0)
        normalized = normalizer(value) if normalizer else value.strip()
        rows.append(
            _entity_row(
                page=page,
                source_url=source_url,
                source_sha256=source_sha256,
                snapshot_date=snapshot_date,
                document_kind=document_kind,
                entity_type=entity_type,
                value=value,
                normalized_value=str(normalized) if normalized is not None else None,
                start_char=match.start(),
                end_char=match.end(),
            )
        )
    return rows


def _entity_row(
    *,
    page: dict[str, Any],
    source_url: str,
    source_sha256: str,
    snapshot_date: str,
    document_kind: str,
    entity_type: str,
    value: str,
    normalized_value: str | None,
    start_char: int,
    end_char: int,
) -> dict[str, Any]:
    raw_evidence = _evidence_window(page["text"], start_char, end_char)
    return {
        "entity_id": stable_id(source_sha256, page["page_number"], entity_type, start_char, value),
        "document_sha256": source_sha256,
        "source_url": source_url,
        "document_kind": document_kind,
        "entity_type": entity_type,
        "entity_subtype": None,
        "text": value.strip(),
        "normalized_value": normalized_value,
        "amount_eur": _amount_to_float(value) if entity_type == "amount" else None,
        "date_value": value if entity_type == "date" else None,
        "page_number": page["page_number"],
        "start_char": start_char,
        "end_char": end_char,
        "bbox": None,
        "extraction_method": "regex",
        "model_name": model_name_from_contract("deputy_document_entity_regex", "regex"),
        "confidence": 0.95,
        "needs_review": False,
        "raw_evidence": raw_evidence,
        "source_file_sha256": source_sha256,
        "snapshot_date": snapshot_date,
    }


def _extraction_rows(
    *,
    pages: list[dict[str, Any]],
    source_url: str,
    source_sha256: str,
    snapshot_date: str,
    document_kind: str,
) -> list[dict[str, Any]]:
    structured_rows = _structured_document_extraction_rows(
        pages=pages,
        source_url=source_url,
        source_sha256=source_sha256,
        snapshot_date=snapshot_date,
        document_kind=document_kind,
    )

    rows: list[dict[str, Any]] = []
    phase = _declaration_phase(source_url, "\n".join(page["text"] for page in pages[:2]))
    for page in pages:
        for clean in _logical_extraction_lines(page["text"], document_kind=document_kind):
            if len(clean) < 8:
                continue
            if _is_boilerplate_extraction_line(clean):
                continue
            if _is_structured_form_noise_line(clean, document_kind):
                continue
            # Mortgage rows are parsed structurally so granted and pending amounts remain distinct.
            if document_kind == "assets_income" and "PRESTAMOHIPOTECARIO" in _compact_ocr(clean):
                continue
            classified = _classify_line(clean, document_kind)
            if classified is None:
                continue
            if not has_patrimony_evidence_signal(
                clean,
                classified,
                document_kind=document_kind,
            ):
                continue
            amount = _first_amount(clean)
            percentage = _first_percentage(clean)
            confidence = _line_confidence(clean, amount, classified)
            rows.append(
                {
                    "extraction_id": stable_id(
                        source_sha256,
                        page["page_number"],
                        classified["item_family"],
                        classified["item_category"],
                        clean,
                    ),
                    "document_sha256": source_sha256,
                    "source_url": source_url,
                    "document_kind": document_kind,
                    "declaration_phase": phase,
                    "item_family": classified["item_family"],
                    "item_category": classified["item_category"],
                    "item_subcategory": classified.get("item_subcategory"),
                    "label": classified["label"],
                    "description": clean,
                    "amount_eur": amount,
                    "percentage": percentage,
                    "currency": "EUR" if amount is not None else None,
                    "date_value": _first_match(DATE_RE, clean),
                    "location": _line_location(clean),
                    "company_name": _first_match(COMPANY_RE, clean),
                    "role": classified.get("role"),
                    "page_number": page["page_number"],
                    "text_span": clean,
                    "bbox": None,
                    "extraction_method": "rules_v1",
                    "model_name": model_name_from_contract("deputy_document_rules", "rules_v1"),
                    "confidence": confidence,
                    "needs_review": confidence
                    < review_threshold_for_model("deputy_document_rules"),
                    "raw_evidence": clean,
                    "source_file_sha256": source_sha256,
                    "snapshot_date": snapshot_date,
                }
            )
    return _merge_extraction_rows(structured_rows, rows)


def _structured_document_extraction_rows(
    *,
    pages: list[dict[str, Any]],
    source_url: str,
    source_sha256: str,
    snapshot_date: str,
    document_kind: str,
) -> list[dict[str, Any]]:
    if document_kind == "assets_income":
        return _structured_assets_income_rows(
            pages=pages,
            source_url=source_url,
            source_sha256=source_sha256,
            snapshot_date=snapshot_date,
        )
    if document_kind == "economic_interests":
        return _structured_economic_interest_rows(
            pages=pages,
            source_url=source_url,
            source_sha256=source_sha256,
            snapshot_date=snapshot_date,
        )
    return []


def _structured_assets_income_rows(
    *,
    pages: list[dict[str, Any]],
    source_url: str,
    source_sha256: str,
    snapshot_date: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in pages:
        lines = [_clean_line(line) for line in str(page.get("text") or "").splitlines()]
        compact_lines = [_compact_ocr(line) for line in lines]
        page_number = int(page.get("page_number") or 0)

        if any("RENTASPERCIBIDASPORELPARLAMENTARIO" in line for line in compact_lines):
            rows.extend(
                _structured_income_rows(
                    lines=lines,
                    page_number=page_number,
                    source_url=source_url,
                    source_sha256=source_sha256,
                    snapshot_date=snapshot_date,
                )
            )
        if any("BIENESPATRIMONIALESDELPARLAMENTARIO" in line for line in compact_lines):
            rows.extend(
                _structured_real_estate_rows(
                    lines=lines,
                    page_number=page_number,
                    source_url=source_url,
                    source_sha256=source_sha256,
                    snapshot_date=snapshot_date,
                )
            )
            rows.extend(
                _structured_financial_asset_rows(
                    lines=lines,
                    page_number=page_number,
                    source_url=source_url,
                    source_sha256=source_sha256,
                    snapshot_date=snapshot_date,
                )
            )
        if any("OTROSBIENESODERECHOS" in line for line in compact_lines):
            rows.extend(
                _structured_other_asset_rows(
                    lines=lines,
                    page_number=page_number,
                    source_url=source_url,
                    source_sha256=source_sha256,
                    snapshot_date=snapshot_date,
                )
            )
        if any("DEUDASYOBLIGACIONESPATRIMONIALES" in line for line in compact_lines):
            rows.extend(
                _structured_liability_rows(
                    lines=lines,
                    page_number=page_number,
                    source_url=source_url,
                    source_sha256=source_sha256,
                    snapshot_date=snapshot_date,
                )
            )
    return rows


def _structured_income_rows(
    *,
    lines: list[str],
    page_number: int,
    source_url: str,
    source_sha256: str,
    snapshot_date: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        amount = _structured_amount_to_float(line)
        if amount is None:
            continue
        description = _structured_neighbor_description(
            lines,
            index,
            prefer_following_high_priority=True,
        )
        if description is None:
            continue
        category = _structured_income_category(description)
        if category is None:
            continue
        label = _structured_income_label(description, category)
        evidence = f"{description}: {_format_eur(amount)}"
        rows.append(
            _structured_extraction_row(
                source_sha256=source_sha256,
                source_url=source_url,
                snapshot_date=snapshot_date,
                document_kind="assets_income",
                item_family="income",
                item_category=category,
                description=label,
                amount_eur=amount,
                currency="EUR",
                page_number=page_number,
                raw_evidence=evidence,
                confidence=0.86,
            )
        )
    return rows


def _structured_real_estate_rows(
    *,
    lines: list[str],
    page_number: int,
    source_url: str,
    source_sha256: str,
    snapshot_date: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        compact = _compact_ocr(lines[index])
        if compact not in {"VIVIENDA", "PLAZADEAPARCAMIENTO"}:
            index += 1
            continue
        item_lines = [lines[index]]
        end = index + 1
        while end < min(len(lines), index + 7):
            item_lines.append(lines[end])
            if "COMPRAVENTA" in _compact_ocr(lines[end]):
                break
            end += 1
        evidence = " ".join(line for line in item_lines if line)
        evidence_compact = _compact_ocr(evidence)
        if "COMPRAVENTA" not in evidence_compact:
            index += 1
            continue
        property_type = "vivienda" if compact == "VIVIENDA" else "plaza_aparcamiento"
        location = _first_known_location(item_lines)
        percentage = _first_percentage(evidence)
        description = _human_join(
            [
                "Vivienda" if property_type == "vivienda" else "Plaza de aparcamiento",
                location,
                _first_year(evidence),
                f"{percentage:g}%" if percentage is not None else None,
                "compraventa",
            ]
        )
        rows.append(
            _structured_extraction_row(
                source_sha256=source_sha256,
                source_url=source_url,
                snapshot_date=snapshot_date,
                document_kind="assets_income",
                item_family="asset",
                item_category="real_estate",
                item_subcategory=property_type,
                description=description,
                percentage=percentage,
                location=location,
                page_number=page_number,
                raw_evidence=evidence,
                confidence=0.78 if location else 0.66,
                needs_review=location is None,
            )
        )
        index = end + 1
    return rows


def _structured_financial_asset_rows(
    *,
    lines: list[str],
    page_number: int,
    source_url: str,
    source_sha256: str,
    snapshot_date: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        amount = _structured_amount_to_float(line)
        if amount is None:
            continue
        description = _structured_neighbor_description(lines, index)
        if description is None:
            continue
        category = _structured_asset_category(description)
        if category is None:
            continue
        label = _structured_asset_label(description, category)
        evidence = f"{description}: {_format_eur(amount)}"
        rows.append(
            _structured_extraction_row(
                source_sha256=source_sha256,
                source_url=source_url,
                snapshot_date=snapshot_date,
                document_kind="assets_income",
                item_family="asset",
                item_category=category,
                description=label,
                amount_eur=amount,
                currency="EUR",
                page_number=page_number,
                raw_evidence=evidence,
                confidence=0.86,
            )
        )
    return rows


def _structured_other_asset_rows(
    *,
    lines: list[str],
    page_number: int,
    source_url: str,
    source_sha256: str,
    snapshot_date: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        amount = _structured_amount_to_float(line)
        if amount is None:
            continue
        description = _structured_neighbor_description(
            lines,
            index,
            prefer_following_high_priority=True,
        )
        if description is None:
            continue
        category = _structured_asset_category(description)
        if category is None:
            continue
        label = _structured_asset_label(description, category)
        needs_review = category == "other_asset"
        rows.append(
            _structured_extraction_row(
                source_sha256=source_sha256,
                source_url=source_url,
                snapshot_date=snapshot_date,
                document_kind="assets_income",
                item_family="asset",
                item_category=category,
                description=label,
                amount_eur=amount,
                currency="EUR",
                page_number=page_number,
                raw_evidence=f"{description}: {_format_eur(amount)}",
                confidence=0.74 if needs_review else 0.82,
                needs_review=needs_review,
            )
        )
    return rows


def _structured_liability_rows(
    *,
    lines: list[str],
    page_number: int,
    source_url: str,
    source_sha256: str,
    snapshot_date: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if "PRESTAMOHIPOTECARIO" not in _compact_ocr(line):
            continue
        window = lines[index : index + 5]
        amounts = [
            amount
            for candidate in window
            if (amount := _structured_amount_to_float(candidate)) is not None
        ]
        pending = amounts[-1] if amounts else None
        granted = amounts[0] if len(amounts) > 1 else None
        description = "Prestamo hipotecario"
        if granted is not None:
            description = f"{description}; importe concedido {_format_eur(granted)}"
        rows.append(
            _structured_extraction_row(
                source_sha256=source_sha256,
                source_url=source_url,
                snapshot_date=snapshot_date,
                document_kind="assets_income",
                item_family="liability",
                item_category="mortgage",
                description=description,
                amount_eur=pending,
                currency="EUR" if pending is not None else None,
                page_number=page_number,
                raw_evidence=" ".join(window),
                confidence=0.86 if pending is not None else 0.66,
                needs_review=pending is None,
            )
        )
    return rows


def _structured_economic_interest_rows(
    *,
    pages: list[dict[str, Any]],
    source_url: str,
    source_sha256: str,
    snapshot_date: str,
) -> list[dict[str, Any]]:
    text = "\n".join(str(page.get("text") or "") for page in pages)
    compact = _compact_ocr(text)
    specs = [
        ("GOBIERNODEESPANIA", "position", "public_office", "Presidente del Gobierno"),
        ("PARTIDOSOCIALISTAOBREROESPANOL", "position", "party", "Secretario General PSOE"),
        ("SECRETARIOGENERALPSOE", "position", "party", "Secretario General PSOE"),
        ("EDICIONES62SA", "income", "royalties", "Derechos de autor por creacion literaria"),
        ("OBRASOCIALNUR", "position", "foundation", "Obra social NUR"),
        ("CUOTAAFILIADO", "position", "party", "Cuota de afiliado a partido"),
        (
            "APORTACIONCOMOCARGOPUBLICO",
            "position",
            "party",
            "Aportacion como cargo publico del PSOE",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for token, family, category, description in specs:
        if token not in compact:
            continue
        rows.append(
            _structured_extraction_row(
                source_sha256=source_sha256,
                source_url=source_url,
                snapshot_date=snapshot_date,
                document_kind="economic_interests",
                item_family=family,
                item_category=category,
                description=description,
                page_number=1
                if token not in {"OBRASOCIALNUR", "APORTACIONCOMOCARGOPUBLICO"}
                else 2,
                raw_evidence=description,
                confidence=0.78,
            )
        )
    return rows


def _structured_extraction_row(
    *,
    source_sha256: str,
    source_url: str,
    snapshot_date: str,
    document_kind: str,
    item_family: str,
    item_category: str,
    description: str,
    page_number: int,
    raw_evidence: str,
    confidence: float,
    item_subcategory: str | None = None,
    amount_eur: float | None = None,
    percentage: float | None = None,
    currency: str | None = None,
    location: str | None = None,
    company_name: str | None = None,
    role: str | None = None,
    needs_review: bool | None = None,
) -> dict[str, Any]:
    review = (
        confidence < review_threshold_for_model("deputy_document_rules")
        if needs_review is None
        else needs_review
    )
    return {
        "extraction_id": stable_id(
            source_sha256,
            page_number,
            item_family,
            item_category,
            description,
            amount_eur,
            percentage,
        ),
        "document_sha256": source_sha256,
        "source_url": source_url,
        "document_kind": document_kind,
        "declaration_phase": _declaration_phase(source_url, raw_evidence),
        "item_family": item_family,
        "item_category": item_category,
        "item_subcategory": item_subcategory,
        "label": description[:120],
        "description": description,
        "amount_eur": amount_eur,
        "percentage": percentage,
        "currency": currency,
        "date_value": _first_match(DATE_RE, raw_evidence),
        "location": location,
        "company_name": company_name,
        "role": role,
        "page_number": page_number,
        "text_span": description,
        "bbox": None,
        "extraction_method": "rules_v1",
        "model_name": model_name_from_contract("deputy_document_rules", "rules_v1"),
        "confidence": confidence,
        "needs_review": review,
        "raw_evidence": raw_evidence,
        "source_file_sha256": source_sha256,
        "snapshot_date": snapshot_date,
    }


def _nearest_previous_amount(lines: list[str], index: int, *, lookback: int = 4) -> float | None:
    for candidate in reversed(lines[max(0, index - lookback) : index]):
        amount = _structured_amount_to_float(candidate)
        if amount is not None:
            return amount
    return None


def _nearest_next_amount(lines: list[str], index: int, *, lookahead: int = 4) -> float | None:
    for candidate in lines[index + 1 : index + 1 + lookahead]:
        amount = _structured_amount_to_float(candidate)
        if amount is not None:
            return amount
    return None


def _structured_neighbor_description(
    lines: list[str],
    index: int,
    *,
    lookaround: int = 3,
    prefer_following_high_priority: bool = False,
) -> str | None:
    current = _strip_structured_amounts(lines[index])
    if _is_structured_data_description(current):
        return current
    if prefer_following_high_priority:
        for step in range(1, lookaround + 1):
            candidate_index = index + step
            if candidate_index >= len(lines):
                break
            candidate = _strip_structured_amounts(lines[candidate_index])
            if _is_high_priority_structured_description(candidate):
                return candidate
    offsets = [offset for step in range(1, lookaround + 1) for offset in (-step, step)]
    for offset in offsets:
        candidate_index = index + offset
        if candidate_index < 0 or candidate_index >= len(lines):
            continue
        if prefer_following_high_priority and offset < 0 and candidate_index > 0:
            previous_to_candidate = _structured_amount_to_float(lines[candidate_index - 1])
            if previous_to_candidate is not None:
                continue
        if (
            prefer_following_high_priority
            and offset < 0
            and any(
                _structured_amount_to_float(lines[position]) is not None
                for position in range(candidate_index + 1, index)
            )
        ):
            continue
        candidate = _strip_structured_amounts(lines[candidate_index])
        if _is_structured_data_description(candidate):
            return candidate
    return None


def _strip_structured_amounts(value: str) -> str:
    text = _clean_line(value)
    text = re.sub(r"\d{1,3}(?:[.,]\d{3})+[.,]\d{1,2}", " ", text)
    text = re.sub(r"\d+(?:,\d{1,2})", " ", text)
    text = re.sub(r"\d{1,3}(?:\.\d{3})+", " ", text)
    return _clean_line(text)


def _is_structured_data_description(value: str) -> bool:
    text = _clean_line(value)
    if len(text) < 4:
        return False
    if _is_pdf_stamp_line(text) or _is_boilerplate_extraction_line(text):
        return False
    compact = _compact_ocr(text)
    if not compact or re.fullmatch(r"\d+", compact):
        return False
    if compact in _STRUCTURED_SECTION_HEADINGS:
        return False
    if any(token in compact for token in _STRUCTURED_SKIP_TOKENS):
        return False
    return any(char.isalpha() for char in text)


def _is_high_priority_structured_description(value: str) -> bool:
    compact = _compact_ocr_loose(value)
    return any(token in compact for token in _HIGH_PRIORITY_STRUCTURED_TOKENS)


def _structured_income_category(description: str) -> str | None:
    compact = _compact_ocr_loose(description)
    if "IRPF" in compact:
        return None
    if any(token in compact for token in ("ARRENDAMIENTO", "ALQUILER")):
        return "rental_income"
    if "DIVIDENDO" in compact:
        return "dividends"
    if "DERECHOSDEAUTOR" in compact:
        return "royalties"
    if "INTERESES" in compact or "RENDIMIENTOSDECUENTAS" in compact:
        return "interest"
    if "PENSION" in compact:
        return "pension"
    if "DOCENCIA" in compact or "UNIVERSIDAD" in compact:
        return "teaching"
    if "CONFERENCIA" in compact:
        return "speaking"
    if "INDEMNIZACION" in compact:
        return "other_income"
    if any(
        token in compact
        for token in (
            "PRESIDENTEDELGOBIERNO",
            "RETRIBUCION",
            "SALARIO",
            "SUELDO",
            "PARLAMENTO",
            "GOBIERNO",
            "AYUNTAMIENTO",
            "MINISTERIO",
            "JUNTA",
            "DIETA",
        )
    ):
        return "public_salary"
    if any(token in compact for token in ("HONORARIOS", "ACTIVIDADPRIVADA")):
        return "private_activity"
    if "OTRASRENTAS" in compact or "OTRASPERCEPCIONES" in compact:
        return "other_income"
    return None


def _structured_income_label(description: str, category: str) -> str:
    compact = _compact_ocr_loose(description)
    known = (
        ("PRESIDENTEDELGOBIERNO", "Presidente del Gobierno"),
        ("DIVIDENDOSYOTROS", "Dividendos y otros"),
        ("ARRENDAMIENTOSDEINMUEBLES", "Arrendamientos de inmuebles"),
        ("DERECHOSDEAUTOR", "Derechos de autor"),
    )
    for token, label in known:
        if token in compact:
            return label
    return description[:120]


def _structured_asset_category(description: str) -> str | None:
    compact = _compact_ocr_loose(description)
    if "CUENTA" in compact:
        return "bank_account"
    if "DEPOSITO" in compact:
        return "deposit"
    if "PLANDEPENSION" in compact:
        return "pension_plan"
    if "FONDODEINVERSION" in compact or "FONDOSDEINVERSION" in compact:
        return "fund"
    if any(token in compact for token in ("ACCIONES", "PARTICIPACIONES", "VALORES")):
        return "shares"
    if "SEGURO" in compact:
        return "insurance"
    if any(token in compact for token in ("VEHICULO", "AUTOMOVIL", "TURISMO", "MOTOCICLETA")):
        return "vehicle"
    if "DONATIVO" in compact:
        return "other_asset"
    return None


def _structured_asset_label(description: str, category: str) -> str:
    compact = _compact_ocr_loose(description)
    known = (
        ("CUENTASCORRIENTES", "Cuentas corrientes"),
        ("PLANDEPENSIONES", "Plan de pensiones"),
        ("FONDODEINVERSION", "Fondo de inversion"),
        ("FONDOSDEINVERSION", "Fondo de inversion"),
        ("ACCIONESCOTIZADASENBOLSA", "Acciones cotizadas en bolsa"),
        ("DONATIVOSAPSOEYOTROS", "Donativos a PSOE y otros"),
    )
    for token, label in known:
        if token in compact:
            return label
    return description[:120]


def _compact_ocr_loose(value: str) -> str:
    compact = _compact_ocr(value)
    return compact.translate(str.maketrans({"0": "O", "6": "O", "1": "I"}))


def _structured_amount_to_float(value: str) -> float | None:
    text = _clean_line(value)
    if re.fullmatch(r"\d{1,2}", text):
        return None
    match = re.search(r"\d{1,3}(?:[.,]\d{3})+[.,]\d{1,2}", text)
    if match:
        return _parse_amount_token(match.group(0))
    match = re.search(r"\d+(?:,\d{1,2})", text)
    if match:
        return _parse_amount_token(match.group(0))
    match = re.search(r"\d{1,3}(?:\.\d{3})+", text)
    if match:
        return _parse_amount_token(match.group(0))
    return None


def _format_eur(amount: float) -> str:
    return f"{amount:,.2f} EUR".replace(",", "_").replace(".", ",").replace("_", ".")


def _compact_ocr(value: str) -> str:
    return re.sub(r"[^A-Z0-9ÁÉÍÓÚÜÑ]", "", _fold(value).upper())


def _first_known_location(lines: list[str]) -> str | None:
    for line in lines:
        compact = _compact_ocr(line)
        if compact in {"MADRID", "BARCELONA", "VALENCIA", "SEVILLA", "ZARAGOZA", "MALAGA"}:
            return compact.title()
    return None


def _first_year(text: str) -> str | None:
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return match.group(0) if match else None


def _human_join(parts: list[str | None]) -> str:
    return ", ".join(part for part in parts if part)


_STRUCTURED_SECTION_HEADINGS = {
    "EUROS",
    "CONCEPTO",
    "PROCEDENCIADE",
    "LASRENTAS",
    "BIENES",
    "CLASEYCARACTERISTICAS",
    "FECHADEADQUISICION",
    "TITULO",
    "SITUACION",
    "DERECHOSOBREELBIEN",
}
_STRUCTURED_SKIP_TOKENS = (
    "INDICARSISTEMA",
    "CANTIDADPAGADAPORIRPF",
    "RENTASPERCIBIDASPORELPARLAMENTARIO",
    "BIENESPATRIMONIALESDELPARLAMENTARIO",
    "OTROSBIENESODERECHOS",
    "DEUDASYOBLIGACIONESPATRIMONIALES",
)
_ECONOMIC_INTEREST_SKIP_TOKENS = (
    "DECLARACIONDEINTERESESECONOMICO",
    "PROTECCIONDEDATOS",
    "FINALIDADLASDECLARACIONES",
    "DESTINATARIOSLASDECLARACIONES",
    "ACTIVIDADESDESARROLLADAS",
    "DONACIONESOBSEQUIOS",
    "FUNDACIONESYOTRASASOCIACIONES",
    "OTROSINTERESES",
    "CODIGODECONDUCTA",
    "CONFLICTOSDEINTERESES",
    "PARTIDOSPOLITICOS",
    "APORTACIONCOMOCARGOPUBLICOESTABLECIDA",
    "TENERLACONDICIONDECARGOPUBLICO",
)
_HIGH_PRIORITY_STRUCTURED_TOKENS = (
    "PRESIDENTEDELGOBIERNO",
    "DIVIDENDOSYOTROS",
    "ARRENDAMIENTOSDEINMUEBLES",
    "DERECHOSDEAUTOR",
    "CUENTASCORRIENTES",
    "PLANDEPENSIONES",
    "FONDODEINVERSION",
    "ACCIONESCOTIZADASENBOLSA",
    "DONATIVOSAPSOEYOTROS",
)


_CONTINUATION_TAIL_RE = re.compile(
    r"(?:^|[\s,])(?:a|al|como|con|de|del|desde|e|el|en|hasta|la|las|los|o|para|por|que|sin|u|un|una|y)$",
    re.I,
)
_TERMINAL_PUNCTUATION_RE = re.compile(r"[.!?;:)]$")
_NEW_ITEM_PREFIX_RE = re.compile(r"^(?:\d+[.)]|[A-Z]\)|[-*])\s+")


def _logical_extraction_lines(text: str, *, document_kind: str) -> list[str]:
    lines: list[str] = []
    current: str | None = None
    for raw_line in text.splitlines():
        clean = _clean_line(raw_line)
        if not clean:
            if current:
                lines.append(current)
                current = None
            continue
        if current and _should_join_extraction_continuation(
            current,
            clean,
            document_kind=document_kind,
        ):
            current = f"{current} {clean}"
            continue
        if current:
            lines.append(current)
        current = clean
    if current:
        lines.append(current)
    return lines


def _should_join_extraction_continuation(
    previous: str,
    current: str,
    *,
    document_kind: str,
) -> bool:
    if _NEW_ITEM_PREFIX_RE.match(current):
        return False
    if _is_boilerplate_extraction_line(previous):
        return False
    previous_folded = _fold(previous).strip(" .,:;")
    if previous.endswith((",", "-")) or _CONTINUATION_TAIL_RE.search(previous_folded):
        return True

    current_classification = _classify_line(current, document_kind)
    if current_classification is not None and has_patrimony_evidence_signal(
        current,
        current_classification,
        document_kind=document_kind,
    ):
        return False
    current_is_boilerplate = _is_boilerplate_extraction_line(current)
    if current_is_boilerplate and not _looks_like_continuation_fragment(current):
        return False

    previous_classification = _classify_line(previous, document_kind)
    if previous_classification is None:
        return False
    if not _TERMINAL_PUNCTUATION_RE.search(previous):
        return True
    if (AMOUNT_RE.search(current) or PERCENT_RE.search(current)) and not _NEW_ITEM_PREFIX_RE.match(
        current
    ):
        return True
    return current[:1].islower() and not _TERMINAL_PUNCTUATION_RE.search(previous)


def _looks_like_continuation_fragment(line: str) -> bool:
    folded = _fold(line).strip(" .,:;")
    return (
        bool(line[:1].islower())
        or folded in {"remuneracion", "retribucion"}
        or folded.startswith(
            (
                "cualquier otra",
                "percibir ",
                "recibir ",
                "referido ",
                "tipo de ",
            )
        )
    )


def _classify_line(line: str, document_kind: str) -> dict[str, str] | None:
    return classify_patrimony_line(line, document_kind)


def _declaration_phase(source_url: str, text: str) -> str:
    lower = _fold(f"{source_url} {text}")
    if any(token in lower for token in ("final", "cese")):
        return "final"
    if any(token in lower for token in ("modificacion", "modifica", "complementaria")):
        return "modification"
    if any(token in lower for token in ("inicial", "alta")):
        return "initial"
    return "unknown"


def _candidate_table_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if _looks_like_table_row(line):
            current.append(line.strip())
            continue
        if len(current) >= 3:
            blocks.append("\n".join(current))
        current = []
    if len(current) >= 3:
        blocks.append("\n".join(current))
    return blocks


def _looks_like_table_row(line: str) -> bool:
    return bool(
        re.search(r"\s{2,}", line.strip()) or AMOUNT_RE.search(line) or PERCENT_RE.search(line)
    )


def _rough_column_count(lines: list[str]) -> int:
    counts = [len(re.split(r"\s{2,}", line.strip())) for line in lines if line.strip()]
    return max(counts) if counts else 0


def _location_matches(text: str) -> list[dict[str, Any]]:
    rows = []
    pattern = re.compile(
        r"\b(?:municipio|provincia|localidad|comunidad autonoma|pais)"
        r"\s+(?:de\s+)?([A-Z][A-Za-z .'-]{2,50})",
        re.I,
    )
    for match in pattern.finditer(text):
        rows.append(
            {
                "value": match.group(1).strip(" .,"),
                "start": match.start(1),
                "end": match.end(1),
            }
        )
    return rows


def _line_location(line: str) -> str | None:
    matches = _location_matches(line)
    return matches[0]["value"] if matches else None


def _first_amount(text: str) -> float | None:
    match = AMOUNT_RE.search(text)
    return _amount_to_float(match.group(1)) if match else None


def _first_percentage(text: str) -> float | None:
    match = PERCENT_RE.search(text)
    if not match:
        return None
    return float(match.group(1).replace(",", "."))


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1) if match.groups() else match.group(0)


def _amount_to_float(value: str) -> float | None:
    match = re.search(r"\d{1,3}(?:[.,]\d{3})+[.,]\d{1,2}", value)
    if match:
        return _parse_amount_token(match.group(0))
    match = re.search(r"\d[\d.\s]*,\d{1,4}", value)
    if not match:
        return None
    return _parse_amount_token(match.group(0))


def _parse_amount_token(value: str) -> float | None:
    """Parse Spanish or mixed OCR number separators without truncating the integer part."""
    token = re.sub(r"\s", "", value).strip(".,")
    if not token or not re.fullmatch(r"\d[\d.,]*", token):
        return None
    separators = [index for index, char in enumerate(token) if char in ".,"]
    if not separators:
        return float(token)
    decimal_index = separators[-1]
    fractional = token[decimal_index + 1 :]
    integer = re.sub(r"[.,]", "", token[:decimal_index])
    normalized = (
        f"{integer}.{fractional}" if len(fractional) in {1, 2} else re.sub(r"[.,]", "", token)
    )
    try:
        return float(normalized)
    except ValueError:
        return None


def _line_confidence(line: str, amount: float | None, classified: dict[str, str]) -> float:
    confidence = 0.62
    if amount is not None:
        confidence += 0.12
    if classified["item_category"] not in {
        "other_income",
        "other_asset",
        "other_liability",
        "other_position",
        "other_company_interest",
    }:
        confidence += 0.12
    if len(line) > 180:
        confidence -= 0.08
    return max(0.35, min(confidence, 0.9))


def _is_boilerplate_extraction_line(line: str) -> bool:
    return is_patrimony_boilerplate(line)


def _is_structured_form_noise_line(line: str, document_kind: str) -> bool:
    compact = _compact_ocr(line)
    if document_kind == "assets_income":
        return any(token in compact for token in _STRUCTURED_SKIP_TOKENS)
    if document_kind == "economic_interests":
        return any(token in compact for token in _ECONOMIC_INTEREST_SKIP_TOKENS)
    return False


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip(" -:\t")


def _fold(value: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char)).lower()


def _evidence_window(text: str, start: int, end: int) -> str:
    left = max(0, start - 80)
    right = min(len(text), end + 80)
    return _clean_line(text[left:right])


def _dedupe_rows(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped[row[key]] = row
    return list(deduped.values())


def _merge_extraction_rows(
    structured_rows: list[dict[str, Any]],
    generic_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = list(structured_rows)
    for row in generic_rows:
        if any(_extraction_rows_overlap(row, structured) for structured in structured_rows):
            continue
        merged.append(row)
    return _dedupe_rows(merged, "extraction_id")


def _extraction_rows_overlap(row: dict[str, Any], structured: dict[str, Any]) -> bool:
    if row.get("document_sha256") != structured.get("document_sha256"):
        return False
    if row.get("page_number") != structured.get("page_number"):
        return False
    if row.get("item_family") != structured.get("item_family"):
        return False
    if (
        row.get("item_category") != structured.get("item_category")
        and row.get("item_family") != "position"
    ):
        return False
    row_amount = row.get("amount_eur")
    structured_amount = structured.get("amount_eur")
    if row_amount is not None and structured_amount is not None:
        return abs(float(row_amount) - float(structured_amount)) < 0.01
    row_percentage = row.get("percentage")
    structured_percentage = structured.get("percentage")
    if row_percentage is not None and structured_percentage is not None:
        return abs(float(row_percentage) - float(structured_percentage)) < 0.01
    row_description = _compact_ocr(str(row.get("description") or ""))
    structured_description = _compact_ocr(str(structured.get("description") or ""))
    return bool(
        row_description
        and structured_description
        and (row_description in structured_description or structured_description in row_description)
    )

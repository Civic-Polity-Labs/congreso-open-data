"""High-level records returned by the domain facade."""

from __future__ import annotations

from typing import Literal

from congreso_open_data.models import ExtractionCandidate, Intervention, SourceRef


class InterventionRecord(Intervention):
    """An intervention plus optional native text and reviewable extractions."""

    text: str | None = None
    text_status: Literal[
        "not_requested",
        "matched",
        "speaker_not_found",
        "document_unavailable",
        "extraction_failed",
    ] = "not_requested"
    text_source: SourceRef | None = None
    text_method: str | None = None
    text_confidence: float | None = None
    extractions: tuple[ExtractionCandidate, ...] = ()

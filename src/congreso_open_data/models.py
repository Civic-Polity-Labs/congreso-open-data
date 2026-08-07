"""Typed public contracts for acquisition, normalization and reviewable extraction."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from datetime import date as Date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SECRET_KEY = re.compile(r"(?:api[_-]?key|token|secret|password|authorization|credential)", re.I)


def redact_parameters(value: Any) -> Any:
    """Return a JSON-compatible copy with credential-like fields removed."""

    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _SECRET_KEY.search(str(key)) else redact_parameters(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_parameters(item) for item in value]
    return value


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True, populate_by_name=True)


class SourceRef(PublicModel):
    requested_url: str
    effective_url: str | None = None
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    parameters: dict[str, Any] = Field(default_factory=dict)
    adapter: str
    adapter_version: str
    normalization_version: str
    method: str | None = None
    model: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @field_validator("parameters", "diagnostics", mode="before")
    @classmethod
    def _redact(cls, value: Any) -> Any:
        return redact_parameters(value or {})


class CatalogResource(PublicModel):
    family: str
    dataset: str
    format: str
    url: str
    snapshot_token: str | None = None
    legislature: str | None = None
    session: str | None = None
    vote_number: str | None = None
    post_data: dict[str, Any] | None = None

    @field_validator("post_data", mode="before")
    @classmethod
    def _redact_post_data(cls, value: Any) -> Any:
        return None if value is None else redact_parameters(value)


class ArtifactManifest(PublicModel):
    schema_version: str = "1.0"
    family: str
    dataset: str
    format: str
    source_url: str
    effective_url: str | None = None
    snapshot_token: str | None = None
    run_date: str
    fetched_at: datetime | str
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    bytes: int = Field(ge=0)
    payload_path: str
    manifest_path: str | None = None
    request_parameters: str | dict[str, Any] | None = None
    adapter: str = "congreso.catalog"
    adapter_version: str = "1.0.0"
    normalization_version: str = "1.0.0"
    content_type: str | None = None
    http_status: int | None = None
    legislature: str | None = None
    session: str | None = None
    vote_number: str | None = None

    @field_validator("request_parameters", mode="before")
    @classmethod
    def _redact_request_parameters(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return redact_parameters(value)
        return value

    @classmethod
    def from_legacy(cls, raw: dict[str, Any]) -> ArtifactManifest:
        """Read old unversioned Bronze sidecars without changing their payload."""

        mapped = dict(raw)
        mapped.setdefault("schema_version", "legacy-0")
        mapped.setdefault("adapter", "congreso.legacy")
        mapped.setdefault("adapter_version", "0")
        mapped.setdefault("normalization_version", "0")
        mapped.setdefault("effective_url", mapped.get("source_url"))
        mapped.setdefault("bytes", mapped.pop("content_length", 0))
        mapped.setdefault("payload_path", mapped.pop("local_path", ""))
        return cls.model_validate(mapped)

    def source_ref(self, *, method: str | None = None, model: str | None = None) -> SourceRef:
        parameters: dict[str, Any]
        if isinstance(self.request_parameters, dict):
            parameters = self.request_parameters
        else:
            parameters = {"canonical": self.request_parameters} if self.request_parameters else {}
        return SourceRef(
            requested_url=self.source_url,
            effective_url=self.effective_url or self.source_url,
            sha256=self.sha256,
            fetched_at=self.fetched_at,
            parameters=parameters,
            adapter=self.adapter,
            adapter_version=self.adapter_version,
            normalization_version=self.normalization_version,
            method=method,
            model=model,
        )


class ExtractionEvidence(PublicModel):
    text: str | None = None
    span_start: int | None = Field(default=None, ge=0)
    span_end: int | None = Field(default=None, ge=0)
    page: int | None = Field(default=None, ge=1)
    bbox: tuple[float, float, float, float] | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    backend: str
    model: str
    version: str
    literal: bool = True
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_span(self) -> ExtractionEvidence:
        if (self.span_start is None) != (self.span_end is None):
            raise ValueError("span_start and span_end must be supplied together")
        if (
            self.span_start is not None
            and self.span_end is not None
            and self.span_end < self.span_start
        ):
            raise ValueError("span_end must be greater than or equal to span_start")
        return self


class ExtractionCandidate(PublicModel):
    candidate_id: str
    kind: str
    value: Any
    evidence: tuple[ExtractionEvidence, ...]
    source: SourceRef
    status: Literal["review_required"] = "review_required"
    inferred: bool = True


class ExtractionSpec(PublicModel):
    engine: Literal["native", "rules", "nlp", "ocr", "llm"]
    backend: str = Field(min_length=1)
    model: str = Field(min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)
    fallback: tuple[ExtractionSpec, ...] = ()

    @field_validator("options", mode="before")
    @classmethod
    def _redact_options(cls, value: Any) -> Any:
        return redact_parameters(value or {})


class ExtractionPlan(PublicModel):
    families: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()
    formats: tuple[str, ...] = ("json", "xml", "csv", "html", "pdf", "zip", "png")
    resources: tuple[CatalogResource, ...] = ()
    output_root: Path
    run_date: str = Field(default_factory=lambda: Date.today().isoformat())
    specs: tuple[ExtractionSpec, ...] = ()
    batch_size: int = Field(default=32, ge=1, le=1000)
    max_workers: int = Field(default=4, ge=1, le=16)
    request_interval_seconds: float = Field(default=0.2, ge=0)
    resume: bool = True
    continue_on_error: bool = True


class ExtractionFailure(PublicModel):
    resource_key: str
    family: str
    dataset: str
    source_url: str
    error_type: str
    error_message: str
    snapshot_token: str | None = None


class ExtractionRun(PublicModel):
    run_id: str
    started_at: datetime
    finished_at: datetime | None = None
    planned: int = 0
    succeeded: int = 0
    reused: int = 0
    failed: int = 0
    manifest_index_path: str | None = None
    checkpoint_path: str | None = None
    failures: tuple[ExtractionFailure, ...] = ()


class CongressRecord(PublicModel):
    source: SourceRef


class Deputy(CongressRecord):
    deputy_id: str
    full_name: str
    legislature: str | None = None
    constituency: str | None = None
    parliamentary_group: str | None = None


class DeputyProfile(CongressRecord):
    deputy_id: str
    full_name: str
    profile_url: str | None = None
    legislature: str | None = None


class InterestDeclaration(CongressRecord):
    declaration_id: str
    deputy_id: str | None = None
    description: str | None = None
    amount_eur: float | None = None


class FinancialDocument(CongressRecord):
    document_id: str
    deputy_id: str | None = None
    document_kind: str
    url: str


class Organ(CongressRecord):
    organ_id: str
    name: str
    organ_type: str | None = None
    legislature: str | None = None


class OrganMembership(CongressRecord):
    membership_id: str
    organ_id: str
    deputy_id: str | None = None
    role: str | None = None


class Initiative(CongressRecord):
    initiative_id: str
    title: str
    file_number: str | None = None
    legislature: str | None = None
    initiative_type: str | None = None


class Intervention(CongressRecord):
    intervention_id: str
    title: str | None = None
    speaker: str | None = None
    legislature: str | None = None


class InterventionOccurrence(CongressRecord):
    occurrence_id: str
    intervention_id: str
    date: Date | None = None
    organ_id: str | None = None


class VoteEvent(CongressRecord):
    vote_id: str
    title: str | None = None
    legislature: str | None = None
    session: str | None = None
    vote_number: str | None = None


class VoteItem(CongressRecord):
    vote_item_id: str
    vote_id: str
    result: str | None = None
    yes: int | None = None
    no: int | None = None
    abstentions: int | None = None


class NominalVote(CongressRecord):
    nominal_vote_id: str
    vote_id: str
    deputy_id: str | None = None
    deputy_name: str | None = None
    position: str


class DocumentAsset(CongressRecord):
    document_id: str
    url: str
    mime_type: str | None = None
    document_kind: str | None = None


class DocumentText(CongressRecord):
    document_id: str
    text: str
    page: int | None = None
    extraction_method: str
    model: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: tuple[ExtractionEvidence, ...] = ()


class SpeechBlock(CongressRecord):
    speech_block_id: str
    document_id: str
    text: str
    speaker: str | None = None
    sequence: int = Field(ge=0)


class SalaryEntitlement(CongressRecord):
    entitlement_id: str
    label: str
    amount_eur: float | None = None
    effective_date: Date | None = None


NormalizedRecord = (
    Deputy
    | DeputyProfile
    | InterestDeclaration
    | FinancialDocument
    | Organ
    | OrganMembership
    | Initiative
    | Intervention
    | InterventionOccurrence
    | VoteEvent
    | VoteItem
    | NominalVote
    | DocumentAsset
    | DocumentText
    | SpeechBlock
    | SalaryEntitlement
)

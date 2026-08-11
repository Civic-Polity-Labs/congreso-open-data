"""Provider-neutral model contracts and callable adapters.

Runtime objects (SDK clients, functions and credentials) deliberately live here,
outside serializable extraction/query specifications.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from congreso_open_data.models import (
    ExtractionCandidate,
    ExtractionEvidence,
    ExtractionSpec,
    SourceRef,
    redact_parameters,
)
from congreso_open_data.protocols import ExtractionContext, ExtractionResult


class ModelPluginError(RuntimeError):
    """Base error for a provider-neutral model backend."""


class ModelUnavailableError(ModelPluginError):
    """The requested runtime model is not installed or configured."""


class ModelTimeoutError(ModelPluginError):
    """A model call exceeded its declared time budget."""


class ModelResponseError(ModelPluginError):
    """A model returned a payload that violates the candidate contract."""


class ModelDescriptor(BaseModel):
    """Stable, non-secret identity recorded for every model call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    model: str = Field(min_length=1)
    version: str = Field(min_length=1)
    provider: str | None = None
    engine: Literal["llm", "nlp", "ocr", "custom"] = "llm"

    @field_validator("name", "model", "version", "provider", mode="before")
    @classmethod
    def _strip(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class ModelRequest(BaseModel):
    """Bounded request supplied to every custom model implementation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    instructions: str
    output_schema: dict[str, Any]
    source: SourceRef
    timeout_seconds: float = Field(default=60.0, gt=0, le=3600)
    max_output_tokens: int = Field(default=2048, ge=1, le=1_000_000)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def _redact_metadata(cls, value: Any) -> dict[str, Any]:
        redacted = redact_parameters(value or {})
        if not isinstance(redacted, dict):
            raise ValueError("metadata must be an object")
        return {str(key): item for key, item in redacted.items()}


class ModelResponse(BaseModel):
    """Provider-neutral response; payload is validated before becoming candidates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    payload: str | dict[str, Any]
    request_id: str | None = None
    usage: dict[str, int | float] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @field_validator("diagnostics", mode="before")
    @classmethod
    def _redact_diagnostics(cls, value: Any) -> dict[str, Any]:
        redacted = redact_parameters(value or {})
        if not isinstance(redacted, dict):
            raise ValueError("diagnostics must be an object")
        return {str(key): item for key, item in redacted.items()}


@runtime_checkable
class ModelBackend(Protocol):
    """Structural protocol implemented by local, cloud and plugin models."""

    @property
    def descriptor(self) -> ModelDescriptor: ...

    def generate(self, request: ModelRequest) -> ModelResponse: ...


ModelCallable = Callable[[ModelRequest], ModelResponse | Mapping[str, Any] | str]


@dataclass(frozen=True, init=False)
class CallableModelBackend:
    """Adapt a synchronous Python callable without weakening the model contract."""

    descriptor: ModelDescriptor
    function: ModelCallable

    def __init__(
        self,
        *,
        name: str,
        model: str,
        version: str,
        function: ModelCallable,
        provider: str | None = None,
        engine: Literal["llm", "nlp", "ocr", "custom"] = "llm",
    ) -> None:
        object.__setattr__(
            self,
            "descriptor",
            ModelDescriptor(
                name=name,
                model=model,
                version=version,
                provider=provider,
                engine=engine,
            ),
        )
        object.__setattr__(self, "function", function)

    def generate(self, request: ModelRequest) -> ModelResponse:
        try:
            response = self.function(request)
        except TimeoutError as exc:
            raise ModelTimeoutError(
                f"Model callable {self.descriptor.name!r} exceeded its time budget"
            ) from exc
        except ModelPluginError:
            raise
        except Exception as exc:
            raise ModelPluginError(
                f"Model callable {self.descriptor.name!r} failed: {type(exc).__name__}"
            ) from exc
        if isinstance(response, ModelResponse):
            return response
        if isinstance(response, Mapping):
            return ModelResponse(payload=dict(response))
        if isinstance(response, str):
            return ModelResponse(payload=response)
        raise ModelResponseError(
            "A callable model must return ModelResponse, a mapping, or a JSON string"
        )


class CandidateValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = Field(min_length=1)
    value: Any
    quote: str
    confidence: float | None = Field(default=None, ge=0, le=1)


class CandidateEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidates: tuple[CandidateValue, ...] = ()


CANDIDATE_ENVELOPE_SCHEMA: dict[str, Any] = CandidateEnvelope.model_json_schema()


class ExtractionLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    timeout_seconds: float = Field(default=60.0, gt=0, le=3600)
    max_output_tokens: int = Field(default=2048, ge=1, le=1_000_000)
    max_input_characters: int = Field(default=32_000, ge=1, le=10_000_000)
    chunk_characters: int = Field(default=24_000, ge=1, le=10_000_000)
    chunk_overlap_characters: int = Field(default=500, ge=0, le=100_000)
    max_chunks: int = Field(default=100, ge=1, le=100_000)

    @field_validator("chunk_overlap_characters")
    @classmethod
    def _valid_overlap(cls, value: int, info: Any) -> int:
        chunk_size = info.data.get("chunk_characters")
        if isinstance(chunk_size, int) and value >= chunk_size:
            raise ValueError("chunk_overlap_characters must be smaller than chunk_characters")
        return value


class ExtractionTask(BaseModel):
    """Serializable task; runtime backends are resolved separately by name."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    backend: ExtractionSpec
    output_schema: dict[str, Any] = Field(
        default_factory=lambda: json.loads(json.dumps(CANDIDATE_ENVELOPE_SCHEMA))
    )
    limits: ExtractionLimits = Field(default_factory=ExtractionLimits)

    @field_validator("output_schema", mode="before")
    @classmethod
    def _json_schema(cls, value: Any) -> dict[str, Any]:
        if isinstance(value, type) and issubclass(value, BaseModel):
            value = value.model_json_schema()
        try:
            serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("output_schema must be JSON serializable") from exc
        parsed = json.loads(serialized)
        if not isinstance(parsed, dict):
            raise ValueError("output_schema must be a JSON object")
        return parsed

    @field_validator("backend")
    @classmethod
    def _serializable_backend(cls, value: ExtractionSpec) -> ExtractionSpec:
        forbidden = {
            str(key)
            for key in value.options
            if str(key).casefold()
            in {"client", "callable", "function", "api_key", "token", "credentials"}
        }
        if forbidden:
            raise ValueError(
                "Runtime clients, callables and credentials belong in ModelRegistry, "
                f"not ExtractionTask.backend.options: {sorted(forbidden)}"
            )
        try:
            json.dumps(value.options, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("ExtractionTask.backend.options must be JSON serializable") from exc
        return value


class StructuredModelExtractor:
    """Turn a provider-neutral generator into the package ExtractionResult."""

    name = "structured-model"
    engine = "llm"
    version = "1.1.0"

    def __init__(self, backend: ModelBackend, task: ExtractionTask) -> None:
        self.backend = backend
        self.task = task

    @property
    def model(self) -> str:
        return self.backend.descriptor.model

    def extract(self, content: bytes, context: ExtractionContext) -> ExtractionResult:
        text = content.decode(str(context.metadata.get("encoding", "utf-8")), errors="replace")
        if len(text) > self.task.limits.max_input_characters:
            raise ValueError(
                f"Model input contains {len(text)} characters; configured maximum is "
                f"{self.task.limits.max_input_characters}"
            )
        chunks = _text_chunks(
            text,
            size=self.task.limits.chunk_characters,
            overlap=self.task.limits.chunk_overlap_characters,
            max_chunks=self.task.limits.max_chunks,
        )
        descriptor = self.backend.descriptor
        source = context.source.model_copy(
            update={"method": descriptor.name, "model": descriptor.model}
        )
        accumulated: dict[
            tuple[str, str],
            tuple[CandidateValue, list[ExtractionEvidence]],
        ] = {}
        calls: list[dict[str, Any]] = []
        for chunk_index, (chunk_start, chunk_text) in enumerate(chunks):
            response = self.backend.generate(
                ModelRequest(
                    text=chunk_text,
                    instructions=self.task.instructions,
                    output_schema=self.task.output_schema,
                    source=context.source,
                    timeout_seconds=self.task.limits.timeout_seconds,
                    max_output_tokens=self.task.limits.max_output_tokens,
                    metadata={
                        **context.metadata,
                        "task": self.task.name,
                        "chunk_index": chunk_index,
                        "chunk_start": chunk_start,
                        "chunk_end": chunk_start + len(chunk_text),
                    },
                )
            )
            envelope = _candidate_envelope(response.payload)
            calls.append(
                {
                    "chunk_index": chunk_index,
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_start + len(chunk_text),
                    "request_id": response.request_id,
                    "usage": dict(response.usage),
                    "provider_diagnostics": response.diagnostics,
                    "candidate_count": len(envelope.candidates),
                }
            )
            for item in envelope.candidates:
                evidence = _quote_evidence(
                    source_text=chunk_text,
                    quote=item.quote,
                    confidence=item.confidence,
                    descriptor=descriptor,
                    offset=chunk_start,
                )
                key = (
                    item.kind,
                    json.dumps(item.value, ensure_ascii=False, sort_keys=True, default=repr),
                )
                _, stored_evidence = accumulated.setdefault(key, (item, []))
                known = {
                    (
                        evidence_item.text,
                        evidence_item.span_start,
                        evidence_item.span_end,
                        evidence_item.literal,
                    )
                    for evidence_item in stored_evidence
                }
                stored_evidence.extend(
                    evidence_item
                    for evidence_item in evidence
                    if (
                        evidence_item.text,
                        evidence_item.span_start,
                        evidence_item.span_end,
                        evidence_item.literal,
                    )
                    not in known
                )
        candidates: list[ExtractionCandidate] = []
        all_evidence: list[ExtractionEvidence] = []
        for index, ((kind, canonical_value), (item, candidate_evidence)) in enumerate(
            accumulated.items()
        ):
            digest = hashlib.sha256(
                (
                    f"{source.sha256}:{descriptor.name}:{descriptor.model}:"
                    f"{descriptor.version}:{self.task.name}:{index}:{kind}:{canonical_value}"
                ).encode()
            ).hexdigest()
            candidates.append(
                ExtractionCandidate(
                    candidate_id=f"{descriptor.name}:{digest[:24]}",
                    kind=item.kind,
                    value=item.value,
                    evidence=tuple(candidate_evidence),
                    source=source,
                )
            )
            all_evidence.extend(candidate_evidence)
        return ExtractionResult(
            texts=(text,),
            candidates=candidates,
            evidence=all_evidence,
            diagnostics={
                "task": self.task.name,
                "backend": descriptor.name,
                "model": descriptor.model,
                "model_version": descriptor.version,
                "chunk_count": len(chunks),
                "calls": calls,
                "request_id": calls[0]["request_id"] if len(calls) == 1 else None,
                "usage": calls[0]["usage"] if len(calls) == 1 else {},
                "provider_diagnostics": (
                    calls[0]["provider_diagnostics"] if len(calls) == 1 else {}
                ),
                "candidate_count": len(candidates),
            },
        )


def _candidate_envelope(payload: str | dict[str, Any]) -> CandidateEnvelope:
    try:
        if isinstance(payload, str):
            return CandidateEnvelope.model_validate_json(payload)
        return CandidateEnvelope.model_validate(payload)
    except Exception as exc:
        raise ModelResponseError("Model response violates CandidateEnvelope") from exc


def _quote_evidence(
    *,
    source_text: str,
    quote: str,
    confidence: float | None,
    descriptor: ModelDescriptor,
    offset: int = 0,
) -> tuple[ExtractionEvidence, ...]:
    spans = (
        [(match.start(), match.end()) for match in re.finditer(re.escape(quote), source_text)]
        if quote
        else []
    )
    if not spans:
        return (
            ExtractionEvidence(
                text=quote or None,
                confidence=confidence,
                backend=descriptor.name,
                model=descriptor.model,
                version=descriptor.version,
                literal=False,
                diagnostics={"quote_occurrences": 0},
            ),
        )
    return tuple(
        ExtractionEvidence(
            text=quote,
            span_start=offset + start,
            span_end=offset + end,
            confidence=confidence,
            backend=descriptor.name,
            model=descriptor.model,
            version=descriptor.version,
            literal=True,
            diagnostics={"quote_occurrences": len(spans), "occurrence_index": index},
        )
        for index, (start, end) in enumerate(spans)
    )


def _text_chunks(
    text: str,
    *,
    size: int,
    overlap: int,
    max_chunks: int,
) -> tuple[tuple[int, str], ...]:
    if not text:
        return ((0, ""),)
    chunks: list[tuple[int, str]] = []
    start = 0
    step = size - overlap
    while start < len(text):
        chunks.append((start, text[start : start + size]))
        if len(chunks) > max_chunks:
            raise ValueError(
                f"Model input requires more than max_chunks={max_chunks}; "
                "raise the explicit extraction limit or reduce the input"
            )
        if start + size >= len(text):
            break
        start += step
    return tuple(chunks)


def model_fingerprint(backend: ModelBackend) -> str:
    """Return a deterministic, non-secret backend identity for checkpoints."""

    payload = backend.descriptor.model_dump(mode="json")
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()

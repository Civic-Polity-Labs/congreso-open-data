"""Dependency-injection contracts for transport, storage and normalization."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

from congreso_open_data.models import (
    ArtifactManifest,
    CatalogResource,
    ExtractionCandidate,
    ExtractionEvidence,
    SourceRef,
)


class AcquisitionResult(Protocol):
    content: bytes
    requested_url: str
    effective_url: str
    status_code: int
    headers: Mapping[str, str]


@runtime_checkable
class HttpTransport(Protocol):
    def get(self, url: str, **kwargs: Any) -> AcquisitionResult: ...

    def post(self, url: str, **kwargs: Any) -> AcquisitionResult: ...


@runtime_checkable
class ArtifactStore(Protocol):
    def persist(
        self,
        *,
        resource: CatalogResource,
        result: AcquisitionResult,
        run_date: str,
    ) -> ArtifactManifest: ...

    def open(self, manifest: ArtifactManifest) -> bytes: ...


@runtime_checkable
class SourceAdapter(Protocol):
    name: str
    version: str

    def catalog(self) -> Iterator[CatalogResource]: ...

    def acquire(self, resource: CatalogResource, *, run_date: str) -> ArtifactManifest: ...


@runtime_checkable
class Normalizer(Protocol):
    name: str
    version: str

    def normalize(self, manifest: ArtifactManifest) -> Iterable[Any]: ...


class ExtractionContext:
    """Immutable context supplied to every extraction backend."""

    __slots__ = ("metadata", "mime_type", "source")

    def __init__(
        self,
        *,
        source: SourceRef,
        mime_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.source = source
        self.mime_type = mime_type
        self.metadata = dict(metadata or {})


class ExtractionResult:
    """Bounded result. Inferred values remain candidates, never canonical facts."""

    __slots__ = ("candidates", "diagnostics", "evidence", "texts")

    def __init__(
        self,
        *,
        texts: Iterable[str] = (),
        candidates: Iterable[ExtractionCandidate] = (),
        evidence: Iterable[ExtractionEvidence] = (),
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        self.texts = tuple(texts)
        self.candidates = tuple(candidates)
        self.evidence = tuple(evidence)
        self.diagnostics = dict(diagnostics or {})


@runtime_checkable
class ExtractorBackend(Protocol):
    name: ClassVar[str]
    engine: ClassVar[str]
    version: ClassVar[str]
    model: str

    def extract(self, content: bytes, context: ExtractionContext) -> ExtractionResult: ...


def ensure_bounded_file(path: Path, *, max_bytes: int) -> bytes:
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"Artifact is {size} bytes; configured limit is {max_bytes}")
    return path.read_bytes()

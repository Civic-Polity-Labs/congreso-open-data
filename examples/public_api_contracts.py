"""Deterministic, executable examples for every stable top-level API symbol.

This example deliberately uses a tiny injected source adapter.  It exercises the
real acquisition, manifest, checkpoint, normalization, facade, streaming and model
contracts without contacting congreso.es.  Use ``verify_live_all_domains.py`` for
the complementary bounded verification against the current official sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import congreso_open_data as public_api
from congreso_open_data import (
    AmbiguousEntityError,
    ArtifactManifest,
    CallableModelBackend,
    CandidateEnvelope,
    CandidateValue,
    CatalogResource,
    Congress,
    CongressClient,
    CongressError,
    CongressQuery,
    Deputy,
    DeputyProfile,
    DeputyQuery,
    DocumentAsset,
    DocumentQuery,
    DocumentText,
    EntityNotFoundError,
    ExtractionCandidate,
    ExtractionEvidence,
    ExtractionFailure,
    ExtractionLimits,
    ExtractionPlan,
    ExtractionRun,
    ExtractionSpec,
    ExtractionTask,
    FinancialDocument,
    FinancialDocumentQuery,
    IncompleteResultError,
    Initiative,
    InitiativeQuery,
    InterestDeclaration,
    InterestQuery,
    Intervention,
    InterventionOccurrence,
    InterventionQuery,
    InterventionRecord,
    ModelBackend,
    ModelDescriptor,
    ModelRegistry,
    ModelRequest,
    ModelResponse,
    NominalVote,
    Organ,
    OrganMembership,
    OrganQuery,
    ProfileQuery,
    QueryRun,
    QueryValidationError,
    RefreshPolicy,
    ResultConsumedError,
    SalaryEntitlement,
    SalaryEntitlementQuery,
    SearchResult,
    SortOrder,
    SourceContractError,
    SourceRef,
    SourceUnavailableError,
    SpeechBlock,
    StructuredModelExtractor,
    TextPolicy,
    VoteEvent,
    VoteItem,
    VoteQuery,
)
from congreso_open_data.plugins import ExtractionContext

# This is intentionally explicit.  Its equality with ``public_api.__all__`` is a
# tested documentation-coverage contract, not a value generated from the package.
PUBLIC_API_EXAMPLES = {
    "AmbiguousEntityError": "errors",
    "ArtifactManifest": "acquisition",
    "CallableModelBackend": "models",
    "CandidateEnvelope": "models",
    "CandidateValue": "models",
    "CatalogResource": "acquisition",
    "Congress": "facade",
    "CongressClient": "acquisition",
    "CongressError": "errors",
    "CongressQuery": "queries",
    "Deputy": "records",
    "DeputyProfile": "records",
    "DeputyQuery": "queries",
    "DocumentAsset": "records",
    "DocumentQuery": "queries",
    "DocumentText": "records",
    "EntityNotFoundError": "errors",
    "ExtractionCandidate": "models",
    "ExtractionEvidence": "models",
    "ExtractionFailure": "acquisition",
    "ExtractionLimits": "models",
    "ExtractionPlan": "acquisition",
    "ExtractionRun": "acquisition",
    "ExtractionSpec": "models",
    "ExtractionTask": "models",
    "FinancialDocument": "records",
    "FinancialDocumentQuery": "queries",
    "IncompleteResultError": "errors",
    "Initiative": "records",
    "InitiativeQuery": "queries",
    "InterestDeclaration": "records",
    "InterestQuery": "queries",
    "Intervention": "records",
    "InterventionOccurrence": "records",
    "InterventionQuery": "queries",
    "InterventionRecord": "records",
    "ModelBackend": "models",
    "ModelDescriptor": "models",
    "ModelRegistry": "models",
    "ModelRequest": "models",
    "ModelResponse": "models",
    "NominalVote": "records",
    "Organ": "records",
    "OrganMembership": "records",
    "OrganQuery": "queries",
    "ProfileQuery": "queries",
    "QueryRun": "results",
    "QueryValidationError": "errors",
    "RefreshPolicy": "queries",
    "ResultConsumedError": "errors",
    "SalaryEntitlement": "records",
    "SalaryEntitlementQuery": "queries",
    "SearchResult": "results",
    "SortOrder": "queries",
    "SourceContractError": "errors",
    "SourceRef": "provenance",
    "SourceUnavailableError": "errors",
    "SpeechBlock": "records",
    "StructuredModelExtractor": "models",
    "TextPolicy": "queries",
    "VoteEvent": "records",
    "VoteItem": "records",
    "VoteQuery": "queries",
}


class DemoAdapter:
    """One bounded official-source-shaped dependency for the offline example."""

    name = "example-adapter"
    version = "1"

    def __init__(self, root: Path) -> None:
        self.root = root

    def catalog(self) -> Iterator[CatalogResource]:
        yield CatalogResource(
            family="diputados",
            dataset="Diputados",
            format="json",
            url="https://example.test/diputados.json",
            legislature="Leg.15",
        )

    def acquire(self, resource: CatalogResource, *, run_date: str) -> ArtifactManifest:
        content = json.dumps(
            [{"nombre": "Ana", "apellidos": "García", "legislatura": "XV"}],
            ensure_ascii=False,
        ).encode()
        relative_path = Path("bronze") / "diputados" / "diputados.json"
        payload_path = self.root / relative_path
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_bytes(content)
        return ArtifactManifest(
            family=resource.family,
            dataset=resource.dataset,
            format=resource.format,
            source_url=resource.url,
            effective_url=resource.url,
            run_date=run_date,
            fetched_at=datetime(2026, 8, 10, tzinfo=UTC),
            sha256=hashlib.sha256(content).hexdigest(),
            bytes=len(content),
            payload_path=str(relative_path),
            adapter=self.name,
            adapter_version=self.version,
            normalization_version="1",
            legislature="Leg.15",
        )


def source_example() -> SourceRef:
    source = SourceRef(
        requested_url="https://example.test/documento",
        effective_url="https://example.test/documento",
        sha256="a" * 64,
        fetched_at=datetime(2026, 8, 10, tzinfo=UTC),
        parameters={"api_token": "never persisted", "legislature": "XV"},
        adapter="example",
        adapter_version="1",
        normalization_version="1",
        method="native-json",
    )
    assert source.parameters["api_token"] == "[REDACTED]"
    return source


def query_examples() -> tuple[CongressQuery, ...]:
    queries: tuple[CongressQuery, ...] = (
        CongressQuery(domain="custom", legislatures=(15,), max_results=1),
        DeputyQuery(name="Ana García", legislatures=(15,), max_results=10),
        ProfileQuery(deputy_id="189", max_results=1),
        InterestQuery(deputy="Pedro Sánchez", declaration_kind="bienes", max_results=10),
        FinancialDocumentQuery(deputy="Pedro Sánchez", document_kind="bienes"),
        InitiativeQuery(file_number="121/000001", title="vivienda"),
        InterventionQuery(
            speaker="Pedro Sánchez",
            last_months=3,
            text_policy=TextPolicy.NATIVE,
            sort=SortOrder.DESCENDING,
            refresh=RefreshPolicy.AUTO,
            max_results=100,
        ),
        VoteQuery(session="193", vote_number="1", max_results=500),
        OrganQuery(name="Comisión", organ_type="Comisión"),
        SalaryEntitlementQuery(role="Secretario General", max_results=50),
        DocumentQuery(
            source_families=("organos",),
            source_datasets=("ComisionesMiembros",),
            mime_type="application/pdf",
        ),
    )
    resolved = queries[6].resolved(today=date(2026, 8, 10))
    assert resolved.date_from == date(2026, 5, 10)
    assert resolved.date_to == date(2026, 8, 10)
    assert resolved.last_months is None
    assert len(resolved.fingerprint()) == 64
    return queries


def record_examples(source: SourceRef) -> tuple[Any, ...]:
    evidence = ExtractionEvidence(
        text="acceso a la vivienda",
        span_start=3,
        span_end=23,
        confidence=0.95,
        backend="example-model",
        model="model-1",
        version="2026-08-10",
    )
    candidate = ExtractionCandidate(
        candidate_id="candidate-1",
        kind="topic",
        value="vivienda",
        evidence=(evidence,),
        source=source,
    )
    return (
        Deputy(source=source, deputy_id="189", full_name="Sánchez Pérez-Castejón, Pedro"),
        DeputyProfile(
            source=source,
            deputy_id="189",
            full_name="Sánchez Pérez-Castejón, Pedro",
        ),
        InterestDeclaration(source=source, declaration_id="interest-1", amount_eur=123.45),
        FinancialDocument(
            source=source,
            document_id="financial-1",
            document_kind="declaración de bienes",
            url="https://example.test/bienes.pdf",
        ),
        Initiative(source=source, initiative_id="initiative-1", title="Ley de vivienda"),
        Intervention(source=source, intervention_id="intervention-1", speaker="Pedro Sánchez"),
        InterventionOccurrence(
            source=source,
            occurrence_id="occurrence-1",
            intervention_id="intervention-1",
        ),
        InterventionRecord(
            source=source,
            intervention_id="intervention-1",
            speaker="Pedro Sánchez",
            text="Respuesta oficial.",
            text_status="matched",
            text_source=source,
            text_method="official_html_transcript",
            extractions=(candidate,),
        ),
        VoteEvent(source=source, vote_id="vote-1", yes_votes=177, no_votes=170),
        VoteItem(source=source, vote_item_id="item-1", vote_id="vote-1", result="aprobado"),
        NominalVote(
            source=source,
            nominal_vote_id="nominal-1",
            vote_id="vote-1",
            deputy_id="189",
            position="Sí",
        ),
        Organ(source=source, organ_id="organ-1", name="Comisión Constitucional"),
        OrganMembership(
            source=source,
            membership_id="membership-1",
            organ_id="organ-1",
            deputy_id="189",
        ),
        DocumentAsset(
            source=source,
            document_id="document-1",
            url="https://example.test/documento.pdf",
            mime_type="application/pdf",
        ),
        DocumentText(
            source=source,
            document_id="document-1",
            text="Texto recuperado.",
            extraction_method="pypdf_text",
            model="pypdf",
            evidence=(evidence,),
        ),
        SpeechBlock(
            source=source,
            speech_block_id="speech-1",
            document_id="document-1",
            text="Señorías, comienza la sesión.",
            sequence=0,
        ),
        SalaryEntitlement(
            source=source,
            entitlement_id="salary-1",
            label="Asignación constitucional",
            amount_eur=3_236.32,
        ),
        candidate,
    )


def model_extension_example(source: SourceRef) -> tuple[ModelRegistry, ExtractionCandidate]:
    def model_callable(request: ModelRequest) -> ModelResponse:
        assert request.output_schema
        return ModelResponse(
            payload={
                "candidates": [
                    {
                        "kind": "topic",
                        "value": "vivienda",
                        "quote": "acceso a la vivienda",
                        "confidence": 0.99,
                    }
                ]
            },
            request_id="request-1",
            usage={"input_tokens": 5, "output_tokens": 8},
        )

    descriptor = ModelDescriptor(
        name="example-model",
        model="model-1",
        version="2026-08-10",
        provider="local",
    )
    backend = CallableModelBackend(
        name=descriptor.name,
        model=descriptor.model,
        version=descriptor.version,
        provider=descriptor.provider,
        function=model_callable,
    )
    assert isinstance(backend, ModelBackend)

    envelope = CandidateEnvelope(
        candidates=(
            CandidateValue(
                kind="topic",
                value="vivienda",
                quote="acceso a la vivienda",
                confidence=0.99,
            ),
        )
    )
    assert envelope.candidates[0].kind == "topic"

    registry = ModelRegistry()
    registry.register("example", backend)
    spec = ExtractionSpec(engine="llm", backend="example", model="model-1")
    assert registry.create(spec) is backend
    task = ExtractionTask(
        name="topics",
        instructions="Extrae temas respaldados por una cita literal.",
        backend=spec,
        limits=ExtractionLimits(
            timeout_seconds=10,
            max_input_characters=1_000,
            chunk_characters=500,
            chunk_overlap_characters=50,
            max_chunks=3,
        ),
    )
    extractor = StructuredModelExtractor(backend, task)
    result = extractor.extract(
        b"El acceso a la vivienda requiere una respuesta.",
        ExtractionContext(source=source, mime_type="text/plain", metadata={"encoding": "utf-8"}),
    )
    assert result.candidates[0].status == "review_required"
    assert result.candidates[0].evidence[0].literal is True
    return registry, result.candidates[0]


def streaming_result_example(deputy: Deputy) -> QueryRun:
    query = DeputyQuery(max_results=1)
    run = QueryRun(
        run_id="query-run-1",
        query_fingerprint=query.fingerprint(),
        started_at=datetime(2026, 8, 10, tzinfo=UTC),
        planned_resources=1,
        succeeded_resources=1,
        raw_records=1,
    )
    result: SearchResult[Deputy] = SearchResult(
        query=query,
        records=lambda: iter((deputy,)),
        run=run,
    )
    assert result.collect(max_items=1) == [deputy]
    assert result.run.complete is True
    assert result.run.normalized_records == 1
    try:
        list(result)
    except ResultConsumedError:
        pass
    else:  # pragma: no cover - an executable assertion for readers
        raise AssertionError("SearchResult must be single-pass")
    return result.run


def error_examples() -> tuple[str, ...]:
    error_types = (
        CongressError,
        QueryValidationError,
        EntityNotFoundError,
        AmbiguousEntityError,
        SourceUnavailableError,
        SourceContractError,
        IncompleteResultError,
        ResultConsumedError,
    )
    caught: list[str] = []
    for error_type in error_types:
        try:
            raise error_type("controlled example")
        except CongressError as exc:
            caught.append(type(exc).__name__)
    return tuple(caught)


def low_level_and_facade_example(root: Path) -> tuple[Deputy, ExtractionRun, Deputy]:
    low_level_root = root / "low-level"
    client = CongressClient(output_root=low_level_root, adapter=DemoAdapter(low_level_root))
    resource = next(client.catalog())
    plan = ExtractionPlan(
        resources=(resource,),
        output_root=low_level_root,
        run_date="2026-08-10",
        batch_size=1,
        max_resources=1,
        max_workers=1,
        request_interval_seconds=0,
        continue_on_error=False,
    )
    manifests = tuple(client.extract(plan))
    low_level_deputy = next(client.deputies(manifests))
    assert client.last_run is not None and client.last_run.failed == 0

    facade_root = root / "facade"
    with Congress(
        data_dir=facade_root,
        adapter=DemoAdapter(facade_root),
        today=date(2026, 8, 10),
    ) as congress:
        result = congress.deputies.search(name="García", legislatures=(15,), max_results=10)
        facade_deputy = result.collect(max_items=10)[0]
        assert result.run.complete is True
    return low_level_deputy, client.last_run, facade_deputy


def acquisition_contract_examples(root: Path, source: SourceRef) -> tuple[Any, ...]:
    artifact = ArtifactManifest(
        family="example",
        dataset="Example",
        format="json",
        source_url=source.requested_url,
        effective_url=source.effective_url,
        run_date="2026-08-10",
        fetched_at=source.fetched_at,
        sha256=source.sha256,
        bytes=2,
        payload_path=str(root / "example.json"),
        adapter=source.adapter,
        adapter_version=source.adapter_version,
        normalization_version=source.normalization_version,
    )
    failure = ExtractionFailure(
        resource_key="example:Example:json",
        family="example",
        dataset="Example",
        source_url=source.requested_url,
        error_type="TimeoutError",
        error_message="bounded example",
    )
    run = ExtractionRun(
        run_id="extraction-run-1",
        started_at=datetime(2026, 8, 10, tzinfo=UTC),
        finished_at=datetime(2026, 8, 10, 0, 0, 1, tzinfo=UTC),
        planned=1,
        failed=1,
        failures=(failure,),
    )
    assert artifact.source_ref().sha256 == source.sha256
    return artifact, failure, run


def run(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    assert set(PUBLIC_API_EXAMPLES) == set(public_api.__all__)

    source = source_example()
    queries = query_examples()
    records = record_examples(source)
    registry, model_candidate = model_extension_example(source)
    query_run = streaming_result_example(records[0])
    low_level_deputy, extraction_run, facade_deputy = low_level_and_facade_example(output_root)
    artifacts = acquisition_contract_examples(output_root, source)
    caught_errors = error_examples()

    assert all(item.model_dump(mode="json") for item in records)
    assert all(item.model_dump(mode="json") for item in artifacts)
    return {
        "status": "passed",
        "public_symbols": len(PUBLIC_API_EXAMPLES),
        "sections": sorted(set(PUBLIC_API_EXAMPLES.values())),
        "queries": len(queries),
        "normalized_record_examples": len(records) - 1,
        "model_backends": list(registry.names()),
        "model_candidate": model_candidate.model_dump(mode="json"),
        "streaming_query_complete": query_run.complete,
        "low_level_run": extraction_run.model_dump(mode="json"),
        "low_level_deputy": low_level_deputy.model_dump(mode="json"),
        "facade_deputy": facade_deputy.model_dump(mode="json"),
        "caught_errors": list(caught_errors),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Keep the tiny Bronze/checkpoint example here; defaults to a temporary directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.output_dir is not None:
        summary = run(args.output_dir.expanduser().resolve())
    else:
        with tempfile.TemporaryDirectory(prefix="congreso-api-example-") as temporary:
            summary = run(Path(temporary))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

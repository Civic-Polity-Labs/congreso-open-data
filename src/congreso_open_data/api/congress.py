"""End-to-end domain facade over acquisition and deterministic normalization."""

from __future__ import annotations

import re
import unicodedata
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from datetime import date as Date
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

from platformdirs import user_data_path
from pydantic import ValidationError

from congreso_open_data.adapters import CongressSourceAdapter
from congreso_open_data.api.errors import (
    AmbiguousEntityError,
    EntityNotFoundError,
    IncompleteResultError,
    QueryValidationError,
    SourceContractError,
    SourceUnavailableError,
)
from congreso_open_data.api.queries import (
    AnyCongressQuery,
    CongressQuery,
    DeputyQuery,
    DocumentQuery,
    FinancialDocumentQuery,
    InitiativeQuery,
    InterestQuery,
    InterventionQuery,
    OrganQuery,
    ProfileQuery,
    RefreshPolicy,
    SalaryEntitlementQuery,
    SortOrder,
    TextPolicy,
    VoteQuery,
    legislature_number,
    legislature_roman,
)
from congreso_open_data.api.records import InterventionRecord
from congreso_open_data.api.results import QueryRun, SearchResult
from congreso_open_data.catalog import DatasetResource
from congreso_open_data.client import CongressClient
from congreso_open_data.extractors.interventions import (
    discover_filtered_intervention_resources,
    discover_intervention_pdf_resources_from_manifest,
    discover_intervention_text_resources_from_manifest,
)
from congreso_open_data.extractors.profiles import (
    deputy_profile_resources_from_payload,
    deputy_profile_search_rows,
)
from congreso_open_data.http import CongresoHttpClient
from congreso_open_data.interventions import match_intervention_text, split_speech_blocks
from congreso_open_data.models import (
    ArtifactManifest,
    CatalogResource,
    CongressRecord,
    Deputy,
    DeputyProfile,
    DocumentAsset,
    ExtractionCandidate,
    ExtractionPlan,
    FinancialDocument,
    Initiative,
    InterestDeclaration,
    Intervention,
    NominalVote,
    Organ,
    SalaryEntitlement,
    VoteEvent,
    VoteItem,
)
from congreso_open_data.normalizers import legacy_manifest
from congreso_open_data.plugins import (
    ExtractionTask,
    ModelBackend,
    ModelRegistry,
    StructuredModelExtractor,
)
from congreso_open_data.protocols import ExtractionContext, SourceAdapter

Q = TypeVar("Q", bound=CongressQuery)
T = TypeVar("T")


class _DomainService(Generic[Q, T]):
    def __init__(self, congress: Congress, query_type: type[Q]) -> None:
        self._congress = congress
        self._query_type = query_type

    def execute(self, query: Q) -> SearchResult[T]:
        return cast(
            SearchResult[T],
            self._congress.execute(cast(AnyCongressQuery, query)),
        )


class _SearchDomainService(_DomainService[Q, T]):
    def _search(self, **filters: Any) -> SearchResult[T]:
        try:
            query = self._query_type(**filters)
        except ValidationError as exc:
            raise QueryValidationError(str(exc)) from exc
        return cast(
            SearchResult[T],
            self._congress.execute(cast(AnyCongressQuery, query)),
        )


class DeputyService(_SearchDomainService[DeputyQuery, Deputy]):
    def search(
        self,
        *,
        name: str | None = None,
        deputy_id: str | None = None,
        constituency: str | None = None,
        parliamentary_group: str | None = None,
        legislatures: str | int | tuple[str | int, ...] = ("XV",),
        refresh: RefreshPolicy | str = RefreshPolicy.AUTO,
        sort: SortOrder | str = SortOrder.ASCENDING,
        max_results: int = 50_000,
        allow_partial: bool = False,
    ) -> SearchResult[Deputy]:
        return self._search(
            name=name,
            deputy_id=deputy_id,
            constituency=constituency,
            parliamentary_group=parliamentary_group,
            legislatures=legislatures,
            refresh=refresh,
            sort=sort,
            max_results=max_results,
            allow_partial=allow_partial,
        )


class ProfileService(_SearchDomainService[ProfileQuery, DeputyProfile]):
    def search(
        self,
        *,
        name: str | None = None,
        deputy_id: str | None = None,
        legislatures: str | int | tuple[str | int, ...] = ("XV",),
        refresh: RefreshPolicy | str = RefreshPolicy.AUTO,
        sort: SortOrder | str = SortOrder.ASCENDING,
        max_results: int = 50_000,
        allow_partial: bool = False,
    ) -> SearchResult[DeputyProfile]:
        return self._search(
            name=name,
            deputy_id=deputy_id,
            legislatures=legislatures,
            refresh=refresh,
            sort=sort,
            max_results=max_results,
            allow_partial=allow_partial,
        )


class InterestService(_SearchDomainService[InterestQuery, InterestDeclaration]):
    def search(
        self,
        *,
        deputy: str | None = None,
        declaration_kind: str | None = None,
        legislatures: str | int | tuple[str | int, ...] = ("XV",),
        refresh: RefreshPolicy | str = RefreshPolicy.AUTO,
        sort: SortOrder | str = SortOrder.ASCENDING,
        max_results: int = 50_000,
        allow_partial: bool = False,
    ) -> SearchResult[InterestDeclaration]:
        return self._search(
            deputy=deputy,
            declaration_kind=declaration_kind,
            legislatures=legislatures,
            refresh=refresh,
            sort=sort,
            max_results=max_results,
            allow_partial=allow_partial,
        )


class FinancialDocumentService(_SearchDomainService[FinancialDocumentQuery, FinancialDocument]):
    def search(
        self,
        *,
        deputy: str | None = None,
        document_kind: str | None = None,
        legislatures: str | int | tuple[str | int, ...] = ("XV",),
        refresh: RefreshPolicy | str = RefreshPolicy.AUTO,
        sort: SortOrder | str = SortOrder.ASCENDING,
        max_results: int = 50_000,
        allow_partial: bool = False,
    ) -> SearchResult[FinancialDocument]:
        return self._search(
            deputy=deputy,
            document_kind=document_kind,
            legislatures=legislatures,
            refresh=refresh,
            sort=sort,
            max_results=max_results,
            allow_partial=allow_partial,
        )


class InitiativeService(_SearchDomainService[InitiativeQuery, Initiative]):
    def search(
        self,
        *,
        title: str | None = None,
        text: str | None = None,
        author: str | None = None,
        file_number: str | None = None,
        initiative_type: str | None = None,
        legislatures: str | int | tuple[str | int, ...] = ("XV",),
        date_from: Date | None = None,
        date_to: Date | None = None,
        last_months: int | None = None,
        refresh: RefreshPolicy | str = RefreshPolicy.AUTO,
        sort: SortOrder | str = SortOrder.ASCENDING,
        max_results: int = 50_000,
        allow_partial: bool = False,
    ) -> SearchResult[Initiative]:
        return self._search(
            title=title,
            text=text,
            author=author,
            file_number=file_number,
            initiative_type=initiative_type,
            legislatures=legislatures,
            date_from=date_from,
            date_to=date_to,
            last_months=last_months,
            refresh=refresh,
            sort=sort,
            max_results=max_results,
            allow_partial=allow_partial,
        )


class VoteService(_SearchDomainService[VoteQuery, VoteEvent | VoteItem | NominalVote]):
    def search(
        self,
        *,
        session: str | None = None,
        vote_number: str | None = None,
        deputy: str | None = None,
        legislatures: str | int | tuple[str | int, ...] = ("XV",),
        date_from: Date | None = None,
        date_to: Date | None = None,
        last_months: int | None = None,
        refresh: RefreshPolicy | str = RefreshPolicy.AUTO,
        sort: SortOrder | str = SortOrder.ASCENDING,
        max_results: int = 50_000,
        allow_partial: bool = False,
    ) -> SearchResult[VoteEvent | VoteItem | NominalVote]:
        return self._search(
            session=session,
            vote_number=vote_number,
            deputy=deputy,
            legislatures=legislatures,
            date_from=date_from,
            date_to=date_to,
            last_months=last_months,
            refresh=refresh,
            sort=sort,
            max_results=max_results,
            allow_partial=allow_partial,
        )


class OrganService(_SearchDomainService[OrganQuery, Organ]):
    def search(
        self,
        *,
        name: str | None = None,
        organ_type: str | None = None,
        legislatures: str | int | tuple[str | int, ...] = ("XV",),
        refresh: RefreshPolicy | str = RefreshPolicy.AUTO,
        sort: SortOrder | str = SortOrder.ASCENDING,
        max_results: int = 50_000,
        allow_partial: bool = False,
    ) -> SearchResult[Organ]:
        return self._search(
            name=name,
            organ_type=organ_type,
            legislatures=legislatures,
            refresh=refresh,
            sort=sort,
            max_results=max_results,
            allow_partial=allow_partial,
        )


class SalaryEntitlementService(_SearchDomainService[SalaryEntitlementQuery, SalaryEntitlement]):
    def search(
        self,
        *,
        role: str | None = None,
        date_from: Date | None = None,
        date_to: Date | None = None,
        refresh: RefreshPolicy | str = RefreshPolicy.AUTO,
        sort: SortOrder | str = SortOrder.ASCENDING,
        max_results: int = 50_000,
        allow_partial: bool = False,
    ) -> SearchResult[SalaryEntitlement]:
        return self._search(
            role=role,
            date_from=date_from,
            date_to=date_to,
            refresh=refresh,
            sort=sort,
            max_results=max_results,
            allow_partial=allow_partial,
        )


class DocumentService(_SearchDomainService[DocumentQuery, DocumentAsset]):
    def search(
        self,
        *,
        source_families: tuple[str, ...] = (),
        source_datasets: tuple[str, ...] = (),
        document_kind: str | None = None,
        entity_id: str | None = None,
        mime_type: str | None = None,
        legislatures: str | int | tuple[str | int, ...] = ("XV",),
        date_from: Date | None = None,
        date_to: Date | None = None,
        last_months: int | None = None,
        refresh: RefreshPolicy | str = RefreshPolicy.AUTO,
        sort: SortOrder | str = SortOrder.ASCENDING,
        max_results: int = 50_000,
        allow_partial: bool = False,
    ) -> SearchResult[DocumentAsset]:
        return self._search(
            source_families=source_families,
            source_datasets=source_datasets,
            document_kind=document_kind,
            entity_id=entity_id,
            mime_type=mime_type,
            legislatures=legislatures,
            date_from=date_from,
            date_to=date_to,
            last_months=last_months,
            refresh=refresh,
            sort=sort,
            max_results=max_results,
            allow_partial=allow_partial,
        )


class InterventionService(_DomainService[InterventionQuery, InterventionRecord]):
    def __init__(self, congress: Congress) -> None:
        super().__init__(congress, InterventionQuery)

    def search(
        self,
        *,
        speaker: str | None = None,
        speaker_id: str | None = None,
        legislatures: str | int | tuple[str | int, ...] = ("XV",),
        date_from: Date | None = None,
        date_to: Date | None = None,
        last_months: int | None = None,
        title: str | None = None,
        text: str | None = None,
        initiative_type: str | None = None,
        initiative_file_number: str | None = None,
        phase: str | None = None,
        body: str | None = None,
        author: str | None = None,
        text_policy: TextPolicy | str = TextPolicy.NATIVE,
        extractions: tuple[ExtractionTask, ...] = (),
        refresh: RefreshPolicy | str = RefreshPolicy.AUTO,
        sort: SortOrder | str = SortOrder.ASCENDING,
        max_results: int = 50_000,
        allow_partial: bool = False,
    ) -> SearchResult[InterventionRecord]:
        try:
            query = InterventionQuery(
                speaker=speaker,
                speaker_id=speaker_id,
                legislatures=legislatures,
                date_from=date_from,
                date_to=date_to,
                last_months=last_months,
                title=title,
                text=text,
                initiative_type=initiative_type,
                initiative_file_number=initiative_file_number,
                phase=phase,
                body=body,
                author=author,
                text_policy=text_policy,
                extractions=extractions,
                refresh=refresh,
                sort=sort,
                max_results=max_results,
                allow_partial=allow_partial,
            )
        except ValidationError as exc:
            raise QueryValidationError(str(exc)) from exc
        return self.execute(query)


class Congress:
    """One cohesive entry point for official Congress data.

    The facade owns package-local Bronze artifacts by default.  Passing ``data_dir``
    makes storage explicit for orchestration and tests.
    """

    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        transport: CongresoHttpClient | None = None,
        adapter: SourceAdapter | None = None,
        models: Mapping[str, ModelBackend] | None = None,
        today: Date | Callable[[], Date] | None = None,
    ) -> None:
        self.data_dir = (
            Path(
                data_dir
                if data_dir is not None
                else user_data_path("congreso-open-data", appauthor=False)
            )
            .expanduser()
            .resolve()
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)
        adapter_transport = getattr(adapter, "transport", None)
        self._owned_transport = transport is None and adapter_transport is None
        self.transport = transport or adapter_transport or CongresoHttpClient()
        resolved_adapter = adapter or CongressSourceAdapter(
            output_root=self.data_dir,
            transport=self.transport,
        )
        self.raw = CongressClient(output_root=self.data_dir, adapter=resolved_adapter)
        self.models = ModelRegistry(models)
        self._today = today
        self._catalog_cache: tuple[CatalogResource, ...] | None = None

        self.interventions = InterventionService(self)
        self.deputies = DeputyService(self, DeputyQuery)
        self.profiles = ProfileService(self, ProfileQuery)
        self.interests = InterestService(self, InterestQuery)
        self.financial_documents = FinancialDocumentService(self, FinancialDocumentQuery)
        self.initiatives = InitiativeService(self, InitiativeQuery)
        self.votes = VoteService(self, VoteQuery)
        self.organs = OrganService(self, OrganQuery)
        self.salary_entitlements = SalaryEntitlementService(self, SalaryEntitlementQuery)
        self.documents = DocumentService(self, DocumentQuery)

    def __enter__(self) -> Congress:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owned_transport:
            self.transport.close()

    def execute(self, query: AnyCongressQuery) -> SearchResult[Any]:
        resolved = query.resolved(today=self._current_date())
        if isinstance(resolved, InterventionQuery):
            return self._intervention_result(resolved)
        return self._generic_result(resolved)

    def _current_date(self) -> Date:
        if callable(self._today):
            return self._today()
        return self._today or Date.today()

    def _intervention_result(
        self,
        query: InterventionQuery,
    ) -> SearchResult[InterventionRecord]:
        started_at = datetime.now(UTC)
        resolved_entities: dict[str, str] = {}
        resources: list[DatasetResource] = []
        official_total = 0
        try:
            for legislature in query.legislatures:
                official_speaker = query.speaker
                if query.speaker is not None or query.speaker_id is not None:
                    identity = self._resolve_deputy(
                        legislature=legislature,
                        name=query.speaker,
                        deputy_id=query.speaker_id,
                    )
                    official_speaker = identity["name"]
                    resolved_entities[f"speaker.{legislature}"] = official_speaker
                    resolved_entities[f"speaker_id.{legislature}"] = identity["id"]
                plan = discover_filtered_intervention_resources(
                    client=self.transport,
                    legislature=str(legislature_number(legislature)),
                    speaker=official_speaker,
                    title=query.title,
                    text=query.text,
                    initiative_type=query.initiative_type,
                    date_from=query.date_from,
                    date_to=query.date_to,
                    initiative_file_number=query.initiative_file_number,
                    phase=query.phase,
                    body=query.body,
                    author=query.author,
                    max_results=max(1, query.max_results - official_total),
                )
                official_total += plan.official_total
                if official_total > query.max_results:
                    raise ValueError(
                        f"Official query contains more than {query.max_results} records"
                    )
                resources.extend(plan.resources)
        except (AmbiguousEntityError, EntityNotFoundError):
            raise
        except ValueError as exc:
            raise SourceContractError(str(exc)) from exc
        except Exception as exc:
            raise SourceUnavailableError(f"Official intervention query failed: {exc}") from exc

        manifests = self._extract_resources(
            resources,
            query=query,
            run_date=self._current_date().isoformat(),
        )
        primary_run = self.raw.last_run
        base_records = list(self.raw.interventions(manifests))
        if len(base_records) != official_total and not query.allow_partial:
            raise IncompleteResultError(
                "Official intervention count does not match normalized rows: "
                f"official={official_total}, normalized={len(base_records)}"
            )

        enriched, document_manifests, text_failures = self._enrich_intervention_text(
            records=base_records,
            source_manifests=manifests,
            query=query,
        )
        enriched = [self._extract_candidates(item, query) for item in enriched]
        enriched.sort(key=_intervention_sort_key, reverse=query.sort == SortOrder.DESCENDING)
        duplicate_records = len(enriched) - len({item.intervention_id for item in enriched})
        unmatched = sum(item.text_status not in {"matched", "not_requested"} for item in enriched)
        all_manifests = (*manifests, *document_manifests)
        primary_failures = tuple(
            f"{item.error_type}: {item.error_message}"
            for item in (primary_run.failures if primary_run else ())
        )
        run = QueryRun(
            run_id=uuid.uuid4().hex,
            query_fingerprint=query.fingerprint(),
            started_at=started_at,
            planned_resources=len(resources),
            succeeded_resources=primary_run.succeeded if primary_run else len(manifests),
            reused_resources=primary_run.reused if primary_run else 0,
            failed_resources=(
                primary_run.failed if primary_run else max(0, len(resources) - len(manifests))
            ),
            raw_records=official_total,
            duplicate_records=duplicate_records,
            unmatched_text_records=unmatched,
            checkpoint_path=(
                str(primary_run.checkpoint_path)
                if primary_run and primary_run.checkpoint_path
                else None
            ),
            event_log_path=(
                str(
                    self.data_dir
                    / "extraction-runs"
                    / self._current_date().isoformat()
                    / f"{primary_run.run_id}.jsonl"
                )
                if primary_run
                else None
            ),
            failures=(*primary_failures, *text_failures),
            resolved_entities=resolved_entities,
        )
        return _result_from_list(
            query=query,
            records=enriched,
            manifests=all_manifests,
            run=run,
            expected=official_total,
        )

    def _extract_resources(
        self,
        resources: list[DatasetResource],
        *,
        query: CongressQuery,
        run_date: str,
    ) -> tuple[ArtifactManifest, ...]:
        if not resources:
            return ()
        public_resources = tuple(
            CatalogResource.model_validate(asdict(resource)) for resource in resources
        )
        plan = ExtractionPlan(
            resources=public_resources,
            output_root=self.data_dir,
            run_date=run_date,
            batch_size=min(16, len(public_resources)),
            max_resources=max(len(public_resources), 1),
            max_workers=1,
            resume=query.refresh != RefreshPolicy.ALWAYS,
            continue_on_error=query.allow_partial,
        )
        try:
            return tuple(self.raw.extract(plan))
        except Exception as exc:
            raise SourceUnavailableError(
                f"Official resource acquisition failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _enrich_intervention_text(
        self,
        *,
        records: list[Intervention],
        source_manifests: tuple[ArtifactManifest, ...],
        query: InterventionQuery,
    ) -> tuple[list[InterventionRecord], tuple[ArtifactManifest, ...], list[str]]:
        if query.text_policy == TextPolicy.NONE or not records:
            return (
                [
                    InterventionRecord.model_validate(
                        {**item.model_dump(), "text_status": "not_requested"}
                    )
                    for item in records
                ],
                (),
                [],
            )
        if len(records) > 5_000:
            raise IncompleteResultError(
                "Native text enrichment is capped at 5,000 interventions per query; "
                "add filters or use text_policy='none'."
            )
        html_resources = _document_resources(
            source_manifests,
            root=self.data_dir,
            discover=discover_intervention_text_resources_from_manifest,
        )
        document_manifests: list[ArtifactManifest] = []
        failures: list[str] = []
        html_manifests = self._extract_document_resources(html_resources, query=query)
        document_manifests.extend(html_manifests)
        if html_resources and self.raw.last_run is not None:
            failures.extend(
                f"HTML transcript {item.error_type}: {item.error_message}"
                for item in self.raw.last_run.failures
            )
        blocks_by_document: dict[str, list[Any]] = defaultdict(list)
        source_by_document: dict[str, Any] = {}
        for manifest in html_manifests:
            try:
                for block in self.raw.speech_blocks((manifest,)):
                    blocks_by_document[block.document_id].append(block)
                    source_by_document[block.document_id] = block.source
            except Exception as exc:
                failures.append(f"HTML transcript {manifest.source_url}: {exc}")

        needed = {item.document_id for item in records if item.document_id}
        missing = needed - set(blocks_by_document)
        if missing:
            pdf_resources = [
                resource
                for resource in _document_resources(
                    source_manifests,
                    root=self.data_dir,
                    discover=discover_intervention_pdf_resources_from_manifest,
                )
                if str(resource.snapshot_token) in missing
            ]
            pdf_manifests = self._extract_document_resources(pdf_resources, query=query)
            document_manifests.extend(pdf_manifests)
            if pdf_resources and self.raw.last_run is not None:
                failures.extend(
                    f"PDF transcript {item.error_type}: {item.error_message}"
                    for item in self.raw.last_run.failures
                )
            token_by_url = {
                resource.url: str(resource.snapshot_token) for resource in pdf_resources
            }
            for manifest in pdf_manifests:
                document_id = token_by_url.get(manifest.source_url) or token_by_url.get(
                    manifest.effective_url or ""
                )
                try:
                    document = next(
                        self.raw.document_texts(
                            (manifest,),
                            use_ocr=query.text_policy == TextPolicy.OCR,
                        )
                    )
                    if document_id:
                        blocks_by_document[document_id].extend(split_speech_blocks(document.text))
                        source_by_document[document_id] = document.source
                except Exception as exc:
                    failures.append(f"PDF transcript {manifest.source_url}: {exc}")

        occurrence_indexes = _occurrence_indexes(records)
        output: list[InterventionRecord] = []
        for item in records:
            blocks = blocks_by_document.get(item.document_id or "", [])
            if not item.document_id or not blocks:
                output.append(
                    InterventionRecord.model_validate(
                        {**item.model_dump(), "text_status": "document_unavailable"}
                    )
                )
                continue
            match = match_intervention_text(
                speaker=item.speaker or "",
                blocks=blocks,
                occurrence_index=occurrence_indexes[item.intervention_id],
            )
            status = "matched" if match.text_fragment is not None else "speaker_not_found"
            source = source_by_document.get(item.document_id)
            output.append(
                InterventionRecord.model_validate(
                    {
                        **item.model_dump(),
                        "text": match.text_fragment,
                        "text_status": status,
                        "text_source": source,
                        "text_method": source.method if source is not None else None,
                        "text_confidence": match.confidence,
                    }
                )
            )
        return output, tuple(document_manifests), failures

    def _extract_document_resources(
        self,
        resources: list[DatasetResource],
        *,
        query: CongressQuery,
    ) -> tuple[ArtifactManifest, ...]:
        if not resources:
            return ()
        document_query = query.model_copy(update={"allow_partial": True})
        return self._extract_resources(
            resources,
            query=document_query,
            run_date=self._current_date().isoformat(),
        )

    def _extract_candidates(
        self,
        record: InterventionRecord,
        query: InterventionQuery,
    ) -> InterventionRecord:
        if not query.extractions or not record.text or record.text_source is None:
            return record
        candidates: list[ExtractionCandidate] = []
        context = ExtractionContext(
            source=record.text_source,
            mime_type="text/plain",
            metadata={"encoding": "utf-8", "intervention_id": record.intervention_id},
        )
        for task in query.extractions:
            backend = self.models.create(task.backend)
            result = StructuredModelExtractor(backend, task).extract(
                record.text.encode(),
                context,
            )
            candidates.extend(result.candidates)
        return record.model_copy(update={"extractions": tuple(candidates)})

    def _resolve_deputy(
        self,
        *,
        legislature: str,
        name: str | None,
        deputy_id: str | None,
    ) -> dict[str, str]:
        rows = deputy_profile_search_rows(
            client=self.transport,
            legislature_number=legislature_number(legislature),
        )
        return _select_deputy_identity(
            rows=rows,
            legislature=legislature,
            name=name,
            deputy_id=deputy_id,
        )

    def _generic_result(self, query: CongressQuery) -> SearchResult[CongressRecord]:
        started_at = datetime.now(UTC)
        normalizer_name, _ = _generic_domain(query)
        normalizer = cast(
            Callable[[tuple[ArtifactManifest, ...]], Iterator[CongressRecord]],
            getattr(self.raw, normalizer_name),
        )
        if isinstance(query, (ProfileQuery, FinancialDocumentQuery)):
            profile_resources = self._profile_resources(query)
            manifests = self._extract_resources(
                profile_resources,
                query=query,
                run_date=self._current_date().isoformat(),
            )
            planned_resources = len(profile_resources)
        elif (
            isinstance(query, DocumentQuery)
            and query.entity_id is not None
            and "intervenciones" in query.source_families
        ):
            document_resources: list[DatasetResource] = []
            for legislature in query.legislatures:
                document_plan = discover_filtered_intervention_resources(
                    client=self.transport,
                    legislature=str(legislature_number(legislature)),
                    date_from=query.date_from,
                    date_to=query.date_to,
                    initiative_file_number=query.entity_id,
                    max_results=query.max_results,
                )
                document_resources.extend(document_plan.resources)
            manifests = self._extract_resources(
                document_resources,
                query=query,
                run_date=self._current_date().isoformat(),
            )
            planned_resources = len(document_resources)
        else:
            catalog_resources = _catalog_resources_for_query(
                query,
                self._catalog(query.refresh),
            )
            manifests = self._extract_resources(
                [DatasetResource(**item.model_dump()) for item in catalog_resources],
                query=query,
                run_date=self._current_date().isoformat(),
            )
            planned_resources = len(catalog_resources)
        try:
            normalized = list(normalizer(manifests))
        except Exception as exc:
            raise SourceContractError(
                f"Official {query.domain} normalization failed: {type(exc).__name__}: {exc}"
            ) from exc
        filtered = _filter_normalized_records(normalized, query)
        records, duplicate_records = _deduplicate_records(filtered)
        if len(records) > query.max_results:
            raise IncompleteResultError(
                f"Query produced more than max_results={query.max_results}; add filters"
            )
        records.sort(key=_record_sort_key, reverse=query.sort == SortOrder.DESCENDING)
        extraction_run = self.raw.last_run if planned_resources else None
        run = QueryRun(
            run_id=extraction_run.run_id if extraction_run else uuid.uuid4().hex,
            query_fingerprint=query.fingerprint(),
            started_at=started_at,
            planned_resources=extraction_run.planned if extraction_run else len(manifests),
            succeeded_resources=extraction_run.succeeded if extraction_run else len(manifests),
            reused_resources=extraction_run.reused if extraction_run else 0,
            failed_resources=extraction_run.failed if extraction_run else 0,
            raw_records=len(normalized),
            duplicate_records=duplicate_records,
            checkpoint_path=(
                str(extraction_run.checkpoint_path)
                if extraction_run and extraction_run.checkpoint_path
                else None
            ),
            event_log_path=(
                str(
                    self.data_dir
                    / "extraction-runs"
                    / self._current_date().isoformat()
                    / f"{extraction_run.run_id}.jsonl"
                )
                if extraction_run
                else None
            ),
            failures=tuple(
                f"{item.error_type}: {item.error_message}"
                for item in (extraction_run.failures if extraction_run else ())
            ),
        )
        return _result_from_list(
            query=query,
            records=records,
            manifests=manifests,
            run=run,
            expected=len(records),
        )

    def _catalog(self, refresh: RefreshPolicy | str) -> tuple[CatalogResource, ...]:
        if self._catalog_cache is None or refresh == RefreshPolicy.ALWAYS:
            self._catalog_cache = tuple(self.raw.catalog())
        return self._catalog_cache

    def _profile_resources(
        self,
        query: ProfileQuery | FinancialDocumentQuery,
    ) -> list[DatasetResource]:
        selected: list[DatasetResource] = []
        name = query.name if isinstance(query, ProfileQuery) else query.deputy
        deputy_id = query.deputy_id if isinstance(query, ProfileQuery) else None
        for legislature in query.legislatures:
            rows = deputy_profile_search_rows(
                client=self.transport,
                legislature_number=legislature_number(legislature),
            )
            if name is not None or deputy_id is not None:
                identity = _select_deputy_identity(
                    rows=rows,
                    legislature=legislature,
                    name=name,
                    deputy_id=deputy_id,
                )
                rows = tuple(
                    row for row in rows if str(row.get("codParlamentario") or "") == identity["id"]
                )
            selected.extend(
                deputy_profile_resources_from_payload(
                    payload={"data": list(rows)},
                    legislature_number=legislature_number(legislature),
                )
            )
        return selected


def _select_deputy_identity(
    *,
    rows: tuple[dict[str, Any], ...],
    legislature: str,
    name: str | None,
    deputy_id: str | None,
) -> dict[str, str]:
    candidates: list[dict[str, str]] = []
    wanted = _identity_tokens(name or "")
    for row in rows:
        code = str(row.get("codParlamentario") or "").strip()
        official_name = str(
            row.get("apellidosNombre")
            or " ".join(str(row.get(key) or "") for key in ("nombre", "apellidos"))
        ).strip()
        id_matches = deputy_id is not None and code == str(deputy_id).strip()
        name_matches = (
            name is not None and bool(wanted) and wanted <= _identity_tokens(official_name)
        )
        if id_matches or name_matches:
            candidates.append({"id": code, "name": official_name})
    unique = {(item["id"], item["name"]): item for item in candidates}
    candidates = list(unique.values())
    if not candidates:
        target = deputy_id or name
        raise EntityNotFoundError(f"No deputy matching {target!r} in legislature {legislature}")
    exact = [
        item for item in candidates if name is not None and _identity_tokens(item["name"]) == wanted
    ]
    if len(exact) == 1:
        return exact[0]
    if len(candidates) > 1:
        choices = ", ".join(f"{item['name']} ({item['id']})" for item in candidates[:10])
        raise AmbiguousEntityError(f"Ambiguous deputy {name!r}: {choices}")
    return candidates[0]


def _generic_domain(
    query: CongressQuery,
) -> tuple[str, str]:
    if isinstance(query, DeputyQuery):
        return "deputies", "diputados"
    if isinstance(query, ProfileQuery):
        return "profiles", "diputados"
    if isinstance(query, InterestQuery):
        return "interests", "diputados"
    if isinstance(query, FinancialDocumentQuery):
        return "financial_documents", "diputados"
    if isinstance(query, InitiativeQuery):
        return "initiatives", "iniciativas"
    if isinstance(query, VoteQuery):
        return "votes", "votaciones"
    if isinstance(query, OrganQuery):
        return "organs", "organos"
    if isinstance(query, SalaryEntitlementQuery):
        return "salary_entitlements", "transparencia"
    if isinstance(query, DocumentQuery):
        return "documents", "*"
    raise TypeError(f"Unsupported query type: {type(query).__name__}")


def _catalog_resources_for_query(
    query: CongressQuery,
    catalog: tuple[CatalogResource, ...],
) -> tuple[CatalogResource, ...]:
    """Select only formats and datasets accepted by a domain normalizer."""

    if isinstance(query, DeputyQuery):
        wanted_datasets: set[str] = set()
        for legislature in query.legislatures:
            number = legislature_number(legislature)
            if number == 15:
                wanted_datasets.update({"DiputadosActivos", "DiputadosDeBaja"})
            elif 0 <= number <= 14:
                wanted_datasets.add(f"odsDiputados{number:02d}")
        official = tuple(
            item
            for item in catalog
            if item.family == "diputados"
            and item.dataset in wanted_datasets
            and item.format == "json"
        )
        if official:
            return official
        # Dependency-injected adapters may expose their own dataset names.
        return tuple(
            item
            for item in catalog
            if item.family == "diputados"
            and item.dataset != "docacteco"
            and item.format in {"json", "csv"}
        )
    if isinstance(query, InterestQuery):
        return tuple(
            item
            for item in catalog
            if item.family == "diputados" and item.dataset == "docacteco" and item.format == "json"
        )
    if isinstance(query, InitiativeQuery):
        return tuple(
            item for item in catalog if item.family == "iniciativas" and item.format == "json"
        )
    if isinstance(query, VoteQuery):
        output: list[CatalogResource] = []
        for item in catalog:
            if item.family != "votaciones" or item.dataset != "Votacion":
                continue
            if item.format != "json":
                continue
            if item.legislature is not None:
                try:
                    if legislature_roman(item.legislature) not in query.legislatures:
                        continue
                except ValueError:
                    continue
            if query.session is not None and not _same_identifier(item.session, query.session):
                continue
            if query.vote_number is not None and not _same_identifier(
                item.vote_number, query.vote_number
            ):
                continue
            output.append(item)
        return tuple(output)
    if isinstance(query, OrganQuery):
        return tuple(item for item in catalog if item.family == "organos" and item.format == "html")
    if isinstance(query, SalaryEntitlementQuery):
        return tuple(
            item
            for item in catalog
            if item.family == "transparencia"
            and item.dataset == "RetribucionesCargosMesa"
            and item.format == "html"
        )
    if isinstance(query, DocumentQuery):
        candidates = tuple(
            item
            for item in catalog
            if (not query.source_families or item.family in query.source_families)
            and (not query.source_datasets or item.dataset in query.source_datasets)
            and item.format in {"json", "csv", "html"}
        )
        return _preferred_document_sources(candidates)
    raise TypeError(f"Unsupported catalog query: {type(query).__name__}")


def _preferred_document_sources(
    resources: tuple[CatalogResource, ...],
) -> tuple[CatalogResource, ...]:
    """Choose one parseable export per logical source, preferring structured JSON."""

    priority = {"json": 0, "csv": 1, "html": 2}
    selected: dict[tuple[str, str, str, str, str], CatalogResource] = {}
    for item in resources:
        key = (
            item.family,
            item.dataset,
            str(item.legislature or ""),
            str(item.session or ""),
            str(item.vote_number or ""),
        )
        current = selected.get(key)
        if current is None or priority[item.format] < priority[current.format]:
            selected[key] = item
    return tuple(
        sorted(
            selected.values(),
            key=lambda item: (
                item.family,
                item.dataset,
                str(item.legislature or ""),
                str(item.session or ""),
                str(item.vote_number or ""),
                item.url,
            ),
        )
    )


def _document_resources(
    manifests: tuple[ArtifactManifest, ...],
    *,
    root: Path,
    discover: Callable[..., list[DatasetResource]],
) -> list[DatasetResource]:
    unique: dict[tuple[str, str], DatasetResource] = {}
    for manifest in manifests:
        for resource in discover(lake_root=root, manifest=legacy_manifest(manifest)):
            unique[(resource.dataset, resource.url)] = resource
    return sorted(
        unique.values(),
        key=lambda item: (str(item.snapshot_token or ""), item.dataset, item.url),
    )


def _identity_tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    ascii_value = "".join(char for char in normalized if not unicodedata.combining(char))
    return {
        token
        for token in re.findall(r"[a-z0-9]+", ascii_value)
        if token not in {"de", "del", "la", "las", "los", "y"}
    }


def _occurrence_indexes(records: list[Intervention]) -> dict[str, int]:
    groups: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    identity_by_id: dict[str, tuple[str, str, str]] = {}
    for item in records:
        group = (item.document_id or "", " ".join(sorted(_identity_tokens(item.speaker or ""))))
        identity = (item.starts_at or "", item.ends_at or "", item.speaker or "")
        identity_by_id[item.intervention_id] = identity
        if identity not in groups[group]:
            groups[group].append(identity)
    output: dict[str, int] = {}
    for item in records:
        group = (item.document_id or "", " ".join(sorted(_identity_tokens(item.speaker or ""))))
        identities = sorted(groups[group])
        output[item.intervention_id] = identities.index(identity_by_id[item.intervention_id])
    return output


def _intervention_sort_key(item: InterventionRecord) -> tuple[Any, ...]:
    return (
        item.session_date or Date.min,
        item.starts_at or "",
        item.document_id or "",
        item.intervention_id,
    )


def _record_sort_key(item: Any) -> tuple[str, str]:
    for field in ("session_date", "vote_date", "presented_at", "effective_date", "date"):
        value = getattr(item, field, None)
        if value is not None:
            return str(value), _record_identity(item)
    return "", _record_identity(item)


def _record_identity(item: Any) -> str:
    for field in (
        "membership_id",
        "nominal_vote_id",
        "vote_item_id",
        "occurrence_id",
        "speech_block_id",
        "intervention_id",
        "document_id",
        "declaration_id",
        "entitlement_id",
        "initiative_id",
        "deputy_id",
        "vote_id",
        "organ_id",
    ):
        value = getattr(item, field, None)
        if value is not None:
            return str(value)
    return repr(item)


def _deduplicate_records(records: list[T]) -> tuple[list[T], int]:
    unique: dict[tuple[str, str], T] = {}
    for item in records:
        key = (type(item).__name__, _record_identity(item))
        current = unique.get(key)
        if current is None or _record_completeness(item) > _record_completeness(current):
            unique[key] = item
    return list(unique.values()), len(records) - len(unique)


def _filter_normalized_records(
    records: list[CongressRecord],
    query: CongressQuery,
) -> list[CongressRecord]:
    if not isinstance(query, VoteQuery):
        return [item for item in records if _record_matches(item, query)]

    event_query = query.model_copy(update={"deputy": None})
    matching_vote_ids = {
        item.vote_id
        for item in records
        if isinstance(item, VoteEvent) and _record_matches(item, event_query)
    }
    scoped = [item for item in records if getattr(item, "vote_id", None) in matching_vote_ids]
    if query.deputy is None:
        return scoped
    wanted = _identity_tokens(query.deputy)
    return [
        item
        for item in scoped
        if isinstance(item, NominalVote)
        and wanted
        and wanted <= _identity_tokens(item.deputy_name or "")
    ]


def _record_completeness(item: Any) -> int:
    dumped = item.model_dump(mode="json", exclude={"source"})
    return sum(value not in (None, "", [], {}, ()) for value in dumped.values())


def _record_matches(item: Any, query: CongressQuery) -> bool:
    dumped = item.model_dump(mode="json")
    record_legislature = dumped.get("legislature")
    if record_legislature is not None:
        try:
            if legislature_roman(str(record_legislature)) not in query.legislatures:
                return False
        except ValueError:
            return False
    ignored = {
        "domain",
        "legislatures",
        "date_from",
        "date_to",
        "last_months",
        "refresh",
        "sort",
        "max_results",
        "allow_partial",
        "extractions",
        "text_policy",
        "source_families",
        "source_datasets",
    }
    aliases = {
        "name": ("name", "full_name", "title", "label"),
        "deputy": ("deputy_name", "name", "full_name"),
        "deputy_id": ("deputy_id",),
        "file_number": ("file_number",),
        "session": ("session",),
        "vote_number": ("vote_number",),
        "role": ("role",),
        "entity_id": ("entity_id",),
        "mime_type": ("mime_type",),
    }
    for key, wanted in query.model_dump(mode="json").items():
        if key in ignored or wanted is None:
            continue
        fields = aliases.get(key, (key,))
        if key == "deputy_id" and isinstance(item, DeputyProfile):
            fields = ("deputy_id", "parliamentary_code")
        values = [dumped.get(field) for field in fields if dumped.get(field) is not None]
        if not values:
            return False
        if key in {"name", "deputy"}:
            wanted_tokens = _identity_tokens(str(wanted))
            if not wanted_tokens or not any(
                wanted_tokens <= _identity_tokens(str(value)) for value in values
            ):
                return False
        elif not any(_fold_text(str(wanted)) in _fold_text(str(value)) for value in values):
            return False
    record_date = next(
        (
            value
            for field in ("session_date", "vote_date", "presented_at", "effective_date", "date")
            if (value := getattr(item, field, None)) is not None
        ),
        None,
    )
    if record_date is not None:
        if query.date_from is not None and record_date < query.date_from:
            return False
        if query.date_to is not None and record_date > query.date_to:
            return False
    return True


def _fold_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _same_identifier(left: str | None, right: str | None) -> bool:
    left_value = str(left or "").strip()
    right_value = str(right or "").strip()
    if left_value.isdigit() and right_value.isdigit():
        return int(left_value) == int(right_value)
    return _fold_text(left_value) == _fold_text(right_value)


def _result_from_list(
    *,
    query: CongressQuery,
    records: list[T],
    manifests: tuple[ArtifactManifest, ...],
    run: QueryRun,
    expected: int,
) -> SearchResult[T]:
    def finish(count: int, completed: bool, failure: BaseException | None) -> QueryRun:
        complete = completed and failure is None and count == expected and run.failed_resources == 0
        failures = run.failures
        if failure is not None:
            failures = (*failures, f"{type(failure).__name__}: {failure}")
        elif count != expected:
            failures = (*failures, f"Expected {expected} records but yielded {count}")
        return run.model_copy(
            update={
                "finished_at": datetime.now(UTC),
                "normalized_records": count,
                "complete": complete,
                "failures": failures,
            }
        )

    return SearchResult(
        query=query,
        records=lambda: iter(records),
        manifests=manifests,
        run=run,
        on_finish=finish,
    )

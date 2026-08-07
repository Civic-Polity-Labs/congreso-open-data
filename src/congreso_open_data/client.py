"""High-level synchronous, streaming-first Congress client."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from congreso_open_data.adapters import CongressSourceAdapter
from congreso_open_data.batch_extract import extract_resource_batch
from congreso_open_data.catalog import DatasetResource
from congreso_open_data.durable_io import append_jsonl_durably
from congreso_open_data.extractors import extract_with_fallback
from congreso_open_data.models import (
    ArtifactManifest,
    CatalogResource,
    Deputy,
    DocumentAsset,
    ExtractionFailure,
    ExtractionPlan,
    ExtractionRun,
    Initiative,
    Intervention,
    NominalVote,
    Organ,
    OrganMembership,
    VoteEvent,
    VoteItem,
)
from congreso_open_data.normalizers import (
    deputies as normalize_deputies,
)
from congreso_open_data.normalizers import (
    documents as normalize_documents,
)
from congreso_open_data.normalizers import (
    initiatives as normalize_initiatives,
)
from congreso_open_data.normalizers import (
    interventions as normalize_interventions,
)
from congreso_open_data.normalizers import (
    legacy_manifest,
    public_manifest,
)
from congreso_open_data.normalizers import (
    organs as normalize_organs,
)
from congreso_open_data.normalizers import (
    votes as normalize_votes,
)
from congreso_open_data.protocols import ExtractionContext, SourceAdapter


class CongressClient:
    def __init__(
        self,
        *,
        output_root: str | Path = "data/lake",
        adapter: SourceAdapter | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.adapter = adapter or CongressSourceAdapter(output_root=self.output_root)
        self._last_run: ExtractionRun | None = None

    @property
    def last_run(self) -> ExtractionRun | None:
        return self._last_run

    def catalog(self) -> Iterator[CatalogResource]:
        yield from self.adapter.catalog()

    def extract(self, plan: ExtractionPlan) -> Iterator[ArtifactManifest]:
        resources = tuple(plan.resources) or tuple(self._selected_catalog(plan))
        legacy_resources = [DatasetResource(**item.model_dump()) for item in resources]
        run_id = uuid.uuid4().hex
        started_at = datetime.now(UTC)
        run_dir = plan.output_root / "extraction-runs" / plan.run_date
        manifest_index = run_dir / f"{run_id}.manifests.json"
        event_log = run_dir / f"{run_id}.jsonl"
        append_jsonl_durably(
            event_log,
            {
                "event": "started",
                "run_id": run_id,
                "at": started_at.isoformat(),
                "planned": len(resources),
            },
        )
        result = extract_resource_batch(
            resources=legacy_resources,
            run_date=plan.run_date,
            output_root=plan.output_root,
            manifest_index_path=manifest_index,
            max_workers=plan.max_workers,
            resume=plan.resume,
            continue_on_error=plan.continue_on_error,
            request_interval_seconds=plan.request_interval_seconds,
            progress=lambda event: append_jsonl_durably(event_log, {"run_id": run_id, **event}),
            extract_one=lambda resource: legacy_manifest(
                self.adapter.acquire(
                    CatalogResource.model_validate(resource.__dict__),
                    run_date=plan.run_date,
                )
            ),
        )
        failures = tuple(ExtractionFailure.model_validate(asdict(item)) for item in result.failures)
        finished_at = datetime.now(UTC)
        self._last_run = ExtractionRun(
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            planned=result.planned,
            succeeded=result.completed,
            reused=result.reused,
            failed=result.failed,
            manifest_index_path=result.manifest_index_path,
            checkpoint_path=result.state_path,
            failures=failures,
        )
        manifests = tuple(public_manifest(item) for item in result.manifests)
        if plan.specs:
            extraction_log = run_dir / f"{run_id}.extractions.jsonl"
            for manifest in manifests:
                path = Path(manifest.payload_path)
                if not path.is_absolute():
                    path = plan.output_root / path
                content = path.read_bytes()
                context = ExtractionContext(
                    source=manifest.source_ref(),
                    mime_type=manifest.content_type,
                    metadata={"format": manifest.format},
                )
                for spec in plan.specs:
                    extracted = extract_with_fallback(spec, content, context)
                    append_jsonl_durably(
                        extraction_log,
                        {
                            "manifest_sha256": manifest.sha256,
                            "spec": spec.model_dump(mode="json"),
                            "texts": list(extracted.texts),
                            "candidates": [
                                item.model_dump(mode="json") for item in extracted.candidates
                            ],
                            "evidence": [
                                item.model_dump(mode="json") for item in extracted.evidence
                            ],
                            "diagnostics": extracted.diagnostics,
                        },
                    )
        append_jsonl_durably(
            event_log,
            {
                "event": "finished",
                "run_id": run_id,
                "at": finished_at.isoformat(),
                **self._last_run.model_dump(mode="json"),
            },
        )
        yield from manifests

    def deputies(self, manifests: Iterable[ArtifactManifest] | None = None) -> Iterator[Deputy]:
        yield from normalize_deputies(
            self._manifests("diputados", manifests), root=self.output_root
        )

    def initiatives(
        self, manifests: Iterable[ArtifactManifest] | None = None
    ) -> Iterator[Initiative]:
        yield from normalize_initiatives(
            self._manifests("iniciativas", manifests), root=self.output_root
        )

    def interventions(
        self, manifests: Iterable[ArtifactManifest] | None = None
    ) -> Iterator[Intervention]:
        yield from normalize_interventions(
            self._manifests("intervenciones", manifests), root=self.output_root
        )

    def votes(
        self, manifests: Iterable[ArtifactManifest] | None = None
    ) -> Iterator[VoteEvent | VoteItem | NominalVote]:
        yield from normalize_votes(self._manifests("votaciones", manifests), root=self.output_root)

    def organs(
        self, manifests: Iterable[ArtifactManifest] | None = None
    ) -> Iterator[Organ | OrganMembership]:
        yield from normalize_organs(self._manifests("organos", manifests), root=self.output_root)

    def documents(
        self, manifests: Iterable[ArtifactManifest] | None = None
    ) -> Iterator[DocumentAsset]:
        source = manifests if manifests is not None else self._manifests("diputados", None)
        yield from normalize_documents(source, root=self.output_root)

    def _manifests(
        self, family: str, manifests: Iterable[ArtifactManifest] | None
    ) -> Iterable[ArtifactManifest]:
        if manifests is not None:
            return manifests
        return self.extract(ExtractionPlan(families=(family,), output_root=self.output_root))

    def _selected_catalog(self, plan: ExtractionPlan) -> Iterator[CatalogResource]:
        for resource in self.catalog():
            if plan.families and resource.family not in plan.families:
                continue
            if plan.datasets and resource.dataset not in plan.datasets:
                continue
            if plan.formats and resource.format not in plan.formats:
                continue
            yield resource

"""Public typed normalization over immutable Bronze artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from congreso_open_data.html import parse_visible_html
from congreso_open_data.models import (
    ArtifactManifest,
    Deputy,
    DeputyProfile,
    DocumentAsset,
    Initiative,
    InterestDeclaration,
    Intervention,
    NominalVote,
    Organ,
    OrganMembership,
    SourceRef,
    VoteEvent,
    VoteItem,
)
from congreso_open_data.normalization import load_records, stable_id
from congreso_open_data.storage import BronzeManifest
from congreso_open_data.transforms import (
    deputy_profile_row,
    deputy_row,
    document_asset_rows_from_links,
    document_asset_rows_from_records,
    initiative_row,
    interest_row,
    intervention_row,
    organ_membership_rows,
    organ_rows_from_links,
    vote_rows,
)


def public_manifest(manifest: BronzeManifest) -> ArtifactManifest:
    return ArtifactManifest(
        family=manifest.family,
        dataset=manifest.dataset,
        format=manifest.format,
        source_url=manifest.requested_url or manifest.source_url,
        effective_url=manifest.source_url,
        snapshot_token=manifest.snapshot_token,
        run_date=manifest.run_date,
        fetched_at=manifest.extracted_at,
        sha256=manifest.sha256,
        bytes=manifest.bytes,
        payload_path=manifest.bronze_path,
        request_parameters=manifest.request_parameters_json,
        content_type=manifest.content_type,
        http_status=manifest.status_code,
        legislature=manifest.legislature,
        session=manifest.session,
        vote_number=manifest.vote_number,
    )


def legacy_manifest(manifest: ArtifactManifest) -> BronzeManifest:
    return BronzeManifest(
        family=manifest.family,
        dataset=manifest.dataset,
        format=manifest.format,
        source_url=manifest.effective_url or manifest.source_url,
        snapshot_token=manifest.snapshot_token,
        run_date=manifest.run_date,
        extracted_at=str(manifest.fetched_at),
        sha256=manifest.sha256,
        bytes=manifest.bytes,
        bronze_path=manifest.payload_path,
        status_code=manifest.http_status or 200,
        requested_url=manifest.source_url,
        legislature=manifest.legislature,
        session=manifest.session,
        vote_number=manifest.vote_number,
        content_type=manifest.content_type,
        request_parameters_json=(
            manifest.request_parameters
            if isinstance(manifest.request_parameters, str)
            else json.dumps(manifest.request_parameters, ensure_ascii=False, sort_keys=True)
            if manifest.request_parameters
            else None
        ),
    )


def _content(manifest: ArtifactManifest, root: Path) -> bytes:
    path = Path(manifest.payload_path)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"Bronze artifact not found: {path}")
    return path.read_bytes()


def _records(manifest: ArtifactManifest, root: Path) -> list[dict[str, Any]]:
    return load_records(_content(manifest, root), manifest.format)


def _source(manifest: ArtifactManifest, *, method: str = "deterministic") -> SourceRef:
    return manifest.source_ref(
        method=method,
        model="congreso-open-data",
    )


def deputies(manifests: Iterable[ArtifactManifest], *, root: Path) -> Iterator[Deputy]:
    for manifest in manifests:
        if manifest.family != "diputados" or manifest.format not in {"json", "csv"}:
            continue
        if manifest.dataset == "docacteco":
            continue
        for raw in _records(manifest, root):
            row = deputy_row(
                raw,
                source_sha256=manifest.sha256,
                snapshot_date=manifest.run_date,
                default_legislature=manifest.legislature,
            )
            yield Deputy.model_validate(
                {
                    **row,
                    "deputy_id": row["person_id"],
                    "source": _source(manifest),
                }
            )


def interests(
    manifests: Iterable[ArtifactManifest], *, root: Path
) -> Iterator[InterestDeclaration]:
    for manifest in manifests:
        if manifest.family != "diputados" or manifest.dataset != "docacteco":
            continue
        for raw in _records(manifest, root):
            row = interest_row(raw, source_sha256=manifest.sha256, snapshot_date=manifest.run_date)
            yield InterestDeclaration.model_validate(
                {
                    **row,
                    "declaration_id": row.get("interest_id") or stable_id(manifest.sha256, row),
                    "deputy_id": row.get("person_id"),
                    "description": row.get("description") or row.get("activity"),
                    "amount_eur": row.get("amount_eur"),
                    "source": _source(manifest),
                }
            )


def profiles(manifests: Iterable[ArtifactManifest], *, root: Path) -> Iterator[DeputyProfile]:
    for manifest in manifests:
        if manifest.family != "diputados" or manifest.dataset != "DeputyProfile":
            continue
        parsed = parse_visible_html(_content(manifest, root), base_url=manifest.source_url)
        row = deputy_profile_row(
            visible_text=parsed.visible_text,
            source_url=manifest.source_url,
            source_sha256=manifest.sha256,
            snapshot_date=manifest.run_date,
        )
        yield DeputyProfile.model_validate(
            {
                **row,
                "deputy_id": row["person_id"],
                "profile_url": manifest.source_url,
                "source": _source(manifest, method="official_html"),
            }
        )


def initiatives(manifests: Iterable[ArtifactManifest], *, root: Path) -> Iterator[Initiative]:
    for manifest in manifests:
        if manifest.family not in {"iniciativas", "initiative_recovery"}:
            continue
        for raw in _records(manifest, root):
            row = initiative_row(
                raw,
                source_sha256=manifest.sha256,
                snapshot_date=manifest.run_date,
                source_dataset=manifest.dataset,
            )
            yield Initiative.model_validate(
                {
                    **row,
                    "title": row.get("subject") or "",
                    "initiative_type": row.get("type"),
                    "source": _source(manifest),
                }
            )


def interventions(manifests: Iterable[ArtifactManifest], *, root: Path) -> Iterator[Intervention]:
    for manifest in manifests:
        if manifest.family != "intervenciones" or manifest.format not in {"json", "csv"}:
            continue
        for ordinal, raw in enumerate(_records(manifest, root)):
            row = intervention_row(
                raw,
                source_sha256=manifest.sha256,
                snapshot_date=manifest.run_date,
                source_record_ordinal=ordinal,
            )
            yield Intervention.model_validate(
                {
                    **row,
                    "title": row.get("initiative_subject"),
                    "speaker": row.get("speaker_name"),
                    "source": _source(manifest),
                }
            )


def votes(
    manifests: Iterable[ArtifactManifest], *, root: Path
) -> Iterator[VoteEvent | VoteItem | NominalVote]:
    for manifest in manifests:
        if manifest.family != "votaciones" or manifest.dataset != "Votacion":
            continue
        payload = json.loads(_content(manifest, root).decode("utf-8-sig"))
        event, nominal_rows, item_rows = vote_rows(
            payload,
            source_sha256=manifest.sha256,
            snapshot_date=manifest.run_date,
            legislature=manifest.legislature,
            source_url=manifest.source_url,
        )
        source = _source(manifest, method="official_json")
        yield VoteEvent.model_validate(
            {
                **event,
                "vote_id": event["vote_event_id"],
                "session": event.get("session_number"),
                "source": source,
            }
        )
        for row in item_rows:
            yield VoteItem.model_validate(
                {**row, "vote_id": row["vote_event_id"], "result": None, "source": source}
            )
        for row in nominal_rows:
            yield NominalVote.model_validate(
                {
                    **row,
                    "vote_id": row["vote_event_id"],
                    "position": row.get("vote") or "unknown",
                    "source": source,
                }
            )


def organs(
    manifests: Iterable[ArtifactManifest], *, root: Path
) -> Iterator[Organ | OrganMembership]:
    for manifest in manifests:
        source = _source(manifest)
        if manifest.family == "organos" and manifest.dataset in {
            "OrganMemberships",
            "OrganMembershipsAjax",
        }:
            payload = json.loads(_content(manifest, root).decode("utf-8-sig"))
            for row in organ_membership_rows(
                payload,
                source_url=manifest.source_url,
                source_sha256=manifest.sha256,
                snapshot_date=manifest.run_date,
            ):
                yield OrganMembership.model_validate(
                    {**row, "deputy_id": row.get("person_id"), "source": source}
                )
        elif manifest.family == "organos" and manifest.format == "html":
            parsed = parse_visible_html(_content(manifest, root), base_url=manifest.source_url)
            for row in organ_rows_from_links(
                links=parsed.links,
                snapshot_date=manifest.run_date,
            ):
                yield Organ.model_validate({**row, "source": source})


def documents(manifests: Iterable[ArtifactManifest], *, root: Path) -> Iterator[DocumentAsset]:
    for manifest in manifests:
        source = _source(manifest)
        rows: list[dict[str, Any]] = []
        if manifest.format in {"json", "csv"}:
            rows = document_asset_rows_from_records(
                _records(manifest, root),
                family=manifest.family,
                dataset=manifest.dataset,
                snapshot_date=manifest.run_date,
            )
        elif manifest.format == "html":
            parsed = parse_visible_html(_content(manifest, root), base_url=manifest.source_url)
            rows = document_asset_rows_from_links(
                links=parsed.links,
                family=manifest.family,
                dataset=manifest.dataset,
                entity_id=manifest.snapshot_token or manifest.sha256,
                snapshot_date=manifest.run_date,
            )
        for row in rows:
            yield DocumentAsset.model_validate({**row, "source": source})


def discover_document_assets(
    *, root: Path, manifest: BronzeManifest, mime_types: set[str] | None = None
) -> list[dict[str, Any]]:
    selected = mime_types or {"application/pdf"}
    return [
        item.model_dump(mode="json")
        for item in documents((public_manifest(manifest),), root=root)
        if item.mime_type in selected
    ]


def manifest_dict(manifest: BronzeManifest) -> dict[str, Any]:
    """Preserve the exact legacy shape for foundry compatibility."""

    return asdict(manifest)

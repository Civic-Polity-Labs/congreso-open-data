"""Public typed normalization over immutable Bronze artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from congreso_open_data.documents import pdf_document_text_row
from congreso_open_data.extractors.initiatives import initiative_empty_scope_is_expected
from congreso_open_data.html import parse_visible_html
from congreso_open_data.interventions import speech_block_rows as intervention_speech_block_rows
from congreso_open_data.models import (
    ArtifactManifest,
    Deputy,
    DeputyProfile,
    DocumentAsset,
    DocumentText,
    ExtractionEvidence,
    FinancialDocument,
    Initiative,
    InterestDeclaration,
    Intervention,
    InterventionOccurrence,
    NominalVote,
    Organ,
    OrganMembership,
    SalaryEntitlement,
    SourceRef,
    SpeechBlock,
    VoteEvent,
    VoteItem,
)
from congreso_open_data.normalization import iter_records, load_records, stable_id
from congreso_open_data.protocols import ensure_bounded_file
from congreso_open_data.storage import BronzeManifest
from congreso_open_data.transforms import (
    approved_law_rows_from_payload,
    deputy_financial_document_rows_from_profile,
    deputy_profile_row,
    deputy_row,
    document_asset_rows_from_links,
    document_asset_rows_from_records,
    historical_initiative_rows_from_list_payload,
    initiative_row,
    interest_row,
    intervention_row,
    organ_membership_rows,
    organ_rows_from_links,
    salary_rows_from_text,
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
        request_method=manifest.request_method,
        request_parameters=manifest.request_parameters_json,
        request_parameters_sha256=manifest.request_parameters_sha256,
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
        request_method=manifest.request_method,
        request_parameters_json=(
            manifest.request_parameters
            if isinstance(manifest.request_parameters, str)
            else json.dumps(manifest.request_parameters, ensure_ascii=False, sort_keys=True)
            if manifest.request_parameters
            else None
        ),
        request_parameters_sha256=manifest.request_parameters_sha256,
    )


def _artifact_path(manifest: ArtifactManifest, root: Path) -> Path:
    path = Path(manifest.payload_path)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"Bronze artifact not found: {path}")
    return path


def _content(
    manifest: ArtifactManifest,
    root: Path,
    *,
    max_bytes: int = 256 * 1024 * 1024,
) -> bytes:
    return ensure_bounded_file(_artifact_path(manifest, root), max_bytes=max_bytes)


def _records(manifest: ArtifactManifest, root: Path) -> Iterable[dict[str, Any]]:
    if manifest.family in {"iniciativas", "initiative_recovery"}:
        return records_for_manifest_content(
            _content(manifest, root, max_bytes=64 * 1024 * 1024),
            manifest=manifest,
        )
    return iter_records(_artifact_path(manifest, root), manifest.format)


def records_for_manifest_content(
    content: bytes,
    *,
    manifest: ArtifactManifest,
) -> list[dict[str, Any]]:
    """Decode one bounded tabular artifact using its dataset-specific contract."""

    if manifest.family == "initiative_recovery" and manifest.format == "json":
        payload = json.loads(content.decode("utf-8-sig"))
        if (
            isinstance(payload, dict)
            and payload.get("provenance_kind") == "verified_multi_source_derivative"
        ):
            record = payload.get("record")
            lineage = payload.get("field_lineage")
            if not isinstance(record, dict) or not isinstance(lineage, dict) or not lineage:
                raise ValueError("Verified initiative recovery derivative is incomplete")
            return [record]
        if isinstance(payload, dict) and "lista_iniciativas" in payload:
            return historical_initiative_rows_from_list_payload(
                payload,
                source_dataset="GeneralInitiatives",
            )
        raise ValueError("Unsupported initiative recovery payload")
    if manifest.family == "iniciativas" and manifest.format == "json":
        payload = json.loads(content.decode("utf-8-sig"))
        if payload == {}:
            if initiative_empty_scope_is_expected(manifest.dataset, manifest.legislature):
                return []
            raise ValueError("Unexpected empty historical initiative payload")
        if isinstance(payload, dict) and "lista_iniciativas" in payload:
            return historical_initiative_rows_from_list_payload(
                payload,
                source_dataset=manifest.dataset,
            )
        if (
            isinstance(payload, dict)
            and "data" in payload
            and manifest.dataset == "IniciativasLegislativasAprobadas"
        ):
            return approved_law_rows_from_payload(
                payload,
                default_year=_source_query_param(
                    manifest.source_url,
                    "_iniciativasLegislativasAprobadas_anyoSelec",
                ),
            )
    return load_records(content, manifest.format)


def _source_query_param(url: str, key: str) -> str | None:
    values = parse_qs(urlparse(url).query).get(key)
    return values[0] if values else None


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


def financial_documents(
    manifests: Iterable[ArtifactManifest], *, root: Path
) -> Iterator[FinancialDocument]:
    for manifest in manifests:
        if manifest.family != "diputados" or manifest.dataset != "DeputyProfile":
            continue
        parsed = parse_visible_html(_content(manifest, root), base_url=manifest.source_url)
        profile = deputy_profile_row(
            visible_text=parsed.visible_text,
            source_url=manifest.source_url,
            source_sha256=manifest.sha256,
            snapshot_date=manifest.run_date,
        )
        for row in deputy_financial_document_rows_from_profile(
            links=parsed.links,
            profile=profile,
            snapshot_date=manifest.run_date,
        ):
            yield FinancialDocument.model_validate(
                {
                    **row,
                    "document_id": row["financial_document_id"],
                    "deputy_id": row.get("person_id"),
                    "url": row["source_url"],
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


def intervention_occurrences(
    manifests: Iterable[ArtifactManifest], *, root: Path
) -> Iterator[InterventionOccurrence]:
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
            yield InterventionOccurrence.model_validate(
                {
                    **row,
                    "occurrence_id": stable_id(manifest.sha256, ordinal),
                    "date": row.get("session_date"),
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
                "session": (
                    str(event["session_number"])
                    if event.get("session_number") is not None
                    else None
                ),
                "vote_number": (
                    str(event["vote_number"]) if event.get("vote_number") is not None else None
                ),
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


def document_texts(
    manifests: Iterable[ArtifactManifest],
    *,
    root: Path,
    use_ocr: bool = False,
) -> Iterator[DocumentText]:
    for manifest in manifests:
        if manifest.format != "pdf" or manifest.family not in {
            "documents",
            "intervention_documents",
        }:
            continue
        row = pdf_document_text_row(
            _content(manifest, root),
            source_url=manifest.source_url,
            source_sha256=manifest.sha256,
            snapshot_date=manifest.run_date,
            use_ocr=use_ocr,
        )
        page_texts = row.get("page_texts") or []
        page_methods = row.get("_page_methods") or row.get("page_methods") or []
        page_confidences = row.get("_page_confidences") or row.get("page_confidences") or []
        evidence = tuple(
            ExtractionEvidence(
                text=str(text),
                page=index,
                confidence=(
                    float(page_confidences[index - 1])
                    if index <= len(page_confidences) and page_confidences[index - 1] is not None
                    else None
                ),
                backend=(
                    str(page_methods[index - 1])
                    if index <= len(page_methods)
                    else str(row["extraction_method"])
                ),
                model=str(row["model_name"]),
                version="1.0.0",
            )
            for index, text in enumerate(page_texts, start=1)
        )
        yield DocumentText.model_validate(
            {
                **row,
                "document_id": row["document_sha256"],
                "model": row["model_name"],
                "confidence": (
                    sum(item.confidence for item in evidence if item.confidence is not None)
                    / sum(item.confidence is not None for item in evidence)
                    if any(item.confidence is not None for item in evidence)
                    else None
                ),
                "evidence": evidence,
                "source": _source(
                    manifest,
                    method=str(row["extraction_method"]),
                ),
            }
        )


def speech_blocks(manifests: Iterable[ArtifactManifest], *, root: Path) -> Iterator[SpeechBlock]:
    for manifest in manifests:
        if (
            manifest.family != "intervention_documents"
            or manifest.dataset != "InterventionFullText"
            or manifest.format != "html"
        ):
            continue
        parsed = parse_visible_html(_content(manifest, root), base_url=manifest.source_url)
        if parsed.content_selector != ".textoIntegro" or not parsed.visible_text.strip():
            raise ValueError(
                "Official intervention HTML is missing its unique .textoIntegro transcript"
            )
        for row in intervention_speech_block_rows(
            visible_text=parsed.visible_text,
            source_url=manifest.source_url,
            source_sha256=manifest.sha256,
            snapshot_date=manifest.run_date,
            legislature=manifest.legislature,
            source_kind="html",
            extraction_method="official_html_transcript",
        ):
            yield SpeechBlock.model_validate(
                {
                    **row,
                    "speaker": row.get("normalized_speaker"),
                    "sequence": row["ordinal"],
                    "source": _source(manifest, method="official_html_transcript"),
                }
            )


def salary_entitlements(
    manifests: Iterable[ArtifactManifest], *, root: Path
) -> Iterator[SalaryEntitlement]:
    for manifest in manifests:
        if (
            manifest.family != "transparencia"
            or manifest.dataset != "RetribucionesCargosMesa"
            or manifest.format != "html"
        ):
            continue
        parsed = parse_visible_html(_content(manifest, root), base_url=manifest.source_url)
        for row in salary_rows_from_text(
            parsed.visible_text,
            source_url=manifest.source_url,
            snapshot_date=manifest.run_date,
        ):
            yield SalaryEntitlement.model_validate(
                {
                    **row,
                    "entitlement_id": row["salary_entitlement_id"],
                    "label": row["concept"],
                    "effective_date": row.get("valid_from"),
                    "source": _source(manifest, method="official_html"),
                }
            )


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

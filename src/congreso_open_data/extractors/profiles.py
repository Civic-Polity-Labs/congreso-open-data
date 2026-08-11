from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from congreso_open_data.batch_extract import extract_resource_batch
from congreso_open_data.catalog import DatasetResource
from congreso_open_data.durable_io import append_jsonl_durably, write_json_atomically
from congreso_open_data.extractors.documents import discover_document_resources_from_manifest
from congreso_open_data.extractors.opendata import extract_resource
from congreso_open_data.http import CongresoHttpClient
from congreso_open_data.storage import (
    BronzeManifest,
    bronze_payload_is_valid,
    canonical_request_parameters,
    content_type_matches_format,
)

PROFILE_SOURCE_VERSION = "leg15-deputy-profile-sources-v2-frozen-search-linked-documents"

PROFILE_SEARCH_URL = (
    "https://www.congreso.es/es/busqueda-de-diputados"
    "?p_p_id=diputadomodule"
    "&p_p_lifecycle=2"
    "&p_p_state=normal"
    "&p_p_mode=view"
    "&p_p_resource_id=searchDiputados"
    "&p_p_cacheability=cacheLevelPage"
    "&_diputadomodule_idLegislatura={legislature_number}"
    "&_diputadomodule_mostrarFicha=false"
)
PROFILE_URL = (
    "https://www.congreso.es/es/busqueda-de-diputados"
    "?p_p_id=diputadomodule"
    "&p_p_lifecycle=0"
    "&p_p_state=normal"
    "&p_p_mode=view"
    "&_diputadomodule_mostrarFicha=true"
    "&codParlamentario={code}"
    "&idLegislatura={legislature_roman}"
)


def deputy_profile_search_rows(
    *,
    client: CongresoHttpClient | None = None,
    legislature_number: int = 15,
) -> tuple[dict[str, Any], ...]:
    """Return the bounded official identity index for one legislature."""

    transport = client or CongresoHttpClient()
    result = transport.post(
        PROFILE_SEARCH_URL.format(legislature_number=legislature_number),
        data=_profile_search_payload(legislature_number),
    )
    payload = json.loads(result.content.decode("utf-8-sig"))
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Deputy profile search must contain a data list")
    rows: list[dict[str, Any]] = []
    for ordinal, row in enumerate(payload["data"]):
        if not isinstance(row, dict):
            raise ValueError(f"Malformed deputy search row at ordinal {ordinal}")
        rows.append(dict(row))
    return tuple(rows)


def discover_deputy_profile_resources(
    *,
    client: CongresoHttpClient | None = None,
    legislature_number: int = 15,
) -> list[DatasetResource]:
    rows = deputy_profile_search_rows(
        client=client,
        legislature_number=legislature_number,
    )
    return deputy_profile_resources_from_payload(
        payload={"data": list(rows)},
        legislature_number=legislature_number,
    )


def deputy_profile_resources_from_payload(
    *,
    payload: Any,
    legislature_number: int = 15,
) -> list[DatasetResource]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Deputy profile search must contain a data list")
    resources: dict[str, DatasetResource] = {}
    for ordinal, row in enumerate(payload["data"]):
        if not isinstance(row, dict):
            raise ValueError(f"Malformed deputy search row at ordinal {ordinal}")
        code = str(row.get("codParlamentario") or "").strip()
        if not code or not code.isdigit():
            raise ValueError(f"Missing deputy code at search ordinal {ordinal}")
        raw_legislature_number = row.get("idLegislatura")
        try:
            row_legislature_number = (
                int(raw_legislature_number)
                if raw_legislature_number is not None
                else legislature_number
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid legislature at deputy search ordinal {ordinal}") from error
        if legislature_number >= 0 and row_legislature_number != legislature_number:
            raise ValueError(
                "Deputy search returned a row outside the requested legislature: "
                f"requested={legislature_number} row={row_legislature_number} code={code}"
            )
        legislature_code = _profile_legislature_code(row_legislature_number)
        url = PROFILE_URL.format(code=code, legislature_roman=legislature_code)
        if url in resources:
            raise ValueError(f"Duplicate deputy profile in official search: {code}")
        resources[url] = DatasetResource(
            family="diputados",
            dataset="DeputyProfile",
            format="html",
            url=url,
            snapshot_token=f"{row_legislature_number}_{code}",
            legislature=legislature_code,
        )
    return list(resources.values())


def deputy_profile_search_resource(legislature_number: int = 15) -> DatasetResource:
    return DatasetResource(
        family="diputados",
        dataset="DeputyProfileSearch",
        format="json",
        url=PROFILE_SEARCH_URL.format(legislature_number=legislature_number),
        snapshot_token=f"Leg{legislature_number}-profile-search",
        legislature=f"Leg.{legislature_number}",
        post_data=_profile_search_payload(legislature_number),
    )


def extract_leg15_deputy_profile_sources(
    *,
    run_date: str,
    lake_root: Path,
    workers: int = 2,
    request_interval_seconds: float = 0.5,
    throttle_backoff_seconds: float = 60.0,
    resume: bool = True,
) -> dict[str, Any]:
    lake_root = lake_root.resolve()
    if not lake_root.is_absolute():
        raise ValueError("lake_root must be absolute")
    plans = lake_root / "plans"
    search_index = plans / f"deputy-profiles-Leg.15-{run_date}.search.json"
    profile_index = plans / f"deputy-profiles-Leg.15-{run_date}.profiles.json"
    document_index = plans / f"deputy-profiles-Leg.15-{run_date}.documents.json"
    resource_plan = plans / f"deputy-profiles-Leg.15-{run_date}.resources.json"
    document_plan = plans / f"deputy-profiles-Leg.15-{run_date}.document-resources.json"
    event_log = plans / f"deputy-profiles-Leg.15-{run_date}.progress.jsonl"
    audit_path = lake_root / "audit" / f"deputy-profiles-Leg.15-{run_date}.json"

    def progress(payload: dict[str, Any]) -> None:
        append_jsonl_durably(event_log, payload)

    search_result = extract_resource_batch(
        resources=[deputy_profile_search_resource()],
        run_date=run_date,
        output_root=lake_root,
        manifest_index_path=search_index,
        max_workers=1,
        resume=resume,
        continue_on_error=False,
        progress=progress,
        request_interval_seconds=request_interval_seconds,
        throttle_backoff_seconds=throttle_backoff_seconds,
    )
    search_manifest = search_result.manifests[0]
    payload = json.loads((lake_root / search_manifest.bronze_path).read_text("utf-8-sig"))
    resources = deputy_profile_resources_from_payload(
        payload=payload,
        legislature_number=15,
    )
    if not resources:
        raise ValueError("Official Legislature XV deputy profile search is empty")
    write_json_atomically(
        resource_plan,
        {
            "version": PROFILE_SOURCE_VERSION,
            "run_date": run_date,
            "legislature": "Leg.15",
            "resources": [asdict(resource) for resource in resources],
        },
    )
    profile_result = extract_resource_batch(
        resources=resources,
        run_date=run_date,
        output_root=lake_root,
        manifest_index_path=profile_index,
        max_workers=workers,
        resume=resume,
        continue_on_error=False,
        progress=progress,
        request_interval_seconds=request_interval_seconds,
        throttle_backoff_seconds=throttle_backoff_seconds,
    )
    document_resources_by_url: dict[str, DatasetResource] = {}
    document_url_by_token: dict[str, str] = {}
    document_profile_tokens: dict[str, set[str]] = {}
    for manifest in profile_result.manifests:
        for resource in discover_document_resources_from_manifest(
            lake_root=lake_root,
            manifest=manifest,
            mime_types={"application/pdf"},
        ):
            scoped_resource = replace(resource, legislature="XV")
            existing = document_resources_by_url.get(scoped_resource.url)
            if existing is not None and existing.snapshot_token != scoped_resource.snapshot_token:
                raise ValueError(
                    "One official document URL produced conflicting document identities: "
                    f"{scoped_resource.url}"
                )
            token = str(scoped_resource.snapshot_token)
            previous_url = document_url_by_token.get(token)
            if previous_url is not None and previous_url != scoped_resource.url:
                raise ValueError(
                    f"One official document identity produced conflicting URLs: {token}"
                )
            document_resources_by_url.setdefault(scoped_resource.url, scoped_resource)
            document_url_by_token[token] = scoped_resource.url
            document_profile_tokens.setdefault(scoped_resource.url, set()).add(
                str(manifest.snapshot_token)
            )
    document_resources = list(document_resources_by_url.values())
    write_json_atomically(
        document_plan,
        {
            "version": PROFILE_SOURCE_VERSION,
            "run_date": run_date,
            "legislature": "Leg.15",
            "resources": [
                {
                    **asdict(resource),
                    "source_profile_tokens": sorted(
                        document_profile_tokens.get(resource.url, set())
                    ),
                }
                for resource in document_resources
            ],
        },
    )
    document_result = extract_resource_batch(
        resources=document_resources,
        run_date=run_date,
        output_root=lake_root,
        manifest_index_path=document_index,
        max_workers=workers,
        resume=resume,
        continue_on_error=False,
        progress=progress,
        request_interval_seconds=request_interval_seconds,
        throttle_backoff_seconds=throttle_backoff_seconds,
    )
    audit = audit_leg15_deputy_profile_sources(
        lake_root=lake_root,
        run_date=run_date,
        search_manifest=search_manifest,
        planned_resources=resources,
        profile_manifests=profile_result.manifests,
        planned_document_resources=document_resources,
        document_manifests=document_result.manifests,
    )
    write_json_atomically(audit_path, audit)
    append_jsonl_durably(event_log, {"event": "terminal_audit", **audit})
    if not audit["passed"]:
        raise ValueError(f"Legislature XV deputy profile audit failed: {audit_path}")
    return {**audit, "audit_path": str(audit_path)}


def audit_leg15_deputy_profile_sources(
    *,
    lake_root: Path,
    run_date: str,
    search_manifest: BronzeManifest,
    planned_resources: list[DatasetResource],
    profile_manifests: tuple[BronzeManifest, ...],
    planned_document_resources: list[DatasetResource],
    document_manifests: tuple[BronzeManifest, ...],
) -> dict[str, Any]:
    expected_search = deputy_profile_search_resource()
    expected = {_profile_resource_identity(resource) for resource in planned_resources}
    actual = {_profile_manifest_identity(manifest) for manifest in profile_manifests}
    expected_documents = {
        _profile_resource_identity(resource) for resource in planned_document_resources
    }
    actual_documents = {_profile_manifest_identity(manifest) for manifest in document_manifests}
    all_manifests = (search_manifest, *profile_manifests, *document_manifests)
    invalid_payloads = 0
    hash_mismatches = 0
    mime_mismatches = 0
    invalid_statuses = 0
    invalid_lineage = 0
    invalid_profile_html = 0
    for manifest in all_manifests:
        path = lake_root / manifest.bronze_path
        if not bronze_payload_is_valid(root=lake_root, manifest=manifest):
            invalid_payloads += 1
        if _sha256_path(path) != manifest.sha256:
            hash_mismatches += 1
        if not content_type_matches_format(manifest.content_type, manifest.format):
            mime_mismatches += 1
        if not 200 <= manifest.status_code < 300:
            invalid_statuses += 1
        if manifest.dataset == "DeputyProfileSearch":
            valid_lineage = bool(
                manifest.request_method == "POST"
                and manifest.request_parameters_json
                and manifest.request_parameters_sha256
                and hashlib.sha256(manifest.request_parameters_json.encode()).hexdigest()
                == manifest.request_parameters_sha256
            )
        else:
            valid_lineage = (
                manifest.request_method == "GET"
                and manifest.request_parameters_json is None
                and manifest.request_parameters_sha256 is None
            )
        if not valid_lineage:
            invalid_lineage += 1
        if manifest.dataset == "DeputyProfile" and not _profile_html_matches_contract(
            path=path,
            manifest=manifest,
        ):
            invalid_profile_html += 1
    duplicate_profiles = len(profile_manifests) - len(actual)
    duplicate_documents = len(document_manifests) - len(actual_documents)
    checks = {
        "search_frozen": (
            search_manifest.family == expected_search.family
            and search_manifest.dataset == expected_search.dataset
            and search_manifest.format == expected_search.format
            and (search_manifest.requested_url or search_manifest.source_url) == expected_search.url
            and search_manifest.snapshot_token == expected_search.snapshot_token
            and search_manifest.legislature == expected_search.legislature
            and search_manifest.request_parameters_json
            == canonical_request_parameters(expected_search)
        ),
        "profiles_reconciled": (
            expected == actual
            and len(profile_manifests) == len(expected)
            and duplicate_profiles == 0
        ),
        "documents_reconciled": (
            bool(planned_document_resources)
            and expected_documents == actual_documents
            and len(document_manifests) == len(expected_documents)
            and duplicate_documents == 0
        ),
        "payload_contracts": invalid_payloads == 0,
        "deep_hashes": hash_mismatches == 0,
        "mime_types": mime_mismatches == 0,
        "http_statuses": invalid_statuses == 0,
        "request_lineage": invalid_lineage == 0,
        "profile_html_structure": invalid_profile_html == 0,
        "legislature_scope": all(
            manifest.legislature == "XV" for manifest in (*profile_manifests, *document_manifests)
        ),
    }
    return {
        "version": PROFILE_SOURCE_VERSION,
        "run_date": run_date,
        "legislature": "Leg.15",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "planned_profiles": len(planned_resources),
            "profile_manifests": len(profile_manifests),
            "missing_profiles": len(expected - actual),
            "unexpected_profiles": len(actual - expected),
            "duplicate_profiles": duplicate_profiles,
            "planned_documents": len(planned_document_resources),
            "document_manifests": len(document_manifests),
            "missing_documents": len(expected_documents - actual_documents),
            "unexpected_documents": len(actual_documents - expected_documents),
            "duplicate_documents": duplicate_documents,
            "invalid_payloads": invalid_payloads,
            "hash_mismatches": hash_mismatches,
            "mime_mismatches": mime_mismatches,
            "invalid_statuses": invalid_statuses,
            "invalid_lineage": invalid_lineage,
            "invalid_profile_html": invalid_profile_html,
        },
    }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_resource_identity(resource: DatasetResource) -> tuple[str, ...]:
    return (
        resource.family,
        resource.dataset,
        resource.format,
        resource.url,
        str(resource.snapshot_token),
        str(resource.legislature),
        "POST" if resource.post_data is not None else "GET",
        canonical_request_parameters(resource) or "",
    )


def _profile_manifest_identity(manifest: BronzeManifest) -> tuple[str, ...]:
    return (
        manifest.family,
        manifest.dataset,
        manifest.format,
        manifest.requested_url or manifest.source_url,
        str(manifest.snapshot_token),
        str(manifest.legislature),
        manifest.request_method,
        manifest.request_parameters_json or "",
    )


def _profile_html_matches_contract(*, path: Path, manifest: BronzeManifest) -> bool:
    requested_url = manifest.requested_url or manifest.source_url
    query = parse_qs(urlparse(requested_url).query)
    code_values = query.get("codParlamentario") or []
    legislature_values = query.get("idLegislatura") or []
    if len(code_values) != 1 or len(legislature_values) != 1:
        return False
    if legislature_values[0] != "XV":
        return False
    if str(manifest.snapshot_token) != f"15_{code_values[0]}":
        return False
    content = path.read_text("utf-8-sig", errors="replace").casefold()
    return "ficha personal" in content and "legislatura" in content


def extract_deputy_profile_resources(
    *,
    run_date: str,
    output_root: Path,
    client: CongresoHttpClient | None = None,
    legislature_number: int = 15,
) -> list[BronzeManifest]:
    client = client or CongresoHttpClient()
    return [
        extract_resource(
            resource=resource,
            run_date=run_date,
            output_root=output_root,
            client=client,
        )
        for resource in discover_deputy_profile_resources(
            client=client,
            legislature_number=legislature_number,
        )
    ]


def _profile_search_payload(legislature_number: int) -> dict[str, str]:
    return {
        "_diputadomodule_idLegislatura": str(legislature_number),
        "_diputadomodule_genero": "",
        "_diputadomodule_grupo": "",
        "_diputadomodule_tipo": "2",
        "_diputadomodule_nombre": "",
        "_diputadomodule_apellidos": "",
        "_diputadomodule_formacion": "",
        "_diputadomodule_filtroProvincias": "[]",
        "_diputadomodule_nombreCircunscripcion": "",
    }


def _romanize(value: int) -> str:
    numerals = (
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    )
    result = []
    remaining = value
    for number, roman in numerals:
        while remaining >= number:
            result.append(roman)
            remaining -= number
    return "".join(result)


def _profile_legislature_code(value: int) -> str:
    return "0" if value == 0 else _romanize(value)
